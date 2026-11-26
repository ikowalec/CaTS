import os
import sys
import stat
from threading import Lock
import time
from types import ModuleType

from similarity.workflows import mace_pool


def test_mpi_object_messages_avoid_matched_receive():
    assert os.environ["MPI4PY_RC_RECV_MPROBE"] == "false"


def test_mpi_entry_refuses_split_single_rank_worlds(monkeypatch):
    comm = type(
        "Comm",
        (),
        {"Get_size": lambda self: 1, "Get_rank": lambda self: 0},
    )()
    mpi4py = ModuleType("mpi4py")
    mpi4py.MPI = type("MPI", (), {"COMM_WORLD": comm})
    monkeypatch.setitem(sys.modules, "mpi4py", mpi4py)
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_PROCID", "0")

    try:
        mace_pool.initialize_mpi_worker_pool()
    except RuntimeError as exc:
        assert "Refusing to run duplicate workflows" in str(exc)
    else:
        raise AssertionError("split MPI worlds must not execute the workflow")


def test_existing_checkpoint_database_is_made_owner_writable(tmp_path):
    path = tmp_path / "relaxed.db"
    path.write_text("checkpoint", encoding="utf-8")
    path.chmod(stat.S_IRUSR)

    mace_pool.prepare_writable_database(path)

    assert path.stat().st_mode & stat.S_IWUSR


def test_cuda_pool_uses_all_visible_devices_once(monkeypatch):
    torch = ModuleType("torch")
    torch.cuda = type("Cuda", (), {"device_count": staticmethod(lambda: 2)})()
    calls = []
    calculators = ModuleType("mace.calculators")
    calculators.mace_mp = lambda **kwargs: calls.append(kwargs) or object()
    mace = ModuleType("mace")
    mace.calculators = calculators
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "mace", mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    monkeypatch.setattr(mace_pool, "_CALCULATORS", [])
    monkeypatch.setattr(mace_pool, "_CALCULATOR_KEY", None)

    first = mace_pool.get_mace_calculators(
        model="model", dispersion=False, default_dtype="float64", device="cuda"
    )
    second = mace_pool.get_mace_calculators(
        model="model", dispersion=False, default_dtype="float64", device="cuda"
    )

    assert first is second
    assert [call["device"] for call in calls] == ["cuda:0", "cuda:1"]


def test_task_runner_leases_calculators_and_prints_only_milestones(capsys):
    calculators = [object(), object()]
    active = set()
    used = set()
    lock = Lock()

    def worker(task, calculator):
        with lock:
            assert id(calculator) not in active
            active.add(id(calculator))
            used.add(id(calculator))
        time.sleep(0.002 if task % 2 else 0.004)
        with lock:
            active.remove(id(calculator))
        return task

    results = mace_pool.run_mace_tasks(
        range(20), calculators, worker, label="test", verbose=True
    )

    lines = capsys.readouterr().out.splitlines()
    assert sorted(results) == list(range(20))
    assert used == {id(calculator) for calculator in calculators}
    assert len(lines) == 10
    assert lines[-1].endswith("[test] Progress: 20/20 completed (100%)")


def test_debug_mode_prints_one_evaluation_summary_per_result(monkeypatch, capsys):
    monkeypatch.setenv("MACE_WORKFLOW_DEBUG", "1")

    mace_pool.run_mace_tasks(
        [1, 2],
        [object()],
        lambda task, _calculator: (
            task,
            object(),
            {
                "row_id": task,
                "nsteps": task,
                "force_evaluations": task + 1,
                "elapsed_seconds": 0.5 * task,
            },
        ),
        label="debug",
        verbose=True,
    )

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert "row=1, steps=1, evaluations=2, elapsed=0.5s" in lines[0]
    assert "row=2, steps=2, evaluations=3, elapsed=1.0s" in lines[1]


def test_task_runner_dispatches_to_mpi_rank_and_local_gpu(monkeypatch):
    class Status:
        def __init__(self):
            self.source = 1

        def Get_source(self):
            return self.source

    class Comm:
        def __init__(self):
            self.responses = []
            self.remote_tasks = []
            self.remote_messages = []

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 2

        def send(self, message, dest, tag):
            self.remote_tasks.append(message["task"])
            self.remote_messages.append(message)
            self.responses.append({"ok": True, "result": message["task"] * 10})

        def Iprobe(self, source, tag, status):
            status.source = 1
            return bool(self.responses)

        def recv(self, source, tag):
            return self.responses.pop(0)

    mpi = ModuleType("mpi4py")
    mpi.MPI = type(
        "MPI",
        (),
        {"ANY_SOURCE": -1, "Status": Status},
    )
    comm = Comm()
    monkeypatch.setitem(sys.modules, "mpi4py", mpi)
    monkeypatch.setattr(mace_pool, "_MPI_COMM", comm)

    results = mace_pool.run_mace_tasks(
        [1, 2],
        [object()],
        lambda task, _calculator: task * 10,
        distributed_operation="module:operation",
        distributed_settings={},
        calculator_settings={},
        verbose=False,
    )

    assert sorted(results) == [10, 20]
    assert comm.remote_tasks == [2]

    cpu_results = mace_pool.run_cpu_tasks(
        [3, 4],
        lambda task, _calculator, _settings: task * 10,
        distributed_operation="module:cpu_operation",
        verbose=False,
    )

    assert sorted(cpu_results) == [30, 40]
    assert comm.remote_tasks == [2, 4]
    assert comm.remote_messages[-1]["calculator_settings"] is None
