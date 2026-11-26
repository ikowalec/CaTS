from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import importlib
import io
import json
import os
from pathlib import Path
from queue import Queue
import sqlite3
import stat
from threading import Lock
import time
import traceback
from typing import Callable, Iterable, TypeVar

# Cray MPICH can truncate mpi4py object messages in the matched-receive path.
os.environ["MPI4PY_RC_RECV_MPROBE"] = "false"

import numpy as np


_CALCULATORS = []
_CALCULATOR_KEY = None
_CALCULATOR_LOCK = Lock()
_MPI_COMM = None
_MPI_SHUTDOWN = False

_MPI_TASK_TAG = 4101
_MPI_RESULT_TAG = 4102
_MPI_SHUTDOWN_TAG = 4103

T = TypeVar("T")
R = TypeVar("R")


def _debug_enabled() -> bool:
    return os.environ.get("MACE_WORKFLOW_DEBUG", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _debug_result_details(result) -> str:
    if not isinstance(result, tuple):
        return ""
    row = next((item for item in reversed(result) if isinstance(item, dict)), None)
    if row is None:
        return ""
    details = []
    if "row_id" in row:
        details.append(f"row={row['row_id']}")
    if "nsteps" in row:
        details.append(f"steps={row['nsteps']}")
    if "force_evaluations" in row:
        details.append(f"evaluations={row['force_evaluations']}")
    if "elapsed_seconds" in row:
        details.append(f"elapsed={float(row['elapsed_seconds']):.1f}s")
    return ("; " + ", ".join(details)) if details else ""


def checkpoint_hash(atoms, settings: dict) -> str:
    payload = {
        "symbols": atoms.get_chemical_symbols(),
        "positions": np.asarray(atoms.positions, dtype=float).round(10).tolist(),
        "cell": np.asarray(atoms.cell, dtype=float).round(10).tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
        "settings": settings,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def prepare_writable_database(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(path.parent, os.W_OK):
        raise PermissionError(f"MACE checkpoint directory is not writable: {path.parent}")
    sqlite_files = [
        path,
        Path(f"{path}-journal"),
        Path(f"{path}-shm"),
        Path(f"{path}-wal"),
    ]
    for sqlite_file in sqlite_files:
        if sqlite_file.exists() and not sqlite_file.stat().st_mode & stat.S_IWUSR:
            sqlite_file.chmod(sqlite_file.stat().st_mode | stat.S_IWUSR)
        if sqlite_file.exists() and not os.access(sqlite_file, os.W_OK):
            raise PermissionError(f"MACE checkpoint file is not writable: {sqlite_file}")
    return path


def write_ase_database_atomically(path: str | Path, records) -> Path:
    from ase.db.sqlite import SQLite3Database

    path = prepare_writable_database(path)
    connection = sqlite3.connect(":memory:")
    database = SQLite3Database(str(path), serial=True)
    database.connection = connection
    database.change_count = 0
    try:
        for atoms, data in records:
            database.write(atoms, data={} if data is None else dict(data))
        connection.commit()
        payload = connection.serialize()
    finally:
        connection.close()
    staged_path = path.with_suffix(path.suffix + ".tmp")
    staged_path.write_bytes(payload)
    staged_path.replace(path)
    return path


def _visible_devices(device: str) -> list[str]:
    device = str(device).strip() or "cpu"
    if device != "cuda":
        return [device]

    import torch

    count = int(torch.cuda.device_count())
    return [f"cuda:{index}" for index in range(count)] or ["cuda"]


def get_mace_calculators(
    *,
    model: str,
    dispersion: bool,
    default_dtype: str,
    device: str,
    enable_cueq: bool = True,
):
    global _CALCULATORS, _CALCULATOR_KEY

    devices = _visible_devices(device)
    key = (
        str(model),
        bool(dispersion),
        str(default_dtype),
        tuple(devices),
        bool(enable_cueq),
    )
    with _CALCULATOR_LOCK:
        if _CALCULATOR_KEY == key:
            return _CALCULATORS

        from mace.calculators import mace_mp

        calculators = []
        for calculator_device in devices:
            kwargs = {
                "model": str(model),
                "dispersion": bool(dispersion),
                "default_dtype": str(default_dtype),
                "device": calculator_device,
                "enable_cueq": bool(enable_cueq),
            }
            try:
                calculators.append(mace_mp(**kwargs))
            except TypeError as exc:
                if "enable_cueq" not in str(exc):
                    raise
                kwargs.pop("enable_cueq")
                calculators.append(mace_mp(**kwargs))
        _CALCULATORS = calculators
        _CALCULATOR_KEY = key
        return _CALCULATORS


def initialize_mpi_worker_pool() -> bool:
    global _MPI_COMM

    slurm_tasks = int(os.environ.get("SLURM_NTASKS", "1") or "1")
    slurm_rank = int(os.environ.get("SLURM_PROCID", "0") or "0")
    try:
        from mpi4py import MPI
    except ImportError as exc:
        if slurm_tasks > 1:
            raise RuntimeError(
                "mpi4py is required for a multi-task Slurm MACE launch."
            ) from exc
        return True

    comm = MPI.COMM_WORLD
    if slurm_tasks > 1 and comm.Get_size() != slurm_tasks:
        raise RuntimeError(
            "Slurm launched "
            f"{slurm_tasks} tasks, but mpi4py COMM_WORLD contains {comm.Get_size()} rank(s) "
            f"on SLURM_PROCID={slurm_rank}. Check that cray-python/mpi4py is using "
            "Cray MPICH and that the launch uses srun. Refusing to run duplicate workflows."
        )
    if comm.Get_size() <= 1:
        return True
    _MPI_COMM = comm
    from ase import parallel as ase_parallel
    ase_parallel.world = ase_parallel.DummyMPI()
    if comm.Get_rank() == 0:
        print(
            f"[MACE MPI] Coordinator active with {comm.Get_size()} ranks "
            f"across {os.environ.get('SLURM_JOB_NUM_NODES', '?')} node(s)."
        )
        atexit.register(shutdown_mpi_workers)
        return True

    while True:
        status = MPI.Status()
        message = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == _MPI_SHUTDOWN_TAG:
            return False
        worker_output = io.StringIO()
        try:
            with redirect_stdout(worker_output), redirect_stderr(worker_output):
                module_name, function_name = message["operation"].split(":", 1)
                operation = getattr(importlib.import_module(module_name), function_name)
                calculator_settings = message.get("calculator_settings")
                calculator = (
                    get_mace_calculators(**calculator_settings)[0]
                    if calculator_settings is not None
                    else None
                )
                result = operation(
                    message["task"], calculator, message["worker_settings"]
                )
            response = {"ok": True, "result": result}
        except BaseException as exc:
            response = {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "worker_output": worker_output.getvalue(),
            }
        comm.send(response, dest=0, tag=_MPI_RESULT_TAG)


def shutdown_mpi_workers() -> None:
    global _MPI_SHUTDOWN

    if _MPI_SHUTDOWN or _MPI_COMM is None or _MPI_COMM.Get_rank() != 0:
        return
    _MPI_SHUTDOWN = True
    for rank in range(1, _MPI_COMM.Get_size()):
        _MPI_COMM.send(None, dest=rank, tag=_MPI_SHUTDOWN_TAG)


def _run_distributed_mace_tasks(
    tasks,
    calculators,
    worker,
    *,
    operation: str,
    worker_settings: dict,
    calculator_settings: dict | None,
    on_result,
    completed: int,
    total: int,
    label: str,
    verbose: bool,
):
    from mpi4py import MPI

    pending = deque(tasks)
    results = []
    active_ranks = set()
    next_fraction = min(10, (int(completed) * 10 // max(total, 1)) + 1)
    debug = _debug_enabled()

    def report(result):
        nonlocal completed, next_fraction
        if verbose and debug:
            next_completed = completed + 1
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[{label}] Completed {next_completed}/{total} "
                f"({round(next_completed * 100 / max(total, 1))}%)"
                f"{_debug_result_details(result)}"
            )
        if on_result is not None:
            on_result(result)
        results.append(result)
        completed += 1
        fraction = min(10, completed * 10 // max(total, 1))
        if verbose and not debug and (fraction >= next_fraction or completed == total):
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[{label}] Progress: {completed}/{total} completed "
                f"({min(100, fraction * 10)}%)"
            )
            next_fraction = fraction + 1

    def send_next(rank: int) -> bool:
        if not pending:
            return False
        _MPI_COMM.send(
            {
                "operation": operation,
                "task": pending.popleft(),
                "worker_settings": worker_settings,
                "calculator_settings": calculator_settings,
            },
            dest=rank,
            tag=_MPI_TASK_TAG,
        )
        active_ranks.add(rank)
        return True

    local_executor = ThreadPoolExecutor(max_workers=1)
    local_future = None

    def start_local():
        nonlocal local_future
        if local_future is None and pending:
            task = pending.popleft()
            local_future = local_executor.submit(worker, task, calculators[0])

    start_local()
    for rank in range(1, _MPI_COMM.Get_size()):
        send_next(rank)
    try:
        while active_ranks or local_future is not None:
            progressed = False
            if local_future is not None and local_future.done():
                report(local_future.result())
                local_future = None
                start_local()
                progressed = True

            status = MPI.Status()
            while _MPI_COMM.Iprobe(
                source=MPI.ANY_SOURCE, tag=_MPI_RESULT_TAG, status=status
            ):
                rank = status.Get_source()
                response = _MPI_COMM.recv(source=rank, tag=_MPI_RESULT_TAG)
                active_ranks.discard(rank)
                if not response["ok"]:
                    raise RuntimeError(
                        f"MPI worker rank {rank} failed: {response['error']}\n"
                        f"{response['traceback']}\n"
                        f"{response.get('worker_output', '')}"
                    )
                report(response["result"])
                send_next(rank)
                progressed = True
            if not progressed:
                time.sleep(0.01)
    except BaseException as exc:
        if verbose:
            print(f"[{label}] Stopped after {completed}/{total} completed: {exc}")
        raise
    finally:
        local_executor.shutdown(wait=True, cancel_futures=True)
    return results


def run_cpu_tasks(
    tasks: Iterable[T],
    worker: Callable[[T, None, dict], R],
    *,
    distributed_operation: str,
    distributed_settings: dict | None = None,
    label: str = "CPU work",
    verbose: bool = True,
) -> list[R]:
    tasks = list(tasks)
    settings = distributed_settings or {}
    if not tasks:
        return []
    if _MPI_COMM is not None and _MPI_COMM.Get_rank() == 0 and _MPI_COMM.Get_size() > 1:
        return _run_distributed_mace_tasks(
            tasks,
            [None],
            lambda task, _calculator: worker(task, None, settings),
            operation=distributed_operation,
            worker_settings=settings,
            calculator_settings=None,
            on_result=None,
            completed=0,
            total=len(tasks),
            label=label,
            verbose=verbose,
        )
    results = []
    next_fraction = 1
    for completed, task in enumerate(tasks, start=1):
        results.append(worker(task, None, settings))
        fraction = min(10, completed * 10 // len(tasks))
        if verbose and (fraction >= next_fraction or completed == len(tasks)):
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[{label}] Progress: {completed}/{len(tasks)} completed "
                f"({min(100, fraction * 10)}%)"
            )
            next_fraction = fraction + 1
    return results


def run_mace_tasks(
    tasks: Iterable[T],
    calculators,
    worker: Callable[[T, object], R],
    *,
    on_result: Callable[[R], None] | None = None,
    completed: int = 0,
    total: int | None = None,
    label: str = "MACE relaxation",
    verbose: bool = True,
    distributed_operation: str | None = None,
    distributed_settings: dict | None = None,
    calculator_settings: dict | None = None,
) -> list[R]:
    tasks = list(tasks)
    total = int(total if total is not None else completed + len(tasks))
    if not tasks:
        if verbose and total:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[{label}] Progress: {total}/{total} completed (100%)"
            )
        return []

    if (
        distributed_operation
        and calculator_settings is not None
        and _MPI_COMM is not None
        and _MPI_COMM.Get_rank() == 0
        and _MPI_COMM.Get_size() > 1
    ):
        return _run_distributed_mace_tasks(
            tasks,
            calculators,
            worker,
            operation=distributed_operation,
            worker_settings=distributed_settings or {},
            calculator_settings=calculator_settings,
            on_result=on_result,
            completed=completed,
            total=total,
            label=label,
            verbose=verbose,
        )

    calculator_queue = Queue()
    for calculator in calculators:
        calculator_queue.put(calculator)

    def leased_worker(task: T) -> R:
        calculator = calculator_queue.get()
        try:
            return worker(task, calculator)
        finally:
            calculator_queue.put(calculator)

    results = []
    next_fraction = min(10, (int(completed) * 10 // max(total, 1)) + 1)
    debug = _debug_enabled()
    with ThreadPoolExecutor(max_workers=len(calculators)) as executor:
        futures = [executor.submit(leased_worker, task) for task in tasks]
        try:
            for future in as_completed(futures):
                result = future.result()
                if verbose and debug:
                    next_completed = completed + 1
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"[{label}] Completed {next_completed}/{total} "
                        f"({round(next_completed * 100 / max(total, 1))}%)"
                        f"{_debug_result_details(result)}"
                    )
                if on_result is not None:
                    on_result(result)
                results.append(result)
                completed += 1
                fraction = min(10, completed * 10 // max(total, 1))
                if verbose and not debug and (
                    fraction >= next_fraction or completed == total
                ):
                    percent = min(100, fraction * 10)
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"[{label}] Progress: {completed}/{total} completed ({percent}%)"
                    )
                    next_fraction = fraction + 1
        except BaseException as exc:
            for future in futures:
                future.cancel()
            if verbose:
                print(f"[{label}] Stopped after {completed}/{total} completed: {exc}")
            raise
    return results
