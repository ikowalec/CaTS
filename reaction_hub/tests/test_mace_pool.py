import sys
from types import SimpleNamespace

from ase import parallel as ase_parallel

from similarity.workflows import mace_pool


def test_mpi_worker_pool_disables_ase_collectives(monkeypatch):
    comm = SimpleNamespace(Get_size=lambda: 2, Get_rank=lambda: 0)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(COMM_WORLD=comm)))
    monkeypatch.setattr(mace_pool, "_MPI_COMM", None)
    monkeypatch.setattr(ase_parallel, "world", SimpleNamespace(size=2))
    monkeypatch.setattr(mace_pool.atexit, "register", lambda _func: None)

    assert mace_pool.initialize_mpi_worker_pool() is True
    assert isinstance(ase_parallel.world, ase_parallel.DummyMPI)
