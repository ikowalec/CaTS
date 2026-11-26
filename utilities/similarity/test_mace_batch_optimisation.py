from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from similarity.run import mace_batch_optimisation


def test_optimise_task_does_not_write_trajectory_by_default(monkeypatch):
    optimizer_arguments = {}

    class RecordingBFGS(BFGS):
        def __init__(self, atoms, **kwargs):
            optimizer_arguments.update(kwargs)
            super().__init__(atoms, **kwargs)

    monkeypatch.setitem(mace_batch_optimisation.OPTIMIZERS, "bfgs", RecordingBFGS)
    task = (
        1,
        Atoms("Cu2", positions=[[0, 0, 0], [3, 0, 0]]),
        {"input_id": 7, "name": "pair"},
    )
    settings = {
        "model": "test-model",
        "optimizer": "bfgs",
        "fmax": 100.0,
        "steps": 1,
        "maxstep": 0.2,
        "trajectory_dir": None,
    }

    index, atoms, row = mace_batch_optimisation.optimise_task(task, EMT(), settings)

    assert index == 1
    assert row["input_id"] == 7
    assert row["energy_eV"] == atoms.get_potential_energy()
    assert optimizer_arguments["trajectory"] is None
