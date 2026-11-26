import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


def test_spectrum_gradient_uses_selected_root_and_preserves_geometry(tmp_path):
    # Load only the function: importing the workflow configures external programs.
    source = Path(__file__).resolve().parents[2] / "cats/working_directory/tddft_active_learning_workflow.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_final_tddft_spectrum")
    captured = {}
    atoms = Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]])
    forces = np.array([[.01, 0, 0], [-.01, 0, 0]])

    def calculator(actual, **kwargs):
        captured.update(kwargs)
        return SinglePointCalculator(actual, energy=-10., forces=forces)

    backend = SimpleNamespace(key="orca", ensure_restart=lambda *a, **k: None,
                              make_tddft_calculator=calculator, output_path=lambda p: p / "orca.out",
                              validate_output=lambda p: True)
    namespace = {"Path": Path, "json": json, "DFT_ROOT_DIR": tmp_path,
                 "REACTION_HUB_FINAL_TDDFT_SPECTRUM_DIRNAME": "spectrum",
                 "REACTION_HUB_FINAL_TDDFT_SPECTRUM_CSV": "spectrum.csv",
                 "REACTION_HUB_FINAL_TDDFT_SPECTRUM_PNG": "spectrum.png",
                 "_export_final_tddft_spectrum_plot_data": lambda *a, **k: {}}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    namespace["_run_final_tddft_spectrum"](final_atoms=atoms, final_opt_dir=tmp_path, cpus=1, mem=100,
                                           nroots=130, basis_keyword="LANL2DZ", xc_functional="CAM-B3LYP",
                                           solvent=None, spin_multiplicity=1, workflow_backend=backend, gradient_root=2)
    result = json.loads((tmp_path / "spectrum/root_gradient_results.json").read_text())
    assert captured["task"] == "EnGrad" and captured["iroot"] == result["iroot"] == 2
    assert np.array_equal(result["positions_A"], atoms.positions)
    assert np.array_equal(result["atom_forces_eV_A"], forces)
    assert result["fmax_eV_A"] == pytest.approx(.01)
