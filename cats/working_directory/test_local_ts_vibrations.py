import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write

from cats.working_directory.active_learning_neb_workflow import (
    _run_dimer_refine_pass, _run_local_ts_vibrations,
)


class LocalTSVibrationsTests(unittest.TestCase):
    def setUp(self):
        atoms = Atoms('Cu3CO2', positions=np.arange(18).reshape(-1, 3))
        atoms.info.update(dimer_neighbor_shells=0, vibration_neighbor_shells=2)
        stream = io.StringIO()
        write(stream, atoms, format='extxyz')
        self.atoms = read(io.StringIO(stream.getvalue()), format='extxyz')

    def test_dimer_accepts_staged_numpy_integer(self):
        class ReachedDimer(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory, patch(
            'cats.working_directory.active_learning_neb_workflow.DimerControl',
            side_effect=ReachedDimer,
        ), self.assertRaises(ReachedDimer):
            _run_dimer_refine_pass(
                atoms=self.atoms, calculator_getter=lambda: None,
                output_dir=Path(directory),
            )

    def test_vibrations_accept_staged_shell_count_and_keep_adsorbates(self):
        module = types.ModuleType('carmm.analyse.neighbours')
        module.neighbours = MagicMock(return_value=(np.array([1, 2], dtype=np.int64), None))
        vibrations = MagicMock()
        vibrations.get_frequencies.return_value = np.array([100j, 250 + 0j])
        vibrations.get_mode.return_value = np.zeros((6, 3))
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            'sys.modules', {'carmm.analyse.neighbours': module}
        ), patch('ase.vibrations.Vibrations', return_value=vibrations):
            result = _run_local_ts_vibrations(
                self.atoms, shells=self.atoms.info['vibration_neighbor_shells'],
                output_dir=Path(directory),
            )
            self.assertEqual(result['active_atom_indices_0based'], [1, 2, 3, 4, 5])
            self.assertEqual(result['frequencies_cm1'], [-100., 250.])
            self.assertEqual(result['imaginary_count'], 1)
            self.assertEqual(json.loads((Path(directory) / 'results.json').read_text()), result)
            module.neighbours.assert_called_once_with(self.atoms, centre=[3, 4, 5], shell=2)
            vibrations.run.assert_called_once()

    def test_all_atom_vibrations_remove_constraints_and_keep_calculator(self):
        self.atoms.set_constraint(FixAtoms(indices=[0, 1]))
        calculator = MagicMock()
        self.atoms.calc = calculator
        vibrations = MagicMock()
        vibrations.get_frequencies.return_value = np.array([100j, 250 + 0j])
        vibrations.get_mode.return_value = np.zeros((6, 3))
        with tempfile.TemporaryDirectory() as directory, patch(
            'ase.vibrations.Vibrations', return_value=vibrations,
        ) as constructor:
            result = _run_local_ts_vibrations(
                self.atoms, shells=np.int64(0), output_dir=Path(directory),
            )
            evaluated_atoms = constructor.call_args.args[0]
            self.assertEqual(evaluated_atoms.constraints, [])
            self.assertIs(evaluated_atoms.calc, calculator)
            self.assertEqual(constructor.call_args.kwargs['indices'], list(range(6)))
            self.assertEqual(result['active_atom_indices_0based'], list(range(6)))
            self.assertEqual(result['method'], 'ASE finite-difference full Hessian')
            self.assertEqual(len(self.atoms.constraints), 1)


if __name__ == '__main__':
    unittest.main()
