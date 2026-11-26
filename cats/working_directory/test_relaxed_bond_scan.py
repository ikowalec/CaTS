import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.calculators.harmonic import SpringCalculator
from ase.io import read

from cats.working_directory.active_learning_neb_workflow import run_active_learning_neb_workflow


class RelaxedBondScanTests(unittest.TestCase):
    def test_only_distance_is_constrained_and_raw_forces_are_retained(self):
        reference = Atoms('Au2O2', positions=[[-2, 0, 0], [2, 0, 0], [-.75, 3, 0], [.75, 3, 0]])
        seed = []
        for distance in [1.4, 1.5, 1.6]:
            atoms = reference.copy()
            atoms.positions[0, 1] += .3
            atoms.set_distance(2, 3, distance, fix=.5)
            atoms.info.update(relaxed_scan_pair=[2, 3], relaxed_scan_target_A=distance)
            seed.append(atoms)
        with tempfile.TemporaryDirectory() as directory, patch(
            'cats.working_directory.active_learning_neb_workflow.build_calculator_factory',
            return_value=lambda: SpringCalculator(reference.positions, k=1),
        ):
            summary = run_active_learning_neb_workflow(
                initial_band_atoms=seed, backend='dft', output_dir=directory,
                fmax=.001, max_steps=40,
            )
            frames = read(summary['scan_path'], ':')
            self.assertEqual(summary['mode'], 'relaxed_bond_scan')
            self.assertTrue(summary['converged'])
            self.assertTrue(Path(summary['workflow_log_path']).is_file())
            self.assertFalse(summary['crest_is_interior'])
            np.testing.assert_allclose([a.get_distance(2, 3) for a in frames], [1.4, 1.5, 1.6], atol=1e-7)
            self.assertLess(abs(frames[-1].positions[0, 1]), .001)
            self.assertGreater(np.linalg.norm(frames[-1].get_forces()[2]), .04)
            self.assertTrue(all(p['constrained_fmax_eV_A'] <= .001 for p in summary['points']))
            self.assertEqual(len(json.loads(Path(directory, 'relaxed_scan_results.json').read_text())), 3)

    def test_invalid_target_is_rejected_before_calculation(self):
        atoms = Atoms('O2', positions=[[0, 0, 0], [1.4, 0, 0]])
        atoms.info.update(relaxed_scan_pair=[0, 1], relaxed_scan_target_A=float('nan'))
        with tempfile.TemporaryDirectory() as directory, patch(
            'cats.working_directory.active_learning_neb_workflow.build_calculator_factory'
        ), self.assertRaisesRegex(ValueError, 'finite distances'):
            run_active_learning_neb_workflow(initial_band_atoms=[atoms, atoms.copy()], backend='dft', output_dir=directory)


if __name__ == '__main__':
    unittest.main()
