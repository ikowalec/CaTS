import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from ase import Atoms
from cats.orca.routines.ts_vibrations_gradient import run_ts_vibrations_gradient
from cats.orca.routines.ts_vibrations_gradient import relax_ts_endpoints


class NumFreqOnlyTest(unittest.TestCase):
    def test_two_signed_relaxations_join_at_exact_ts(self):
        from ase.calculators.calculator import Calculator, all_changes
        from ase.io import read

        class DoubleWell(Calculator):
            implemented_properties = ["energy", "forces"]
            def calculate(self, atoms=None, properties=None, system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                x = self.atoms.positions[0, 0]
                forces = np.zeros_like(self.atoms.positions)
                forces[0, 0] = -4*x*(x*x - 1)
                self.results = {"energy": (x*x - 1)**2,
                                "forces": forces}

        hessian = {"frequencies_cm1": [-200.], "normal_modes": np.array([[1.], [0.], [0.]])}
        with tempfile.TemporaryDirectory() as directory:
            result = relax_ts_endpoints(Atoms("H", positions=[[0, 0, 0]]), hessian,
                                       lambda path: DoubleWell(), directory)
            self.assertEqual(result["status"], "completed")
            frames = read(str(Path(directory) / result["path"]), ":")
            self.assertLess(frames[0].positions[0, 0], -.9)
            self.assertGreater(frames[-1].positions[0, 0], .9)
            np.testing.assert_array_equal(frames[result["ts_index"]].positions, [[0, 0, 0]])
            self.assertEqual(sum(bool(f.info["is_ts"]) for f in frames), 1)
            self.assertTrue(all(f.get_potential_energy() < 1 for f in (frames[0], frames[-1])))
            oriented = relax_ts_endpoints(
                Atoms("H2", positions=[[0, 0, 0], [3, 0, 0]]),
                {"frequencies_cm1": [-200.], "normal_modes": np.array([[-1.], [0.], [0.], [0.], [0.], [0.]])},
                lambda path: DoubleWell(), Path(directory) / "oriented", reaction_bond=[0, 1],
            )
            self.assertEqual(oriented["status"], "completed")
            self.assertGreater(oriented["signed_normalised_cartesian_mode"][0][0], 0)
            hessian["frequencies_cm1"] = [100.]
            blocked = relax_ts_endpoints(Atoms("H"), hessian, lambda path: self.fail("No calculator allowed"), directory)
            self.assertEqual(blocked["status"], "blocked")

    def test_frequency_only_never_runs_excited_gradient(self):
        module = "cats.orca.routines.ts_vibrations_gradient"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.xyz"
            Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.95]]).write(source)
            hessian = {"frequencies_cm1": [-200., 100.], "imaginary_indices": [0], "normal_modes": np.ones((6, 2))}
            with patch(module + ".get_orca_path", return_value="orca"), patch(module + ".subprocess.run", return_value=MagicMock(returncode=0)) as run, patch(module + ".parse_orca_hess", return_value=hessian), patch(module + ".parse_orca_engrad") as gradient:
                result = run_ts_vibrations_gradient(source, output_dir=root / "result", ground_state_only=True, hpc="falcon", basis_set="LANL2DZ")
            run.assert_called_once()
            gradient.assert_not_called()
            self.assertEqual(result["imaginary_count"], 1)
            self.assertEqual(result["all_frequencies_cm1"], [-200., 100.])
            self.assertFalse(any((root / "result").glob("s*_engrad")))
            self.assertTrue((root / "result/reaction_hub_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
