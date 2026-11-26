"""Run locally with the project interpreter: -m unittest cats.multiwfn.test_nto_overlap."""
import unittest
import os
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.units import Bohr

from cats.multiwfn.nto import get_nto_cube_routine_config, nto_routine
from cats.multiwfn.nto_overlap import compare, compare_nto_density_sets, compare_nto_pair


def orbital(axis=0, formula="Au20"):
    spacing = np.eye(3) * 0.15
    origin = np.full(3, -3.0)
    xyz = origin + np.indices((41, 41, 41)).reshape(3, -1).T @ spacing
    data = (xyz[:, axis] * np.exp(-np.sum(xyz ** 2, axis=1))).reshape((41, 41, 41))
    data /= np.sqrt(np.sum(data ** 2) * abs(np.linalg.det(spacing)) / Bohr ** 3)
    atoms = Atoms(formula)
    atoms.positions = np.random.default_rng(1).normal(size=(len(atoms), 3))
    return {"atoms": atoms, "data": data, "origin": origin, "spacing": spacing}


class OverlapTests(unittest.TestCase):
    def test_beta_export_uses_explicit_spin(self):
        root = Path.cwd()
        menus = []
        def runner(config, log):
            menu = Path(config).read_text()
            menus.append(menu)
            if config.endswith('state_config.txt'):
                Path('state.mwfn').write_text('test wavefunction')
            else:
                Path('MOvalue.cub').write_text('test amplitude')
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                result = nto_routine('input.molden', 'orca.out', 10, output_prefix='state',
                                     hole_selector='hb', particle_selector='lb', runner=runner)
                self.assertTrue(Path(result['hole_cube']).is_file())
                self.assertIn('hb', menus[1])
                self.assertIn('lb', menus[2])
                self.assertIsNone(result['dominant_pair_weight'])
            finally:
                os.chdir(root)

    def test_identical_phase_and_orthogonal(self):
        ref, candidate = orbital(), orbital()
        self.assertAlmostEqual(compare(ref, candidate)["squared_overlap_mean"], 1.0)
        candidate["data"] *= -1
        self.assertAlmostEqual(compare(ref, candidate)["squared_overlap_mean"], 1.0)
        self.assertAlmostEqual(compare(ref, orbital(1))["squared_overlap_mean"], 0.0)

    def test_rigid_rotation_translation_and_explicit_mapping(self):
        ref, candidate = orbital(), orbital()
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        shift = np.array([2., 3., -1.])
        candidate["atoms"].positions = candidate["atoms"].positions @ rotation + shift
        candidate["origin"] = candidate["origin"] @ rotation + shift
        candidate["spacing"] = candidate["spacing"] @ rotation
        candidate["atoms"] = candidate["atoms"][::-1]
        result = compare(ref, candidate, list(range(19, -1, -1)))
        self.assertLess(result["au_rmsd_A"], 1e-12)
        self.assertAlmostEqual(result["squared_overlap_mean"], 1.0, places=7)

    def test_spin_selectors_and_invalid_mapping(self):
        self.assertIn("hb", get_nto_cube_routine_config("state_010_nto.mwfn", "hb"))
        with self.assertRaises(ValueError):
            compare(orbital(), orbital(), [0] * 20)

    def test_cu13_matched_hole_particle_pair(self):
        hole = orbital(0, "Cu13CO")
        particle = orbital(1, "Cu13CO")
        result = compare_nto_pair(hole, particle, orbital(0, "Cu13CO"), orbital(1, "Cu13CO"))
        self.assertAlmostEqual(result["paired_overlap"], 1.0)

    def test_weighted_nto_density_is_phase_insensitive(self):
        reference = [{"weight": 0.6, "hole": orbital(0, "Cu13CO"), "particle": orbital(1, "Cu13CO")}]
        candidate = [{"weight": 0.6, "hole": orbital(0, "Cu13CO"), "particle": orbital(1, "Cu13CO")}]
        candidate[0]["hole"]["data"] *= -1
        result = compare_nto_density_sets(reference, candidate)
        self.assertAlmostEqual(result["paired_density_overlap"], 1.0)

    def test_cross_pair_order_flip_resolved(self):
        from cats.multiwfn.state_tracker import StateTracker
        tracker = StateTracker(root_state=25)

        h1, p1 = orbital(0, "Cu13CO"), orbital(1, "Cu13CO")
        h2, p2 = orbital(2, "Cu13CO"), orbital(0, "Cu13CO")

        # Ref has pair 1 = (h1, p1), pair 2 = (h2, p2)
        ref_cubes = {
            "nto_hole": h1, "nto_particle": p1,
            "multi_pairs": [
                {"hole": h1, "particle": p1, "weight": 0.6},
                {"hole": h2, "particle": p2, "weight": 0.4},
            ],
            "cdd": None,
        }
        # Candidate flipped order: pair 1 = (h2, p2), pair 2 = (h1, p1)
        cand_cubes = {
            "nto_hole": h2, "nto_particle": p2,
            "multi_pairs": [
                {"hole": h2, "particle": p2, "weight": 0.6},
                {"hole": h1, "particle": p1, "weight": 0.4},
            ],
            "cdd": None,
        }

        res = tracker.score_candidate(ref_cubes, cand_cubes, 25, 23)
        # Should resolve pair 1 of ref matching pair 2 of candidate -> overlap near 1.0!
        self.assertAlmostEqual(res.paired_score, 1.0, places=5)
        self.assertAlmostEqual(res.spatial_score, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
