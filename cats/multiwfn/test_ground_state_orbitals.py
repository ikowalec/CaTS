import os
import tempfile
import unittest
from pathlib import Path

from cats.multiwfn.nto import ground_state_orbital_cubes


class GroundStateOrbitalExportTest(unittest.TestCase):
    def test_exports_canonical_orbitals_without_nto_generation(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                Path("ground.molden.input").write_text("[Molden Format]\n", encoding="utf-8")
                selectors = []

                def runner(config, log):
                    routine = Path(config).read_text().splitlines()
                    self.assertEqual(routine[1:3], ["5", "4"])
                    self.assertNotIn("18", routine)
                    selectors.append(routine[3])
                    Path("MOvalue.cub").write_text("exported amplitude", encoding="utf-8")
                    Path(log).write_text("completed", encoding="utf-8")

                result = ground_state_orbital_cubes("ground.molden.input", runner=runner)
                self.assertEqual(selectors, ["ha", "hb", "la", "lb"])
                self.assertEqual(result["orbital_kind"], "ground_state_canonical")
                self.assertTrue(all(Path(row["cube"]).is_file() for row in result["orbitals"]))
                self.assertEqual(len(result["wavefunction_sha256"]), 64)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
