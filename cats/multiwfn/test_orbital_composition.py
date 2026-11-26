import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cats.multiwfn.orbital_composition import mulliken_orbital_composition


class OrbitalCompositionTests(unittest.TestCase):
    def test_signed_channels_spin_and_settings_restoration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            source = root / 'nto.mwfn'
            source.write_text('Nbasis= 3\n$Shell types\n0 1 2\n$Shell centers\n1 1 1\n$Primitive exponents\n1\nIndex= 1\nType= 1\nOcc= 1\nIndex= 2\nType= 1\nOcc= 0\nIndex= 3\nType= 2\nOcc= 1\nIndex= 4\nType= 2\nOcc= 0\n')
            ini = root / 'settings.ini'
            ini.write_text('compthres= 0.5\n')
            original = ini.read_bytes()
            def runner(config, log):
                self.assertEqual(ini.read_text(), 'compthres= -1.0\n')
                routine = Path(config).read_text().splitlines()
                self.assertEqual(routine[3], 'hb')
                self.assertEqual(routine[4:], ['0', '-10', 'q'])
                Path(log).write_text('Orbital: 4 Energy(a.u.): 0 Occ: 1 Type: Beta\n1 S 1(Au) 1 35.0% 35.0% 70.0%\n2 X 1(Au) 2 20.0% 20.0% 40.0%\n3 XX 1(Au) 3 1.0% -11.0% -10.0%\n')
            with patch('cats.multiwfn.orbital_composition.Path', wraps=Path) as paths:
                paths.return_value = ini
                # Only redirect the relative settings path, preserving other paths.
                paths.side_effect = lambda value: ini if value == 'settings.ini' else Path(value)
                result = mulliken_orbital_composition(source, spin='beta', output_prefix=str(root/'result'), runner=runner)
                def incomplete(config, log):
                    Path(log).write_text('Orbital: 4 Type: Beta\n1 S 1(Au) 1 35.0% 35.0% 70.0%\n')
                with self.assertRaisesRegex(ValueError, 'Incomplete AO composition'):
                    mulliken_orbital_composition(source, spin='beta', output_prefix=str(root/'incomplete'), runner=incomplete)
            self.assertEqual(result['atoms']['1'], {'symbol':'Au','s':70.,'p':40.,'d':-10.})
            self.assertEqual(result['orbital_index'], 3)
            self.assertEqual(result['native_orbital_index'], 4)
            def electron(config, log):
                self.assertEqual(Path(config).read_text().splitlines()[3], 'lb')
                Path(log).write_text('Orbital: 5 Type: Beta\n1 S 1(Au) 1 35.0% 35.0% 70.0%\n2 X 1(Au) 2 20.0% 20.0% 40.0%\n3 XX 1(Au) 3 1.0% -11.0% -10.0%\n')
            with patch('cats.multiwfn.orbital_composition.Path', side_effect=lambda value: ini if value == 'settings.ini' else Path(value)):
                particle = mulliken_orbital_composition(source, spin='beta', role='electron',output_prefix=str(root/'electron'),runner=electron)
            self.assertEqual(particle['orbital_index'], 4)
            self.assertEqual(particle['native_orbital_index'], 5)
            self.assertEqual(ini.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
