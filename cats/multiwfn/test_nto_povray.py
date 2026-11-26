from pathlib import Path
import numpy as np
import pytest
from ase.build import molecule
from ase.io.cube import write_cube

from cats.multiwfn.nto_povray import (
    write_structure_pov_scene,
    write_nto_cube_pov_scene,
    export_density_analysis_pov_scenes,
)


def test_windows_missing_profile_never_launches_renderer(monkeypatch):
    import os
    if os.name != "nt":
        pytest.skip("Windows user-profile handling")
    import winreg
    from cats.multiwfn import nto_povray
    def denied(*args, **kwargs):
        raise PermissionError("User settings unavailable")
    monkeypatch.setattr(winreg, "OpenKey", denied)
    def unexpected_launch(*args, **kwargs):
        pytest.fail("Renderer started despite missing user profile")
    monkeypatch.setattr(nto_povray.subprocess, "run", unexpected_launch)
    with pytest.raises(RuntimeError, match="launch stopped to prevent setup dialogs"):
        nto_povray.render_pov_scene(Path(__file__).with_suffix('.ini'),
                                   povray_executable="pvengine64.exe")


def _make_dummy_cube(path: Path):
    atoms = molecule("H2O")
    data = np.zeros((6, 6, 6), dtype=float)
    # Add positive and negative lobes
    data[2, 2, 2] = 0.05
    data[3, 3, 3] = -0.05
    with path.open("w", encoding="utf-8") as f:
        write_cube(f, atoms, data=data)
    return path


def test_write_structure_pov_scene(tmp_path: Path):
    atoms = molecule("H2O")
    pov_p, ini_p = write_structure_pov_scene(atoms, tmp_path / "h2o")

    assert pov_p.is_file()
    assert ini_p.is_file()
    ini_content = ini_p.read_text(encoding="utf-8")
    assert "Pause_When_Done=False" in ini_content
    pov_content = pov_path = pov_p.read_text(encoding="utf-8")
    assert "sphere" in pov_content or "cylinder" in pov_content


def test_write_nto_cube_pov_scene(tmp_path: Path):
    cube_path = _make_dummy_cube(tmp_path / "test.cub")
    pov_p, ini_p = write_nto_cube_pov_scene(cube_path, tmp_path / "test_scene", isovalue=0.02)

    assert pov_p.is_file()
    assert ini_p.is_file()
    pov_text = pov_p.read_text(encoding="utf-8")
    assert "mesh2" in pov_text or "triangle" in pov_text or "smooth_triangle" in pov_text


def test_export_density_analysis_pov_scenes_with_cube_cleanup(tmp_path: Path):
    cdd = _make_dummy_cube(tmp_path / "state_001_cdd.cub")
    hole = _make_dummy_cube(tmp_path / "state_001_nto_hole.cub")
    particle = _make_dummy_cube(tmp_path / "state_001_nto_particle.cub")

    states = [
        {
            "excitation_number": 1,
            "cdd_cube": "density_analysis/state_001_cdd.cub",
            "nto_hole_cube": "density_analysis/state_001_nto_hole.cub",
            "nto_particle_cube": "density_analysis/state_001_nto_particle.cub",
        }
    ]

    updated_states = export_density_analysis_pov_scenes(
        tmp_path,
        states,
        cdd_isovalue=0.01,
        nto_isovalue=0.01,
        delete_cubes=True,
    )

    # 1. Structure scene created
    assert (tmp_path / "structure.pov").is_file()
    assert (tmp_path / "structure.ini").is_file()

    # 2. CDD and NTO scenes created
    assert (tmp_path / "state_001_cdd.pov").is_file()
    assert (tmp_path / "state_001_cdd.ini").is_file()
    assert (tmp_path / "state_001_hole.pov").is_file()
    assert (tmp_path / "state_001_hole.ini").is_file()
    assert (tmp_path / "state_001_particle.pov").is_file()
    assert (tmp_path / "state_001_particle.ini").is_file()

    # 3. Bulky cube files deleted
    assert not cdd.is_file()
    assert not hole.is_file()
    assert not particle.is_file()

    # 4. State metadata references the scenes
    s = updated_states[0]
    assert "cdd_pov_scene" in s
    assert "cdd_ini" in s
    assert "nto_hole_pov_scene" in s
    assert "nto_particle_pov_scene" in s
    assert "cdd_cube" not in s
