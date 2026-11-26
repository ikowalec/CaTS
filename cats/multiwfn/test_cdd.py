from pathlib import Path

from cats.multiwfn.cdd import (
    charge_density_difference_routine,
    get_charge_density_difference_routine_config,
)


def test_cdd_config_uses_multiwfn_hole_electron_export():
    config = get_charge_density_difference_routine_config(
        "orca.molden.input", "orca.out", 7, grid_quality=2
    )
    assert config[1:8] == [
        "18 // electron excitation analysis",
        "1 // hole-electron and transition-density analysis",
        "orca.out",
        "7 // excited state",
        "1 // calculate hole-electron grid data",
        "2 // grid quality",
        "15 // export charge density difference as CDD.cub",
    ]


def test_cdd_routine_renames_multiwfn_cube(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_runner(_config_name, log_name):
        Path("CDD.cub").write_text("cube", encoding="utf-8")
        Path(log_name).write_text("done", encoding="utf-8")

    result = charge_density_difference_routine(
        "orca.molden.input",
        "orca.out",
        7,
        output_prefix="state_007_cdd",
        runner=fake_runner,
    )
    assert Path(result["cdd_cube"]).read_text(encoding="utf-8") == "cube"
    assert result["sign_convention"].startswith("positive=electron accumulation")
