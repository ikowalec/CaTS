from pathlib import Path

import pytest

from cats.multiwfn.nto import (
    get_nto_cube_routine_config,
    get_nto_routine_config,
    nto_routine,
    parse_nto_eigenvalue_blocks,
    parse_nto_eigenvalues,
    select_nto_pairs,
)


def test_nto_configs_are_separate_and_select_the_dominant_pair():
    nto = get_nto_routine_config("orca.molden.input", "orca.out", 7, "S7.mwfn")
    assert nto[:7] == [
        "orca.molden.input",
        "18",
        "6",
        "orca.out",
        "7",
        "3",
        "S7.mwfn",
    ]
    assert get_nto_cube_routine_config("S7.mwfn", "h")[3] == "h"
    assert get_nto_cube_routine_config("S7.mwfn", "l")[3] == "l"


def test_parse_nto_eigenvalues(tmp_path):
    log = tmp_path / "nto.log"
    log.write_text(
        "The highest 3 eigenvalues of NTO pairs:\n 0.91 0.07 0.02\n"
        "Sum of all eigenvalues: 1.000\n",
        encoding="utf-8",
    )
    assert parse_nto_eigenvalues(log) == [0.91, 0.07, 0.02]


def test_selects_fewest_spin_resolved_pairs_reaching_target(tmp_path):
    log = tmp_path / "nto.log"
    log.write_text(
        "The highest 3 eigenvalues of alpha NTO pairs:\n 0.50 0.10 0.00\n"
        "Sum of all alpha eigenvalues: 0.60\n"
        "The highest 3 eigenvalues of beta NTO pairs:\n 0.40 0.00 0.00\n"
        "Sum of all beta eigenvalues: 0.40\n",
        encoding="utf-8",
    )

    assert parse_nto_eigenvalue_blocks(log)["beta"]["total"] == pytest.approx(0.4)
    pairs = select_nto_pairs(log, max_pairs=3, target_coverage=0.8)
    assert [(pair["spin"], pair["spin_rank"]) for pair in pairs] == [("alpha", 1), ("beta", 1)]
    assert pairs[-1]["cumulative_coverage"] == pytest.approx(0.9)


def test_nto_routine_exports_selected_spin_pairs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_runner(config_name, log_name):
        commands = Path(config_name).read_text(encoding="utf-8").splitlines()
        if commands[2] == "6":
            Path(commands[6]).write_text("NTO wavefunction", encoding="utf-8")
            Path(log_name).write_text(
                "The highest 2 eigenvalues of alpha NTO pairs:\n 0.50 0.10\n"
                "Sum of all alpha eigenvalues: 0.60\n"
                "The highest 2 eigenvalues of beta NTO pairs:\n 0.40 0.00\n"
                "Sum of all beta eigenvalues: 0.40\n",
                encoding="utf-8",
            )
        else:
            Path("MOvalue.cub").write_text(commands[3], encoding="utf-8")

    result = nto_routine(
        "orca.molden.input", "orca.out", 7, output_prefix="S7",
        max_pairs=3, target_coverage=0.8, runner=fake_runner,
    )

    assert len(result["pairs"]) == 2
    assert [pair["hole_selector"] for pair in result["pairs"]] == ["ha", "hb"]
    assert [pair["particle_selector"] for pair in result["pairs"]] == ["la", "lb"]


def test_nto_routine_exports_named_hole_and_particle_cubes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_runner(config_name, log_name):
        commands = Path(config_name).read_text(encoding="utf-8").splitlines()
        if commands[2] == "6":
            Path(commands[6]).write_text("NTO wavefunction", encoding="utf-8")
            Path(log_name).write_text(
                "The highest 2 eigenvalues of NTO pairs:\n 0.95 0.05\n"
                "Sum of all eigenvalues: 1.0\n",
                encoding="utf-8",
            )
        else:
            Path("MOvalue.cub").write_text(commands[3], encoding="utf-8")

    result = nto_routine(
        "orca.molden.input",
        "orca.out",
        7,
        output_prefix="S7",
        grid_quality=1,
        runner=fake_runner,
    )

    assert Path(result["hole_cube"]).read_text(encoding="utf-8") == "h"
    assert Path(result["particle_cube"]).read_text(encoding="utf-8") == "l"
    assert result["dominant_pair_weight"] == pytest.approx(0.95)
