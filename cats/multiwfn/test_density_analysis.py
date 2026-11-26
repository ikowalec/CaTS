import json

from cats.multiwfn import density_analysis
from cats.multiwfn.density_analysis import select_relevant_excitations


def test_selects_brightest_excitation_in_each_manuscript_band():
    states = [
        {"state": 60, "nm": 343.9, "f": 0.8},
        {"state": 59, "nm": 350.0, "f": 0.2},
        {"state": 25, "nm": 413.0, "f": 0.5},
        {"state": 11, "nm": 505.2, "f": 0.4},
        {"state": 10, "nm": 520.0, "f": 0.1},
        {"state": 4, "nm": 618.7, "f": 0.3},
    ]

    selected = select_relevant_excitations(states)

    assert [row["excitation_number"] for row in selected] == [60, 25, 11, 4]


def test_explicit_states_use_bounded_parallel_workers(tmp_path, monkeypatch):
    calls = []

    class Future:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class Executor:
        def __init__(self, max_workers):
            calls.append(("workers", max_workers))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def submit(self, function, *args):
            calls.append(("state", args[3]["excitation_number"]))
            return Future(function(*args))

    def fake_generate(*args):
        return {**args[3], "cdd_cube": f"state_{args[3]['excitation_number']:03d}.cub"}

    molden = tmp_path / "orca.molden.input"
    output = tmp_path / "orca.out"
    molden.touch()
    output.touch()
    monkeypatch.setattr(density_analysis, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(density_analysis, "_generate_state_density", fake_generate)

    result = density_analysis.generate_excitation_density_analysis(
        molden,
        output,
        [],
        output_directory=tmp_path / "density",
        excitation_numbers=[23, 24],
        export_pov_scenes=False,
        max_workers=8,
    )

    assert calls == [("workers", 2), ("state", 23), ("state", 24)]
    assert result["max_workers"] == 2
    manifest = json.loads((tmp_path / "density" / "density_analysis_manifest.json").read_text())
    assert manifest["max_workers"] == 2


def test_state_worker_isolates_multiwfn_scratch_files(tmp_path, monkeypatch):
    output = tmp_path / "density"
    output.mkdir()
    working_directories = []

    def fake_cdd(*_args, output_prefix, **_kwargs):
        working_directories.append(density_analysis.Path.cwd())
        density_analysis.Path(f"{output_prefix}.cub").touch()
        density_analysis.Path("CDD.cub").touch()
        return {"cdd_cube": f"{output_prefix}.cub", "sign_convention": "test"}

    def fake_nto(*_args, output_prefix, **_kwargs):
        density_analysis.Path(f"{output_prefix}_hole.cub").touch()
        density_analysis.Path(f"{output_prefix}_particle.cub").touch()
        density_analysis.Path("MOvalue.cub").touch()
        return {
            "hole_cube": f"{output_prefix}_hole.cub",
            "particle_cube": f"{output_prefix}_particle.cub",
            "dominant_pair_weight": 0.9,
        }

    monkeypatch.setattr(density_analysis, "charge_density_difference_routine", fake_cdd)
    monkeypatch.setattr(density_analysis, "nto_routine", fake_nto)

    result = density_analysis._generate_state_density(
        str(tmp_path / "orca.molden.input"),
        str(tmp_path / "orca.out"),
        str(output),
        {"excitation_number": 23},
        3,
        True,
        "ha",
        "la",
    )

    assert working_directories[0].parent == output
    assert not working_directories[0].exists()
    assert (output / "state_023_cdd.cub").is_file()
    assert (output / "state_023_nto_hole.cub").is_file()
    assert (output / "state_023_nto_particle.cub").is_file()
    assert not (output / "CDD.cub").exists()
    assert result["dominant_nto_pair_weight"] == 0.9
