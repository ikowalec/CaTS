import csv
from pathlib import Path
from types import SimpleNamespace

import ase.parallel
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

from utilities.similarity.alloy_nanocluster_growth_workflow import (
    _above_z_cutoff,
    _filter_concentration_window,
    _select_restart_beam,
    _size_tightened_similarity_threshold,
    _stage_generation_count,
    _write_csv,
    _write_db,
)


def test_above_z_cutoff_checks_only_added_species():
    atoms = Atoms("TiOCuPb", positions=[[0, 0, 1], [0, 0, 2], [0, 0, 6.2], [0, 0, 6.1]])

    assert _above_z_cutoff(atoms, {"Cu", "Pb"}, 6.0)
    assert not _above_z_cutoff(atoms, {"Cu", "Pb"}, 6.15)


def test_write_csv_accepts_batched_and_unbatched_report_rows(tmp_path):
    path = tmp_path / "similarity.csv"
    _write_csv(
        path,
        [
            {"row_id": 1, "kept": True},
            {
                "row_id": 2,
                "kept": False,
                "batch_row_id": 1,
                "batch_kept": True,
                "recombine_row_id": 1,
            },
        ],
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["batch_row_id"] == ""
    assert rows[1]["batch_row_id"] == "1"


def test_write_db_does_not_enter_ase_mpi_collective(tmp_path, monkeypatch):
    monkeypatch.setattr(ase.parallel, "world", SimpleNamespace(size=8, rank=0))
    monkeypatch.setattr(
        ase.parallel,
        "broadcast",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("coordinator DB writes must not use ASE MPI")
        ),
    )
    path = tmp_path / "candidates.db"

    _write_db(path, [Atoms("Cu"), Atoms("Pb")])

    assert connect(str(path), serial=True).count() == 2


def _with_energy(formula, energy):
    atoms = Atoms(formula)
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    return atoms


def test_stage_generation_count_matches_au10cu20_example():
    seed = Atoms("Au10Cu20")

    assert _stage_generation_count(
        seed,
        concentration_symbol="Au",
        concentration_symbols=["Au", "Cu"],
        concentration_span=0.05,
    ) == 6


def test_similarity_threshold_tightens_with_alloy_size():
    baseline = 0.9999

    assert _size_tightened_similarity_threshold(baseline, 0) == baseline
    assert _size_tightened_similarity_threshold(baseline, 1) == baseline
    assert abs(
        _size_tightened_similarity_threshold(baseline, 5)
        - (1.0 - 0.0001 / 5)
    ) < 1e-12
    assert abs(
        _size_tightened_similarity_threshold(baseline, 20)
        - (1.0 - 0.0001 / 20)
    ) < 1e-12
    assert abs(
        _size_tightened_similarity_threshold(0.995, 60) - (1.0 - 0.005 / 60)
    ) < 1e-12


def test_nearest_concentration_window_at_36_atoms_keeps_au10_through_au14():
    candidates = [Atoms(f"Au{count}Cu{36 - count}") for count in range(9, 16)]

    kept = _filter_concentration_window(
        candidates,
        concentration_symbol="Au",
        concentration_symbols=["Au", "Cu"],
        lower=10 / 30 - 0.05,
        upper=10 / 30 + 0.05,
    )

    assert [atoms.get_chemical_symbols().count("Au") for atoms in kept] == [10, 11, 12, 13, 14]


def test_restart_beam_prefers_three_distinct_compositions():
    candidates = [
        _with_energy("Au3Cu", -10.0),
        _with_energy("Au3Cu", -9.9),
        _with_energy("Au2Cu2", -9.7),
        _with_energy("AuCu3", -9.6),
    ]

    selected, rows = _select_restart_beam(
        candidates,
        beam_width=3,
        concentration_symbols=["Au", "Cu"],
        reference_energies_eV={"Au": 0.0, "Cu": 0.0},
    )

    assert len(selected) == 3
    assert {row["composition"] for row in rows} == {"Au1Cu3", "Au2Cu2", "Au3Cu1"}


def test_restart_beam_targets_atomization_by_default():
    candidates = [
        _with_energy("Au3Cu", -10.0),
        _with_energy("AuCu3", -9.6),
    ]

    selected, rows = _select_restart_beam(
        candidates,
        beam_width=1,
        concentration_symbols=["Au", "Cu"],
        reference_energies_eV={"Au": 0.0, "Cu": 0.0},
        bulk_reference_energies_eV_per_atom={"Au": -3.0, "Cu": -1.0},
    )

    assert selected[0].get_chemical_formula() == "Au3Cu"
    assert rows[0]["selection_criterion"] == "atomization_energy_per_atom_eV"


def test_restart_beam_can_target_bulk_mixing_across_compositions():
    candidates = [
        _with_energy("Au3Cu", -10.0),
        _with_energy("AuCu3", -9.6),
    ]

    selected, rows = _select_restart_beam(
        candidates,
        beam_width=1,
        concentration_symbols=["Au", "Cu"],
        reference_energies_eV={"Au": 0.0, "Cu": 0.0},
        bulk_reference_energies_eV_per_atom={"Au": -3.0, "Cu": -1.0},
        target_mixing_energy=True,
    )

    assert selected[0].get_chemical_formula() == "AuCu3"
    assert rows[0]["selection_criterion"] == "bulk_mixing_energy_per_atom_eV"
    assert abs(float(rows[0]["bulk_mixing_energy_per_atom_eV"]) + 0.9) < 1e-12


def test_phase_scatter_colours_by_mixing_energy_when_available(tmp_path, monkeypatch):
    import matplotlib.axes
    import utilities.similarity.alloy_nanocluster_growth_workflow as workflow

    captured = {}
    original_scatter = matplotlib.axes.Axes.scatter

    def capture_scatter(self, *args, **kwargs):
        captured["colours"] = kwargs["c"]
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", capture_scatter)
    path = tmp_path / "phase.png"
    workflow._write_phase_scatter(
        path,
        [
            {
                "x_Cu": 0.25,
                "atomization_energy_per_atom_eV": 2.0,
                "mixing_energy_per_atom_eV": -0.1,
                "n_atoms": 4,
            },
            {
                "x_Cu": 0.5,
                "atomization_energy_per_atom_eV": 2.2,
                "mixing_energy_per_atom_eV": -0.3,
                "n_atoms": 6,
            },
        ],
        ["Au", "Cu"],
    )

    assert captured["colours"] == [-0.1, -0.3]
    assert path.exists()


def test_reference_energies_use_distributed_mace_pool(monkeypatch):
    import similarity.workflows.mace_pool as mace_pool
    import utilities.similarity.alloy_nanocluster_growth_workflow as workflow

    captured = {}
    monkeypatch.setattr(mace_pool, "get_mace_calculators", lambda **_kwargs: [object()])

    def fake_run(tasks, _calculators, _worker, **kwargs):
        captured.update(kwargs)
        return [(symbol, -float(index)) for index, symbol in enumerate(tasks, start=1)]

    monkeypatch.setattr(mace_pool, "run_mace_tasks", fake_run)
    references = workflow.mace_single_atom_reference_energies(
        ["Cu", "Au", "Cu"], model_path="model", device="cuda"
    )

    assert references == {"Au": -1.0, "Cu": -2.0}
    assert captured["distributed_operation"].endswith(":_single_atom_reference_mpi_task")


def test_bulk_reference_skips_model_without_stress():
    from ase.calculators.calculator import Calculator
    import utilities.similarity.alloy_nanocluster_growth_workflow as workflow

    class EnergyOnlyCalculator(Calculator):
        implemented_properties = ["energy"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": -float(len(atoms))}

    symbol, atoms, row = workflow._bulk_reference_mpi_task(
        "Au",
        EnergyOnlyCalculator(),
        {"fmax": 0.01, "max_steps": 5},
    )

    assert symbol == "Au"
    assert atoms is None
    assert row["stress_supported"] is False


def test_bulk_mixing_minima_selects_lowest_energy_per_concentration(tmp_path):
    import utilities.similarity.alloy_nanocluster_growth_workflow as workflow

    rows = [
        {"x_Cu": 0.5, "bulk_mixing_energy_per_atom_eV": -0.2, "n_atoms": 4},
        {"x_Cu": 0.5, "bulk_mixing_energy_per_atom_eV": -0.4, "n_atoms": 10},
        {"x_Cu": 0.75, "bulk_mixing_energy_per_atom_eV": -0.1, "n_atoms": 8},
    ]
    png_path = tmp_path / "minima.png"
    csv_path = tmp_path / "minima.csv"
    workflow._write_bulk_mixing_minima_plot(
        png_path,
        csv_path,
        rows,
        ["Au", "Cu"],
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        minima = list(csv.DictReader(handle))
    assert len(minima) == 2
    assert float(minima[0]["bulk_mixing_energy_per_atom_eV"]) == -0.4
    assert int(minima[0]["n_atoms"]) == 10
    assert png_path.exists()


def test_seedless_staged_campaign_runs_unrestricted_then_branches(tmp_path, monkeypatch):
    import utilities.similarity.alloy_nanocluster_growth_workflow as workflow

    monkeypatch.setattr(
        workflow,
        "mace_single_atom_reference_energies",
        lambda *_args, **_kwargs: {"Au": 0.0, "Cu": 0.0},
    )
    monkeypatch.setattr(
        workflow,
        "mace_bulk_reference_energies",
        lambda *_args, **_kwargs: {"Au": -1.5, "Cu": -1.0},
    )
    checkpoint_thresholds = []

    def fake_deduplicate(atoms_list, **kwargs):
        checkpoint_thresholds.append(kwargs["similarity_threshold"])
        return list(atoms_list)

    monkeypatch.setattr(workflow, "_deduplicate_by_composition", fake_deduplicate)
    calls = []

    def fake_growth(seed, *, n_additions, output_root, concentration_window=None, **_kwargs):
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        if len(seed) == 0:
            candidates = [_with_energy("AuCu2", -5.0), _with_energy("Au2Cu", -4.9)]
        else:
            added_symbol = "Cu" if seed.get_chemical_symbols().count("Au") == 1 else "Au"
            symbols = seed.get_chemical_symbols() + [added_symbol] * int(n_additions)
            candidates = [_with_energy(symbols, -6.0)]
        generation_dir = output_root / f"generation_{int(n_additions):03d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        workflow._write_db(generation_dir / "final_unique.db", candidates)
        ranked = workflow._rank_by_atomization_energy(
            candidates,
            concentration_symbols=["Au", "Cu"],
            reference_energies_eV={"Au": 0.0, "Cu": 0.0},
        )
        _, ranked_rows = workflow._select_per_composition(ranked, keep_per_composition=1)
        workflow._write_csv(generation_dir / "ranked.csv", ranked_rows)
        stable_db = output_root / "stable_alloy_structures.db"
        workflow._write_db(stable_db, candidates)
        calls.append((len(seed), int(n_additions), concentration_window))
        return {"output_root": output_root, "stable_db_path": stable_db}

    monkeypatch.setattr(workflow, "grow_alloy_nanocluster_by_atom_addition", fake_growth)

    result = workflow.grow_alloy_nanocluster_staged(
        Atoms(),
        add_symbols=["Au", "Cu"],
        max_atoms=4,
        staging_start_atoms=3,
        concentration_symbol="Au",
        concentration_span=0.1,
        restart_beam_width=2,
        output_root=tmp_path / "campaign",
        verbose=False,
    )

    assert calls[0] == (0, 3, None)
    assert len(calls) == 3
    assert all(call[1] == 1 and call[2] is not None for call in calls[1:])
    assert abs(checkpoint_thresholds[0] - (1.0 - 0.0001 / 3)) < 1e-12
    assert abs(checkpoint_thresholds[-1] - (1.0 - 0.0001 / 4)) < 1e-12
    assert result["campaign_path"].exists()
    assert result["selection_criterion"].exists()
    assert result["combined_stable_db"].exists()
    assert workflow._read_db(result["combined_stable_db"])[0].get_potential_energy() < 0
    assert result["explored_compositions_png"].exists()
    assert result["stable_atomization_vs_cross_coordination_png"].exists()
    assert result["stable_atomization_vs_first_metal_coordination_png"].exists()
    assert result["stable_atomization_vs_second_metal_coordination_png"].exists()
    assert result["stable_bulk_mixing_vs_cross_coordination_png"].exists()
    assert result["stable_bulk_mixing_vs_first_metal_coordination_png"].exists()
    assert result["stable_bulk_mixing_vs_second_metal_coordination_png"].exists()
    assert result["bulk_mixing_energy_phase_diagram_png"].exists()
    assert result["bulk_mixing_energy_concentration_minima_png"].exists()
    assert result["bulk_mixing_energy_concentration_minima_csv"].exists()
    assert not (tmp_path / "campaign" / "bulk_mixing_energy_phase_lines.png").exists()
    checkpoint_ranking = tmp_path / "campaign" / "checkpoint_N003" / "restart_ranking.csv"
    with checkpoint_ranking.open(newline="", encoding="utf-8") as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    assert checkpoint_rows[0]["selection_criterion"] == "atomization_energy_per_atom_eV"
    with result["stable_coordination_csv"].open(newline="", encoding="utf-8") as handle:
        coordination_rows = list(csv.DictReader(handle))
    assert "mean_total_metal_coordination" in coordination_rows[0]
    assert "mean_cross_metal_coordination" in coordination_rows[0]
    assert "mean_Au_metal_coordination" in coordination_rows[0]
    assert "mean_Cu_metal_coordination" in coordination_rows[0]
    assert coordination_rows[0]["bulk_mixing_energy_per_atom_eV"]
    stage_dir = tmp_path / "campaign" / "stage_000_N003_N004"
    assert (stage_dir / "stable_atomization_vs_Au_coordination.png").exists()
    assert (stage_dir / "stable_atomization_vs_Cu_coordination.png").exists()
    assert (stage_dir / "stable_bulk_mixing_energy_vs_Au_coordination.png").exists()
    assert (stage_dir / "stable_bulk_mixing_energy_vs_Cu_coordination.png").exists()
    assert (stage_dir / "bulk_mixing_energy_phase_diagram.png").exists()
    assert (stage_dir / "bulk_mixing_energy_concentration_minima.png").exists()
    assert not (stage_dir / "bulk_mixing_energy_phase_lines.png").exists()
    obsolete_lines = tmp_path / "campaign" / "bulk_mixing_energy_phase_lines.png"
    obsolete_stage_lines = stage_dir / "bulk_mixing_energy_phase_lines.png"
    obsolete_lines.write_text("obsolete", encoding="utf-8")
    obsolete_stage_lines.write_text("obsolete", encoding="utf-8")

    monkeypatch.setattr(
        workflow,
        "mace_single_atom_reference_energies",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed restart must reuse stored references")
        ),
    )
    monkeypatch.setattr(
        workflow,
        "grow_alloy_nanocluster_by_atom_addition",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed restart must not repeat growth")
        ),
    )
    restarted = workflow.grow_alloy_nanocluster_staged(
        Atoms(),
        add_symbols=["Au", "Cu"],
        max_atoms=4,
        staging_start_atoms=3,
        concentration_symbol="Au",
        concentration_span=0.1,
        restart_beam_width=2,
        output_root=tmp_path / "campaign",
        verbose=False,
    )
    assert restarted["explored_compositions_csv"].exists()
    assert not obsolete_lines.exists()
    assert not obsolete_stage_lines.exists()
