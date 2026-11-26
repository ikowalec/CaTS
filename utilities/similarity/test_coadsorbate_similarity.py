import sys
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

# This test exercises relaxation persistence, not the optional REMatch stack.
memory_rematch = ModuleType("similarity.rematch.memory_rematch")
memory_rematch.deduplicate_atoms_by_similarity = None
memory_rematch.suggest_similarity_threshold = None
sys.modules.setdefault("similarity.rematch.memory_rematch", memory_rematch)

from similarity.workflows import coadsorbate_similarity


class _Calculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


def test_support_relaxation_indices_include_requested_neighbor_shells():
    atoms = Atoms(
        "Au4", positions=[[0, 0, 0], [2.8, 0, 0], [5.6, 0, 0], [8.4, 0, 0]]
    )

    assert coadsorbate_similarity._support_relaxation_indices(atoms, 0, 0) == [0]
    assert coadsorbate_similarity._support_relaxation_indices(atoms, 0, 2) == [0, 1, 2]


def test_relaxation_commits_each_endpoint_before_next_structure(monkeypatch):
    calculators = ModuleType("mace.calculators")
    calculators.mace_mp = lambda **kwargs: _Calculator()
    mace = ModuleType("mace")
    mace.calculators = calculators
    monkeypatch.setitem(sys.modules, "mace", mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)

    class InterruptSecondOptimisation:
        count = 0

        def __init__(self, atoms, **kwargs):
            type(self).count += 1
            self.nsteps = 0

        def run(self, **kwargs):
            if type(self).count == 2:
                raise KeyboardInterrupt

    monkeypatch.setattr(coadsorbate_similarity, "BFGS", InterruptSecondOptimisation)
    db_path = Path("scratch") / f"test-relaxed-{uuid4().hex}.db"

    try:
        with pytest.raises(KeyboardInterrupt):
            coadsorbate_similarity._relax_adsorbates_with_mace(
                [Atoms("Au"), Atoms("Au")],
                n_fixed_atoms=1,
                model_path="model",
                fmax=0.05,
                max_steps=10,
                device="cpu",
                dispersion=False,
                default_dtype="float64",
                output_db_path=db_path,
                verbose=False,
            )

        assert connect(db_path).count() == 1

        class CompleteOptimisation:
            count = 0

            def __init__(self, atoms, **kwargs):
                type(self).count += 1
                self.nsteps = 0

            def run(self, **kwargs):
                pass

        monkeypatch.setattr(coadsorbate_similarity, "BFGS", CompleteOptimisation)
        relaxed, _ = coadsorbate_similarity._relax_adsorbates_with_mace(
            [Atoms("Au"), Atoms("Au")],
            n_fixed_atoms=1,
            model_path="model",
            fmax=0.05,
            max_steps=10,
            device="cpu",
            dispersion=False,
            default_dtype="float64",
            output_db_path=db_path,
            verbose=False,
        )

        assert len(relaxed) == 2
        assert CompleteOptimisation.count == 1
        assert connect(db_path).count() == 2
    finally:
        db_path.unlink(missing_ok=True)


def test_single_adsorbate_modes_are_pruned_before_pairing(monkeypatch):
    modes = [Atoms("AuH"), Atoms("AuH"), Atoms("AuH")]
    threshold = type("Threshold", (), {"attrs": {"suggested_threshold": 0.8}})()
    monkeypatch.setattr(
        coadsorbate_similarity,
        "_suggest_threshold_safe",
        lambda *args, **kwargs: threshold,
    )
    monkeypatch.setattr(
        coadsorbate_similarity,
        "deduplicate_atoms_by_similarity",
        lambda atoms_list, **kwargs: (
            atoms_list[:1],
            __import__("pandas").DataFrame(
                {"row_id": [1, 2, 3], "kept": [True, False, False]}
            ),
        ),
    )

    kept, used_threshold, kept_row_ids, _ = coadsorbate_similarity._prune_single_adsorbate_modes(
        modes,
        reference_row_id=1,
        similarity_threshold=None,
        similarity_relaxation=0.0,
        similarity_kwargs={},
    )

    assert len(kept) == 1
    assert used_threshold == 0.8
    assert kept_row_ids == [1]


def test_single_adsorbate_modes_use_fixed_threshold(monkeypatch):
    modes = [Atoms("AuH"), Atoms("AuH")]
    seen = {}
    monkeypatch.setattr(
        coadsorbate_similarity,
        "_suggest_threshold_safe",
        lambda *args, **kwargs: pytest.fail("fixed threshold should not be estimated"),
    )

    def deduplicate(atoms_list, **kwargs):
        seen["threshold"] = kwargs["similarity_threshold"]
        return atoms_list, __import__("pandas").DataFrame(
            {"row_id": [1, 2], "kept": [True, True]}
        )

    monkeypatch.setattr(
        coadsorbate_similarity,
        "deduplicate_atoms_by_similarity",
        deduplicate,
    )

    _, used_threshold, kept_row_ids, _ = coadsorbate_similarity._prune_single_adsorbate_modes(
        modes,
        reference_row_id=1,
        similarity_threshold=0.99,
        similarity_relaxation=0.02,
        similarity_kwargs={},
    )

    assert seen["threshold"] == 0.99
    assert used_threshold == 0.99
    assert kept_row_ids == [1, 2]


def test_similarity_cutoff_one_skips_all_similarity_work(monkeypatch):
    atoms = [Atoms("H"), Atoms("H")]
    monkeypatch.setattr(
        coadsorbate_similarity,
        "_suggest_threshold_safe",
        lambda *args, **kwargs: pytest.fail(
            "cutoff 1 should not build SOAP descriptors"
        ),
    )
    monkeypatch.setattr(
        coadsorbate_similarity,
        "deduplicate_atoms_by_similarity",
        lambda *args, **kwargs: pytest.fail("cutoff 1 should not run similarity"),
    )

    threshold_df, suggested, uncapped, threshold = (
        coadsorbate_similarity._resolve_duplicate_cutoff(
            atoms,
            reference_row_id=1,
            explicit_cutoff=1,
            suggested_cutoff_scale=1,
            similarity_kwargs={},
        )
    )
    kept, used_threshold, kept_row_ids, report = (
        coadsorbate_similarity._prune_single_adsorbate_modes(
            atoms,
            reference_row_id=1,
            similarity_threshold=1,
            similarity_relaxation=0,
            similarity_kwargs={},
            batch_size=1,
        )
    )

    assert (suggested, uncapped, threshold) == (1.0, 1.0, 1.0)
    assert threshold_df.attrs["suggested_threshold"] == 1.0
    assert kept == atoms
    assert used_threshold == 1.0
    assert kept_row_ids == [1, 2]
    assert report["kept"].all()
    assert report["max_similarity_to_kept"].isna().all()


def test_similarity_deduplication_skips_recombination_when_survivors_exceed_batch(monkeypatch):
    calls = []

    def deduplicate(atoms_list, **kwargs):
        calls.append(len(atoms_list))
        report = pd.DataFrame(
            {
                "row_id": range(1, len(atoms_list) + 1),
                "kept": [True] * (len(atoms_list) - 1) + [False],
                "max_similarity_to_kept": np.nan,
                "closest_kept_row_id": np.nan,
            }
        )
        return list(atoms_list[:-1]), report

    monkeypatch.setattr(coadsorbate_similarity, "_deduplicate_atoms_safe", deduplicate)
    atoms = [Atoms("H") for _ in range(450)]

    kept, report = coadsorbate_similarity._deduplicate_atoms_in_batches(
        atoms,
        batch_size=200,
        similarity_threshold=0.99,
        similarity_kwargs={},
    )

    assert calls == [200, 200, 50]
    assert len(kept) == 447
    assert report["row_id"].tolist() == list(range(1, 451))
    assert report["recombine_row_id"].notna().sum() == 447
    assert report["kept"].sum() == 447


def test_h_probe_equivalents_recover_nonclashing_pair_without_extra_relaxation():
    support = Atoms("Au2", positions=[[0, 0, 0], [4, 0, 0]])
    probe_modes = [
        support + Atoms("H", positions=[[0, 0, 2]]),
        support + Atoms("H", positions=[[4, 0, 2]]),
    ]
    probe_report = pd.DataFrame(
        {
            "row_id": [1, 2],
            "kept": [True, False],
            "closest_kept_row_id": [np.nan, 1],
        }
    )
    a_mode = support + Atoms("C", positions=[[0.2, 0, 2.1]])
    b_mode = support + Atoms("O", positions=[[0, 0.2, 2.1]])

    expanded_a = coadsorbate_similarity._expand_modes_to_probe_equivalents(
        [a_mode],
        [{"site_indices": "0"}],
        n_support_atoms=2,
        representative_site_indices=[2],
        probe_modes=probe_modes,
        probe_site_indices=[2, 5],
        probe_report=probe_report,
    )
    expanded_b = coadsorbate_similarity._expand_modes_to_probe_equivalents(
        [b_mode],
        [{"site_indices": "0"}],
        n_support_atoms=2,
        representative_site_indices=[2],
        probe_modes=probe_modes,
        probe_site_indices=[2, 5],
        probe_report=probe_report,
    )
    combined, _ = coadsorbate_similarity.combine_single_adsorbate_modes(
        expanded_a,
        expanded_b,
        n_support_atoms=2,
        skip_abnormal_bond_screening=True,
        allow_lateral_spacing_fallback=False,
    )
    assert len(expanded_a) == len(expanded_b) == 2
    assert len(combined) == 2
    assert np.allclose(expanded_a[0].positions[-1], [0.2, 0, 2.1])
    assert np.allclose(expanded_a[1].positions[-1], probe_modes[1].positions[-1])
    assert np.allclose(expanded_b[1].positions[-1], probe_modes[1].positions[-1])


def test_site_probe_is_screened_once_and_shared_by_both_adsorbates(monkeypatch):
    prefix = Path("scratch") / f"test-site-probe-{uuid4().hex}"
    calls = []
    relaxation_calls = []
    seen_cutoffs = []

    def generate(_atoms, *, adsorbates, output_db, **kwargs):
        calls.append((adsorbates[0].get_chemical_symbols(), kwargs.get("allowed_site_indices")))
        if len(calls) == 1:
            with connect(output_db) as db:
                for site_index in (2, 5, 7):
                    db.write(
                        Atoms("AuH"),
                        data={"site_indices": str(site_index), "placements": f"H@{site_index}"},
                    )
            return [Atoms("AuH") for _ in range(3)]
        return [Atoms("AuH")]

    def prune(modes, *, similarity_threshold, **kwargs):
        seen_cutoffs.append(similarity_threshold)
        report = pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "kept": [True, False, True],
                "max_similarity_to_kept": [None, 0.98, 0.90],
                "closest_kept_row_id": [None, 1, 1],
            }
        )
        return modes[:2], similarity_threshold, [1, 3], report

    monkeypatch.setattr(coadsorbate_similarity, "run_coadsorbate_generation", generate)
    monkeypatch.setattr(coadsorbate_similarity, "_prune_single_adsorbate_modes", prune)
    def stop_after_generation(*args, **kwargs):
        relaxation_calls.append(kwargs)
        raise RuntimeError("stop after generation")

    monkeypatch.setattr(
        coadsorbate_similarity, "_relax_adsorbates_with_mace", stop_after_generation
    )

    try:
        with pytest.raises(RuntimeError, match="stop after generation"):
            coadsorbate_similarity.run_coadsorbate_similarity_workflow(
                Atoms("Au"),
                adsorbates=[Atoms("C"), Atoms("O")],
                candidates_db_path=prefix.with_suffix(".db"),
                pre_relax_db_path=prefix.with_name(prefix.name + "-pre.db"),
                relaxed_db_path=prefix.with_name(prefix.name + "-relaxed.db"),
                final_db_path=prefix.with_name(prefix.name + "-final.db"),
                site_probe_report_csv_path=prefix.with_name(prefix.name + "-probe.csv"),
                candidate_report_csv_path=prefix.with_name(prefix.name + "-candidate.csv"),
                relaxed_minimum_report_csv_path=prefix.with_name(prefix.name + "-minimum.csv"),
                relaxation_report_csv_path=prefix.with_name(prefix.name + "-relax.csv"),
                summary_csv_path=prefix.with_name(prefix.name + "-summary.csv"),
                summary_txt_path=prefix.with_name(prefix.name + "-summary.txt"),
                site_probe_symbol="H",
                site_probe_similarity_cutoff=0.97,
                support_relaxation_origin_atom_index=0,
                support_relaxation_neighbor_shells=0,
                mace_enable_cueq=False,
                verbose=False,
            )

        assert seen_cutoffs == [0.97]
        assert calls == [(["H"], None), (["C"], [2, 7]), (["O"], [2, 7])]
        assert relaxation_calls[0]["relaxed_support_indices"] == [0]
        assert relaxation_calls[0]["max_steps"] == 200
        assert relaxation_calls[0]["enable_cueq"] is False
        probe_report = pd.read_csv(prefix.with_name(prefix.name + "-probe.csv"))
        assert probe_report["site_index"].tolist() == [2, 5, 7]
        assert probe_report["closest_kept_site_index"].iloc[1] == 2
    finally:
        for path in prefix.parent.glob(prefix.name + "*"):
            path.unlink(missing_ok=True)


def test_single_adsorbate_uses_site_probe_before_generation(monkeypatch):
    prefix = Path("scratch") / f"test-single-site-probe-{uuid4().hex}"
    calls = []

    def generate(_atoms, *, adsorbates, output_db, **kwargs):
        calls.append((adsorbates[0].get_chemical_symbols(), kwargs.get("allowed_site_indices")))
        if len(calls) == 1:
            with connect(output_db) as db:
                for site_index in (2, 5, 7):
                    db.write(
                        Atoms("AuH"),
                        data={"site_indices": str(site_index), "placements": f"H@{site_index}"},
                    )
            return [Atoms("AuH") for _ in range(3)]
        raise RuntimeError("stop after single generation")

    def prune(modes, **_kwargs):
        report = pd.DataFrame(
            {
                "row_id": [1, 2, 3],
                "kept": [True, False, True],
                "max_similarity_to_kept": [None, 0.98, 0.90],
                "closest_kept_row_id": [None, 1, 1],
            }
        )
        return modes[:2], 0.97, [1, 3], report

    monkeypatch.setattr(coadsorbate_similarity, "run_coadsorbate_generation", generate)
    monkeypatch.setattr(coadsorbate_similarity, "_prune_single_adsorbate_modes", prune)

    try:
        with pytest.raises(RuntimeError, match="stop after single generation"):
            coadsorbate_similarity.run_coadsorbate_similarity_workflow(
                Atoms("Au"),
                adsorbates=[Atoms("C")],
                candidates_db_path=prefix.with_suffix(".db"),
                pre_relax_db_path=prefix.with_name(prefix.name + "-pre.db"),
                relaxed_db_path=prefix.with_name(prefix.name + "-relaxed.db"),
                final_db_path=prefix.with_name(prefix.name + "-final.db"),
                site_probe_report_csv_path=prefix.with_name(prefix.name + "-probe.csv"),
                candidate_report_csv_path=prefix.with_name(prefix.name + "-candidate.csv"),
                relaxed_minimum_report_csv_path=prefix.with_name(prefix.name + "-minimum.csv"),
                relaxation_report_csv_path=prefix.with_name(prefix.name + "-relax.csv"),
                summary_csv_path=prefix.with_name(prefix.name + "-summary.csv"),
                summary_txt_path=prefix.with_name(prefix.name + "-summary.txt"),
                site_probe_similarity_cutoff=0.97,
                verbose=False,
            )

        assert calls == [(["H"], None), (["C"], [2, 7])]
    finally:
        for path in prefix.parent.glob(prefix.name + "*"):
            path.unlink(missing_ok=True)


def test_relaxed_structures_are_filtered_then_sorted_by_energy():
    initial = [
        Atoms("AuH", positions=[[0, 0, 0], [1.5, 0, 0]]),
        Atoms("AuH", positions=[[0, 0, 0], [1.5, 0, 0]]),
        Atoms("AuH", positions=[[0, 0, 0], [1.5, 0, 0]]),
    ]
    relaxed = [atoms.copy() for atoms in initial]
    for atoms, energy in zip(relaxed, (2.0, 1.0, 0.0)):
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((2, 3)))
    report = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "energy_eV": [2.0, 1.0, 0.0],
            "converged": [True, True, False],
        }
    )

    records, kept = coadsorbate_similarity._filter_and_sort_relaxed(
        initial,
        relaxed,
        report,
        [{"name": "high"}, {"name": "low"}, {"name": "unconverged"}],
        n_support_atoms=1,
        require_converged=True,
        require_bound=True,
    )

    assert len(records) == 3
    assert [record[3]["name"] for record in kept] == ["low", "high"]


def test_relaxed_filter_checks_coadsorbate_integrity_per_adsorbate():
    initial = Atoms(
        "AuCHOH",
        positions=[[0, 0, 0], [2, 0, 0], [3.1, 0, 0], [2, 4, 0], [3, 4, 0]],
    )
    relaxed = initial.copy()
    relaxed.positions[3:] -= [0, 2.9, 0]
    relaxed.calc = SinglePointCalculator(relaxed, energy=0.0, forces=np.zeros((5, 3)))
    report = pd.DataFrame({"row_id": [1], "energy_eV": [0.0], "converged": [True]})

    _, kept = coadsorbate_similarity._filter_and_sort_relaxed(
        [initial],
        [relaxed],
        report,
        [{"a_adsorbate_n_atoms": 2, "b_adsorbate_n_atoms": 2}],
        n_support_atoms=1,
        require_converged=True,
        require_bound=True,
    )

    assert len(kept) == 1


def test_relaxation_checkpoint_rejects_changed_input_geometry(monkeypatch):
    calculators = ModuleType("mace.calculators")
    calculators.mace_mp = lambda **kwargs: _Calculator()
    mace = ModuleType("mace")
    mace.calculators = calculators
    monkeypatch.setitem(sys.modules, "mace", mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)

    class CompleteOptimisation:
        count = 0

        def __init__(self, atoms, **kwargs):
            type(self).count += 1
            self.nsteps = 0

        def run(self, **kwargs):
            pass

    monkeypatch.setattr(coadsorbate_similarity, "BFGS", CompleteOptimisation)
    db_path = Path("scratch") / f"test-checkpoint-hash-{uuid4().hex}.db"
    kwargs = dict(
        n_fixed_atoms=1,
        model_path="model",
        fmax=0.05,
        max_steps=10,
        device="cpu",
        dispersion=False,
        default_dtype="float64",
        output_db_path=db_path,
        verbose=False,
    )

    try:
        coadsorbate_similarity._relax_adsorbates_with_mace(
            [Atoms("Au", positions=[[0, 0, 0]])], **kwargs
        )
        coadsorbate_similarity._relax_adsorbates_with_mace(
            [Atoms("Au", positions=[[1, 0, 0]])], **kwargs
        )

        assert CompleteOptimisation.count == 2
        assert connect(db_path).count() == 2
    finally:
        db_path.unlink(missing_ok=True)


def test_filtered_single_mode_is_written_for_dft_without_recalculation():
    db_path = Path("scratch") / f"test-single-mode-dft-{uuid4().hex}.db"
    atoms = Atoms("AuO", positions=[[0, 0, 0], [0, 0, 2]])
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((2, 3)))
    report = pd.DataFrame(
        {
            "row_id": [1, 2],
            "kept": [False, True],
            "max_similarity_to_kept": [None, 0.95],
        }
    )

    try:
        coadsorbate_similarity._write_single_mode_dft_candidates(
            db_path,
            [atoms],
            [{"energy_eV": -2.0}, {"energy_eV": -1.0}],
            [2],
            report,
            [],
        )

        with connect(db_path) as db:
            row = db.get(1)
        assert row.data.energy_eV == -1.0
        assert row.data.single_mode_max_similarity_to_kept == 0.95
    finally:
        db_path.unlink(missing_ok=True)
