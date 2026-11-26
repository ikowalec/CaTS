import numpy as np
import pytest
from inspect import signature
from ase import Atoms

import cats.working_directory.coadsorbate_generator as generator
from cats.working_directory.coadsorbate_generator import (
    AdsorbateSpec,
    _adsorbate_orientations,
    _adjust_contact_distance,
    _is_carbonaceous_periodic_support,
    _passes_bonding_filter,
    _sites_near_origin,
    _triangulated_com_sites,
    adsorbates_bound_to_support,
    adsorbate_integrity_preserved,
    combine_single_adsorbate_modes,
    run_coadsorbate_generation,
)


def test_single_atom_growth_checks_only_new_clashes():
    candidate = Atoms("Cu3", positions=[[0, 0, 0], [0.1, 0, 0], [3, 0, 0]])
    assert _passes_bonding_filter(
        candidate, skip_abnormal_bond_screening=True, n_added=1
    )

    candidate.positions[-1] = [0.2, 0, 0]
    assert not _passes_bonding_filter(
        candidate, skip_abnormal_bond_screening=True, n_added=1
    )


def test_generator_exposes_allowed_site_indices():
    assert "allowed_site_indices" in signature(run_coadsorbate_generation).parameters
    assert "force_triangulated_com_fallback" in signature(run_coadsorbate_generation).parameters
    assert "skip_geometry_screening" in signature(run_coadsorbate_generation).parameters


def test_generator_can_defer_geometry_screening(monkeypatch):
    monkeypatch.setattr(
        generator,
        "_passes_bonding_filter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("geometry screened")),
    )
    atoms = Atoms("Cu3", positions=[[0, 0, 0], [2, 0, 0], [1, 1.7, 0]])

    candidates = run_coadsorbate_generation(
        atoms,
        adsorbates=["H"],
        output_db=None,
        passive_symbols=[],
        max_structures=1,
        force_triangulated_com_fallback=True,
        skip_geometry_screening=True,
        verbose=False,
    )

    assert len(candidates) == 1


def test_sites_near_origin_uses_periodic_distance_and_validates_paired_options():
    atoms = Atoms("Au2", positions=[[0, 0, 0], [5, 0, 0]], cell=[10, 10, 10], pbc=True)
    sites = [
        {"position": np.array([9.5, 0, 0])},
        {"position": np.array([5.0, 0, 0])},
    ]

    assert _sites_near_origin(sites, atoms, 0, 1.0) == [sites[0]]
    with pytest.raises(ValueError, match="provided together"):
        _sites_near_origin(sites, atoms, 0, None)


def test_carbonaceous_periodic_fallback_includes_all_site_types():
    atoms = Atoms(
        "C3Au",
        positions=[[0, 0, 0], [2.4, 0, 0], [1.2, 2.0, 0], [1.2, 0.7, 2.0]],
        cell=[8, 8, 12],
        pbc=[True, True, False],
    )

    assert _is_carbonaceous_periodic_support(atoms)
    sites = _triangulated_com_sites(atoms, neighbor_cutoff=3.0, top_surface_only=True)
    assert {site["site"] for site in sites} == {"fcc", "bridge", "ontop"}
    capped = _triangulated_com_sites(
        atoms, neighbor_cutoff=3.0, max_sites=3, top_surface_only=True
    )
    assert {site["site"] for site in capped} == {"fcc", "bridge", "ontop"}


def test_periodic_fallback_constructs_sites_across_cell_boundaries():
    atoms = Atoms(
        "C3",
        positions=[[0.2, 0.0, 0.0], [9.8, 0.0, 0.0], [0.0, 0.4, 0.0]],
        cell=[10.0, 10.0, 12.0],
        pbc=[True, True, False],
    )

    sites = _triangulated_com_sites(atoms, neighbor_cutoff=1.0, top_surface_only=True)

    hollow = next(site for site in sites if site["site"] == "fcc")
    bridge = next(
        site for site in sites
        if site["site"] == "bridge" and set(site["indices"]) == {0, 1}
    )
    assert set(hollow["indices"]) == {0, 1, 2}
    assert np.isclose(atoms.get_distance(0, 1, mic=True), 0.4)
    assert np.isclose(abs(bridge["position"][0] % 10.0), 0.0)


def test_forced_fallback_bypasses_acat(monkeypatch, capsys):
    atoms = Atoms(
        "Cu3",
        positions=[[0.5, 0.0, 0.0], [9.5, 0.0, 0.0], [0.0, 0.9, 0.0]],
        cell=[10.0, 10.0, 12.0],
        pbc=[True, True, False],
    )
    monkeypatch.setattr(
        generator,
        "_slab_sites",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ACAT called")),
    )
    monkeypatch.setattr(
        generator,
        "connect",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("DB opened")),
    )

    candidates = run_coadsorbate_generation(
        atoms,
        adsorbates=["H"],
        output_db=None,
        site_symbols=["Cu"],
        passive_symbols=[],
        max_structures=1,
        force_triangulated_com_fallback=True,
        triangle_neighbor_cutoff=1.5,
        skip_abnormal_bond_screening=True,
        verbose=True,
    )

    assert len(candidates) == 1
    output = capsys.readouterr().out
    assert "Detecting adsorption sites with Delaunay triangulation" in output
    assert "(Delaunay triangulation)" in output
    assert "fallback" not in output.lower()


def test_candidate_database_is_written_in_one_transaction(monkeypatch):
    events = []

    class MemoryDB:
        def __enter__(self):
            events.append("begin")
            return self

        def __exit__(self, *_args):
            events.append("commit")

        def write(self, *_args, **_kwargs):
            events.append("write")

    monkeypatch.setattr(generator, "connect", lambda *_args, **_kwargs: MemoryDB())
    atoms = Atoms("Cu3", positions=[[0, 0, 0], [2, 0, 0], [1, 1.7, 0]])

    candidates = run_coadsorbate_generation(
        atoms,
        adsorbates=["H"],
        output_db="unused.db",
        passive_symbols=[],
        max_structures=2,
        force_triangulated_com_fallback=True,
        skip_abnormal_bond_screening=True,
        verbose=False,
    )

    assert len(candidates) == 2
    assert events == ["begin", "write", "write", "commit"]


def test_adsorbate_components_must_each_contact_support():
    bound = Atoms("AuCO", positions=[[0, 0, 0], [2.0, 0, 0], [3.1, 0, 0]])
    detached = Atoms("AuCO", positions=[[0, 0, 0], [5.0, 0, 0], [6.1, 0, 0]])

    assert adsorbates_bound_to_support(bound, 1)
    assert not adsorbates_bound_to_support(detached, 1)


def test_adsorbate_integrity_rejects_dissociation_and_collapse():
    initial = Atoms(
        "AuCOH", positions=[[0, 0, 0], [2, 0, 0], [3.1, 0, 0], [0, 2, 0]]
    )
    intact = initial.copy()
    dissociated = initial.copy()
    dissociated.positions[2] = [6, 0, 0]
    collapsed = initial.copy()
    collapsed.positions[3] = [3.1, 0.7, 0]
    compressed = initial.copy()
    compressed.positions[2] = [2.4, 0, 0]

    assert adsorbate_integrity_preserved(initial, intact, 1)
    assert not adsorbate_integrity_preserved(initial, dissociated, 1)
    assert not adsorbate_integrity_preserved(initial, collapsed, 1)
    assert not adsorbate_integrity_preserved(initial, compressed, 1)


def test_adsorbate_integrity_ignores_new_bonds_between_distinct_adsorbates():
    initial = Atoms(
        "AuCHOH",
        positions=[[0, 0, 0], [2, 0, 0], [3.1, 0, 0], [2, 4, 0], [3, 4, 0]],
    )
    relaxed = initial.copy()
    relaxed.positions[3:] -= [0, 2.9, 0]

    assert not adsorbate_integrity_preserved(initial, relaxed, 1)
    assert adsorbate_integrity_preserved(
        initial,
        relaxed,
        1,
        adsorbate_groups=((1, 2), (3, 4)),
    )


def test_single_adsorbate_modes_are_paired():
    a_modes = [Atoms("AuH", positions=[[0, 0, 0], [2, 0, 0]])]
    b_modes = [Atoms("AuO", positions=[[0, 0, 0], [0, 2, 0]])]

    combined, metadata = combine_single_adsorbate_modes(
        a_modes,
        b_modes,
        n_support_atoms=1,
        skip_abnormal_bond_screening=True,
    )

    assert len(combined) == 1
    assert combined[0].get_chemical_symbols() == ["Au", "H", "O"]
    assert metadata == [
        {
            "a_mode_row_id": 1,
            "b_mode_row_id": 1,
            "a_adsorbate_n_atoms": 1,
            "b_adsorbate_n_atoms": 1,
            "generation": "sequential_single_mode_pairing",
        }
    ]


def test_clashing_single_modes_use_lateral_spacing_fallback():
    a_modes = [Atoms("AuH", positions=[[0, 0, 0], [1, 0, 2]])]
    b_modes = [Atoms("AuO", positions=[[0, 0, 0], [1, 0, 2]])]
    unshifted = a_modes[0].copy()
    unshifted += b_modes[0][1:]
    system_com = unshifted.get_center_of_mass()

    combined, metadata = combine_single_adsorbate_modes(
        a_modes,
        b_modes,
        n_support_atoms=1,
        skip_abnormal_bond_screening=True,
    )

    assert len(combined) == 1
    a_shift = combined[0].positions[-2] - a_modes[0].positions[-1]
    b_shift = combined[0].positions[-1] - b_modes[0].positions[-1]
    assert np.isclose(np.linalg.norm(a_shift), np.linalg.norm(b_shift))
    assert combined[0].get_distance(1, 2) > 1.10 * (0.31 + 0.66)
    assert np.linalg.norm(combined[0].positions[-2, :2] - system_com[:2]) > np.linalg.norm(
        a_modes[0].positions[-1, :2] - system_com[:2]
    )
    assert np.linalg.norm(combined[0].positions[-1, :2] - system_com[:2]) > np.linalg.norm(
        b_modes[0].positions[-1, :2] - system_com[:2]
    )
    assert metadata[0]["generation"] == "sequential_pairing_with_lateral_spacing"
    assert "a_lateral_shift_A" in metadata[0]
    assert "b_lateral_shift_A" in metadata[0]


def test_adsorbate_orientations_sample_z_rotations_and_180_flip():
    adsorbate = Atoms("CO", positions=[[1, 2, 3], [2, 2, 4]])
    variants = _adsorbate_orientations(
        AdsorbateSpec("CO", "co", adsorbate),
        rotation_step_degrees=30,
        include_flip=True,
    )

    assert len(variants) == 24
    assert {variant.rotation_z_degrees for variant in variants} == set(range(0, 360, 30))
    assert all(np.allclose(variant.payload.positions[0], [1, 2, 3]) for variant in variants)
    flipped = next(
        variant
        for variant in variants
        if variant.flipped and variant.rotation_z_degrees == 0
    )
    assert np.allclose(flipped.payload.positions[1], [2, 2, 2])


def test_single_atom_adsorbate_has_one_orientation():
    adsorbate = AdsorbateSpec("O", "o", Atoms("O"))

    variants = _adsorbate_orientations(
        adsorbate,
        rotation_step_degrees=30,
        include_flip=True,
    )

    assert variants == [adsorbate]


def test_contact_distance_uses_nearest_atom_after_flip():
    site = {"normal": np.array([0.0, 0.0, 1.0])}

    hydrogen = Atoms("CH", positions=[[0, 0, 0], [0, 0, 1.8]])
    contact = _adjust_contact_distance(
        hydrogen,
        adsorbate_indices=[1],
        n_support_atoms=1,
        site=site,
        radius_scale=1.0,
    )
    assert contact["adsorbate_symbol"] == "H"
    assert np.isclose(hydrogen.get_distance(0, 1), 0.76 + 0.31)

    methoxy = Atoms("CCO", positions=[[0, 0, 0], [0, 0, 1.8], [0, 0, 0.8]])
    contact = _adjust_contact_distance(
        methoxy,
        adsorbate_indices=[1, 2],
        n_support_atoms=1,
        site=site,
        radius_scale=1.0,
    )
    assert contact["adsorbate_symbol"] == "O"
    assert np.isclose(methoxy.get_distance(0, 2), 0.76 + 0.66)
