import importlib.util
from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

SCRIPT = Path(__file__).parents[2] / "cats" / "working_directory" / "mace_surface_wulff.py"
SPEC = importlib.util.spec_from_file_location("mace_surface_wulff", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_surface_energy_equations_and_hkl_parser():
    cleave, relax, surface = MODULE.paper_surface_energy(
        unrelaxed_slab_eV=-46.0,
        relaxed_slab_eV=-47.0,
        bulk_eV_per_atom=-5.0,
        atom_count=10,
        area_A2=10.0,
    )

    assert MODULE.parse_hkl("322") == (3, 2, 2)
    assert MODULE.parse_hkl("(1, 1, 0)") == (1, 1, 0)
    assert len(MODULE.cubic_surface_families(4)) == 22
    assert len(MODULE.cubic_surface_families(5)) > 22
    assert (cleave, relax, surface) == pytest.approx((0.2, -0.1, 0.1))


def test_completed_surface_rows():
    path = Path(__file__).with_name("_surface_rows.csv")
    try:
        MODULE.append_csv_row(path, {"h": 5, "k": 5, "l": 4, "surface_J_m2": 1.2})
        rows = MODULE.completed_surface_rows(path)
        assert rows[(5, 5, 4)]["surface_J_m2"] == "1.2"
    finally:
        path.unlink(missing_ok=True)


def test_wulff_clusters_are_written_incrementally():
    trajectory_path = Path(__file__).with_name("_clusters.traj")
    csv_path = Path(__file__).with_name("_clusters.csv")
    try:
        rows = MODULE.build_wulff_series(
            symbol="Pd",
            structure="fcc",
            lattice_constant_A=3.89,
            surfaces=list(MODULE.DEFAULT_SURFACES[:3]),
            surface_energies_J_m2=[1.0, 1.1, 1.2],
            min_atoms=1,
            max_atoms=21,
            size_step=20,
            calculator=EMT(),
            trajectory_path=trajectory_path,
            csv_path=csv_path,
        )

        clusters = MODULE.read(trajectory_path, ":")
        assert clusters
        assert csv_path.read_text().count("\n") == len(rows) + 1
        assert all(not cluster.pbc.any() and cluster.cell.rank == 0 for cluster in clusters)
        assert all("mace_energy_eV" in row and "mace_energy_eV_per_atom" in row for row in rows)
    finally:
        trajectory_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)


def test_wulff_accepts_all_fcc_surfaces_through_index_five():
    factory = MODULE.WULFF_FACTORIES["fcc"]
    factory.set_atomic_numbers("Cu")
    factory.lattice_constant = 3.63
    factory.set_basis()
    assert factory.get_layer_distance((5, 2, 2)) == pytest.approx(
        3.63 / (2 * MODULE.np.sqrt(33))
    )

    cluster = MODULE.wulff_cluster(
        symbol="Cu",
        structure="fcc",
        lattice_constant_A=3.63,
        surfaces=list(MODULE.cubic_surface_families(5)),
        energies=[1.0] * 40,
        size=20,
    )

    assert len(cluster) > 0


def test_high_index_slab_matches_100_thickness_and_fixed_depth():
    optimized_bulk = bulk("Pd", "fcc", cubic=True)
    reference = MODULE.make_slab(
        optimized_bulk,
        (1, 0, 0),
        layers=7,
        minimum_thickness_A=None,
        vacuum_A=5.0,
        repeat_xy=(1, 1),
        layer_tolerance_A=0.2,
    )
    reference_thickness = MODULE.np.ptp(MODULE.surface_heights(reference))
    reference_fixed_depth = MODULE.bottom_layer_depth(
        reference,
        bottom_layers=3,
        layer_tolerance_A=0.2,
    )
    slab = MODULE.make_slab(
        optimized_bulk,
        (3, 2, 2),
        layers=7,
        minimum_thickness_A=reference_thickness,
        vacuum_A=5.0,
        repeat_xy=(1, 1),
        layer_tolerance_A=0.2,
    )
    _, _, fixed_depth = MODULE.constrain_bottom_depth(
        slab,
        minimum_depth_A=reference_fixed_depth,
        layer_tolerance_A=0.2,
    )

    slab_thickness = MODULE.np.ptp(MODULE.surface_heights(slab))
    assert slab_thickness >= reference_thickness - 0.2
    assert fixed_depth >= reference_fixed_depth - 0.2

    tags, _ = MODULE.surface_layers(slab, 0.2)
    fixed = set(slab.constraints[0].get_indices())
    assert all(
        all(index in fixed for index in (tags == tag).nonzero()[0])
        or all(index not in fixed for index in (tags == tag).nonzero()[0])
        for tag in set(tags)
    )
