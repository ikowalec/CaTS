from ase import Atoms

from core.mep_render import analyze_structure_connectivity
from workflows.views import _generate_mlneb_comparison_plot


def test_connectivity_uses_known_cluster_formula_for_supported_cluster():
    atoms = Atoms(
        "AuCeO2CO",
        positions=[
            (0, 0, 0),
            (2.5, 0, 0),
            (2.5, 0, 1.8),
            (2.5, 1.8, 0),
            (8, 0, 0),
            (8, 0, 1.1),
        ],
    )

    assert analyze_structure_connectivity(atoms, "AuCeO2") == ("AuCeO2", "CO")


def test_comparison_plot_supports_two_image_paths():
    mep = {
        "run_id": 1,
        "shifts": [0.0, 1.0],
        "rel_e": [0.0, 0.2],
        "ts_shift": 1.0,
        "forward_barrier_eV": 0.2,
        "reverse_barrier_eV": 0.1,
    }

    assert _generate_mlneb_comparison_plot(mep, {**mep, "run_id": 2})
