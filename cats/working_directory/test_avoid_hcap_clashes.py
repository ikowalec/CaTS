import importlib.util
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

spec = importlib.util.spec_from_file_location(
    "avoid_hcap_clashes", Path(__file__).with_name("avoid_hcap_clashes.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
expand_qm_to_avoid_clash = module.expand_qm_to_avoid_clash
place_hcaps = module.place_hcaps


def test_place_hcaps_maps_cluster_indices_to_qm_indices():
    cluster = Atoms("SiOAl", positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    qm = cluster[[1, 2]]

    capped = place_hcaps(qm, [(1, 2)], cluster=cluster)

    assert len(capped) == 3
    np.testing.assert_allclose(capped[-1].position, (0.05, 0, 0))
    with pytest.raises(IndexError):
        place_hcaps(qm, [(1, 2)])


def test_expanded_qm_keeps_cluster_order():
    cluster = Atoms("SiOAlH", positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
    qm = cluster[[3, 1]]

    expanded = expand_qm_to_avoid_clash(qm, cluster, [(2, [0])])

    np.testing.assert_allclose(expanded.positions, cluster.positions)
