"""Seedless Cu/Pb growth with unrestricted exploration followed by staged beam search."""

import os
from pathlib import Path

from ase import Atoms

from utilities.similarity.alloy_nanocluster_growth_workflow import (
    grow_alloy_nanocluster_staged,
)


def main() -> None:
    grow_alloy_nanocluster_staged(
        Atoms(),
        add_symbols=["Cu", "Pb"],
        max_atoms=60,
        staging_start_atoms=30,
        concentration_symbol="Cu",
        concentration_span=0.05,
        restart_beam_width=3,
        staging_require_alloy=True,
        output_root=Path("scratch/CuPb_cluster_staged"),
        keep_per_composition=1,
        min_per_composition=1,
        max_structures_per_parent_symbol=100,
        mace_model_path=os.environ["MODEL_PATH"],
        mace_device="cuda",
        mace_fmax=0.05,
        mace_max_steps=200,
        mace_default_dtype="float64",
        mace_dispersion=True,
        mace_enable_cueq=False,
        similarity_threshold=0.9999,
        secondary_similarity_threshold=0.9999,
        similarity_batch_size=100,
        site_symbols=["Cu", "Pb"],
        force_triangulated_fallback=True,
        rattle_parent=True,
    )


if __name__ == "__main__":
    main()
