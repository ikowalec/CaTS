import os
import sys

from ase import Atoms
from ase.build import molecule
from ase.io import read

sys.path.append(os.getcwd())

from similarity.workflows.coadsorbate_similarity import run_coadsorbate_similarity_workflow

structure = read("atoms.traj")
co = molecule("CO")
o = Atoms("O", positions=[[0, 0, 0]])  # Simple single-atom second adsorbate.

result = run_coadsorbate_similarity_workflow(
    structure,
    adsorbates=[co, o],
    candidates_db_path="scratch/coadsorption_candidates.db",
    pre_relax_db_path="scratch/coadsorption_prefilter.db",
    relaxed_db_path="scratch/coadsorption_relaxed.db",
    final_db_path="scratch/coadsorption_final_for_dft.db",
    single_a_final_db_path="scratch/coadsorption_single_a_final_for_dft.db",
    single_b_final_db_path="scratch/coadsorption_single_b_final_for_dft.db",
    site_probe_report_csv_path="scratch/coadsorption_site_probe_similarity.csv",
    candidate_report_csv_path="scratch/coadsorption_candidate_similarity.csv",
    relaxed_minimum_report_csv_path="scratch/coadsorption_relaxed_minimum_similarity.csv",
    relaxation_report_csv_path="scratch/coadsorption_relaxation_report.csv",
    summary_csv_path="scratch/coadsorption_workflow_summary.csv",
    summary_txt_path="scratch/coadsorption_workflow_summary.txt",
    candidate_reference_row_id=1,
    relaxed_minimum_reference_row_id=1,
    site_probe_symbol="H",
    site_probe_similarity_cutoff=0.99,
    single_mode_duplicate_cutoff=0.99,
    candidate_duplicate_cutoff=None,
    candidate_suggested_cutoff_scale=1.0,
    relaxed_minimum_duplicate_cutoff=None,
    relaxed_minimum_suggested_cutoff_scale=1.0,
    mace_model_path="similarity/run/mace-omat-0-medium.model",
    mace_fmax=0.01,
    mace_max_steps=200,
    mace_device="cuda",
    mace_dispersion=False,
    mace_default_dtype="float64",
    keep_bound_only=False,
    require_converged=True,
    generation_kwargs={
        "site_symbols": ["Ti", "Au", "Cu"],
        "passive_symbols": ["Ce", "O"],
        "height": 2.0,
        "z_cutoff": None,  # optional
        "z_buffer": None,  # optional
        "origin_atom_index": None,  # optional 0-based input atom index
        "site_cutoff": None,  # Angstrom; set together with origin_atom_index
    },
    similarity_kwargs={
        "support_species": ["Zn", "Ce", "Ti", "O"],
        "slab_shave_distance": None,  # optional
        "kernel_gamma": 1.0,
        "kernel_alpha": 1.0,
        "kernel_threshold": 1e-6,
    },
    show_final=False,
    verbose=True,
)

print(result["summary_df"])
