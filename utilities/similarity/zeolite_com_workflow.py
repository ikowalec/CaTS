import os
import sys

from ase.build import molecule
from ase.io import read

sys.path.append(os.getcwd())

from similarity.workflows.zeolite_ring_adsorbate import run_zeolite_com_similarity_workflow


zeolite = read("zeolite.traj")

# Replace with your cage/ring center coordinate in Angstrom.
cage_com = [12.0, 8.5, 14.2]

result = run_zeolite_com_similarity_workflow(
    zeolite_atoms=zeolite,
    cage_com=cage_com,
    adsorbates=[molecule("H2O"), molecule("CH3OH")],
    site_symbols=["H", "O"],
    target_radius=3.2,
    use_framework_cage_boundary=True,
    cage_boundary_n_nearest=12,
    cage_boundary_margin=0.0,
    relax_cage_boundary_if_empty=True,
    relaxed_cage_boundary_radius=None,
    insertion_distance=1.2,
    cage_padding=0.2,
    max_site_candidates=24,
    max_structures=200,
    min_site_separation=1.5,
    min_initial_adsorbate_framework_distance=0.8,
    min_initial_adsorbate_adsorbate_distance=1.0,
    enforce_nonbonded_coadsorbates=True,
    nonbonded_coadsorbate_scale=1.10,
    resolve_initial_overlaps=True,
    overlap_resolution_max_steps=32,
    placement_progress_interval=20,
    overlap_detailed_logging=False,
    overlap_progress_interval=5,
    min_physical_adsorbate_framework_distance=1.0,
    enforce_inside_target_radius=True,
    relax_inside_radius_filter_if_empty=True,
    inside_radius_tolerance=0.15,
    similarity_threshold=None,
    reference_row_id=1,
    similarity_kwargs={
        "support_species": ["Si", "Al", "O", "H"],
        "slab_shave_distance": None,
        "kernel_gamma": 1.0,
        "kernel_alpha": 1.0,
        "kernel_threshold": 1e-6,
    },
    candidates_db_path="scratch/zeolite_com_candidates.db",
    prefilter_db_path="scratch/zeolite_com_prefilter.db",
    constrained_relaxed_db_path="scratch/zeolite_com_constrained_relaxed.db",
    final_relaxed_db_path="scratch/zeolite_com_final_relaxed.db",
    similarity_report_csv_path="scratch/zeolite_com_similarity_report.csv",
    constrained_report_csv_path="scratch/zeolite_com_constrained_report.csv",
    final_report_csv_path="scratch/zeolite_com_final_report.csv",
    summary_txt_path="scratch/zeolite_com_summary.txt",
    mace_model_path="similarity/run/mace-omat-0-medium.model",
    mace_device="cpu",
    mace_dispersion=False,
    mace_default_dtype="float64",
    crude_fmax=0.20,
    crude_max_steps=500,
    crude_mace_maxstep=0.01,
    final_fmax=0.05,
    final_max_steps=1000,
    final_mace_maxstep=0.01,
    constrain_adsorbate_angles=False,
    adsorbate_bond_constraint_mode="nearest",
    fixinternals_epsilon=None,
    fixinternals_maxiter=400,
    fixinternals_relaxed_retry=True,
    fixinternals_relaxed_epsilon=1.0e-4,
    fixinternals_relaxed_maxiter=800,
    fixinternals_relaxed_step_scale=0.5,
    mace_trajectory_dir="scratch/zeolite_com_mace_trajectories",
    show_final=False,
    verbose=True,
)

print(result["final_report_df"])
