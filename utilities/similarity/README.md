# Co-adsorbate Workflow (Beginner-Friendly Guide)

For shared multi-GPU execution, restart, logging, and checkpoint
behavior across cluster growth, alloy growth, and co-adsorbate placement, see
[`mace_parallel_workflows.md`](mace_parallel_workflows.md).

This guide is written for users who are new to coding/computational workflows.

## What this script does

You give it:
- one input structure (your slab/support),
- one or more adsorbate files.

It will:
1. Create many trial adsorption structures.
2. Remove obvious duplicates.
3. Relax the remaining structures with MACE.
4. Remove duplicates again after relaxation.
5. Save a final set for DFT.

## Quick start

1. Put these files in your working folder:
- `input_structure.extxyz`
- `adsorbates/adsorbate_001.xyz` (and more adsorbates if needed)

2. Activate your environment (example):

```bash
source /srv/research/mace_venv/bin/activate
```

3. Save your Python script as `run_workflow_submit.py`.

4. Run it:

```bash
python run_workflow_submit.py
```

5. Optional: submit on Slurm with your usual submission script:

```bash
sbatch your_submission_script.slurm
```

## Example script (copy/paste)

```python
from pathlib import Path
from ase.io import read
from similarity.workflows.coadsorbate_similarity import run_coadsorbate_similarity_workflow

structure = read("input_structure.extxyz")
adsorbate_paths = ["adsorbates/adsorbate_001.xyz"]
adsorbates = [read(path) for path in adsorbate_paths]

candidates_db_path = Path("scratch/coadsorption_candidates.db")
pre_relax_db_path = Path("scratch/coadsorption_prefilter.db")
relaxed_db_path = Path("scratch/coadsorption_relaxed.db")
final_db_path = Path("scratch/coadsorption_final_for_dft.db")
single_a_final_db_path = Path("scratch/coadsorption_single_a_final_for_dft.db")
single_b_final_db_path = Path("scratch/coadsorption_single_b_final_for_dft.db")
site_probe_report_csv_path = Path("scratch/coadsorption_site_probe_similarity.csv")
candidate_report_csv_path = Path("scratch/coadsorption_candidate_similarity.csv")
relaxed_minimum_report_csv_path = Path("scratch/coadsorption_relaxed_minimum_similarity.csv")
relaxation_report_csv_path = Path("scratch/coadsorption_relaxation_report.csv")
summary_csv_path = Path("scratch/coadsorption_workflow_summary.csv")
summary_txt_path = Path("scratch/coadsorption_workflow_summary.txt")

result = run_coadsorbate_similarity_workflow(
    structure,
    adsorbates=adsorbates,
    candidates_db_path=candidates_db_path,
    pre_relax_db_path=pre_relax_db_path,
    relaxed_db_path=relaxed_db_path,
    final_db_path=final_db_path,
    single_a_final_db_path=single_a_final_db_path,
    single_b_final_db_path=single_b_final_db_path,
    site_probe_report_csv_path=site_probe_report_csv_path,
    candidate_report_csv_path=candidate_report_csv_path,
    relaxed_minimum_report_csv_path=relaxed_minimum_report_csv_path,
    relaxation_report_csv_path=relaxation_report_csv_path,
    summary_csv_path=summary_csv_path,
    summary_txt_path=summary_txt_path,
    candidate_reference_row_id=1,
    relaxed_minimum_reference_row_id=1,
    site_probe_symbol="H",
    site_probe_similarity_cutoff=0.99,
    single_mode_duplicate_cutoff=0.99,
    candidate_duplicate_cutoff=None,
    candidate_suggested_cutoff_scale=1.0,
    relaxed_minimum_duplicate_cutoff=None,
    relaxed_minimum_suggested_cutoff_scale=1.0,
    mace_model_path="/srv/research/mace_models/mace-omat-0-medium.model",
    mace_fmax=0.05,
    mace_max_steps=200,
    mace_device="cuda",
    mace_dispersion=False,
    mace_default_dtype="float64",
    generation_kwargs={
        "site_symbols": ("O", "Ce"),
        "passive_symbols": ("O", "Ce"),
        "height": 3.0,
        "z_cutoff": 5.0,
        "z_buffer": None,
        "skip_abnormal_bond_screening": False,
    },
    similarity_kwargs={
        "support_species": ("O", "Ce"),
        "slab_shave_distance": 4.0,
        "kernel_gamma": 1.0,
        "kernel_alpha": 1.0,
        "kernel_threshold": 1e-06,
    },
    show_final=False,
)

summary_df = result.get("summary_df")
if summary_df is not None:
    print(summary_df)
```

## Most important settings to change

If you are unsure, start here:

- `adsorbate_paths`
  - Add/remove adsorbate files.
- `mace_model_path`
  - Set to your model path.
- `mace_device`
  - Use `"cuda"` for GPU or `"cpu"` for CPU.
- `mace_fmax`
  - Smaller = stricter relaxation (often slower).
- `mace_max_steps`
  - Increase if many structures do not converge.
- `generation_kwargs["height"]`
  - Initial adsorbate distance from site.
- `generation_kwargs["z_cutoff"]`
  - Set `None` to disable z filtering.
- `generation_kwargs["skip_abnormal_bond_screening"]`
  - `True` uses only the 0.5 Angstrom clash check (skips radii-based bond screening).

## Full option explanations

### Main inputs

- `structure`
  - Main support/slab structure (ASE `Atoms`).
- `adsorbates`
  - List of adsorbates (ASE `Atoms`) loaded from xyz files.

### Output files

All paths in this example go to `scratch/`.

- `candidates_db_path`
  - Raw generated candidates.
- `pre_relax_db_path`
  - Structures after primary similarity filtering.
- `relaxed_db_path`
  - Structures after MACE relaxation.
- `final_db_path`
  - Final structures for DFT.
- `single_a_final_db_path` / `single_b_final_db_path`
  - Reused single-adsorbate MACE results after validity and similarity filtering, with DFT constraints restored.
- `candidate_report_csv_path`
  - Keep/prune report for unrelaxed combined candidates.
- `site_probe_report_csv_path`
  - Original site index, keep/prune decision, closest retained site, and similarity for every bond-valid probe placement.
- `relaxed_minimum_report_csv_path`
  - Energy-ordered keep/prune report for relaxed minima.
- `relaxation_report_csv_path`
  - MACE metrics (energy, max force, convergence, optimiser steps, force
    evaluations, and elapsed time).
- `summary_csv_path`
  - Compact numeric summary by stage.
- `summary_txt_path`
  - Human-readable summary.

### Similarity threshold controls

- `site_probe_symbol="H"`
  - Temporary atom used to compare local adsorption-site environments before full adsorbates are generated.
- `site_probe_similarity_cutoff=0.99`
  - Probe environments with similarity at or above this cutoff are duplicate sites.
- `single_mode_duplicate_cutoff=0.99`
  - Fixed cutoff for independently relaxed single-adsorbate modes.
- `candidate_duplicate_cutoff=None`
  - Fixed cutoff for unrelaxed combined candidates, or `None` for automatic selection.
- `candidate_suggested_cutoff_scale=1.0`
  - Multiplier applied only when the combined-candidate cutoff is automatic.
- `relaxed_minimum_duplicate_cutoff=None`
  - Fixed cutoff for relaxed minima, or `None` for automatic selection.
- `relaxed_minimum_suggested_cutoff_scale=1.0`
  - Multiplier applied only when the relaxed-minimum cutoff is automatic.
- `candidate_reference_row_id=1` / `relaxed_minimum_reference_row_id=1`
  - Explicit, validated reference rows for automatic cutoff estimation.

### MACE settings

- `mace_model_path`
  - Path to MACE model file.
- `mace_fmax`
  - Force convergence target in eV/A.
- `mace_max_steps`
  - Maximum optimization steps.
- `mace_device`
  - `"cuda"` or `"cpu"`.
- `mace_dispersion`
  - Enable/disable dispersion correction.
- `mace_default_dtype`
  - Numeric precision (`"float64"` usually stable).
- Full optimisation trajectories are not written. Completed endpoints and
  metrics are stored in `relaxed_db_path` and the relaxation report.

### Candidate generation settings (`generation_kwargs`)

- `site_symbols=("O", "Ce")`
  - Species used for adsorption site detection.
- `passive_symbols=("O", "Ce")`
  - Used when `site_symbols` is not set.
- `height=3.0`
  - Initial placement height (Angstrom).
- `z_cutoff=5.0`
  - Rejects candidates where new adsorbate atoms are below this z.
- `z_buffer=None`
  - Used only when `z_cutoff=None`.
  - Then z cutoff is computed automatically from host geometry.
- `skip_abnormal_bond_screening=False`
  - `False`: use both quick clash and radii-based abnormal-bond screening.
  - `True`: use only the quick clash check.

### Similarity engine settings (`similarity_kwargs`)

- `support_species=("O", "Ce")`
  - Support species kept during slab pre-processing.
- `slab_shave_distance=4.0`
  - Trims deeper slab atoms for similarity calculation.
- `kernel_gamma=1.0`
  - REMatch kernel parameter.
- `kernel_alpha=1.0`
  - REMatch regularization parameter.
- `kernel_threshold=1e-06`
  - Numerical solver threshold.

### Display setting

- `show_final=False`
  - Set to `True` to open an ASE viewer for final structures.

## Common problems and fixes

- `FileNotFoundError`
  - Check file names and paths for structure/adsorbates.
- No GPU found
  - Change `mace_device="cpu"`.
- No candidates generated
  - Try different `site_symbols`, lower `z_cutoff`, or adjust `height`.
- Too few final structures
  - Relax thresholds (or set manual threshold values).

## Important notes

- Existing output files at the chosen output paths are removed at run start.
- Slab site detection may fall back to cluster detection if optional dependencies are missing.
- MACE tries `enable_cueq=True`; if unsupported by your installed MACE version, it falls back automatically.
