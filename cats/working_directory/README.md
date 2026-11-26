# TDDFT Active Learning Workflow

This document explains how to run and customize the TDDFT active-learning workflow implemented in `cats/working_directory/tddft_active_learning_workflow.py`.

**Overview**
The workflow combines TDDFT scans and active-learning geometry optimization to follow a chosen excited state (IROOT) while optionally scanning a bond length. It:
- Runs an initial TDDFT scan to select an IROOT near a target wavelength.
- Iterates TDDFT scan + active-learning optimization steps.
- Tracks IROOT continuity using CT metrics.
- Optionally advances a bond length constraint each iteration.
- Writes logs and plots for post-analysis.

**Requirements**
- ORCA installed and available in the environment.
- `ORCADIR` set so `orca_2mkl` can be run during CT analysis.
- Python packages: `ase`, `catlearn`, `numpy`, `pandas`, `matplotlib`.
- Multiwfn and the Multiwfn helpers used by `cats.multiwfn.*`.

**Files and Directories**
The workflow creates and expects these directories:
- `DFT/iter_0/` or `DFT/` with `orca.gbw` for the initial ground-state restart.
- `DFT/opt_iter_<i>/` for DFT-only optimisation iterations (`use_tddft_optimisation=False`).
- `TDDFT/scan_iter_0/` for the initial TDDFT scan (skipped when `use_tddft_optimisation=False`).
- `TDDFT/scan_iter_<i>/` and `TDDFT/opt_iter_<i>/` for TDDFT scan/optimisation iterations (`use_tddft_optimisation=True`).
- `TDDFT/scan_iter_<i>_geom_<step>/` for scan geometry steps inside an iteration.
- `TDDFT/scan_iter_<i>/` created after the DFT-only optimisation when `use_tddft_optimisation=False` (TDDFT scans are run once per optimized geometry).
- `TDDFT_active_learning_shared/` for shared logs.
- `DFT/scan_plots/` for DFT-mode plots.
- `TDDFT/scan_plots/` for TDDFT-mode plots.

Key log files:
- `TDDFT_active_learning_shared/iroot_scan_endpoints.csv`
- `TDDFT_active_learning_shared/iroot_geometry_log.csv`
- `TDDFT_active_learning_shared/iroot_scan_summary.csv`
- `TDDFT/opt_iter_<i>/active_learning/iroot_geometry_summary.csv` (or `DFT/opt_iter_<i>/active_learning/iroot_geometry_summary.csv` in DFT mode)
- `TDDFT_active_learning_shared/workflow_root.txt`

**Quick Start**
1. Place your starting geometry in `orca.xyz` in the run directory.
2. Run a ground-state ORCA geometry optimisation and place `orca.gbw` in `DFT/iter_0/` or `DFT/`, or let the workflow generate it automatically.
3. Run the workflow.

Example:
```python
from ase.io import read
from cats.working_directory.tddft_active_learning_workflow import run_workflow
from cats.orca.tddft.scan import BondScanSettings

atoms = read("orca.xyz")

bond_scan = BondScanSettings(
    enabled=True,
    bond_pair_indices=(13, 15),  # or use bond_symbol_pair=("O", "O")
    increment=0.05,
    fix=0.5,
    fixed_symbols=("Cu",),
)

run_workflow(
    atoms,
    n_iter=10,
    bond_scan=bond_scan,
    xc_functional="pbe0",   # or "cam-b3lyp"
    solvent="Water",        # or e.g. "Methanol", "CPCM(Acetonitrile)", or None
    nodes=2,
    cpus_per_node=32,
)
```

**Configuration Parameters**
Edit constants at the top of `cats/working_directory/tddft_active_learning_workflow.py`:
- `TARGET_WAVELENGTH_NM`: the target wavelength for IROOT selection.
- `TDDFT_SCAN_ROOTS`: number of TDDFT roots per scan.
- `IROOT_SCORE_WEIGHTS` and `IROOT_*_SIGMA_*`: scoring parameters.
- `IROOT_MIN_SIMILARITY`: minimum similarity required to switch IROOT; below this the previous state is kept.
- `nodes`, `cpus_per_node`: default ORCA resource settings.
- `CLUSTER_SYMBOLS`: element symbols used to define fragment 1.

Runtime parameters for `run_workflow(...)`:
- `n_iter`: total number of iterations.
- `bond_scan`: `BondScanSettings` (see below).
- `al_steps_per_iter`: active-learning local-step chunk size per `LocalAL.run(...)` call.
- `optimize_to_convergence`: whether to keep stepping until convergence.
- `max_al_steps_per_iter`: optional total cap for one optimisation iteration when `optimize_to_convergence=True`.
  Set `None` (default) or `<=0` to disable the total cap.
- `fmax`: force convergence threshold used throughout the workflow.
- `initial_fmax`: stricter force threshold for the initial DFT-optimized geometry (defaults to `fmax`).
- `run_initial_dft_if_missing`: run an initial DFT if no `orca.gbw` is found.
- `use_tddft_optimisation`: use TDDFT engrad for the optimisation steps. When `False`, the workflow runs DFT-only active learning and then performs TDDFT scans once per optimized geometry.
- `ensure_initial_dft_converged`: ensure the starting geometry is DFT-optimized before the workflow begins (uses active learning).
- `use_scan_geometry`: update the optimisation starting geometry from the TDDFT scan output (useful when scans are constrained bond-length relaxations).
  When `use_tddft_optimisation=True`, this also seeds TDDFT optimisation from an existing DFT geometry (`DFT/opt_iter_<i>/orca.out`) and, if missing, performs a DFT pre-optimisation first.
- `xc_functional`: exchange-correlation functional for DFT/TDDFT steps. Supported: `"pbe0"` and `"cam-b3lyp"` (case-insensitive).
- `solvent`: solvation setting passed to ORCA. Use a solvent name (e.g. `"Water"`, `"Methanol"`), an explicit keyword (e.g. `"CPCM(Acetonitrile)"`), or disable with `None`/`"none"`/`"vacuum"`.
- `tddft_scan_nroots`: number of excited states (`NRoots`) used for each TDDFT scan sweep and IROOT selection.
- `nodes`: number of nodes used to compute ORCA CPU resources for this run.
- `cpus_per_node`: CPUs per node used to compute total CPUs (`nodes * cpus_per_node`) for this run.
  `memory_per_cpu` is derived as `int(1300 * 128 / cpus_per_node)`.

**Fragment Selection**
Fragments are built dynamically from the current structure:
- Fragment 1: all `Cu` or `Au` atoms (defined by `CLUSTER_SYMBOLS`).
- Fragment 2: every other atom.

Indices are 1-based (atom 0 in Python is 1 in the fragment definition) and are compressed into ranges like `1-13` where possible.

**Bond Scan Settings**
Bond scans use a fixed bond constraint plus optional fixed atoms. The bond is updated once per iteration.

`BondScanSettings` fields:
- `enabled`: enable or disable bond scanning.
- `bond_pair_indices`: explicit atom indices for the scanned bond.
- `bond_symbol_pair`: alternative symbol-based selection if indices are not supplied.
- `increment`: fixed increment per iteration when no target is set.
- `target_distance`: if set, increments are computed to approach this target.
- `distance_change_fraction`: fraction of the total distance change to apply over the run.
- `fix`: ASE `set_distance(..., fix=...)` parameter.
- `fixed_symbols`: atoms to freeze with `FixAtoms`.

**Target Distance Mode**
If `target_distance` is set, the increment is computed from:
- the current bond length at the start of the run
- the remaining iteration count
- `distance_change_fraction`

Example: reach 2.80 Angstrom over the run
```python
bond_scan = BondScanSettings(
    enabled=True,
    bond_pair_indices=(13, 15),
    target_distance=2.80,
    distance_change_fraction=1.0,
)
```

Example: move halfway to the target
```python
bond_scan = BondScanSettings(
    enabled=True,
    bond_pair_indices=(13, 15),
    target_distance=2.80,
    distance_change_fraction=0.5,
)
```

Notes:
- The increment is applied once per iteration after the optimization step.
- The last optimized structure will land on the target when `distance_change_fraction=1.0`.
- The increment is clamped to avoid overshooting the target.

**Restart Behavior**
You can safely re-run the script from the same directory:
- It detects completed iterations using `iteration_complete.txt` and valid `orca.out` files.
- It resumes from the next incomplete iteration.
- It warns if the run directory has moved, using `workflow_root.txt`, but continues the restart.
- If you switch from DFT-only mode (`use_tddft_optimisation=False`) to TDDFT mode (`True`), existing DFT-only iteration outputs are treated as starting geometries, not completed TDDFT iterations.
- If all requested iterations are already complete, re-submission still runs scan postprocessing (TS smoothing/refinement). It also detects additional contiguous completed `opt_iter_*` outputs on disk and uses the full completed scan for smoothing.

If you intentionally moved the run directory, no manual action is required; the workflow logs the mismatch and continues.

**Outputs and Plots**
At the end of the run the workflow attempts to generate plots in:
- `DFT/scan_plots/` when `use_tddft_optimisation=False`
- `TDDFT/scan_plots/` when `use_tddft_optimisation=True`

Plots include:
- Per-iteration CT plots:
  - overall intensity view: `ct_spectrum_iter_XXX_overall.png`
  - transfer-only (UV removed, y-axis rescaled): `ct_spectrum_iter_XXX_transfer_only.png`
  - compatibility copy of overall view: `ct_spectrum_iter_XXX.png`
- Stacked UV-vis summary: `stacked_uvvis.png`, `stacked_uvvis_wide.png`.
- Stacked CT summary: `stacked_ct.png`, `stacked_ct_wide.png`.
- Energy vs bond length.
- Plot manifest: `multiwfn_scan_plot_manifest.csv`.

When `use_tddft_optimisation=False`, TDDFT scans are run once per optimized geometry after the DFT-only optimisation, enabling the reaction-coordinate UV-vis plot.

**Default Wavelength Window (Important)**
- Default analysis window is now `300-800 nm`.
- By default, UV-vis and CT Gaussian fitting ignores signals outside this window.
- Stacked UV-vis plots default to x-limits `300-800 nm`.

This applies to:
- UV-vis intensity analysis and excitation selection in Multiwfn plotting (`get_uvvis_plot`).
- CT/UV Gaussian fitting in `plot_charge_transfer_spectrum`.
- Stacked UV-vis display limits in scan summaries.

**CT Readability Modes**
Each iteration now writes two CT views to improve readability:
- Overall CT spectrum view:
  keeps all components and scales y-axis to the full plotted intensity range.
- CT-only view:
  removes UV-labelled components (i.e., focuses on transfer to/from fragments)
  and rescales y-axis to this filtered signal only.

This is specifically intended for cases where UV components dominate the
vertical scale and obscure transfer features.

**Keyword Controls**
Plot controls are passed through `UVVisPlotSettings` used by `ScanSummaryPlotter`.

Main window controls:
- `analysis_min_wavelength_nm` (default `300.0`)
- `analysis_max_wavelength_nm` (default `800.0`)

Stacked UV-vis display controls:
- `stacked_uvvis_xmin` (default `300.0`)
- `stacked_uvvis_xmax` (default `800.0`)
- `stacked_uvvis_offset_scale`
- `stacked_uvvis_linewidth_nm` (default `20.0`; Gaussian FWHM in nm)
- `stacked_uvvis_normalize`

Stacked CT controls:
- `stacked_ct_offset_scale`
- `stacked_ct_normalize`
- `stacked_ct_xmin`
- `stacked_ct_xmax`

Multiwfn CT Gaussian fitting controls:
- `multiwfn_ct_min_peak_intensity_fraction` (default `1.0`)
- `multiwfn_ct_min_peak_distance_points` (default `1`)
- `multiwfn_ct_min_peak_width_points` (default `1`)
- `multiwfn_ct_min_fwhm_abs` (default `15.0`)
- `multiwfn_ct_max_gaussians_per_spectrum` (default `20`)
- `multiwfn_ct_plot_individual_components` (default `False`)

To disable wavelength clipping, set either bound to `None`.

Example override:
```python
from cats.orca.tddft.scan import ScanSummaryPlotter, UVVisPlotSettings

plotter = ScanSummaryPlotter(
    uvvis_settings=UVVisPlotSettings(
        analysis_min_wavelength_nm=350.0,
        analysis_max_wavelength_nm=1200.0,
        stacked_uvvis_xmin=350.0,
        stacked_uvvis_xmax=1200.0,
    )
)

run_workflow(..., plotter=plotter)
```

**Latest-Per-Distance Rule For Stacked Plots**
- Stacked UV-vis and stacked CT select only the latest available scan for each bond length.
- Entries are then ordered by increasing bond length before plotting.
- This avoids duplicate traces from restarts or repeated calculations at the same distance.

**Common Issues**
- Missing `orca.gbw`: the workflow will run an initial DFT geometry optimisation if none is found; set `run_initial_dft_if_missing=False` to require a precomputed GBW.
- Initial DFT optimisation not converged: inspect `DFT/iter_0/orca.out` and relax convergence/SCF settings if needed.
- `ORCADIR` not set: required for `orca_2mkl` and CT data generation.
- “No valid IROOT candidates”: check CT outputs and scoring thresholds.
- “Workflow root mismatch”: the workflow now logs a warning and continues.

**Where to Customize**
- ORCA input settings: `get_orca_tddft_calculator(...)` in `cats/working_directory/tddft_active_learning_workflow.py`.
- Bond scan logic: `cats/orca/tddft/scan.py`.
- Active learning behavior: `cats/orca/tddft/active_learning.py`.
