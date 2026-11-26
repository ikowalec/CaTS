# MACE Trajectory Dataset Screening Protocol

## Overview

Training robust Machine Learning Atomic Cluster Environment ([MACE](https://github.com/ACEsuit/mace)) neural network potentials requires high-quality, non-redundant reference quantum mechanical data. Raw concatenated relaxation trajectories from DFT or semi-empirical calculations often contain hundreds of nearly identical structures near local minima, leading to severe computational bloat, slow training epochs, and biased loss functions. Conversely, naive uniform downsampling risks discarding high-force barrier configurations critical for stable molecular dynamics and transition state exploration.

The **MACE Trajectory Screener** provides an intelligent, automated pipeline and graphical user interface (`gui_tools/MACE_Trajectory_Screener.exe`) to curate combined `.xyz`, `.extxyz`, `.traj`, and `.db` datasets.

```
                    ┌────────────────────────────────────────┐
                    │  Input .xyz / .extxyz / .traj / .db    │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │       Trajectory Partitioning          │
                    │  (metadata tags, formula/atom count,   │
                    │         pair distance jumps)           │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
             ┌────────────────────────────────────────────────────────┐
             │            Trajectory-Aware Screening Filter           │
             ├──────────────────────────┬─────────────────────────────┤
             │  Preserve Endpoints      │  • Initial state (t0)       │
             │                          │  • Final state (t_final)    │
             ├──────────────────────────┼─────────────────────────────┤
             │  Preserve High Forces    │  • Fmax >= 5.0 eV/Å         │
             │                          │  • Non-converged barrier    │
             ├──────────────────────────┼─────────────────────────────┤
             │  Preserve High Energy    │  • ΔE >= 0.50 eV/atom       │
             ├──────────────────────────┼─────────────────────────────┤
             │  Similarity Screening    │  • REMatch Sim < 0.98       │
             │  (Intermediate frames)   │  • ΔE >= 0.02 eV/atom       │
             │                          │  • Max intermediate <= 5    │
             └──────────────────────────┬─────────────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │     Standardize Property Tagging       │
                    │     • info["REF_energy"]               │
                    │     • arrays["REF_forces"]             │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │    Screened .xyz Output & Audit CSV    │
                    │    (Significant Memory/Disk Savings)   │
                    └────────────────────────────────────────┘
```

---

## 1. Algorithmic Workflow

### 1.1 Trajectory Partitioning
Combined `.xyz` files aggregate multiple relaxation runs into a flat sequence of frames. The screener partitions the stream into contiguous trajectories using:
1. **Explicit Metadata Keys**: `source_frame`, `trajectory_id`, `run_id`, or `step_index` in `atoms.info`. When step counts reset to 0 or drop, a new trajectory boundary is declared.
2. **Chemical Formula & Atom Count**: Shifts in chemical formula or number of atoms immediately split trajectories.
3. **Geometric Continuity**: If all-atom pair distance RMSD between adjacent frames exceeds 1.5 Å, a new trajectory is declared to prevent conflating distinct reaction cascades.

### 1.2 Mandatory Endpoint Retention
- **Initial Frame (‖F‖t_0‖F‖)**: Captures the non-equilibrium starting geometry (often perturbed, adsorbed, or pre-reaction complexes).
- **Final Frame (‖F‖t_{\text{final}}‖F‖)**: Captures the relaxed local minimum with near-zero residual forces (‖F‖F_{\max} \to 0‖F‖).

### 1.3 High-Force Barrier State Retention
Frames exhibiting maximum atomic force magnitude:
‖F‖‖F‖\max_i \|\mathbf{F}_i\| \ge F_{\text{thresh}} \quad (\text{default: } 5.0\text{ eV/\AA})‖F‖‖F‖
are preserved regardless of similarity downsampling. These structures sample repulsive walls, bond breaking/forming events, and high-energy saddle regions essential for MD stability without unphysical cluster explosion.

### 1.4 High-Energy State Retention
Frames with per-atom potential energy exceeding the trajectory baseline by ‖F‖\ge 0.50\text{ eV/atom}‖F‖ are prioritized to ensure adequate sampling of excited or distorted configurations.

### 1.5 Geometric Similarity Screening (SOAP REMatch Kernel)
Intermediate frames between endpoints are filtered sequentially against the set of already accepted structures following the repository's established REMatch similarity rules (`similarity.rematch.memory_rematch`):
1. **SOAP Descriptor**:
   ‖F‖‖F‖\text{SOAP}(r_{\text{cut}} = 6.0\text{ \AA}, n_{\max} = 8, l_{\max} = 6)‖F‖‖F‖
   constructed across the unique chemical species in the dataset and normalized via ‖F‖\ell_2‖F‖ norm.
2. **REMatch Kernel Similarity & Dynamic Per-Trajectory Threshold**:
   ‖F‖‖F‖K(\mathbf{A}, \mathbf{B}) \in [0, 1] \quad (\text{RBF metric, } \gamma = 1.0, \alpha = 1.0)‖F‖‖F‖
   Rather than applying a brittle uniform threshold across all trajectories, the screener computes an **adaptive threshold per trajectory** via the repository's elbow gap / ‖F‖p_{95}‖F‖ percentile heuristic (`suggest_similarity_threshold`). This scales with the trajectory's intrinsic structural spread, ensuring small-displacement relaxations are not entirely dropped and large structural reorganizations are not over-sampled:
   ‖F‖‖F‖\max_{S \in \text{kept}} K(\mathbf{A}, S) \ge K_{\text{threshold}}^{\text{(dynamic)}} \implies \text{redundant duplicate}‖F‖‖F‖
3. **Energy Difference Override**:
   ‖F‖‖F‖\frac{|E_{\mathbf{A}} - E_{\mathbf{B}}|}{N} \ge 0.02\text{ eV/atom}‖F‖‖F‖
   retains structures with significant electronic/potential energy deviation even when borderline geometric similarity occurs.
4. **Max Intermediate Cap**: By default, at most 5 intermediate frames are retained per trajectory to prevent over-representing lengthy, slow-converging relaxations.
5. **Fallback Metric**: All-atom pair distance RMSD (‖F‖\text{RMSD} \ge 0.08\text{ \AA}‖F‖) serves as fallback if DScribe is unavailable.

---

## 2. Standardized Reference Properties for MACE

MACE expects consistent reference energy and force keys. The screener standardizes all outputs:
- **`REF_energy`**: Extracted from ASE `SinglePointCalculator.energy` or existing `energy`/`free_energy` info tags, stored as scalar float in `atoms.info["REF_energy"]`.
- **`REF_forces`**: Extracted from ASE `SinglePointCalculator.forces` or `forces` array, stored as an ‖F‖(N, 3)‖F‖ float array in `atoms.arrays["REF_forces"]`.

Both keys are written into standard ASE `extxyz` format, ready for direct ingestion by `mace_run_train`.

---

## 3. Graphical User Interface (`gui_tools`)

A desktop application is provided in `gui_tools/`:

```
gui_tools/
├── MACE_Trajectory_Screener.exe   # Standalone single-file executable with custom icon
├── MACETrajectoryScreener.exe     # Renamed copy to bypass Windows icon cache
├── mace_trajectory_screener_gui.py# Tkinter/ttk desktop GUI implementation
├── screener_core.py               # Core screening & trajectory partitioning library
├── build_screener_exe.py          # PyInstaller packaging automation
├── generate_icon.py               # Multi-resolution icon generation script
├── icon.ico                       # Embedded multi-resolution Windows icon (16-256px)
└── icon.png                       # High-resolution 256x256 icon asset
```

### Key GUI Features
1. **Input File Chooser & Fast Inspection**: Instantly analyzes selected `.xyz` files, displaying total structure count, file size (MB), element list, and sample chemical formula.
2. **Customizable Parameters**:
   - High Force Barrier Threshold (‖F‖F_{\max}‖F‖, default: 5.0 eV/Å)
   - Pair Distance RMSD Cutoff (default: 0.08 Å)
   - Energy Variance Cutoff (default: ‖F‖0.02\text{ eV/atom}‖F‖)
   - Max Intermediate Frames (default: ‖F‖5‖F‖)
   - Checkbox to toggle dynamic per-trajectory similarity thresholding.
   - Checkbox to export detailed audit CSV report.
   - Automatic dual tagging: standard calculator properties (`forces`, `energy`) and MACE reference tags (`REF_forces`, `REF_energy`) are always attached automatically.
3. **Responsive Non-Blocking Worker**: Multi-threaded execution prevents UI freezes during large dataset processing, reporting real-time progress and status.
4. **Memory Savings Dashboard**:
   - Initial structures vs Screened structures.
   - Percentage structure reduction (typically 60% – 85%).
   - Disk and memory savings in MB and percentage.
   - Detailed breakdown of retained endpoints, high-force frames, and intermediate frames.
   - **"Open Output Folder"** button to jump directly to the generated files.

---

## 4. Running the Tool

### Running the Pre-Packaged Executable
Double-click `gui_tools/MACE_Trajectory_Screener.exe` (or `gui_tools/MACETrajectoryScreener.exe`). No Python installation or external dependencies are required.

### Running from Python Source
```powershell
.venv\Scripts\python.exe gui_tools\mace_trajectory_screener_gui.py
```

### Headless / CLI Execution
The core screening library can also be invoked directly in Python scripts:
```python
from gui_tools.screener_core import run_screening_workflow

results = run_screening_workflow(
    input_path="mace_training.xyz",
    output_path="mace_training_screened.xyz",
    high_force_ev_a=5.0,
    min_pair_distance_rmsd_a=0.08,
    min_energy_change_per_atom_ev=0.02,
    max_intermediate_frames=5,
    tag_ref_properties=True,
    audit_csv_path="screening_audit.csv",
)

print(f"Retained {results['final_count']}/{results['initial_count']} structures.")
print(f"Memory / disk savings: {results['file_savings_pct']:.1f}%")
```

---

## 5. Typical Memory & Performance Gains

Benchmarked on `mace_training.xyz` (1,020 raw ORCA relaxation structures across 324 trajectories):
- **Raw Structures**: 1,020 frames (‖F‖3.45\text{ MB}‖F‖)
- **Retained Endpoints**: 586 frames (‖F‖t_0‖F‖ and ‖F‖t_{\text{final}}‖F‖)
- **Retained High-Force States**: 10 frames (‖F‖F_{\max} \ge 5.0\text{ eV/\AA}‖F‖)
- **Retained Intermediate States**: 60 frames (‖F‖\text{RMSD} \ge 0.08\text{ \AA}‖F‖)
- **Screened Total**: 656 frames (‖F‖2.22\text{ MB}‖F‖)
- **Structure Reduction**: -35.7% (up to -80% on long unrelaxed MD trajectories)
- **Speedup Factor**: ‖F‖\sim 1.55\times‖F‖ faster MACE training epochs with improved validation metrics due to de-correlated data.
