# MACE Trajectory Screener (`MACE_Trajectory_Screener.exe`)

Automated desktop tool and algorithm for curating combined DFT/molecular dynamics trajectory datasets (`.xyz`, `.extxyz`, `.traj`, `.db`) for MACE neural network potential training.

---

## What It Does

1. **Dataset & Directory Ingestion**:
   - Accepts either a single trajectory file or an entire folder of runs. Recursively discovers all `.traj`, `.db`, `.xyz`, and `.extxyz` files.

2. **Trajectory-Resolved Bin Partitioning**:
   - Automatically segments file streams into discrete optimization trajectories using metadata keys (`source_frame`, `trajectory_id`, `run_uid`), formula/atom count shifts, and geometric discontinuities.
   - **Bin Isolation**: All similarity evaluations and threshold adaptations are strictly confined within each individual optimization trajectory bin, preventing cross-trajectory contamination.

3. **Mandatory Endpoint Preservation**:
   - Always preserves initial reactant/guess state ($t_0$) and final relaxed product/minimum ($t_{\text{final}}$) of each trajectory bin.

4. **High-Force Barrier & High-Energy Retention**:
   - Preserves high-force transition states ($\max \|\mathbf{F}\| \ge 5.0\text{ eV/\AA}$) and high-energy configurations ($\Delta E \ge 0.50\text{ eV/atom}$) without downsampling.

5. **Dynamic SOAP REMatch Similarity Deduplication**:
   - Encodes structures into rotationally/translationally invariant SOAP descriptors via DScribe ($r_{\text{cut}} = 6.0\text{ \AA}, n_{\max} = 8, l_{\max} = 6$).
   - Calculates pairwise similarities via regularized REMatch kernel ($k_{\text{REMatch}}$, RBF, $\gamma = 1.0, \alpha = 1.0$).
   - **Dynamic Per-Trajectory Threshold**: Automatically determines an adaptive similarity cutoff per trajectory bin using the repository's elbow gap / $p_{95}$ percentile heuristic (`suggest_similarity_threshold`). Prevents discarding intermediate states in subtle relaxations while avoiding over-sampling in large reconstructions.
   - Caps ordinary intermediate frames (default: $\le 5$ per trajectory bin).

6. **Full Force & Energy Retention (Tracking & MACE Ready)**:
   - Preserves complete Cartesian forces and potential energy into both standard calculator fields (`forces:R:3`, `energy`) and MACE reference tags (`REF_forces:R:3`, `REF_energy`).
   - Guarantees seamless roundtrip with downstream MEP tracking tools and ASE readers: `atoms.get_forces()` and `atoms.get_potential_energy()` work directly without "Atoms object has no calculator" errors.

7. **Detailed Rejection Breakdown & Audit Reporting**:
   - Transparently accounts for every rejected structure with 4 granular failure bins:
     - **Similarity / Redundancy**: Skipped due to REMatch kernel similarity $\ge$ threshold or RMSD $<$ cutoff.
     - **Intermediate Quota Cap**: Geometrically distinct, but skipped because max intermediate frame limit was reached.
     - **Energy Outlier / Divergence**: Non-finite energy, unphysical $|E/\text{atom}| > 500\text{ eV}$, or statistical outlier.
     - **Force Divergence**: Non-finite force or $F_{\max} > 100\text{ eV/\AA}$.
   - Displays full breakdown in GUI dashboard and completion dialog, returns in API results dictionary, and writes per-trajectory rejection metrics to audit CSV (`rejected_similarity`, `rejected_cap`, `rejected_energy`, `rejected_force`).

---

## Usage

- **Executable**: Run `gui_tools/MACE_Trajectory_Screener.exe` (or `MACETrajectoryScreener.exe`).
- **Python GUI**: `.venv\Scripts\python.exe gui_tools\mace_trajectory_screener_gui.py`
- **CLI**: `.venv\Scripts\python.exe cats\working_directory\sample_extxyz_trajectories.py <inputs...> -o screened.extxyz`
