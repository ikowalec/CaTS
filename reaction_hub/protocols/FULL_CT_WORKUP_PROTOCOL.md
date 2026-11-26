# Full charge-transfer workup

**Purpose.** Define the smallest reproducible calculation set needed to relate ground-state adsorption, optical absorption and vertical charge-transfer character for an adsorbate–nanocluster structure. The protocol applies to stationary structures and stored MEP images. It does not include excited-state geometry optimisation, dynamics, vibrational analysis, ZPE or BSSE corrections.

## 1. Input contract

Start from a Reaction Hub source workflow, not from an untracked geometry. Record:

- source workflow ID and immutable geometry;
- total charge, recorded spin multiplicity and calculation-settings profile;
- adsorbate identity and complete, non-overlapping fragment definitions covering every atom;
- required wavelength window and any explicitly requested excitation states;
- source/output paths and checksums for the ORCA output and wavefunction.

For a stationary structure, use its converged ground-state geometry. For an MEP, use each selected stored image without reoptimisation. Use the lowest admissible multiplicity by default and do not perform routine multiplicity screening; any limited sensitivity test is a separate SI calculation. Gas-phase references retain their chemically appropriate multiplicities.

## 2. Preflight and reuse

Before submitting calculations:

1. verify that the source geometry and electronic calculation completed successfully;
2. locate the retained `.gbw`, `.molden.input` or equivalent wavefunction;
3. verify that geometry, charge, multiplicity and method metadata agree;
4. check Reaction Hub for an equivalent completed, running or queued analysis and reuse it;
5. classify each missing result as Mulliken-only, spectral/IFCT, energy-reference or spatial-density work.

Missing downstream analysis must not trigger a new geometry optimisation. Preserve source-run and geometry lineage in every derived workflow.

## 3. Ordered workup

### A. Ground-state Mulliken charges

Run Multiwfn Mulliken population analysis on the retained ground-state wavefunction. Store atomic and fragment charges with `charge_method=mulliken`. Keep any historical Löwdin values separately labelled; they must not replace the default Mulliken aliases.

**Rationale:** ground-state charge establishes the direction and magnitude of charge redistribution already present before excitation and uses the same partitioning family as the subsequent IFCT analysis.

Mulliken-only recovery uses `standard`/`short` (or equivalent active HPC queue), one task, 20 minutes, through the Celery `submissions` queue. At most four active short-QoS jobs may exist; excess work remains locally queued and retries with delay.

### B. Frozen-fragment energy reference

Where adsorption or interaction strength is part of the comparison, calculate the bare cluster at the complex geometry with the same electronic settings, then evaluate

$$
E_{\mathrm{ref}} = E_{\mathrm{complex}}-E_{\mathrm{cluster@complex}}-\sum_j E_j^{\mathrm{gas}}.
$$

Gas-phase energies must be obtained from identified Reaction Hub reference workflows at matching settings, not copied as constants into the protocol. Report electronic energies to 2 dp and state that BSSE, ZPE and thermal corrections are absent.

**Rationale:** this separates binding from strain of the cluster while retaining a consistent gas-phase reference convention. It is a reference interaction energy, not a strict counterpoise-corrected frozen-fragment interaction energy.

### C. Vertical TDA spectrum

Run vertical TDA-TDDFT at the immutable complex geometry using the recorded campaign method. Use the repository's dynamic root count and verify afterwards that the requested wavelength range is covered; extend the root count only if coverage is incomplete. Never replace this step with full TDDFT.

For a complex-minus-cluster optical comparison, also run TDA on the frozen bare cluster using matching geometry and electronic settings. Interpolate both spectra onto the same wavelength grid before subtraction and retain the two parent workflow IDs.

**Rationale:** the complex spectrum identifies optically accessible states, while the frozen-cluster control distinguishes adsorbate-induced spectral changes from geometric distortion of the metal core.

### D. Same-calculation IFCT decomposition

Run Multiwfn IFCT using the Molden wavefunction and ORCA excitation output belonging to the same TDA calculation. The IFCT state count must equal the completed TDA root count. Store, for every state, wavelength, oscillator strength and all directional terms:

- intrafragment redistribution, e.g. `Redis(1)`;
- nanocluster → adsorbate and adsorbate → nanocluster;
- inter-adsorbate terms for coadsorbed systems.

For a wavelength interval $B$, report the oscillator-strength-weighted contribution

$$
W_B(A\rightarrow B)=\frac{\sum_{i\in B} f_i p_i(A\rightarrow B)}{\sum_{i\in B}f_i},
$$

only when $\sum_{i\in B}f_i>0$. Retain both the per-state data and the stated interval; do not call 300–800 nm the complete visible range.

**Rationale:** IFCT supplies directionality that an absorption spectrum alone cannot provide.

### E. Bounded spatial analysis

Generate CDD and dominant hole/electron NTO cubes for a small, deduplicated state set:

1. the brightest state in each populated 300–400, 400–500, 500–600 and 600–800 nm band;
2. the state contributing most strongly to the dominant directional CT feature;
3. any explicitly justified mechanistic state.

Use the repository defaults of Multiwfn grid quality 3, CDD isovalue 0.001 a.u. and NTO isovalue 0.03 a.u. Render the structure, CDD, hole NTO and electron NTO in the same orientation. Record the CDD sign convention and NTO pair weight. A separate transition-density image is not currently part of the standard pipeline.

**Rationale:** spatial plots test whether the fragment-level IFCT assignment is physically located on the proposed adsorption site without generating cubes for every root.

## 4. Completion record

A full workup is complete only when Reaction Hub contains:

- source and derived workflow IDs with geometry lineage;
- charge, multiplicity, method, basis and actual root count;
- Multiwfn Mulliken atomic and fragment charges;
- complex spectrum and, where required, frozen-cluster spectrum and difference spectrum;
- per-state IFCT terms and declared weighted wavelength summaries;
- selected-state rationale, CDD/NTO cubes and rendered panels;
- separate statuses for locally queued, submitted to Slurm, completed and ingested work.

Raw numerical data, source paths, figure inputs and checksums must remain recoverable for later repository deposition.

## 5. Interpretation limits

- `Redis(1)` means intracluster redistribution; it is not by itself proof of plasmonic character.
- IFCT percentages describe a normalized vertical excitation. Thus 5% corresponds approximately to 0.05 e of transition-resolved directional redistribution, not a ground-state charge, persistent injected population or quantum yield.
- CDD and NTO plots describe vertical electron-density reorganisation, not nuclear relaxation.
- Ground-state charge, interaction energy and vertical CT should be compared as correlated observables; none alone establishes causation.
- Do not infer bond elongation, energy transfer, barrier lowering, reaction kinetics or product formation without the corresponding geometry/path calculations.

Campaign-specific mechanistic expectations belong in a separate analysis plan and must be labelled as hypotheses until supported by calculated structures, paths or excited-state optimisation.
