# Energy Terms Used in Results

This document defines the energy values shown on `/results/` and `/results/energy/<id>/`.

## Core Terms

- `Esystem` (`total_energy_eV`): final single-point energy of the full adsorbate+cluster system.
- `Eref_cluster` (`cluster_initial_energy_eV`): reference nanocluster energy used in adsorption-energy construction.
- `Efinal_cluster` (`cluster_final_energy_eV`): single-point energy of the isolated cluster fragment extracted from the final adsorbed geometry.
- `Eads` (`adsorption_energy_eV`): adsorption energy computed from system and reference components.
- `Ed` (`ed_energy_eV`): distortion energy of the cluster.
- `Eint` (`eint_energy_eV`): adsorbate-cluster interaction energy after distortion correction.

## Formulas

### Adsorption Energy

For one adsorbate:

`Eads = Esystem - (Eref_cluster + Eref_adsorbate)`

For multiple adsorbates/components:

`Eads = Esystem - (Eref_cluster + Σ_i n_i * Eref_adsorbate_i)`

where `n_i` is the stoichiometric factor for adsorbate component `i`.

### Distortion Energy

`Ed = Efinal_cluster - Eref_cluster`

Interpretation:
- `Ed > 0`: cluster is destabilized (higher energy) in the final adsorbed geometry.
- `Ed < 0`: cluster is stabilized (lower energy) in the final adsorbed geometry.

### Interaction Energy

`Eint = Eads - Ed`

Equivalent form:

`Eint = Eads - (Efinal_cluster - Eref_cluster)`

## Metadata Keys

When cluster-fragment energy is available, result metadata includes:

- `cluster_initial_energy_eV`
- `cluster_final_energy_eV`
- `cluster_distortion_energy_eV` (legacy-compatible, same numeric value as `Ed`)
- `ed_energy_eV`
- `eint_energy_eV`
- `distortion_corrected_adsorption_energy_eV` (legacy-compatible alias of `Eint`)
- `adsorbate_cluster_interaction_energy_eV` (legacy-compatible alias of `Eint`)

## UI Labels

- `Eads`: Adsorption Energy
- `Ed`: Energy of Distortion
- `Eint`: Interaction Energy
