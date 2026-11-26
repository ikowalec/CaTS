# Subgroup discovery for Au CO oxidation

The preparation command reuses the archived reaction-hub analysis and produces one row per geometry. Connected optimisation, energy, charge and TDDFT workups supply fields on that row. Spectral transitions retain their own table and are linked by geometry ID. Run subgroup discovery separately for CO and molecular O₂.

## Run from the repository root

Use the reaction-hub Python environment for preparation. It needs NumPy, pandas and SciPy.

```powershell
.venv/Scripts/python.exe reaction_hub/scripts/prepare_au_subgroups.py scratch/au_sgd --database reaction_hub/db.sqlite3
```

The default input is `reaction_hub/reports/au_co_oxidation_light_driven_20260823`. Use `--report` for another export of the same format. Omit `--database` for archive-only preparation. The optional SQLite connection is read-only and supplies geometry, energy and charge records for five Au₁₃ workflows. Missing snapshots are reported in the audit table.

Add `--include-reviewed-optical` with `--database` to include reviewed optical workflows outside the report archive. The current geometry-to-workup pairs are 2894 → 3226 (Au₅₅–O₂) and 3437 → 3437 (Au₂₀–O). Both use settings profile 2, CAM-B3LYP with the mixed 6-31G(d,p)/LANL2DZ basis and Au ECP. The option checks workflow completion, settings profile and workup provenance before extracting properties. The manifest records the added pairs. The provenance table retains the corresponding database property IDs. Additional geometries require review of the source settings, uniqueness, fragment partition and band coverage before registration.

Use a separate Python environment for subgroup search so pySubgroup can resolve its scikit-learn dependency independently of Reaction Hub.

```powershell
python -m venv scratch/au_sgd_env
scratch/au_sgd_env/Scripts/python.exe -m pip install -r reaction_hub/scripts/requirements-au-sgd.txt
scratch/au_sgd_env/Scripts/python.exe reaction_hub/scripts/run_au_subgroups.py scratch/au_sgd scratch/au_sgd/co_search --family CO
scratch/au_sgd_env/Scripts/python.exe reaction_hub/scripts/run_au_subgroups.py scratch/au_sgd scratch/au_sgd/o2_search --family O2
scratch/au_sgd_env/Scripts/python.exe reaction_hub/scripts/run_au_subgroups.py scratch/au_sgd scratch/au_sgd/o_search --family O
scratch/au_sgd_env/Scripts/python.exe reaction_hub/scripts/run_au_subgroups.py scratch/au_sgd scratch/au_sgd/o_o_search --family O+O
```

Use a dedicated output directory for each run. Add `--direction ads_to_au`, `--window red`, `--nuclearity 20` or `--max-delta-e 0.5` to change the search. For sensitivity to negative excitation energies, repeat in a new directory with `--include-negative-roots`.

## Input and provenance

| Input under the report directory | Use |
| --- | --- |
| `raw_data/report_source_properties.jsonl` | Cartesian snapshots, archived properties and spectra keyed by workflow run and property name |
| `raw_data/workflow_run_provenance.jsonl` | Workflow settings and links to legacy inputs |
| `workflow_analysis.csv` | Workflow stage, energies, electronic-structure settings and documented duplicate IDs |
| `scientific_revision_20261106/provenance/supplemental_properties.json` | Explicit geometry-to-workup mapping and additional spectral properties |
| `au13_co_o_outcome_classification.csv` | Au₁₃ outcome energies |
| `converged_mep_profiles.csv`, `au20_mep3326_refined_segments.csv` | Optional archived reaction-path tables carried through as supplied |

For a new workup, export it under its geometry workflow or add an explicit mapping from the geometry workflow to its workup in the supplemental source map. Where the spectral record gives a source geometry ID, it must agree with the join. A matching formula, site label or energy is insufficient. Preserve the source atom ordering because the IFCT fragment indices refer to that ordering. Keep alternative method, spin or geometry calculations as distinct records unless the archive documents that they belong to the same geometry row. The current adapter selects one final TDDFT dataset per geometry and gives the explicitly mapped supplement precedence.

`provenance.csv` includes the status of each candidate workflow, duplicate representative, snapshot checksum and, where applicable, workup ID and spectral line count. Workflows 3080 and 3138 are linked to 3081 after an exact snapshot comparison. `manifest.json` records source file checksums, preparation cutoffs and the local supplement IDs. Archive-specific IDs, paths and aliases are declared in the preparation script. Review them when porting this adapter to a new report.

## Output and conventions

| Output | Contents |
| --- | --- |
| `geometries.csv` | Canonical geometry rows, including scans and candidates with incomplete optical data |
| `CO.csv`, `O.csv`, `O2.csv`, `O_O.csv` | Optimised-candidate views for separate CO, atomic O, molecular O₂ and dissociated O+O searches |
| `CO_O.csv`, `CO2.csv` | Carbon-containing outcome geometries |
| `scans.csv` | Geometries identified as scan or transition-state records by archive metadata |
| `transitions.csv` | One row per archived line in a validated spectrum from a supported family. `line_index` is zero-based archive order |
| `eligible.csv`, `rules.csv`, `overlap.csv`, `search.json` | Search cohort, rules, intersections and settings |

- Cluster descriptors include Au nuclearity, radius of gyration, maximum Au–Au separation, gyration-tensor axis ratios and asphericity. Au connectivity uses a 3.3 Å cutoff. Atom GCN is the sum of neighbouring Au coordination numbers divided by 12. Site GCN summaries use the Au atoms within 2.7 Å of the relevant adsorbate element. Atop, bridge and hollow are contact-count labels. For O₂, retain the two nearest O–Au distances, contact asymmetry and shared Au contacts to help distinguish binding configurations.
- Molecular O₂ uses O–O ≤ 1.6 Å. CO₂-like connectivity uses a second C–O distance ≤ 1.5 Å. These are geometric criteria. Mulliken quantities use matching atom symbols and ordering from the archived property when available.
- Energies are in eV. `eref_ev` and `archived_delta_e_ev` retain the archive's interaction and relative energies. For optional energy screening, `delta_e_within_archive_family_ev` is the electronic energy above the lowest available optimised candidate with the same composition, charge, multiplicity, functional and basis. A matched group requires at least two energies. This is a sample-relative energy screen. Establish metastability with independent structural evidence.
- Optical windows are 400 ≤ λ < 500 nm (`blue`), 500 ≤ λ < 600 nm (`visible`), 600 ≤ λ ≤ 625 nm (`red`), and 600 ≤ λ ≤ 700 nm (`red_extended`). The red windows overlap by design. A window is covered when aligned total and directional arrays extend to its upper photon energy. Negative-energy lines remain visible in the line table, contribute to none of the windows and flag the entire geometry for the default search exclusion.
- For each window and direction the three targets are total brightness, Σf, directional IFCT intensity, Σ(fp), and oscillator-strength-weighted directional fraction, Σ(fp)/Σf. Here p is the dimensionless IFCT fraction. This fraction is distinct from the ground-state Mulliken charge. An uncovered window or a fraction with zero denominator is blank. A covered window with no lines has zero summed intensity.

## Search and interpretation

To maximise the product of directional CT magnitude in percent and oscillator strength for a single transition, use:

```powershell
scratch/au_sgd_env/Scripts/python.exe reaction_hub/scripts/run_au_subgroups.py scratch/au_sgd scratch/au_sgd/co_peak --family CO --target peak-product --depth 3 --numeric-candidates thresholds
```

`peak-product` takes the largest `100 * p * f` across transitions in the selected window. `sum-product` sums the same products to test whether enrichment extends across the window. Here `p` is the nonnegative IFCT fraction for the selected direction and `f` is the line oscillator strength. The default direction is Au to adsorbate, and the default window is 500–600 nm. Both targets use the archived CT-weighted intensities directly, rescaled by 100.

Threshold mode uses complementary `<=` and `>` tests on ordered descriptors. PySubgroup supplies bin boundaries for continuous inputs. For low-cardinality ordered inputs the cut points are observed levels below the maximum. Categorical descriptors use equality. Missingness itself is excluded as a rule condition, and a rule can use each descriptor once. The runner keeps up to 20 distinct geometry memberships from the ranked candidates. Use `--exclude-size` to omit nuclearity, radius of gyration and span, and `--nuclearity` to test a specific Au size where the cohort permits.

Use `--family combined` to pool CO, O, O₂ and O+O with one target per geometry. Shared contact descriptors use C for CO, O for atomic O, and the union of Au contacts across both O atoms for O₂ and O+O. The minimum distance is the nearest Au distance over the selected atom or atoms. Adsorbate family is recorded as subgroup composition and is excluded from candidate selectors. The pooled baseline weights each geometry equally, so larger families contribute more observations. CO+O and CO₂ are absent from the pool because the current export has no equivalent optical targets for those families.

To select subgroups under coverage constraints and visualise the trade-off, run `au_subgroup_pareto.py DATASET OUTPUT` in the pySubgroup environment with Matplotlib installed. It uses pySubgroup threshold selectors and enumerates all distinct memberships of up to three descriptors with at least three geometries and less than full-cohort coverage. The Pareto front maximises support and mean target simultaneously. The selected rule has the highest positive enrichment in the 25–35% individual-cohort band or the 10–20% pooled band. If three members already exceed the individual band, the best enriched rule at the smallest achievable support is flagged as a fallback. A coverage-constrained winner can lie off the unrestricted front. Outputs include peak and summed PNG figures, exact rules, member IDs, family counts and selection status.

For O₂ and O+O, the descriptors include each O atom's nearest Au distance, contact count and mean site GCN. For O+O, IFCT refers to the joint oxygen fragment. Exact-snapshot reruns 3419 and 3423 are linked to 3364 and 3368, respectively. CO+O is available as a structural cohort. Its archived three-fragment spectra require separate CO and O channel definitions for an optical search.

In comparison mode, the three target searches use the same eligible geometry IDs. Candidate selectors come from an explicit structural and ground-state descriptor list. PySubgroup discretises numerical descriptors into up to four bins by default. The default depth is two, minimum support is three and positive scores are calculated as the square root of subgroup size times the increase in target mean. The runner collapses identical geometry memberships among the ranked candidates and reports up to 20 rules per target. The output records support, subgroup and cohort means, Au sizes and member IDs. Brightness and directional-fraction rules are compared by intersection size and Jaccard overlap, and the mean directional intensity is reported for each intersection.

The reported search is exploratory. A rule may primarily distinguish nuclearity, especially in a small cohort. Inspect member IDs and nuclearities, compare within a nuclearity where support permits, and rerun with and without flagged spectra. The archived MEP profiles and outcome geometries are kept for reference. A reaction-barrier or pathway-specific optical search requires a separate eligible cohort with explicit geometry and reaction-step references.
