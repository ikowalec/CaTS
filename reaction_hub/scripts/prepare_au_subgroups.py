"""Prepare provenance-linked Au CO-oxidation data for subgroup discovery."""
import argparse
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

REPORT = Path(__file__).resolve().parents[1] / "reports/au_co_oxidation_light_driven_20260823"
WINDOWS = {"blue": (400, 500), "visible": (500, 600),
           "red": (600, 625), "red_extended": (600, 700)}
OPTICAL_FAMILIES = {"CO", "O", "O2", "O+O"}
HC_EV_NM = 1239.841984
AU_NEIGHBOUR = 3.3
CONTACT = 2.7
MOLECULAR_OO_MAX = 1.6
ARCHIVE_SUPPLEMENT = (47, 48, 165, 174, 175)
REVIEWED_DATABASE_OPTICAL = {2894: 3226, 3437: 3437}
EXACT_GEOMETRY_ALIASES = {3080: 3081, 3138: 3081, 3419: 3364, 3423: 3368}


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def geometry_features(symbols, xyz):
    au = np.flatnonzero(symbols == "Au")
    ads = np.flatnonzero(symbols != "Au")
    metals = xyz[au]
    delta = metals[:, None] - metals
    distance = np.linalg.norm(delta, axis=-1)
    neighbours = (distance <= AU_NEIGHBOUR) & (distance > 0)
    cn = neighbours.sum(axis=1)
    gcn = neighbours @ cn / 12
    _, components = connected_components(neighbours, directed=False)
    component = np.bincount(components)
    centred = metals - metals.mean(axis=0)
    eigen = np.linalg.eigvalsh(centred.T @ centred / len(au)).clip(0)
    shape = {
        "au_n": len(au), "au_rg_a": np.sqrt(eigen.sum()), "au_span_a": distance.max(),
        "au_axis_short_long": np.sqrt(eigen[0] / eigen[-1]),
        "au_axis_middle_long": np.sqrt(eigen[1] / eigen[-1]),
        "au_asphericity": 1.5 * np.sum((eigen - eigen.mean()) ** 2) / eigen.sum() ** 2,
        "au_largest_component_n": component.max(), "au_component_count": len(component),
        "au_gcn_mean": gcn.mean(), "au_gcn_std": gcn.std(),
        "au_fraction_cn_le6": np.mean(cn <= 6),
    }
    d = np.linalg.norm(xyz[ads, None] - metals, axis=2)
    labels = symbols[ads]
    for element in ("C", "O"):
        if not (labels == element).any():
            continue
        ds = d[labels == element]
        contacts = (ds <= CONTACT).any(axis=0)
        shape.update({
            f"{element}_au_min_a": ds.min(),
            f"{element}_au_contacts": contacts.sum(),
            f"{element}_au_gcn_min": gcn[contacts].min() if contacts.any() else np.nan,
            f"{element}_au_gcn_mean": gcn[contacts].mean() if contacts.any() else np.nan,
            f"{element}_au_gcn_spread": np.ptp(gcn[contacts]) if contacts.any() else np.nan,
            f"{element}_au_cn_mean": cn[contacts].mean() if contacts.any() else np.nan,
            f"{element}_site": {0: "detached", 1: "atop", 2: "bridge", 3: "hollow"}.get(
                int(contacts.sum()), "multiple"),
        })
    o = xyz[symbols == "O"]
    c = xyz[symbols == "C"]
    if len(o) == 2:
        shape["oo_a"] = np.linalg.norm(o[0] - o[1])
        od = d[labels == "O"]
        proximal, distal = np.argsort(od.min(axis=1), kind="stable")
        low, high = od.min(axis=1)[[proximal, distal]]
        shape.update(o_proximal_au_a=low, o_distal_au_a=high, o_contact_asymmetry_a=high-low,
                     o_shared_au=int(np.all(od <= CONTACT, axis=0).sum()))
        for label, index in (("proximal", proximal), ("distal", distal)):
            contacts = od[index] <= CONTACT
            shape[f"o_{label}_au_contacts"] = contacts.sum()
            shape[f"o_{label}_au_gcn_mean"] = gcn[contacts].mean() if contacts.any() else np.nan
        shape["o_contact_mode"] = {0: "detached", 1: "one_O_contact", 2: "two_O_contacts"}[
            int((od.min(axis=1) <= CONTACT).sum())]
    if len(c):
        co = np.sort(np.linalg.norm(o - c[0], axis=1))
        shape["co_a"] = co[0]
        if len(co) == 2:
            shape["second_co_a"] = co[1]
            vectors = o - c[0]
            shape["oco_angle_deg"] = np.degrees(np.arccos(np.clip(
                vectors[0] @ vectors[1] / np.prod(co), -1, 1)))
    return shape


def classify(symbols, geometry):
    c, o = (np.count_nonzero(symbols == e) for e in ("C", "O"))
    if (c, o) == (0, 2):
        return "O2" if geometry["oo_a"] <= MOLECULAR_OO_MAX else "O+O"
    if (c, o) == (1, 2):
        return "CO2" if geometry["second_co_a"] <= 1.5 else "CO+O"
    return {(1, 1): "CO", (0, 1): "O"}.get((c, o), "other")


def spectral_rows(dataset, symbols):
    """Use aligned complete intensity arrays, preserving archive order."""
    fragments = dataset["fragments"]["fragments_0based"]
    au_keys = [k for k, indices in fragments.items() if set(indices) == set(np.flatnonzero(symbols == "Au"))]
    if len(au_keys) != 1 or len(fragments) != 2:
        raise ValueError("Requires two fragments with a complete Au partition")
    au = au_keys[0]
    ads = next(k for k in fragments if k != au)
    if set(fragments[ads]) != set(np.flatnonzero(symbols != "Au")):
        raise ValueError("Adsorbate atom partition differs from geometry")
    directions = {}
    for series in dataset["ct_spectra"]:
        name = series["property"]
        if name == "UV-Vis":
            directions["f"] = series["points"]
        if name.startswith("Electron transfer "):
            left, arrow, right = re.split("(->|<-)", name.split()[-1])
            start, end = (left, right) if arrow == "->" else (right, left)
            if (start, end) in [(au, ads), (ads, au)]:
                directions["au_to_ads" if start == au else "ads_to_au"] = series["points"]
    if set(directions) != {"f", "au_to_ads", "ads_to_au"}:
        raise ValueError("Incomplete directional arrays")
    if len({len(series) for series in directions.values()}) != 1:
        raise ValueError("Unequal line counts")
    rows = []
    for total, forward, reverse in zip(*(directions[k] for k in ("f", "au_to_ads", "ads_to_au"))):
        energy = total["energy_eV"]
        if any(abs(point["energy_eV"] - energy) > 1e-7 for point in (forward, reverse)):
            raise ValueError("Directional line energies are misaligned")
        f, a, b = (float(point["intensity"]) for point in (total, forward, reverse))
        if not np.isfinite([energy, f, a, b]).all() or (energy > 0 and (min(f, a, b) < 0 or a + b > f + 2e-5)):
            raise ValueError("Invalid directional intensity")
        rows.append(dict(energy_ev=energy, wavelength_nm=HC_EV_NM / energy if energy > 0 else np.nan,
                         oscillator_strength=f, au_to_ads_weight=a, ads_to_au_weight=b,
                         au_to_ads_fraction=a/f if f > 0 and energy > 0 else np.nan,
                         ads_to_au_fraction=b/f if f > 0 and energy > 0 else np.nan))
    return rows


def window_targets(lines):
    result = {}
    for name, (low, high) in WINDOWS.items():
        within = lines[lines.wavelength_nm.between(
            low, high, inclusive="left" if name in {"blue", "visible"} else "both")]
        covered = lines.energy_ev.max() >= HC_EV_NM / low
        f = within.oscillator_strength.sum()
        result[f"{name}_covered"] = bool(covered)
        result[f"{name}_f_sum"] = f if covered else np.nan
        for direction in ("au_to_ads", "ads_to_au"):
            w = within[f"{direction}_weight"]
            prefix = f"{name}_{direction}"
            result[f"{prefix}_weighted"] = w.sum() if covered else np.nan
            result[f"{prefix}_fraction"] = w.sum() / f if covered and f else np.nan
            result[f"{prefix}_best_root"] = w.max() if covered and len(w) else 0.0 if covered else np.nan
            result[f"{prefix}_best_wavelength_nm"] = (
                within.loc[w.idxmax(), "wavelength_nm"] if covered and len(w) and w.max() > 0 else np.nan)
    return result


def prepare(report, out, database, include_reviewed_optical=False):
    props = defaultdict(dict)
    for record in jsonl(report / "raw_data/report_source_properties.jsonl"):
        props[record["workflow_run"]][record["name"]] = record
    archived = set(props)
    supplement = json.loads((report / "scientific_revision_20261106/provenance/supplemental_properties.json")
                            .read_text(encoding="utf-8"))
    supplement_by_id = {p["workflow_run_id"]: p for p in supplement["properties"]}
    overrides = {int(k): v for k, v in supplement["source_map"].items()}
    workflow = {r["workflow_run"]: r for r in jsonl(report / "raw_data/workflow_run_provenance.jsonl")}
    records = pd.read_csv(report / "workflow_analysis.csv").set_index("workflow_run")
    extra = pd.read_csv(report / "au13_co_o_outcome_classification.csv").set_index("workflow_run_id")
    supplemental_charge = {}
    reviewed = {}
    if database:
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            for i in ARCHIVE_SUPPLEMENT:
                charge_row = conn.execute("select system_charge from workflows_workflowrun where id=?", (i,)).fetchone()
                if charge_row:
                    supplemental_charge[i] = charge_row[0]
                for name, val, metadata in conn.execute(
                    """select name,value_float,metadata from results_computedproperty
                    where workflow_run_id=? and name in ('final_structure_snapshot',
                    'final_single_point_energy_eV','multiwfn_ground_mulliken_charges')""", (i,)):
                    props[i].setdefault(name, dict(value_float=val, metadata=json.loads(metadata)))
            if include_reviewed_optical:
                for geometry, workup in REVIEWED_DATABASE_OPTICAL.items():
                    charge, profile, state, metadata = conn.execute(
                        "select system_charge,settings_profile_id,status,metadata "
                        "from workflows_workflowrun where id=?", (geometry,)
                    ).fetchone()
                    workup_profile, workup_state, workup_metadata = conn.execute(
                        "select settings_profile_id,status,metadata from workflows_workflowrun where id=?",
                        (workup,),
                    ).fetchone()
                    workup_metadata = json.loads(workup_metadata)
                    if (profile != 2 or workup_profile != profile
                        or state != "completed" or workup_state != "completed"
                        or (geometry != workup and workup_metadata.get("source_run_id") != geometry)):
                        raise ValueError(f"Reviewed optical workflow no longer matches its provenance: {geometry}")
                    metadata = json.loads(metadata)
                    reviewed[geometry] = dict(
                        system_charge=charge, settings_profile_id=profile,
                        xc_functional="CAM-B3LYP", basis_set="6-31G(d,p)/LANL2DZ(ECP on Au)",
                        interaction_energy_eV=metadata.get("result_summary", {}).get("final", {})
                        .get("interaction_energy_eV"),
                    )
                    for prop_id, name, value, payload in conn.execute(
                        """select id,name,value_float,metadata from results_computedproperty
                           where workflow_run_id=? and name in ('final_structure_snapshot',
                           'final_single_point_energy_eV','multiwfn_ground_mulliken_charges','spectral_dataset')""",
                        (geometry,),
                    ):
                        props[geometry][name] = dict(id=prop_id, value_float=value, metadata=json.loads(payload))
    rows, spectra, audit = [], [], []
    for i in sorted(set(records.index) | set(ARCHIVE_SUPPLEMENT) | set(reviewed)):
        row = records.loc[i].to_dict() if i in records.index else reviewed.get(i, {})
        context = workflow.get(i, {})
        path = context.get("legacy_source", {}).get("orca_out", "")
        scan = str(row.get("activation_stage", "")).casefold()
        cohort = "scan" if (
            "scan" in scan or "transition-state" in scan or "ct/tddft structure" in scan
            or "fbl" in path.casefold() or context.get("submission", {}).get("scan_config")
        ) else "optimised_candidate"
        provenance = dict(geometry_id=i, source="archive" if i in archived else "local_snapshot_supplement",
                          cohort=cohort, duplicate_of=row.get("duplicate_of_workflow_run"))
        if i in reviewed:
            provenance.update(source="reviewed_database_optical", settings_profile_id=row["settings_profile_id"],
                              spectrum_property_id=props[i].get("spectral_dataset", {}).get("id"),
                              geometry_property_id=props[i].get("final_structure_snapshot", {}).get("id"))
        audit.append(provenance)
        if pd.notna(provenance["duplicate_of"]):
            provenance["status"] = "documented_duplicate"
            continue
        snapshot = props[i].get("final_structure_snapshot", {}).get("metadata")
        if i in EXACT_GEOMETRY_ALIASES:
            representative = EXACT_GEOMETRY_ALIASES[i]
            assert snapshot == props[representative]["final_structure_snapshot"]["metadata"], i
            provenance.update(status="same_snapshot_alias", duplicate_of=representative)
            continue
        if not snapshot:
            provenance["status"] = "missing_geometry"
            continue
        symbols = np.array(snapshot["symbols"])
        xyz = np.asarray(snapshot["positions_A"], dtype=float)
        assert xyz.shape == (len(symbols), 3) and np.isfinite(xyz).all(), i
        features = geometry_features(symbols, xyz)
        family = classify(symbols, features)
        provenance.update(family=family, geometry_sha256=hashlib.sha256(
            json.dumps(snapshot, sort_keys=True).encode()).hexdigest(), status="geometry_only")
        coords_charge = context.get("legacy_source", {}).get("input_text", "")
        match = re.search(r"(?im)^\s*\*\s*xyz\s+(-?\d+)\s+(\d+)", coords_charge)
        multiplicity = row.get("spin_multiplicity")
        if pd.isna(multiplicity):
            multiplicity = context.get("submission", {}).get("spin_multiplicity") or (
                int(match[2]) if match else np.nan)
        energy = row.get("final_energy_eV", props[i].get("final_single_point_energy_eV", {}).get("value_float"))
        if i in extra.index:
            energy = extra.loc[i, "final_energy_eV"]
        charge = row.get("system_charge", supplemental_charge.get(i, int(match[1]) if match else np.nan))
        charge_prop = props[i].get("multiwfn_ground_mulliken_charges", {}).get("metadata", {})
        q = np.asarray(charge_prop.get("charges", []))
        charge_ok = list(charge_prop.get("symbols", [])) == list(symbols) and q.shape == symbols.shape
        geom = dict(geometry_id=i, family=family, cohort=cohort, system_charge=charge,
                    spin_multiplicity=multiplicity, eref_ev=row.get("interaction_energy_eV"),
                    archived_delta_e_ev=row.get("relative_energy_eV"), final_energy_ev=energy,
                    final_formula="".join(e + (str(n) if n > 1 else "") for e, n in zip(*np.unique(symbols, return_counts=True))), xc_functional=row.get("xc_functional"),
                    basis_set=row.get("basis_set"),
                    ads_mulliken_e=q[symbols != "Au"].sum() if charge_ok else np.nan,
                    o_mulliken_asymmetry_e=abs(np.diff(q[symbols == "O"])[0])
                    if charge_ok and np.count_nonzero(symbols == "O") == 2 else np.nan,
                    **features)
        rows.append(geom)
        spectral = supplement_by_id.get(overrides.get(i)) or props[i].get("spectral_dataset")
        if family not in OPTICAL_FAMILIES or not spectral:
            continue
        dataset = spectral["metadata"].get("datasets", {}).get("final_tddft", {})
        if not dataset.get("ct_spectra"):
            provenance["status"] = "missing_directional_arrays"
            continue
        source_id = dataset.get("provenance", {}).get("source_run_id")
        if source_id is not None and int(source_id) != i:
            provenance["status"] = "spectral_geometry_provenance_mismatch"
            continue
        try:
            lines = pd.DataFrame(spectral_rows(dataset, symbols))
        except (ValueError, KeyError, TypeError) as exc:
            provenance["status"] = str(exc)
            continue
        if lines.empty:
            provenance["status"] = "empty_directional_arrays"
            continue
        workup = dataset.get("provenance", {}).get("charge_transfer_run_id") or overrides.get(i, i)
        if i in reviewed and workup != REVIEWED_DATABASE_OPTICAL[i]:
            raise ValueError(f"Reviewed optical spectrum has an unexpected workup: {i}")
        provenance.update(status="aligned_archive_arrays", workup_id=workup, line_count=len(lines))
        geom.update(window_targets(lines))
        geom["negative_energy_line_present"] = bool((lines.energy_ev < 0).any())
        lines.insert(0, "geometry_id", i)
        lines.insert(1, "workup_id", workup)
        lines.insert(2, "line_index", np.arange(len(lines)))
        spectra.append(lines)
    data = pd.DataFrame(rows)
    extracted_reviewed = {row["geometry_id"] for row in audit if row["source"] == "reviewed_database_optical"
                          and row["status"] == "aligned_archive_arrays"}
    if extracted_reviewed != set(reviewed):
        raise ValueError(f"Missing aligned arrays for reviewed geometries: {set(reviewed) - extracted_reviewed}")
    if not data.geometry_id.is_unique:
        raise ValueError("Geometry IDs are not unique")
    if not spectra:
        raise ValueError("No aligned spectra found for the supported adsorbate families")
    reference = ["final_formula", "system_charge", "spin_multiplicity", "xc_functional", "basis_set"]
    optimised = data.cohort.eq("optimised_candidate")
    matched = data.loc[optimised].groupby(reference).final_energy_ev
    data.loc[optimised, "delta_e_within_archive_family_ev"] = (
        data.loc[optimised, "final_energy_ev"] - matched.transform("min")
    ).where(matched.transform("count").ge(2))
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "geometries.csv", index=False)
    pd.concat(spectra, ignore_index=True).to_csv(out / "transitions.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "provenance.csv", index=False)
    for family in ("CO", "O", "O2", "O+O", "CO+O", "CO2"):
        view = data[(data.family == family) & (data.cohort == "optimised_candidate")]
        view.to_csv(out / f"{family.replace('+', '_')}.csv", index=False)
        print(family, len(view), "visible coverage", int(view.visible_covered.eq(True).sum()))
    data[data.cohort == "scan"].to_csv(out / "scans.csv", index=False)
    profiles = {}
    for filename in ("converged_mep_profiles.csv", "au20_mep3326_refined_segments.csv"):
        path = report / filename
        if path.exists():
            profiles[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            pd.read_csv(path).to_csv(out / filename, index=False)
    manifest = dict(windows_nm=WINDOWS, au_neighbour_a=AU_NEIGHBOUR, contact_a=CONTACT,
                    molecular_oo_max_a=MOLECULAR_OO_MAX, au_gcn_bulk_normalisation=12,
                    archive_profiles_sha256=profiles, exact_geometry_aliases=EXACT_GEOMETRY_ALIASES,
                    local_snapshot_supplement=sorted(supplemental_charge),
                    reviewed_database_optical={i: REVIEWED_DATABASE_OPTICAL[i] for i in reviewed},
                    inputs_sha256={name: hashlib.sha256((report / name).read_bytes()).hexdigest()
                                   for name in ("raw_data/report_source_properties.jsonl",
                                                "raw_data/workflow_run_provenance.jsonl",
                                                "workflow_analysis.csv", "au13_co_o_outcome_classification.csv",
                                                "scientific_revision_20261106/provenance/supplemental_properties.json")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--database", type=Path, help="Read-only Au13 snapshot supplement")
    parser.add_argument("--include-reviewed-optical", action="store_true",
                        help="Include additional reviewed, settings-matched optical workflows from the database")
    args = parser.parse_args()
    if args.include_reviewed_optical and not args.database:
        parser.error("--include-reviewed-optical requires --database")
    prepare(args.report, args.out, args.database, args.include_reviewed_optical)

