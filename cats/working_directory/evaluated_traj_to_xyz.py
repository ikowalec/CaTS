from __future__ import annotations

import argparse
import csv
import json
from math import isfinite
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.io.formats import UnknownFileTypeError

TARGET_SETTINGS = {
    "settings_profile": "cam-b3lyp-lanl2dz",
    "basis_keyword": "lanl2dz_mixed",
    "xc_functional": "CAM-B3LYP",
}
TARGET_BASIS_KEYWORDS = {"lanl2dz-mixed", "lanl2dz", "mixed", "mixed-lanl2dz"}
METALS = {"Cu", "Au"}
SKIP_PATH_PARTS = {"NEB"}
MAX_FRAME_FMAX_EV_A = 50.0
MAX_ABS_ENERGY_PER_ATOM_EV = 10000.0
MAX_ENERGY_OUTLIER_PER_ATOM_EV = 100.0
HIGH_FORCE_EV_A = 5.0
HIGH_ENERGY_PER_ATOM_EV = 0.5
MIN_ENERGY_CHANGE_PER_ATOM_EV = 0.02
DEFAULT_SIMILARITY_THRESHOLD = 0.98
MIN_PAIR_DISTANCE_RMSD_A = 0.08
INTERMEDIATE_FRAMES = 5
CONSTRAINT_WORDS = (
    "FixAtoms",
    "FixBond",
    "FixInternals",
    "set_constraint",
    "constraints.append",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from sklearn.preprocessing import normalize
    from similarity.rematch.get_rematch import get_cached_rematch_kernel
    from similarity.rematch.memory_rematch import _build_shared_soap_descriptor, get_soap
    HAS_REMATCH = True
except Exception:
    HAS_REMATCH = False


def norm(value: object) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def cache_uuid(root: Path, path: Path) -> str | None:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    for part in parts:
        compact = part.replace("-", "")
        if re.fullmatch(r"[0-9a-fA-F]{32}", compact):
            return compact.lower()
    return None


def load_run_metadata(root: Path) -> dict[str, dict]:
    db_path = next(
        (
            path
            for path in (
                root / "reaction_hub" / "db.sqlite3",
                root / "db.sqlite3",
                root.parent / "db.sqlite3",
            )
            if path.exists()
        ),
        None,
    )
    if db_path is None:
        return {}
    query = """
        select
            w.run_uid,
            w.run_name,
            c.name,
            c.xc_functional,
            c.basis_set,
            w.metadata,
            w.reference_adsorbate_id,
            w.reference_nanocluster_id
        from workflows_workflowrun w
        join references_calculationsettingsprofile c on w.settings_profile_id = c.id
    """
    runs = {}
    with sqlite3.connect(db_path) as con:
        for (
            run_uid,
            run_name,
            profile,
            xc,
            basis,
            metadata_text,
            reference_adsorbate_id,
            reference_nanocluster_id,
        ) in con.execute(query):
            try:
                metadata = json.loads(metadata_text or "{}")
            except json.JSONDecodeError:
                metadata = {}
            runs[str(run_uid).replace("-", "").lower()] = {
                "run_name": run_name,
                "settings_profile": profile,
                "xc_functional": xc,
                "basis_set": basis,
                "metadata": metadata,
                "reference_adsorbate_id": reference_adsorbate_id,
                "reference_nanocluster_id": reference_nanocluster_id,
            }
    return runs


def setting_payloads(root: Path, traj_path: Path, run_meta: dict | None) -> list[dict]:
    payloads = []
    if run_meta:
        meta = run_meta.get("metadata") if isinstance(run_meta.get("metadata"), dict) else {}
        result_summary = meta.get("result_summary") if isinstance(meta.get("result_summary"), dict) else {}
        settings = result_summary.get("settings")
        if isinstance(settings, dict):
            payloads.append(settings)

    for parent in [traj_path.parent, *traj_path.parents]:
        if root not in (parent, *parent.parents):
            break
        summary = parent / "reaction_hub_summary.json"
        if summary.exists():
            try:
                data = json.loads(summary.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            settings = data.get("settings")
            if isinstance(settings, dict):
                payloads.append(settings)

        for input_path in (parent / "input.py", parent / "run_workflow_submit.py"):
            if input_path.exists():
                text = input_path.read_text(encoding="utf-8", errors="replace")
                payloads.append(
                    {
                        "basis_keyword": _quoted_arg(text, "basis_keyword"),
                        "xc_functional": _quoted_arg(text, "xc_functional"),
                    }
                )
    return payloads


def _quoted_arg(text: str, name: str) -> str | None:
    match = re.search(rf"{re.escape(name)}\s*=\s*['\"]([^'\"]+)['\"]", text)
    return None if match is None else match.group(1)


def settings_status(
    root: Path,
    traj_path: Path,
    run_meta: dict | None,
    *,
    settings_profiles: set[str] | None = None,
    xc_functional: str | None = TARGET_SETTINGS["xc_functional"],
    basis_keywords: set[str] | None = TARGET_BASIS_KEYWORDS,
) -> str:
    allowed_profiles = {
        norm(profile)
        for profile in (settings_profiles or {TARGET_SETTINGS["settings_profile"]})
        if str(profile).strip()
    }
    profile_matches = bool(
        run_meta and norm(run_meta.get("settings_profile")) in allowed_profiles
    )
    if settings_profiles is not None and not profile_matches:
        return "settings_mismatch"
    allowed_basis = None if basis_keywords is None else {norm(keyword) for keyword in basis_keywords}
    payloads = setting_payloads(root, traj_path, run_meta)
    has_informative_payload = False
    has_matching_payload = False
    for payload in payloads:
        basis = payload.get("basis_keyword")
        xc = payload.get("xc_functional")
        has_basis = basis is not None and str(basis).strip() != ""
        has_xc = xc is not None and str(xc).strip() != ""
        if not has_basis and not has_xc:
            continue

        has_informative_payload = True
        basis_matches = (not has_basis) or allowed_basis is None or norm(basis) in allowed_basis
        xc_matches = (not has_xc) or xc_functional is None or norm(xc) == norm(xc_functional)
        if not basis_matches or not xc_matches:
            return "settings_conflict" if profile_matches else "settings_mismatch"
        if has_basis and has_xc:
            has_matching_payload = True

    if profile_matches or has_matching_payload:
        return "ok"
    return "settings_missing" if not has_informative_payload else "settings_mismatch"


def active_single_constraints(config: object) -> bool:
    if not isinstance(config, dict):
        return False
    return bool(
        config.get("fixed_symbols")
        or config.get("bond_pair_indices")
        or config.get("target_distance") is not None
    )


def has_constraint_metadata(run_meta: dict | None) -> bool:
    if not run_meta:
        return False
    metadata = run_meta.get("metadata", {})
    if not isinstance(metadata, dict):
        return False
    submission = metadata.get("submission") if isinstance(metadata.get("submission"), dict) else {}
    run_mode = str(submission.get("run_mode") or "").lower()
    if any(word in run_mode for word in ("neb", "barrier")):
        return True
    if active_single_constraints(submission.get("single_constraints_config")):
        return True
    for key in ("scan_config", "neb_config"):
        value = submission.get(key)
        if isinstance(value, dict) and value:
            return True
    text = json.dumps(metadata, sort_keys=True).lower()
    return any(word.lower() in text for word in CONSTRAINT_WORDS)


def has_constraint_input(root: Path, traj_path: Path) -> bool:
    names = {"input.py", "run_workflow_submit.py", "orca.inp"}
    for parent in [traj_path.parent, *traj_path.parents]:
        if root not in (parent, *parent.parents):
            break
        for candidate in parent.iterdir():
            if candidate.is_file() and (candidate.name in names or candidate.suffix.lower() == ".inp"):
                text = candidate.read_text(encoding="utf-8", errors="replace")
                if any(word in text for word in CONSTRAINT_WORDS):
                    return True
    return False


def has_frame_constraints(images) -> bool:
    return any(bool(getattr(atoms, "constraints", None)) for atoms in images)


def is_ground_state_path(traj_path: Path) -> bool:
    parts = {part.upper() for part in traj_path.parts}
    return not bool(parts & SKIP_PATH_PARTS)


def is_tddft_optimisation(run_meta: dict | None) -> bool:
    if not run_meta:
        return False
    metadata = run_meta.get("metadata", {})
    if not isinstance(metadata, dict):
        return False
    submission = metadata.get("submission") if isinstance(metadata.get("submission"), dict) else {}
    result_summary = (
        metadata.get("result_summary") if isinstance(metadata.get("result_summary"), dict) else {}
    )
    return bool(submission.get("use_tddft_optimisation")) or norm(result_summary.get("mode")) == "tddft"


def is_reference_adsorbate_run(run_meta: dict | None) -> bool:
    if not run_meta:
        return False
    if run_meta.get("reference_adsorbate_id") is not None:
        return True
    metadata = run_meta.get("metadata", {})
    if not isinstance(metadata, dict):
        return False
    payloads = [metadata]
    submission = metadata.get("submission")
    if isinstance(submission, dict):
        payloads.append(submission)
    result_summary = metadata.get("result_summary")
    if isinstance(result_summary, dict):
        payloads.append(result_summary)
    return any(
        payload.get("source") == "reference_adsorbate"
        or payload.get("reference_adsorbate_id") is not None
        for payload in payloads
    )


def gas_reference_key(run_meta: dict | None, atoms) -> str:
    metadata = run_meta.get("metadata", {}) if run_meta else {}
    payloads = [metadata] if isinstance(metadata, dict) else []
    for payload in list(payloads):
        submission = payload.get("submission")
        if isinstance(submission, dict):
            payloads.append(submission)
        result_summary = payload.get("result_summary")
        if isinstance(result_summary, dict):
            payloads.append(result_summary)
    for payload in payloads:
        for key in ("reference_adsorbate_id", "reference_adsorbate_name", "reference_adsorbate_formula"):
            value = payload.get(key)
            if value:
                return f"{key}:{value}"
    if run_meta and run_meta.get("reference_adsorbate_id") is not None:
        return f"reference_adsorbate_id:{run_meta['reference_adsorbate_id']}"
    return f"formula:{atoms.get_chemical_formula()}"


def final_energy_ev(run_meta: dict | None, atoms) -> float | None:
    for key in ("energy_eV", "energy", "potential_energy"):
        value = atoms.info.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    try:
        return float(atoms.get_potential_energy())
    except (RuntimeError, TypeError, ValueError):
        pass
    metadata = run_meta.get("metadata", {}) if run_meta else {}
    result_summary = metadata.get("result_summary") if isinstance(metadata, dict) else {}
    final = result_summary.get("final") if isinstance(result_summary, dict) else {}
    value = final.get("energy_eV") if isinstance(final, dict) else None
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def frame_energy_ev(atoms) -> float | None:
    for key in ("energy_eV", "energy", "potential_energy"):
        value = atoms.info.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    try:
        return float(atoms.get_potential_energy())
    except (RuntimeError, TypeError, ValueError):
        return None


def frame_fmax_ev_a(atoms) -> float | None:
    forces = None
    for kwargs in ({"apply_constraint": False}, {}):
        try:
            forces = atoms.get_forces(**kwargs)
            break
        except Exception:
            continue
    if forces is None:
        return None

    max_force = 0.0
    try:
        for vector in forces:
            fx = float(vector[0])
            fy = float(vector[1])
            fz = float(vector[2])
            if not (isfinite(fx) and isfinite(fy) and isfinite(fz)):
                return float("inf")
            magnitude = (fx * fx + fy * fy + fz * fz) ** 0.5
            if magnitude > max_force:
                max_force = magnitude
    except Exception:
        return None
    return max_force


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def sensible_frame(
    atoms,
    *,
    reference_energy_per_atom: float | None = None,
    max_abs_energy_per_atom: float | None = MAX_ABS_ENERGY_PER_ATOM_EV,
) -> tuple[bool, str]:
    fmax = frame_fmax_ev_a(atoms)
    if fmax is not None:
        if not isfinite(float(fmax)):
            return False, "bad_forces"
        if float(fmax) > MAX_FRAME_FMAX_EV_A:
            return False, f"fmax>{MAX_FRAME_FMAX_EV_A:g}"

    energy = frame_energy_ev(atoms)
    if energy is None:
        return True, ""
    if not isfinite(float(energy)):
        return False, "bad_energy"
    if len(atoms) > 0:
        energy_per_atom = float(energy) / float(len(atoms))
        if max_abs_energy_per_atom is not None and abs(energy_per_atom) > float(max_abs_energy_per_atom):
            return False, f"abs_energy_per_atom>{float(max_abs_energy_per_atom):g}"
        if (
            reference_energy_per_atom is not None
            and abs(energy_per_atom - float(reference_energy_per_atom)) > MAX_ENERGY_OUTLIER_PER_ATOM_EV
        ):
            return False, f"energy_outlier_per_atom>{MAX_ENERGY_OUTLIER_PER_ATOM_EV:g}"
    return True, ""


def trajectory_reference_energy_per_atom(
    images,
    *,
    max_abs_energy_per_atom: float | None = MAX_ABS_ENERGY_PER_ATOM_EV,
) -> float | None:
    values = []
    for atoms in images:
        energy = frame_energy_ev(atoms)
        if energy is None or not isfinite(float(energy)) or len(atoms) <= 0:
            continue
        energy_per_atom = float(energy) / float(len(atoms))
        if max_abs_energy_per_atom is None or abs(energy_per_atom) <= float(max_abs_energy_per_atom):
            values.append(energy_per_atom)
    return median(values)


def in_prefixed_folder(root: Path, traj_path: Path, prefix: str | None) -> bool:
    if prefix is None:
        return True
    if root.name.startswith(prefix):
        return True
    return any(part.startswith(prefix) for part in traj_path.relative_to(root).parts[:-1])


def pair_distance_rmsd(a, b) -> float:
    if len(a) != len(b) or a.get_chemical_symbols() != b.get_chemical_symbols():
        return float("inf")
    if len(a) < 2:
        return 0.0
    delta = a.get_all_distances(mic=any(a.get_pbc())) - b.get_all_distances(mic=any(b.get_pbc()))
    return float(np.sqrt(np.mean(delta * delta)))


def suggest_trajectory_threshold(
    soaps: dict[int, np.ndarray],
    sensible_indices: list[int],
    rematch_kernel,
    fallback_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> float:
    """Calculate trajectory-specific dynamic REMatch similarity threshold via elbow/percentile heuristic."""
    if len(sensible_indices) <= 2 or rematch_kernel is None:
        return fallback_threshold

    ref_idx = sensible_indices[-1]
    if ref_idx not in soaps:
        return fallback_threshold

    ref_soap = soaps[ref_idx]
    sims = []
    for idx in sensible_indices[:-1]:
        if idx in soaps:
            sim = float(rematch_kernel.create([ref_soap, soaps[idx]])[0, 1])
            sims.append(sim)

    if not sims:
        return fallback_threshold

    values = np.sort(np.asarray(sims, dtype=float))[::-1]
    if len(values) == 1:
        return float(values[0])

    search_end = min(len(values) - 1, max(3, int(np.ceil(0.5 * len(values)))))
    diffs = values[:search_end] - values[1 : search_end + 1]
    best_idx = int(np.argmax(diffs))
    best_gap = float(diffs[best_idx])
    suggested = float(0.5 * (values[best_idx] + values[best_idx + 1]))
    if best_gap < 1e-3:
        suggested = float(np.percentile(values, 95))

    return float(min(1.0 - 1e-6, max(0.0, suggested)))


def selected_indices(
    images,
    reference_energy_per_atom: float | None = None,
    *,
    sample_interval: int | None = None,
    max_abs_energy_per_atom: float | None = MAX_ABS_ENERGY_PER_ATOM_EV,
    similarity_threshold: float | str | None = "dynamic",
    shared_soap=None,
) -> tuple[tuple[int, ...], list[str]]:
    if isinstance(images, int):
        n_images = images
        if n_images <= 2:
            return tuple(range(n_images))
        step = sample_interval or 10
        n_steps = n_images - 1
        n_selected = max(2, (n_steps + step - 1) // step)
        return tuple(
            dict.fromkeys(round(i * (n_images - 1) / (n_selected - 1)) for i in range(n_selected))
        )

    sensible = []
    rejected = []
    energies_per_atom = []
    for index, atoms in enumerate(images):
        ok, reason = sensible_frame(
            atoms,
            reference_energy_per_atom=reference_energy_per_atom,
            max_abs_energy_per_atom=max_abs_energy_per_atom,
        )
        if not ok:
            rejected.append(reason)
            continue
        sensible.append(index)
        energy = frame_energy_ev(atoms)
        if energy is not None and len(atoms):
            energies_per_atom.append(float(energy) / len(atoms))
    if not sensible:
        return (), rejected

    # Precompute SOAPs for sensible frames if REMatch available
    soaps: dict[int, np.ndarray] = {}
    rematch_kernel = None
    if HAS_REMATCH and len(sensible) > 2:
        try:
            if shared_soap is None:
                shared_soap = _build_shared_soap_descriptor([images[i] for i in sensible])
            for index in sensible:
                raw_soap = get_soap(images[index], soap_descriptor=shared_soap)
                soaps[index] = normalize(raw_soap)
            rematch_kernel = get_cached_rematch_kernel()
        except Exception:
            soaps = {}
            rematch_kernel = None

    # Determine dynamic per-trajectory similarity threshold or use fixed value
    if similarity_threshold is None or similarity_threshold == "dynamic":
        traj_threshold = suggest_trajectory_threshold(
            soaps, sensible, rematch_kernel, fallback_threshold=DEFAULT_SIMILARITY_THRESHOLD
        )
    else:
        traj_threshold = float(similarity_threshold)

    energy_floor = min(energies_per_atom) if energies_per_atom else None
    endpoint_indices = {sensible[0], sensible[-1]}
    unusual = set()
    ordinary = []
    kept_so_far = [sensible[0]]
    previous = sensible[0]
    previous_energy = frame_energy_ev(images[previous])
    for index in sensible[1:-1]:
        atoms = images[index]
        energy = frame_energy_ev(atoms)
        energy_per_atom = None if energy is None or not len(atoms) else float(energy) / len(atoms)
        fmax = frame_fmax_ev_a(atoms)
        is_unusual = (
            (fmax is not None and fmax >= HIGH_FORCE_EV_A)
            or (
                energy_floor is not None
                and energy_per_atom is not None
                and energy_per_atom - energy_floor >= HIGH_ENERGY_PER_ATOM_EV
            )
        )

        if is_unusual:
            unusual.add(index)
            kept_so_far.append(index)
            previous = index
            previous_energy = energy
            continue

        # Similarity screening against already-kept structures (following deduplicate_atoms_by_similarity)
        is_duplicate = False
        if rematch_kernel is not None and index in soaps:
            max_sim = max(
                (float(rematch_kernel.create([soaps[index], soaps[k]])[0, 1]) for k in kept_so_far if k in soaps),
                default=0.0,
            )
            if max_sim >= traj_threshold:
                is_duplicate = True
        else:
            if pair_distance_rmsd(images[previous], atoms) < MIN_PAIR_DISTANCE_RMSD_A:
                is_duplicate = True

        previous_energy_per_atom = (
            None
            if previous_energy is None or not len(images[previous])
            else float(previous_energy) / len(images[previous])
        )
        if is_duplicate and previous_energy_per_atom is not None and energy_per_atom is not None:
            if abs(energy_per_atom - previous_energy_per_atom) >= MIN_ENERGY_CHANGE_PER_ATOM_EV:
                is_duplicate = False

        if not is_duplicate and len(ordinary) < INTERMEDIATE_FRAMES:
            ordinary.append(index)
            kept_so_far.append(index)
            previous = index
            previous_energy = energy
    return tuple(sorted(endpoint_indices | unusual | set(ordinary))), rejected


def collect_images(
    root: Path,
    folder_prefix: str | None = None,
    species: set[str] | None = None,
    settings_profiles: set[str] | None = None,
    xc_functional: str | None = TARGET_SETTINGS["xc_functional"],
    basis_keywords: set[str] | None = TARGET_BASIS_KEYWORDS,
    max_abs_energy_per_atom: float | None = MAX_ABS_ENERGY_PER_ATOM_EV,
    sample_interval: int | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
):
    root = root.resolve()
    run_metadata = load_run_metadata(root)
    images = []
    audit = []
    gas_reference_candidates = []
    for traj_path in sorted(root.rglob("evaluated.traj")):
        if not in_prefixed_folder(root, traj_path, folder_prefix):
            audit.append((traj_path, "skipped", "prefix"))
            continue
        run_meta = run_metadata.get(cache_uuid(root, traj_path) or "")
        if not is_ground_state_path(traj_path):
            audit.append((traj_path, "skipped", "neb_path"))
            continue
        if is_tddft_optimisation(run_meta):
            audit.append((traj_path, "skipped", "tddft_optimisation"))
            continue
        try:
            traj_images = read(traj_path, index=":")
        except (UnknownFileTypeError, EOFError, OSError, ValueError) as exc:
            audit.append((traj_path, "skipped", f"unreadable: {exc}"))
            continue
        if not traj_images:
            audit.append((traj_path, "skipped", "empty"))
            continue
        frame_species = set(traj_images[0].get_chemical_symbols())
        if species is not None and not frame_species <= species:
            audit.append((traj_path, "skipped", "species_filter"))
            continue
        has_target_metal = bool(frame_species & METALS)
        is_gas_reference = not has_target_metal and is_reference_adsorbate_run(run_meta)
        if not has_target_metal and not is_gas_reference:
            audit.append((traj_path, "skipped", "no_cu_or_au_or_reference_adsorbate"))
            continue
        settings_reason = settings_status(
            root,
            traj_path,
            run_meta,
            settings_profiles=settings_profiles,
            xc_functional=xc_functional,
            basis_keywords=basis_keywords,
        )
        if settings_reason != "ok":
            audit.append((traj_path, "skipped", settings_reason))
            continue
        if has_constraint_metadata(run_meta) or has_constraint_input(root, traj_path):
            audit.append((traj_path, "skipped", "constraints_in_input_or_metadata"))
            continue
        if has_frame_constraints(traj_images):
            audit.append((traj_path, "skipped", "constraints_in_traj"))
            continue

        reference_energy_per_atom = trajectory_reference_energy_per_atom(
            traj_images,
            max_abs_energy_per_atom=max_abs_energy_per_atom,
        )
        if is_gas_reference:
            final_atoms = traj_images[-1]
            sensible, reason = sensible_frame(
                final_atoms,
                reference_energy_per_atom=reference_energy_per_atom,
                max_abs_energy_per_atom=max_abs_energy_per_atom,
            )
            if not sensible:
                audit.append((traj_path, "skipped", f"nonsensical_final_frame:{reason}"))
                continue
            gas_reference_candidates.append(
                {
                    "path": traj_path,
                    "key": gas_reference_key(run_meta, final_atoms),
                    "atoms": final_atoms,
                    "energy": final_energy_ev(run_meta, final_atoms),
                }
            )
            continue

        if sample_interval is not None:
            indices = selected_indices(len(traj_images), sample_interval=sample_interval)
            selected = []
            bad_reasons = []
            for i in indices:
                atoms = traj_images[i]
                sensible, reason = sensible_frame(
                    atoms,
                    reference_energy_per_atom=reference_energy_per_atom,
                    max_abs_energy_per_atom=max_abs_energy_per_atom,
                )
                if sensible:
                    selected.append(atoms)
                else:
                    bad_reasons.append(reason)
        else:
            indices, bad_reasons = selected_indices(
                traj_images,
                reference_energy_per_atom,
                max_abs_energy_per_atom=max_abs_energy_per_atom,
                similarity_threshold=similarity_threshold,
            )
            selected = [traj_images[i] for i in indices]
        if not selected:
            audit.append((traj_path, "skipped", "no_sensible_frames"))
            continue
        images.extend(selected)
        suffix = f"{len(selected)} frames"
        if bad_reasons:
            suffix += f"; skipped {len(bad_reasons)} nonsensical"
        audit.append((traj_path, "included", suffix))

    selected_gas_references = {}
    for candidate in gas_reference_candidates:
        energy = candidate["energy"]
        sort_key = (energy is None, float("inf") if energy is None else energy, str(candidate["path"]))
        key = candidate["key"]
        if key not in selected_gas_references or sort_key < selected_gas_references[key][0]:
            selected_gas_references[key] = (sort_key, candidate)
    selected_paths = {candidate["path"] for _, candidate in selected_gas_references.values()}
    for candidate in gas_reference_candidates:
        if candidate["path"] in selected_paths:
            images.append(candidate["atoms"])
            energy = candidate["energy"]
            suffix = "unknown energy" if energy is None else f"{energy:.8g} eV"
            audit.append((candidate["path"], "included", f"reference_adsorbate_final_frame; {suffix}"))
        else:
            audit.append((candidate["path"], "skipped", "less_stable_reference_adsorbate"))
    return images, audit


def write_audit(path: Path, audit) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory", "status", "reason"])
        writer.writerows(audit)


def parse_species(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    species = {symbol.strip() for value in values for symbol in value.split(",") if symbol.strip()}
    return species or None


def parse_csv_values(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    parsed = {item.strip() for value in values for item in value.split(",") if item.strip()}
    return parsed or None


def parse_optional_float(value: str | None) -> float | None:
    text = str(value or "").strip()
    return None if not text else float(text)


def _self_check() -> None:
    frames = []
    for index in range(10):
        atoms = Atoms("Au2", positions=[[0, 0, 0], [2.5 + (0.2 if index >= 3 else 0), 0, 0]])
        energy = 0.0 if index == 3 else -2.0
        force = 6.0 if index == 2 else 0.1
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=[[force, 0, 0], [-force, 0, 0]],
        )
        frames.append(atoms)
    indices, rejected = selected_indices(frames, trajectory_reference_energy_per_atom(frames))
    assert not rejected
    assert {0, 2, 3, 9} <= set(indices)
    assert 8 not in indices


def main() -> None:
    if "--self-check" in sys.argv:
        _self_check()
        print("self-check passed")
        return
    parser = argparse.ArgumentParser(
        description="Collect sampled images from eligible evaluated.traj files into extxyz."
    )
    parser.add_argument("folder", type=Path, help="Folder to search recursively")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("mace_training.xyz"),
        help="Output XYZ file (default: mace_training.xyz)",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        help="Only include trajectories inside folders whose name starts with this prefix",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("mace_training_audit.csv"),
        help="Audit CSV path (default: mace_training_audit.csv)",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        help="Only include frames whose chemical symbols are all in this list, e.g. --species Au Cu",
    )
    parser.add_argument(
        "--settings-profile",
        nargs="+",
        help=(
            "Only include runs using these settings profile names. "
            f"Default: {TARGET_SETTINGS['settings_profile']}"
        ),
    )
    parser.add_argument(
        "--xc-functional",
        default=TARGET_SETTINGS["xc_functional"],
        help="Require this XC functional in discovered settings payloads; use empty string to disable.",
    )
    parser.add_argument(
        "--basis-keyword",
        nargs="+",
        default=sorted(TARGET_BASIS_KEYWORDS),
        help="Accepted basis keywords in discovered settings payloads; use empty string to disable.",
    )
    parser.add_argument(
        "--max-abs-energy-per-atom",
        default=str(MAX_ABS_ENERGY_PER_ATOM_EV),
        help="Reject frames above this absolute eV/atom; use empty string to disable for all-electron totals.",
    )
    args = parser.parse_args()
    xc_functional = str(args.xc_functional or "").strip() or None
    basis_keywords = parse_csv_values(args.basis_keyword)
    images, audit = collect_images(
        args.folder,
        args.prefix,
        parse_species(args.species),
        settings_profiles=parse_csv_values(args.settings_profile),
        xc_functional=xc_functional,
        basis_keywords=basis_keywords,
        max_abs_energy_per_atom=parse_optional_float(args.max_abs_energy_per_atom),
    )
    write_audit(args.audit, audit)
    if not images:
        raise SystemExit(
            f"No eligible images found in evaluated.traj files under {args.folder}; "
            f"see {args.audit}"
        )

    write(args.output, images, format="extxyz")
    included = sum(1 for _, status, _ in audit if status == "included")
    skipped = len(audit) - included
    print(f"Wrote {len(images)} images from {included} trajectories to {args.output}")
    print(f"Skipped {skipped} trajectories; audit: {args.audit}")


if __name__ == "__main__":
    main()
