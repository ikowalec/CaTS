from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CplasRunCandidate:
    run_dir: Path
    workflow_name: str


def _skip_scan_plot_image(path: Path) -> bool:
    return path.name.startswith("uvvis_iter_")


def _is_numeric_job_archive_name(name: str) -> bool:
    return name.isdigit() and len(name) >= 4


def _is_shadowed_legacy_archive_artifact(run_dir: Path, artifact_path: Path) -> bool:
    try:
        rel_path = artifact_path.resolve().relative_to(run_dir.resolve())
    except ValueError:
        return False
    parts = rel_path.parts
    if not parts or not _is_numeric_job_archive_name(parts[0]):
        return False
    root_equivalent = run_dir.joinpath(*parts[1:])
    return root_equivalent.exists()


def _summary_db_paths(summary_payload: dict) -> list[str]:
    results: list[str] = []

    def _collect(raw) -> None:
        if isinstance(raw, str):
            value = raw.strip()
            if value:
                results.append(value)
            return
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, str):
                    value = entry.strip()
                    if value:
                        results.append(value)

    final = summary_payload.get("final")
    if isinstance(final, dict):
        for key in ("dft_db_paths", "database_paths", "final_db_paths", "dft_db_path", "database_path", "final_db_path"):
            _collect(final.get(key))

    artifacts = summary_payload.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("db_paths", "database_paths", "db_path", "database_path"):
            _collect(artifacts.get(key))

    for key in ("dft_db_paths", "database_paths", "db_paths", "dft_db_path", "database_path", "db_path"):
        _collect(summary_payload.get(key))

    return results


def discover_cplas_tddft_runs(search_root: Path) -> list[CplasRunCandidate]:
    """Discover TDDFT active-learning runs from cplas outputs (read-only)."""
    root = search_root.resolve()
    candidates: dict[Path, CplasRunCandidate] = {}

    for marker in root.rglob("TDDFT_active_learning_shared/workflow_root.txt"):
        run_dir = marker.parent.parent.resolve()
        candidates[run_dir] = CplasRunCandidate(
            run_dir=run_dir,
            workflow_name="cplas_tddft_active_learning",
        )

    for d in root.rglob("TDDFT"):
        if not d.is_dir():
            continue
        run_dir = d.parent.resolve()
        if (run_dir / "DFT").exists() or (run_dir / "TDDFT_active_learning_shared").exists():
            candidates.setdefault(
                run_dir,
                CplasRunCandidate(
                    run_dir=run_dir,
                    workflow_name="cplas_tddft_active_learning",
                ),
            )

    # Also discover runs that only expose Reaction Hub summary outputs.
    # This covers minimal single-optimisation runs where TDDFT directory
    # markers may be absent but reaction_hub_summary.json is present.
    for summary in root.rglob("reaction_hub_summary.json"):
        if not summary.is_file():
            continue
        parent = summary.parent.resolve()
        if parent.name == "TDDFT_active_learning_shared":
            run_dir = parent.parent.resolve()
        elif _is_numeric_job_archive_name(parent.name):
            run_dir = parent.parent.resolve()
        else:
            run_dir = parent
        candidates.setdefault(
            run_dir,
            CplasRunCandidate(
                run_dir=run_dir,
                workflow_name="cplas_tddft_active_learning",
            ),
        )

    return sorted(candidates.values(), key=lambda item: str(item.run_dir))


def collect_cplas_artifacts(run_dir: Path) -> list[tuple[str, Path, str]]:
    """Return (kind, absolute_path, label) tuples."""
    candidates = [
        ("json", run_dir / "reaction_hub_summary.json", "reaction_hub_summary"),
        ("structure", run_dir / "reaction_hub_final.xyz", "structure:reaction_hub_final.xyz"),
        ("log", run_dir / "reaction_hub_workflow.log", "workflow_log:reaction_hub_workflow.log"),
        ("database", run_dir / "TDDFT_active_learning_shared" / "scan_endpoints.db", "scan_endpoints_db"),
        (
            "json",
            run_dir / "TDDFT_active_learning_shared" / "reaction_hub_summary.json",
            "reaction_hub_summary",
        ),
        (
            "structure",
            run_dir / "TDDFT_active_learning_shared" / "reaction_hub_final.xyz",
            "structure:reaction_hub_final.xyz",
        ),
        ("csv", run_dir / "TDDFT_active_learning_shared" / "iroot_scan_endpoints.csv", "iroot_scan_endpoints"),
        ("csv", run_dir / "TDDFT_active_learning_shared" / "iroot_geometry_log.csv", "iroot_geometry_log"),
    ]

    for summary_json in run_dir.rglob("reaction_hub_summary.json"):
        candidates.append(("json", summary_json, "reaction_hub_summary"))
        try:
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for raw_path in _summary_db_paths(payload):
                db_path = Path(raw_path)
                if not db_path.is_absolute():
                    db_path = (summary_json.parent / db_path).resolve()
                name = db_path.name
                if name == "scan_endpoints.db":
                    label = "scan_endpoints_db"
                elif name.startswith("coadsorption_"):
                    label = f"coadsorption_db:{name}"
                else:
                    label = f"database:{name}"
                candidates.append(("database", db_path, label))
    for scan_db in run_dir.rglob("scan_endpoints.db"):
        candidates.append(("database", scan_db, "scan_endpoints_db"))
    for db_path in run_dir.rglob("*.db"):
        name = db_path.name
        if name == "scan_endpoints.db":
            continue
        if name.startswith("nwchem"):
            candidates.append(("database", db_path, f"nwchem_db:{name}"))
            continue
        if name.startswith("coadsorption_"):
            candidates.append(("database", db_path, f"coadsorption_db:{name}"))
    for csv_path in run_dir.rglob("coadsorption_*.csv"):
        candidates.append(("csv", csv_path, f"coadsorption_report:{csv_path.name}"))
    for txt_path in run_dir.rglob("coadsorption_*.txt"):
        candidates.append(("other", txt_path, f"coadsorption_report:{txt_path.name}"))

    if any(run_dir.rglob("CT_multiple.txt")):
        ct_names = {
            "IFCTdata.txt",
            "IFCTmajor.txt",
            "config.txt",
            "log.txt",
            "spectrum_line.txt",
            "spectrum_curve.txt",
            "transinfo.txt",
        }
        for txt_path in run_dir.rglob("*.txt"):
            if txt_path.name in ct_names or "CT_multiple" in txt_path.parts:
                candidates.append(("other", txt_path, f"ct_data:{txt_path.name}"))
        for csv_path in run_dir.rglob("CT_*.csv"):
            candidates.append(("csv", csv_path, f"ct_data:{csv_path.name}"))
        for image_path in run_dir.rglob("CT_*.png"):
            candidates.append(("image", image_path, f"plot:{image_path.name}"))
        for molden_path in run_dir.rglob("*.molden.input"):
            candidates.append(("other", molden_path, f"molden:{molden_path.name}"))

    for cube_path in run_dir.rglob("*.cub"):
        candidates.append(("other", cube_path, f"density_cube:{cube_path.name}"))
    for cube_path in run_dir.rglob("*.cube"):
        candidates.append(("other", cube_path, f"density_cube:{cube_path.name}"))
    for pov_path in run_dir.rglob("*.pov"):
        candidates.append(("other", pov_path, f"povray_scene:{pov_path.name}"))
    for ini_path in run_dir.rglob("*.ini"):
        candidates.append(("other", ini_path, f"povray_ini:{ini_path.name}"))

    for out in run_dir.rglob("orca.out"):
        candidates.append(("log", out, f"orca_out:{out.parent.name}"))
    for out in run_dir.rglob("*.nwo"):
        if out.name.startswith("nwchem"):
            candidates.append(("log", out, f"nwchem_out:{out.stem}"))
    for inp in run_dir.rglob("*.nwi"):
        if inp.name.startswith("nwchem"):
            candidates.append(("other", inp, f"nwchem_input:{inp.stem}"))
    for molden in run_dir.rglob("*.molden"):
        if molden.name.startswith("nwchem"):
            candidates.append(("other", molden, f"molden:{molden.name}"))
    for movecs in run_dir.rglob("*.movecs"):
        if movecs.name.startswith("nwchem"):
            candidates.append(("other", movecs, f"nwchem_movecs:{movecs.name}"))
    for extxyz_name in ("nwchem_final.extxyz", "input_structure_final.extxyz"):
        for extxyz in run_dir.rglob(extxyz_name):
            candidates.append(("structure", extxyz, f"structure:{extxyz.name}"))
    for xyz in run_dir.rglob("*.xyz"):
        candidates.append(("structure", xyz, f"structure:{xyz.name}"))
    for traj in run_dir.rglob("*.traj"):
        candidates.append(("trajectory", traj, f"trajectory:{traj.name}"))
    for render_name in ("test.svg", "orca.gif"):
        for render_path in run_dir.rglob(render_name):
            candidates.append(("image", render_path, f"render:{render_path.name}"))

    plot_dirs = (
        run_dir / "TDDFT" / "scan_plots",
        run_dir / "DFT" / "scan_plots",
    )
    key_plot_labels = {
        "stacked_uvvis.png": "plot:stacked_uvvis",
        "stacked_ct.png": "plot:stacked_ct",
        "energy_vs_reaction_coordinate.png": "plot:energy_vs_reaction_coordinate",
        "final_tddft_spectrum.png": "plot:final_tddft_spectrum",
        "cluster_tddft_spectrum.png": "plot:cluster_tddft_spectrum",
        "cluster_adsorbate_difference_spectrum.png": "plot:cluster_adsorbate_difference_spectrum",
    }
    final_spectrum_dir = run_dir / "DFT" / "final_tddft_spectrum"
    if final_spectrum_dir.exists():
        for image_path in final_spectrum_dir.glob("*.png"):
            candidates.append(("image", image_path, key_plot_labels.get(image_path.name, f"plot:{image_path.name}")))
        for csv_path in final_spectrum_dir.glob("*.csv"):
            candidates.append(("csv", csv_path, f"spectrum:{csv_path.name}"))
    for plot_dir in plot_dirs:
        if not plot_dir.exists():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
            for image_path in plot_dir.glob(ext):
                if _skip_scan_plot_image(image_path):
                    continue
                label = key_plot_labels.get(image_path.name, f"plot:{image_path.name}")
                candidates.append(("image", image_path, label))

    existing = []
    seen = set()
    for kind, path, label in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if _is_shadowed_legacy_archive_artifact(run_dir, resolved):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        existing.append((kind, resolved, label))
    return existing
