from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Optional

from ase.constraints import FixAtoms, FixBondLength

from .ct import load_scan_log
from .io import bond_length_from_orca, energy_from_orca


DEFAULT_BOND_SCAN_INCREMENT = 0.05
DEFAULT_BOND_SCAN_FIX = 0.5
DEFAULT_FIXED_SYMBOLS = ("Cu",)
DEFAULT_BOND_SYMBOL_PAIR = ("O", "O")


@dataclass
class BondScanSettings:
    enabled: bool = True
    increment: float = DEFAULT_BOND_SCAN_INCREMENT
    fix: float = DEFAULT_BOND_SCAN_FIX
    fixed_symbols: tuple[str, ...] = DEFAULT_FIXED_SYMBOLS
    bond_pair_indices: tuple[int, int] | None = None
    bond_symbol_pair: tuple[str, str] = DEFAULT_BOND_SYMBOL_PAIR
    target_distance: float | None = None
    distance_change_fraction: float = 1.0
    enable_ts_smoothing: bool = False
    refine_points_each_side: int = 0
    restart_extend_steps: int = 0
    restart_extend_increment_A: float | None = None


@dataclass
class UVVisPlotSettings:
    units: str = "nm"
    linewidth_nm: float = 20.0
    linewidth_wn: float = 1000.0
    step: float = 1.0
    startx: float | None = None
    endx: float | None = None
    normalize: bool = False
    normalize_factor: float = 1.0
    cache_name: str = "uvvis_spectrum.npz"
    use_cache: bool = True
    allow_multiwfn_fallback: bool = False
    use_multiwfn_plotting: bool = True
    multiwfn_relative_intensity_threshold: float = 0.1
    multiwfn_generate_ct_plot: bool = True
    multiwfn_ct_nroots: int | None = None
    multiwfn_ct_min_peak_intensity_fraction: float = 1.0
    multiwfn_ct_min_peak_distance_points: int = 1
    multiwfn_ct_min_peak_width_points: int = 1
    multiwfn_ct_min_fwhm_abs: float = 15.0
    multiwfn_ct_max_gaussians_per_spectrum: int = 20
    multiwfn_ct_plot_individual_components: bool = False
    analysis_min_wavelength_nm: float | None = 300.0
    analysis_max_wavelength_nm: float | None = 800.0
    generate_stacked_uvvis: bool = True
    generate_stacked_ct: bool = True
    stacked_axis_label: str = "Bond length / Angstrom"
    stacked_uvvis_offset_scale: float = 0.1
    stacked_uvvis_linewidth_nm: float = 20.0
    stacked_uvvis_normalize: bool = False
    stacked_uvvis_xmin: float | None = 300.0
    stacked_uvvis_xmax: float | None = 800.0
    stacked_ct_offset_scale: float = 0.1
    stacked_ct_normalize: bool = True
    stacked_ct_xmin: float | None = None
    stacked_ct_xmax: float | None = None


@dataclass
class ScanSummaryPlotter:
    """Plotter hook for scan summaries; override read_uvvis_spectrum for custom sources."""
    uvvis_settings: UVVisPlotSettings = field(default_factory=UVVisPlotSettings)

    def read_uvvis_spectrum(self, scan_dir: Path):
        spectrum = _read_uvvis_spectrum_orca(scan_dir, self.uvvis_settings)
        if spectrum is None and self.uvvis_settings.allow_multiwfn_fallback:
            spectrum = _read_uvvis_spectrum_multiwfn(scan_dir)
        return spectrum


def _format_stack_colorbar_ticks(ticks, axis_label: str) -> list[str]:
    label = str(axis_label or "").strip().lower()
    is_bond_length_axis = ("bond" in label and "length" in label) and ("angstrom" in label or "angstroem" in label)
    if is_bond_length_axis:
        return [f"{float(tick):.2f}" for tick in ticks]
    return [f"{float(tick):g}" for tick in ticks]


def resolve_bond_indices(atoms, settings: BondScanSettings) -> tuple[int, int]:
    if settings.bond_pair_indices is not None:
        return settings.bond_pair_indices

    sym_a, sym_b = settings.bond_symbol_pair
    indices_a = [atom.index for atom in atoms if atom.symbol == sym_a]
    if sym_a == sym_b:
        if len(indices_a) < 2:
            raise ValueError(f"Unable to locate two atoms with symbol '{sym_a}' for bond scan.")
        return indices_a[0], indices_a[1]

    indices_b = [atom.index for atom in atoms if atom.symbol == sym_b]
    if not indices_a or not indices_b:
        raise ValueError(
            f"Unable to locate required atoms for bond scan: {sym_a} count={len(indices_a)}, {sym_b} count={len(indices_b)}."
        )
    return indices_a[0], indices_b[0]


def apply_bond_constraints(atoms, fixed_symbols: tuple[str, ...], bond_pair: tuple[int, int]):
    constraints = []
    if fixed_symbols:
        fixed_indices = [atom.index for atom in atoms if atom.symbol in fixed_symbols]
        if fixed_indices:
            constraints.append(FixAtoms(indices=fixed_indices))
    constraints.append(FixBondLength(*bond_pair))
    atoms.set_constraint(constraints)


def resolve_bond_increment(
    atoms,
    bond_pair: tuple[int, int],
    settings: BondScanSettings,
    remaining_iterations: int,
) -> float:
    if settings.target_distance is None:
        return settings.increment
    if remaining_iterations <= 0:
        return 0.0
    if settings.distance_change_fraction <= 0.0:
        raise ValueError("distance_change_fraction must be positive when target_distance is set.")
    current = atoms.get_distance(*bond_pair)
    delta = settings.target_distance - current
    steps = max(remaining_iterations - 1, 1)
    return delta * settings.distance_change_fraction / steps


def increment_bond_length(
    atoms,
    bond_pair: tuple[int, int],
    increment: float,
    fix: float,
    target_distance: float | None = None,
):
    atoms.set_constraint(None)
    current = atoms.get_distance(*bond_pair)
    new_distance = current + increment
    if target_distance is not None:
        if increment >= 0.0 and new_distance > target_distance:
            new_distance = target_distance
        elif increment < 0.0 and new_distance < target_distance:
            new_distance = target_distance
    atoms.set_distance(bond_pair[0], bond_pair[1], new_distance, fix=fix)


def _is_primary_scan_log_row(row: dict[str, object]) -> bool:
    point_kind = str(row.get("point_kind") or "").strip()
    if not point_kind:
        return True
    # No-overlap fallback rows are still primary scan iterations and should
    # remain visible in scan summaries (including converged-state wavelength
    # plots), even though they carry a non-empty point_kind marker.
    return _is_no_overlap_fallback_marker(point_kind, row.get("point_label"))


def _is_postprocess_scan_log_row_for_window(
    row: dict[str, object],
    *,
    max_iter: int,
) -> bool:
    point_kind = str(row.get("point_kind") or "").strip()
    if not point_kind:
        return False
    source_iteration = _parse_optional_int(row.get("source_iteration"))
    if source_iteration is None:
        return False
    return 1 <= int(source_iteration) <= int(max_iter)


def _is_no_overlap_fallback_marker(point_kind: object, point_label: object) -> bool:
    kind = str(point_kind or "").strip().lower()
    label = str(point_label or "").strip().lower()
    if kind == "fallback_no_orbital_overlap":
        return True
    return "no_orbital_overlap" in label or "no-overlap" in label


def _collect_no_overlap_fallback_labels(
    entries: list[dict[str, object]],
) -> list[str]:
    labels: list[str] = []
    seen: set[tuple[object, object, str]] = set()
    for entry in entries:
        if not _is_no_overlap_fallback_marker(entry.get("point_kind"), entry.get("point_label")):
            continue
        iteration = _parse_optional_int(entry.get("iteration"))
        bond_length = _parse_optional_float(entry.get("bond_length"))
        scan_dir = str(entry.get("scan_dir") or "").strip()
        key = (iteration, bond_length, scan_dir)
        if key in seen:
            continue
        seen.add(key)
        if iteration is not None and bond_length is not None:
            labels.append(f"iter {iteration} ({bond_length:.3f} A)")
        elif iteration is not None:
            labels.append(f"iter {iteration}")
        elif bond_length is not None:
            labels.append(f"{bond_length:.3f} A")
        elif scan_dir:
            labels.append(Path(scan_dir).name)
        else:
            labels.append("unknown point")
    return labels


def _style_no_overlap_fallback_stacked_traces(ax, flags: list[bool]) -> None:
    if not any(flags):
        return
    lines = list(ax.get_lines())
    for idx, flagged in enumerate(flags):
        if not flagged or idx >= len(lines):
            continue
        line = lines[idx]
        line.set_linestyle("--")
        line.set_linewidth(max(float(line.get_linewidth()), 1.8))
        line.set_zorder(max(float(line.get_zorder()), 3.0))
    try:
        from matplotlib.lines import Line2D
    except Exception:
        return
    handles, labels = ax.get_legend_handles_labels()
    fallback_label = "No-overlap fallback trace"
    if fallback_label in labels:
        return
    handles.append(
        Line2D(
            [0],
            [0],
            color="#dc2626",
            linestyle="--",
            linewidth=1.8,
            label=fallback_label,
        )
    )
    labels.append(fallback_label)
    ax.legend(handles=handles, labels=labels, loc="upper right")


def _scan_dir_mtime(path: Path) -> float:
    try:
        marker = path / "orca.out"
        if marker.exists():
            return float(marker.stat().st_mtime)
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _resolve_scan_dir_for_iteration(
    iteration: int,
    *,
    preferred_scan_dir: Path | None = None,
) -> Path | None:
    candidates_with_out: list[Path] = []
    candidates_existing: list[Path] = []

    def _collect_candidate(path: Path) -> None:
        if not path.exists():
            return
        if (path / "orca.out").exists():
            candidates_with_out.append(path)
        else:
            candidates_existing.append(path)

    if preferred_scan_dir is not None:
        _collect_candidate(preferred_scan_dir)

    modern_base_candidates = [
        Path("TDDFT") / f"scan_iter_{iteration}",
        Path("DFT") / f"scan_iter_{iteration}",
    ]
    legacy_base_candidates = [
        Path(f"TDDFT_scan_iter_{iteration}"),
        Path(f"DFT_scan_iter_{iteration}"),
    ]
    for candidate in [*modern_base_candidates, *legacy_base_candidates]:
        _collect_candidate(candidate)

    for pattern in (
        f"TDDFT/scan_iter_{iteration}_geom_*",
        f"DFT/scan_iter_{iteration}_geom_*",
        f"TDDFT_scan_iter_{iteration}_geom_*",
        f"DFT_scan_iter_{iteration}_geom_*",
    ):
        for candidate in Path(".").glob(pattern):
            _collect_candidate(candidate)

    if candidates_with_out:
        return max(candidates_with_out, key=lambda item: (_scan_dir_mtime(item), str(item)))
    if candidates_existing:
        return max(candidates_existing, key=lambda item: (_scan_dir_mtime(item), str(item)))
    return None


def _collect_scan_endpoints(
    *,
    max_iter: int,
    scan_log_path: Path,
    bond_pair: tuple[int, int] | None,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    log_rows = load_scan_log(scan_log_path)
    if log_rows:
        for iteration in sorted(log_rows, key=lambda k: int(k) if str(k).isdigit() else 10**9):
            try:
                iter_num = int(iteration)
            except ValueError:
                continue
            row = log_rows[iteration]
            point_kind = str(row.get("point_kind") or "").strip()
            point_label = str(row.get("point_label") or "").strip()
            source_iteration = _parse_optional_int(row.get("source_iteration"))
            if _is_primary_scan_log_row(row):
                if iter_num > max_iter:
                    continue
            elif not _is_postprocess_scan_log_row_for_window(row, max_iter=max_iter):
                continue
            scan_dir_text = str(row.get("scan_dir") or "").strip()
            preferred_scan_dir = Path(scan_dir_text) if scan_dir_text else None
            scan_dir = _resolve_scan_dir_for_iteration(
                iter_num,
                preferred_scan_dir=preferred_scan_dir,
            )
            if scan_dir is None:
                continue
            bond_length = _parse_optional_float(row.get("bond_length"))
            if bond_length is None and bond_pair is not None:
                bond_length = bond_length_from_orca(scan_dir, bond_pair)
            replacement_iteration = (
                int(source_iteration)
                if point_kind == "ts_smoothing" and source_iteration is not None
                else int(iter_num)
            )
            entry = {
                "iteration": replacement_iteration,
                "scan_dir": scan_dir,
                "bond_length": bond_length,
                "point_kind": point_kind,
                "point_label": point_label,
            }
            if point_kind == "ts_smoothing" and source_iteration is not None:
                replaced = False
                for idx, existing in enumerate(entries):
                    try:
                        existing_iteration = int(existing.get("iteration", -1))
                    except Exception:
                        continue
                    if existing_iteration == int(source_iteration):
                        entries[idx] = entry
                        replaced = True
                        break
                if not replaced:
                    entries.append(entry)
                continue
            entries.append(entry)

        present_primary_iterations = {
            int(entry.get("iteration", -1))
            for entry in entries
            if 1 <= int(entry.get("iteration", -1)) <= int(max_iter)
        }
        for iter_num in range(1, int(max_iter) + 1):
            if int(iter_num) in present_primary_iterations:
                continue
            scan_dir = _resolve_scan_dir_for_iteration(iter_num, preferred_scan_dir=None)
            if scan_dir is None:
                continue
            bond_length = None
            if bond_pair is not None:
                bond_length = bond_length_from_orca(scan_dir, bond_pair)
            entries.append(
                {
                    "iteration": int(iter_num),
                    "scan_dir": scan_dir,
                    "bond_length": bond_length,
                    "point_kind": "",
                    "point_label": "",
                }
            )
        entries.sort(
            key=lambda item: int(item.get("iteration", 10**9))
            if str(item.get("iteration", "")).strip().lstrip("-").isdigit()
            else 10**9
        )
        return entries

    for i in range(0, max_iter + 1):
        scan_dir = _resolve_scan_dir_for_iteration(i, preferred_scan_dir=None)
        if scan_dir is None:
            continue
        bond_length = None
        if bond_pair is not None:
            bond_length = bond_length_from_orca(scan_dir, bond_pair)
        entries.append(
            {
                "iteration": i,
                "scan_dir": scan_dir,
                "bond_length": bond_length,
                "point_kind": "",
                "point_label": "",
            }
        )
    return entries


def _read_uvvis_spectrum_multiwfn(scan_dir: Path):
    curve_path = scan_dir / "spectrum_curve.txt"
    line_path = scan_dir / "spectrum_line.txt"
    if not curve_path.exists() and not line_path.exists():
        return None
    source = curve_path if curve_path.exists() else line_path
    try:
        xs: list[float] = []
        ys: list[float] = []
        with open(source, "r", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except ValueError:
                    continue
        if not xs:
            return None
        order = sorted(range(len(xs)), key=lambda idx: xs[idx])
        xs_sorted = [xs[idx] for idx in order]
        ys_sorted = [ys[idx] for idx in order]
        return xs_sorted, ys_sorted
    except Exception as exc:
        print(f"Warning: Failed to read UV-vis spectrum from {source}: {exc}")
        return None


def _format_uvvis_xlabel(units: str) -> str:
    key = units.strip().lower().replace("_", "").replace(" ", "")
    if key in {"nm", "nanometer", "nanometers", "wavelength"}:
        return "Wavelength (nm)"
    return "Energy (cm^-1)"


def _resolve_analysis_wavelength_limits(settings: UVVisPlotSettings) -> tuple[float | None, float | None]:
    min_nm = settings.analysis_min_wavelength_nm
    max_nm = settings.analysis_max_wavelength_nm
    if min_nm is None:
        min_nm = settings.stacked_uvvis_xmin
    if max_nm is None:
        max_nm = settings.stacked_uvvis_xmax
    return min_nm, max_nm


def _resolve_stacked_ct_xlim(settings: UVVisPlotSettings) -> tuple[float | None, float | None]:
    xmin = settings.stacked_ct_xmin
    xmax = settings.stacked_ct_xmax
    if xmin is None:
        xmin = settings.stacked_uvvis_xmin
    if xmax is None:
        xmax = settings.stacked_uvvis_xmax
    return xmin, xmax


def _smooth_profile_fwhm(profile, *, axis_step: float, fwhm_nm: float):
    try:
        import numpy as np
    except Exception:
        return profile

    arr = np.asarray(profile, dtype=float)
    if arr.size < 3:
        return arr
    try:
        step = float(axis_step)
        fwhm = float(fwhm_nm)
    except (TypeError, ValueError):
        return arr
    if step <= 0.0 or fwhm <= 0.0:
        return arr

    sigma_nm = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_pts = sigma_nm / step
    if sigma_pts < 0.25:
        return arr

    radius = int(np.ceil(4.0 * sigma_pts))
    radius = max(radius, 1)
    kernel_x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / sigma_pts) ** 2)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        return arr
    kernel /= kernel_sum
    return np.convolve(arr, kernel, mode="same")


def _uvvis_cache_matches(
    meta: dict[str, object],
    settings: UVVisPlotSettings,
    orca_out: Path | None,
) -> bool:
    try:
        import numpy as np
    except Exception:
        return False

    if int(meta.get("version", 0)) != 1:
        return False
    if str(meta.get("units", "")).lower() != settings.units.lower():
        return False

    def _matches_optional_float(key: str, value: float | None) -> bool:
        cached = meta.get(key)
        if value is None and cached is None:
            return True
        if value is None or cached is None:
            return False
        return bool(np.isclose(float(cached), float(value)))

    if not _matches_optional_float("startx", settings.startx):
        return False
    if not _matches_optional_float("endx", settings.endx):
        return False
    try:
        if not np.isclose(float(meta.get("linewidth_nm", -1.0)), float(settings.linewidth_nm)):
            return False
        if not np.isclose(float(meta.get("linewidth_wn", -1.0)), float(settings.linewidth_wn)):
            return False
        if not np.isclose(float(meta.get("step", -1.0)), float(settings.step)):
            return False
        if bool(meta.get("normalize", False)) != bool(settings.normalize):
            return False
        if not np.isclose(
            float(meta.get("normalize_factor", -1.0)), float(settings.normalize_factor)
        ):
            return False
    except (TypeError, ValueError):
        return False

    if orca_out is not None and orca_out.exists():
        cached_mtime = meta.get("orca_out_mtime")
        if cached_mtime is None:
            return False
        try:
            if float(cached_mtime) + 1.0e-6 < orca_out.stat().st_mtime:
                return False
        except OSError:
            return False

    return True


def _load_uvvis_cache(
    cache_path: Path,
    settings: UVVisPlotSettings,
    orca_out: Path | None,
):
    try:
        import numpy as np
    except Exception:
        return None

    try:
        data = np.load(cache_path, allow_pickle=False)
    except Exception:
        return None

    if "x" not in data or "y" not in data or "meta" not in data:
        return None
    meta_raw = data["meta"]
    if isinstance(meta_raw, np.ndarray):
        meta_text = meta_raw.item()
    else:
        meta_text = meta_raw
    if isinstance(meta_text, (bytes, bytearray)):
        meta_text = meta_text.decode("utf-8", "replace")
    try:
        meta = json.loads(str(meta_text))
    except Exception:
        return None

    if not _uvvis_cache_matches(meta, settings, orca_out):
        return None

    return data["x"], data["y"]


def _save_uvvis_cache(
    cache_path: Path,
    spectrum_x,
    spectrum_y,
    settings: UVVisPlotSettings,
    orca_out: Path | None,
) -> None:
    try:
        import numpy as np
    except Exception:
        return

    try:
        meta = {
            "version": 1,
            "units": settings.units,
            "linewidth_nm": settings.linewidth_nm,
            "linewidth_wn": settings.linewidth_wn,
            "step": settings.step,
            "startx": settings.startx,
            "endx": settings.endx,
            "normalize": settings.normalize,
            "normalize_factor": settings.normalize_factor,
            "orca_out_mtime": orca_out.stat().st_mtime if orca_out and orca_out.exists() else None,
        }
        np.savez(cache_path, x=spectrum_x, y=spectrum_y, meta=json.dumps(meta))
    except Exception as exc:
        print(f"Warning: Failed to write UV-vis cache {cache_path}: {exc}")


def _read_uvvis_spectrum_orca(scan_dir: Path, settings: UVVisPlotSettings):
    orca_out = scan_dir / "orca.out"
    if not orca_out.exists():
        return None

    cache_path: Optional[Path] = None
    if settings.use_cache and settings.cache_name:
        cache_path = scan_dir / settings.cache_name
        if cache_path.exists():
            cached = _load_uvvis_cache(cache_path, settings, orca_out)
            if cached is not None:
                return cached

    try:
        from cats.visualise.uvvis import extract_uv_spectrum
    except Exception as exc:
        print(f"Warning: Failed to import UV-vis extractor: {exc}")
        return None

    try:
        spectrum = extract_uv_spectrum(
            str(orca_out),
            units=settings.units,
            linewidth_nm=settings.linewidth_nm,
            linewidth_wn=settings.linewidth_wn,
            startx=settings.startx,
            endx=settings.endx,
            step=settings.step,
            normalize=settings.normalize,
            normalize_factor=settings.normalize_factor,
        )
    except Exception as exc:
        print(f"Warning: Failed to extract UV-vis spectrum from {orca_out}: {exc}")
        return None

    if cache_path is not None:
        _save_uvvis_cache(cache_path, spectrum.x, spectrum.y, settings, orca_out)

    return spectrum.x, spectrum.y


def _ensure_ct_spectrum(scan_dir: Path) -> bool:
    ct_index = scan_dir / "CT_multiple" / "CT_multiple.txt"
    if ct_index.exists():
        return True
    try:
        from cats.multiwfn.tools import get_ct_info as get_ct_info_tool
    except Exception as exc:
        print(f"Warning: Failed to import CT spectrum tools: {exc}")
        return False

    parent_dir = Path.cwd()
    os.chdir(scan_dir)
    try:
        get_ct_info_tool()
    except Exception as exc:
        print(f"Warning: Failed to generate CT spectrum data in {scan_dir}: {exc}")
    finally:
        os.chdir(parent_dir)
    return ct_index.exists()


def _read_ct_transfer_spectrum_fitted(
    scan_dir: Path,
    settings: UVVisPlotSettings,
):
    if not _ensure_ct_spectrum(scan_dir):
        return None
    try:
        import numpy as np
        from matplotlib import pyplot as plt
        from cats.multiwfn.plot_uvvis import plot_charge_transfer_spectrum
    except Exception as exc:
        print(f"Warning: Failed to import fitted CT spectrum dependencies: {exc}")
        return None

    # Align stacked CT fitting controls with stacked UV-vis controls.
    min_nm, max_nm = _resolve_stacked_ct_xlim(settings)
    if min_nm is None and max_nm is None:
        min_nm, max_nm = _resolve_analysis_wavelength_limits(settings)
    fit_fwhm = float(settings.stacked_uvvis_linewidth_nm)
    if fit_fwhm <= 0.0:
        fit_fwhm = float(settings.multiwfn_ct_min_fwhm_abs)
    fig = None
    try:
        ct_plot, _ = plot_charge_transfer_spectrum(
            str(scan_dir),
            use_eV=False,
            min_wavelength_nm=min_nm,
            max_wavelength_nm=max_nm,
            min_peak_intensity_fraction=float(settings.multiwfn_ct_min_peak_intensity_fraction),
            min_peak_distance_points=int(settings.multiwfn_ct_min_peak_distance_points),
            min_peak_width_points=int(settings.multiwfn_ct_min_peak_width_points),
            min_fwhm_abs=fit_fwhm,
            max_gaussians_per_spectrum=int(settings.multiwfn_ct_max_gaussians_per_spectrum),
            verbose=False,
            plot_individual_components=False,
            nroots=settings.multiwfn_ct_nroots,
        )
        ax = ct_plot.gca()
        fig = ax.figure
        axis_ref: Optional[np.ndarray] = None
        transfer_sum: Optional[np.ndarray] = None
        transfer_count = 0
        for line in ax.get_lines():
            label = str(line.get_label() or "").lower()
            # Keep transfer traces (including fallback-fit traces) for robust stacking.
            if "transfer" not in label:
                continue
            x_data = np.asarray(line.get_xdata(), dtype=float)
            y_data = np.asarray(line.get_ydata(), dtype=float)
            if x_data.size == 0 or y_data.size == 0:
                continue
            order = np.argsort(x_data)
            x_data = x_data[order]
            y_data = y_data[order]
            if axis_ref is None:
                axis_ref = x_data
                transfer_sum = np.zeros_like(axis_ref)
            else:
                y_data = np.interp(axis_ref, x_data, y_data)
            transfer_sum += y_data
            transfer_count += 1

        if axis_ref is None or transfer_sum is None or transfer_count == 0:
            return None
        return axis_ref, transfer_sum
    except Exception:
        return None
    finally:
        try:
            if fig is not None:
                plt.close(fig)
            else:
                plt.close("all")
        except Exception:
            pass


def _read_ct_transfer_spectrum(
    scan_dir: Path,
    settings: UVVisPlotSettings | None = None,
):
    if settings is not None:
        fitted_spectrum = _read_ct_transfer_spectrum_fitted(scan_dir, settings)
        if fitted_spectrum is not None:
            return fitted_spectrum

    if not _ensure_ct_spectrum(scan_dir):
        return None
    ct_index = scan_dir / "CT_multiple" / "CT_multiple.txt"
    if not ct_index.exists():
        print(f"Warning: Missing CT spectrum index in {scan_dir}")
        return None
    try:
        import pandas as pd
        import numpy as np
    except Exception as exc:
        print(f"Warning: Failed to import CT spectrum dependencies: {exc}")
        return None

    try:
        files = pd.read_csv(ct_index, header=None, index_col=None)
    except Exception as exc:
        print(f"Warning: Failed to read {ct_index}: {exc}")
        return None
    if files.empty:
        return None
    files["files"] = files[0].apply(lambda x: str(x).split(" ")[0])
    files["property"] = files[0].apply(lambda x: " ".join(str(x).split(" ")[1:]))

    energy_ref: Optional[np.ndarray] = None
    transfer_sum: Optional[np.ndarray] = None
    transfer_count = 0
    fallback_sum: Optional[np.ndarray] = None

    for _, row in files.iterrows():
        file_path = scan_dir / row["files"]
        if not file_path.exists():
            continue
        try:
            temp_df = pd.read_csv(file_path, header=0, index_col=None)
        except Exception:
            continue
        if temp_df.empty:
            continue

        column_name = None
        for col in temp_df.columns:
            series = temp_df[col].dropna()
            if series.empty:
                continue
            sample = str(series.iloc[0]).strip()
            parts = sample.split()
            if len(parts) < 2:
                continue
            try:
                float(parts[0])
                float(parts[1])
            except ValueError:
                continue
            column_name = col
            break
        if column_name is None:
            continue

        energies: list[float] = []
        intensities: list[float] = []
        for value in temp_df[column_name].values:
            parts = str(value).strip().split()
            if len(parts) < 2:
                continue
            try:
                energies.append(float(parts[0]))
                intensities.append(float(parts[1]))
            except ValueError:
                continue
        if not energies:
            continue

        energy_arr = np.array(energies, dtype=float)
        intensity_arr = np.array(intensities, dtype=float)

        if energy_ref is None:
            energy_ref = energy_arr
        elif energy_arr.shape != energy_ref.shape or not np.allclose(energy_arr, energy_ref, rtol=1e-4, atol=1e-6):
            intensity_arr = np.interp(energy_ref, energy_arr, intensity_arr)

        if fallback_sum is None:
            fallback_sum = np.zeros_like(energy_ref)
        fallback_sum += intensity_arr

        prop = str(row.get("property", "")).lower()
        if "transfer" in prop:
            if transfer_sum is None:
                transfer_sum = np.zeros_like(energy_ref)
            transfer_sum += intensity_arr
            transfer_count += 1

    if energy_ref is None:
        return None
    spectrum = transfer_sum if transfer_count > 0 else fallback_sum
    if spectrum is None:
        return None

    median_energy = float(np.median(energy_ref))
    if median_energy < 50.0:
        wavelength = 1239.84193 / energy_ref
    else:
        wavelength = energy_ref
    order = np.argsort(wavelength)
    return wavelength[order], spectrum[order]


def _scan_db_row_value(row, *keys: str):
    row_data = getattr(row, "data", {}) or {}
    for key in keys:
        try:
            if key in row:
                return row[key]
        except Exception:
            pass
        if key in row_data:
            return row_data[key]
    return None


def _energy_profile_entries_from_scan_db(
    scan_endpoints_db_path: Path | None,
    *,
    bond_pair: tuple[int, int] | None,
) -> list[dict[str, object]]:
    if scan_endpoints_db_path is None or not scan_endpoints_db_path.exists():
        return []

    try:
        from ase.db import connect
    except Exception as exc:
        print(f"Warning: Failed to import ASE DB reader for scan endpoint energies: {exc}")
        return []

    profile_entries: list[dict[str, object]] = []
    try:
        db = connect(str(scan_endpoints_db_path))
        rows = list(db.select())
    except Exception as exc:
        print(f"Warning: Failed to read scan endpoint energies from {scan_endpoints_db_path}: {exc}")
        return []

    for row in rows:
        energy = _scan_db_row_value(
            row,
            "total_energy_eV",
            "scan_energy_ev",
            "energy_eV",
            "energy",
        )
        try:
            energy_eV = float(energy)
        except (TypeError, ValueError):
            continue

        bond_length = _scan_db_row_value(row, "scan_bond_length", "bond_length")
        if bond_pair is not None:
            try:
                atoms = row.toatoms()
                bond_length = float(atoms.get_distance(*bond_pair))
            except Exception:
                pass
        try:
            bond_length_A = float(bond_length)
        except (TypeError, ValueError):
            continue

        iteration = _scan_db_row_value(row, "iteration", "scan_iteration")
        max_force = _parse_optional_float(_scan_db_row_value(row, "max_force_eV_A"))
        profile_entries.append(
            {
                "iteration": iteration,
                "bond_length": bond_length_A,
                "energy_eV": energy_eV,
                "max_force_eV_A": max_force,
                "point_kind": _scan_db_row_value(row, "scan_point_kind", "point_kind"),
                "point_label": _scan_db_row_value(row, "scan_point_label", "point_label"),
            }
        )
    return profile_entries


def _plot_energy_vs_reaction_coordinate(
    entries: list[dict[str, object]],
    output_dir: Path,
    *,
    bond_pair: tuple[int, int] | None = None,
    scan_endpoints_db_path: Path | None = None,
) -> None:
    try:
        import numpy as np
        from matplotlib import pyplot as plt
    except Exception as exc:
        print(f"Warning: Plotting dependencies unavailable: {exc}")
        return

    profile_entries = _energy_profile_entries_from_scan_db(
        scan_endpoints_db_path,
        bond_pair=bond_pair,
    )
    selected_entries = _select_latest_entries_by_bond_length(profile_entries or entries)
    if not selected_entries:
        print("Warning: No valid bond-length endpoints available for reaction-coordinate plot.")
        return
    source_count = len(profile_entries or entries)
    if len(selected_entries) < source_count:
        print(
            "Info: Collapsed near-duplicate bond-length endpoints for reaction-coordinate "
            f"plot ({source_count} -> {len(selected_entries)})."
        )

    energy_lengths: list[float] = []
    energies: list[float] = []
    for entry in selected_entries:
        bond_length = entry.get("bond_length")
        if bond_length is None:
            continue
        energy = entry.get("energy_eV")
        if energy is None:
            scan_dir = entry["scan_dir"]
            energy = energy_from_orca(scan_dir)
        if energy is None:
            continue
        energy_lengths.append(float(bond_length))
        energies.append(float(energy))

    if not energies or not energy_lengths:
        print("Warning: No energies available for reaction-coordinate plot.")
        return

    order = np.argsort(energy_lengths)
    energy_lengths = [energy_lengths[idx] for idx in order]
    energies = [energies[idx] for idx in order]
    min_energy = min(energies)
    rel_energies = [energy - min_energy for energy in energies]
    fig, ax = plt.subplots()
    ax.plot(energy_lengths, rel_energies, marker="o", linestyle="-")
    ax.set_xlabel("Bond length (Angstrom)")
    ax.set_ylabel("Relative energy (eV)")
    ax.set_title("Energy over reaction coordinate")
    fig.tight_layout()
    fig.savefig(output_dir / "energy_vs_reaction_coordinate.png", dpi=300)
    plt.close(fig)


def _collect_converged_state_wavelength_points(
    *,
    max_iter: int,
    scan_log_path: Path,
) -> list[dict[str, object]]:
    rows = load_scan_log(scan_log_path)
    if not rows:
        return []

    points: list[dict[str, object]] = []
    for iteration in sorted(rows, key=lambda k: int(k) if str(k).isdigit() else 10**9):
        try:
            iter_num = int(iteration)
        except ValueError:
            continue

        row = rows[iteration]
        point_kind = str(row.get("point_kind") or "").strip()
        source_iteration = _parse_optional_int(row.get("source_iteration"))
        if _is_primary_scan_log_row(row):
            if iter_num > max_iter:
                continue
        elif not _is_postprocess_scan_log_row_for_window(row, max_iter=max_iter):
            continue
        wavelength_nm = _parse_optional_float(row.get("wavelength_nm"))
        if wavelength_nm is None:
            continue
        point = {
            "iteration": (
                int(source_iteration)
                if point_kind == "ts_smoothing" and source_iteration is not None
                else int(iter_num)
            ),
            "wavelength_nm": float(wavelength_nm),
            "iroot": _parse_optional_int(row.get("iroot")),
            "ct_net": _parse_optional_float(row.get("ct_net")),
            "score": _parse_optional_float(row.get("score")),
            "point_kind": point_kind,
            "point_label": str(row.get("point_label") or "").strip(),
        }
        if point_kind == "ts_smoothing" and source_iteration is not None:
            replaced = False
            for idx, existing in enumerate(points):
                try:
                    existing_iteration = int(existing.get("iteration", -1))
                except Exception:
                    continue
                if existing_iteration == int(source_iteration):
                    points[idx] = point
                    replaced = True
                    break
            if not replaced:
                points.append(point)
            continue
        points.append(point)

    if not points:
        return []

    positive = [point for point in points if int(point.get("iteration", 0)) > 0]
    if positive:
        points = positive

    points.sort(key=lambda point: int(point.get("iteration", 0)))
    for image_number, point in enumerate(points, start=1):
        point["image_number"] = image_number
    return points


def _plot_converged_state_wavelengths(
    *,
    max_iter: int,
    scan_log_path: Path,
    output_dir: Path,
) -> None:
    try:
        import math
        from matplotlib import colors as mcolors
        from matplotlib import pyplot as plt
    except Exception as exc:
        print(f"Warning: Plotting dependencies unavailable: {exc}")
        return

    points = _collect_converged_state_wavelength_points(
        max_iter=max_iter,
        scan_log_path=scan_log_path,
    )
    if not points:
        print("Warning: No converged-state wavelength data available for endpoint-state plot.")
        return

    x_values = [int(point["image_number"]) for point in points]
    y_values = [float(point["wavelength_nm"]) for point in points]
    iteration_labels = [str(int(point["iteration"])) for point in points]
    no_overlap_fallback_flags = [
        _is_no_overlap_fallback_marker(point.get("point_kind"), point.get("point_label"))
        for point in points
    ]
    ct_values_raw = [point.get("ct_net") for point in points]
    ct_values: list[float | None] = []
    for value in ct_values_raw:
        if value is None:
            ct_values.append(None)
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            ct_values.append(None)
            continue
        if not math.isfinite(parsed):
            ct_values.append(None)
            continue
        ct_values.append(parsed)
    finite_ct = [value for value in ct_values if value is not None]
    ct_color_kwargs: dict[str, object] | None = None

    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, color="#64748b", linestyle="-", linewidth=1.25, zorder=1)
    ax.set_xlabel("Image number (ticks = iteration)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Converged excited-state wavelength by image (CT-colored)")
    ax.grid(alpha=0.25)

    if len(x_values) <= 20:
        xticks = x_values
    else:
        step = max(1, len(x_values) // 12)
        xticks = [x_values[idx] for idx in range(0, len(x_values), step)]
        if xticks[-1] != x_values[-1]:
            xticks.append(x_values[-1])
    xticklabels = [iteration_labels[tick - 1] for tick in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    if finite_ct:
        cmin = min(float(value) for value in finite_ct)
        cmax = max(float(value) for value in finite_ct)
        if cmin == cmax:
            span = max(abs(cmin), 1.0)
            cmin = -span
            cmax = span
        if cmin >= 0.0:
            cmin = -max(cmax, 1.0e-9)
        if cmax <= 0.0:
            cmax = max(abs(cmin), 1.0e-9)
        norm = mcolors.TwoSlopeNorm(vmin=cmin, vcenter=0.0, vmax=cmax)
        scatter_values = [float(value) if value is not None else 0.0 for value in ct_values]
        ct_color_kwargs = {
            "c": scatter_values,
            "cmap": "coolwarm",
            "norm": norm,
        }
        scatter = ax.scatter(
            x_values,
            y_values,
            **ct_color_kwargs,
            s=36,
            edgecolors="black",
            linewidths=0.45,
            zorder=2,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Net CT (%)  negative: 1->2, positive: 1<-2")
    else:
        ax.scatter(
            x_values,
            y_values,
            color="#0ea5e9",
            s=36,
            edgecolors="black",
            linewidths=0.45,
            zorder=2,
        )

    if any(no_overlap_fallback_flags):
        fallback_x = [
            x_values[idx]
            for idx, flagged in enumerate(no_overlap_fallback_flags)
            if flagged
        ]
        fallback_y = [
            y_values[idx]
            for idx, flagged in enumerate(no_overlap_fallback_flags)
            if flagged
        ]
        fallback_color_kwargs: dict[str, object]
        if ct_color_kwargs is not None:
            fallback_color_kwargs = dict(ct_color_kwargs)
            fallback_color_kwargs["c"] = [
                scatter_values[idx]
                for idx, flagged in enumerate(no_overlap_fallback_flags)
                if flagged
            ]
        else:
            fallback_color_kwargs = {"color": "#0ea5e9"}
        ax.scatter(
            fallback_x,
            fallback_y,
            **fallback_color_kwargs,
            marker="*",
            s=135,
            edgecolors="black",
            linewidths=0.7,
            zorder=4,
            label="No-overlap fallback",
        )
        ax.legend(loc="best")

    for idx, point in enumerate(points):
        x = int(point["image_number"])
        y = float(point["wavelength_nm"])
        iroot = point.get("iroot")
        label = f"S{int(iroot)}" if iroot is not None else "S?"
        if no_overlap_fallback_flags[idx]:
            label += "*"
        y_offset = 8 if idx % 2 == 0 else -12
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, y_offset), fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "converged_state_wavelengths.png", dpi=300)
    plt.close(fig)

    csv_path = output_dir / "converged_state_wavelengths.csv"
    fieldnames = [
        "image_number",
        "iteration",
        "iroot",
        "wavelength_nm",
        "ct_net",
        "score",
        "point_kind",
        "point_label",
        "no_overlap_fallback",
    ]
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for point in points:
                writer.writerow(
                    {
                        "image_number": int(point["image_number"]),
                        "iteration": int(point["iteration"]),
                        "iroot": "" if point.get("iroot") is None else int(point["iroot"]),
                        "wavelength_nm": float(point["wavelength_nm"]),
                        "ct_net": "" if point.get("ct_net") is None else float(point["ct_net"]),
                        "score": "" if point.get("score") is None else float(point["score"]),
                        "point_kind": str(point.get("point_kind") or ""),
                        "point_label": str(point.get("point_label") or ""),
                        "no_overlap_fallback": int(
                            _is_no_overlap_fallback_marker(
                                point.get("point_kind"),
                                point.get("point_label"),
                            )
                        ),
                    }
                )
    except OSError as exc:
        print(f"Warning: Failed to write converged-state wavelength CSV {csv_path}: {exc}")


def _plot_multiwfn_scan_summaries(
    entries: list[dict[str, object]],
    output_dir: Path,
    settings: UVVisPlotSettings,
) -> bool:
    try:
        from cats.multiwfn.plot_uvvis import plot_charge_transfer_spectrum
    except Exception as exc:
        print(f"Warning: Failed to import Multiwfn plotting utilities: {exc}")
        return False

    output_dir_abs = output_dir.resolve()
    output_dir_abs.mkdir(parents=True, exist_ok=True)
    generated_any = False
    manifest_rows: list[dict[str, object]] = []
    min_nm, max_nm = _resolve_analysis_wavelength_limits(settings)
    if (
        min_nm is not None
        and max_nm is not None
        and float(max_nm) <= float(min_nm)
    ):
        print(
            "Warning: Invalid analysis wavelength window "
            f"[{min_nm}, {max_nm}] nm; expected max > min."
        )
        return False

    def _autoscale_y_axis(ax, *, zero_floor: bool = True) -> None:
        try:
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        except Exception:
            return
        try:
            ymin, ymax = ax.get_ylim()
        except Exception:
            return
        if not isinstance(ymin, (int, float)) or not isinstance(ymax, (int, float)):
            return
        if ymax <= ymin:
            return
        if zero_floor and ymax > 0.0:
            top = ymax * 1.05
            if top <= 0.0:
                top = ymax
            ax.set_ylim(bottom=0.0, top=top)

    def _remove_uv_components(ax) -> bool:
        removed = False
        bar_containers = []
        for container in list(ax.containers):
            label = str(container.get_label() or "")
            if "UV" in label:
                bar_containers.append(container)
        for container in bar_containers:
            try:
                container.remove()
                removed = True
            except Exception:
                continue

        for line in list(ax.get_lines()):
            label = str(line.get_label() or "")
            if "UV" in label:
                try:
                    line.remove()
                    removed = True
                except Exception:
                    continue
        return removed

    for entry in entries:
        scan_dir = Path(entry["scan_dir"]).resolve()
        iter_raw = entry.get("iteration")
        try:
            iteration = int(iter_raw)
        except Exception:
            iteration = -1
        bond_length = entry.get("bond_length")
        point_kind = str(entry.get("point_kind") or "").strip()
        point_label = str(entry.get("point_label") or "").strip()
        no_overlap_fallback = _is_no_overlap_fallback_marker(point_kind, point_label)

        iter_label = f"{iteration:03d}" if iteration >= 0 else scan_dir.name
        ct_path = output_dir_abs / f"ct_spectrum_iter_{iter_label}.png"
        ct_overall_path = output_dir_abs / f"ct_spectrum_iter_{iter_label}_overall.png"
        ct_transfer_only_path = output_dir_abs / f"ct_spectrum_iter_{iter_label}_transfer_only.png"
        uvvis_status = "disabled"
        ct_status = "skipped"
        ct_overall_status = "skipped"
        ct_transfer_only_status = "skipped"

        if settings.multiwfn_generate_ct_plot:
            ct_fit_fwhm = float(settings.stacked_uvvis_linewidth_nm)
            if ct_fit_fwhm <= 0.0:
                ct_fit_fwhm = float(settings.multiwfn_ct_min_fwhm_abs)
            try:
                ct_plot, _ = plot_charge_transfer_spectrum(
                    str(scan_dir),
                    use_eV=False,
                    min_wavelength_nm=min_nm,
                    max_wavelength_nm=max_nm,
                    min_peak_intensity_fraction=float(settings.multiwfn_ct_min_peak_intensity_fraction),
                    min_peak_distance_points=int(settings.multiwfn_ct_min_peak_distance_points),
                    min_peak_width_points=int(settings.multiwfn_ct_min_peak_width_points),
                    min_fwhm_abs=ct_fit_fwhm,
                    max_gaussians_per_spectrum=int(settings.multiwfn_ct_max_gaussians_per_spectrum),
                    verbose=False,
                    plot_individual_components=bool(settings.multiwfn_ct_plot_individual_components),
                    nroots=settings.multiwfn_ct_nroots,
                )
                title = f"Iteration {iteration}" if iteration >= 0 else scan_dir.name
                if bond_length is not None:
                    title += f" | bond={float(bond_length):.3f} A"
                if no_overlap_fallback:
                    title += " | NO-OVERLAP FALLBACK"
                try:
                    ct_ax = ct_plot.gca()
                    ct_ax.set_title(title)
                    _autoscale_y_axis(ct_ax, zero_floor=True)
                except Exception:
                    pass
                ct_plot.savefig(ct_overall_path, dpi=300, bbox_inches="tight")
                ct_plot.savefig(ct_path, dpi=300, bbox_inches="tight")
                ct_overall_status = "ok"
                ct_status = "ok"
                generated_any = True
            except Exception as exc:
                ct_overall_status = f"error: {exc}"
                ct_status = ct_overall_status
                print(f"Warning: Failed Multiwfn CT plot for {scan_dir}: {exc}")
            finally:
                try:
                    from matplotlib import pyplot as plt

                    plt.close("all")
                except Exception:
                    pass

            try:
                ct_plot_transfer, _ = plot_charge_transfer_spectrum(
                    str(scan_dir),
                    use_eV=False,
                    min_wavelength_nm=min_nm,
                    max_wavelength_nm=max_nm,
                    min_peak_intensity_fraction=float(settings.multiwfn_ct_min_peak_intensity_fraction),
                    min_peak_distance_points=int(settings.multiwfn_ct_min_peak_distance_points),
                    min_peak_width_points=int(settings.multiwfn_ct_min_peak_width_points),
                    min_fwhm_abs=ct_fit_fwhm,
                    max_gaussians_per_spectrum=int(settings.multiwfn_ct_max_gaussians_per_spectrum),
                    verbose=False,
                    plot_individual_components=bool(settings.multiwfn_ct_plot_individual_components),
                    nroots=settings.multiwfn_ct_nroots,
                )
                ct_transfer_ax = ct_plot_transfer.gca()
                transfer_title = (
                    f"Iteration {iteration} | CT-only"
                    if iteration >= 0
                    else f"{scan_dir.name} | CT-only"
                )
                if bond_length is not None:
                    transfer_title += f" | bond={float(bond_length):.3f} A"
                if no_overlap_fallback:
                    transfer_title += " | NO-OVERLAP FALLBACK"
                ct_transfer_ax.set_title(transfer_title)
                removed = _remove_uv_components(ct_transfer_ax)
                if not removed:
                    print(
                        "Warning: No UV-labelled components were removed for "
                        f"{scan_dir}; CT-only plot may still include UV traces."
                    )
                handles, labels = ct_transfer_ax.get_legend_handles_labels()
                if handles:
                    ct_transfer_ax.legend(loc="upper right")
                _autoscale_y_axis(ct_transfer_ax, zero_floor=True)
                ct_plot_transfer.savefig(ct_transfer_only_path, dpi=300, bbox_inches="tight")
                ct_transfer_only_status = "ok"
                generated_any = True
            except Exception as exc:
                ct_transfer_only_status = f"error: {exc}"
                print(f"Warning: Failed CT-only plot for {scan_dir}: {exc}")
            finally:
                try:
                    from matplotlib import pyplot as plt

                    plt.close("all")
                except Exception:
                    pass

        manifest_rows.append(
            {
                "iteration": iteration if iteration >= 0 else "",
                "scan_dir": str(scan_dir),
                "bond_length": "" if bond_length is None else float(bond_length),
                "point_kind": point_kind,
                "point_label": point_label,
                "no_overlap_fallback": int(no_overlap_fallback),
                "uvvis_plot": "",
                "uvvis_status": uvvis_status,
                "ct_plot": str(ct_path) if ct_status == "ok" else "",
                "ct_status": ct_status,
                "ct_plot_overall": str(ct_overall_path) if ct_overall_status == "ok" else "",
                "ct_overall_status": ct_overall_status,
                "ct_plot_transfer_only": (
                    str(ct_transfer_only_path) if ct_transfer_only_status == "ok" else ""
                ),
                "ct_transfer_only_status": ct_transfer_only_status,
            }
        )

    if manifest_rows:
        manifest_path = output_dir_abs / "multiwfn_scan_plot_manifest.csv"
        fieldnames = [
            "iteration",
            "scan_dir",
            "bond_length",
            "point_kind",
            "point_label",
            "no_overlap_fallback",
            "uvvis_plot",
            "uvvis_status",
            "ct_plot",
            "ct_status",
            "ct_plot_overall",
            "ct_overall_status",
            "ct_plot_transfer_only",
            "ct_transfer_only_status",
        ]
        try:
            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(manifest_rows)
        except OSError as exc:
            print(f"Warning: Failed to write Multiwfn plot manifest {manifest_path}: {exc}")

    return generated_any


def _entry_sequence_rank(entry: dict[str, object]) -> tuple[int, float]:
    iteration_raw = entry.get("iteration")
    try:
        iteration = int(iteration_raw)
    except Exception:
        iteration = -1

    scan_dir = Path(entry.get("scan_dir", ""))
    output_file = scan_dir / "orca.out"
    mtime = 0.0
    try:
        if output_file.exists():
            mtime = float(output_file.stat().st_mtime)
        elif scan_dir.exists():
            mtime = float(scan_dir.stat().st_mtime)
    except OSError:
        mtime = 0.0
    return iteration, mtime


def _select_latest_entries_by_bond_length(
    entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    numeric_entries: list[tuple[float, tuple[int, float], dict[str, object]]] = []
    for entry in entries:
        bond_length_raw = entry.get("bond_length")
        try:
            bond_length = float(bond_length_raw)
        except (TypeError, ValueError):
            continue
        rank = _entry_sequence_rank(entry)
        numeric_entries.append((bond_length, rank, entry))

    if not numeric_entries:
        return []

    numeric_entries.sort(key=lambda item: float(item[0]))
    positive_spacings = [
        float(numeric_entries[idx + 1][0] - numeric_entries[idx][0])
        for idx in range(len(numeric_entries) - 1)
        if float(numeric_entries[idx + 1][0] - numeric_entries[idx][0]) > 0.0
    ]
    if positive_spacings:
        ordered_spacings = sorted(positive_spacings)
        median_idx = len(ordered_spacings) // 2
        if len(ordered_spacings) % 2 == 1:
            median_spacing = float(ordered_spacings[median_idx])
        else:
            median_spacing = 0.5 * float(
                ordered_spacings[median_idx - 1] + ordered_spacings[median_idx]
            )
    else:
        median_spacing = 0.0

    # Collapse restart/recalculation duplicates while preserving intentionally
    # dense scans. The tolerance scales with the scan's typical spacing and is
    # bounded to avoid over-merging.
    dedupe_tolerance = max(1.0e-6, min(1.0e-2, 0.05 * median_spacing))
    grouped: list[list[tuple[float, tuple[int, float], dict[str, object]]]] = []
    for record in numeric_entries:
        if not grouped:
            grouped.append([record])
            continue
        previous_bond = float(grouped[-1][-1][0])
        current_bond = float(record[0])
        if abs(current_bond - previous_bond) <= dedupe_tolerance:
            grouped[-1].append(record)
            continue
        grouped.append([record])

    selected = [max(group, key=lambda item: item[1])[2] for group in grouped]
    selected.sort(key=lambda row: float(row.get("bond_length", 0.0)))
    return selected


def _stacked_uvvis_transitions_cache_signature(
    folders: list[str],
) -> list[tuple[str, bool, int, int]]:
    signature: list[tuple[str, bool, int, int]] = []
    for folder in folders:
        folder_path = Path(folder)
        orca_out = folder_path / "orca.out"
        probe_path = orca_out if orca_out.exists() else folder_path
        mtime_ns = 0
        size_bytes = 0
        try:
            stat = probe_path.stat()
            mtime_ns = int(stat.st_mtime_ns)
            size_bytes = int(stat.st_size)
        except OSError:
            mtime_ns = 0
            size_bytes = 0
        signature.append((str(folder_path), bool(orca_out.exists()), mtime_ns, size_bytes))
    return signature


def _plot_stacked_uvvis_from_spectra(
    selected_entries: list[dict[str, object]],
    output_dir: Path,
    settings: UVVisPlotSettings,
) -> bool:
    if not selected_entries:
        return False

    try:
        import matplotlib.colors as mcolors
        import numpy as np
        from matplotlib import pyplot as plt
    except Exception as exc:
        print(f"Warning: Failed to import fallback stacked UV-vis dependencies: {exc}")
        return False

    axis_ref: Optional[np.ndarray] = None
    profiles: list[np.ndarray] = []
    label_values: list[float] = []
    plotted_entries: list[dict[str, object]] = []

    for entry in selected_entries:
        scan_dir = Path(entry["scan_dir"])
        try:
            bond_length = float(entry["bond_length"])
        except (TypeError, ValueError):
            continue

        spectrum = _read_uvvis_spectrum_orca(scan_dir, settings)
        if spectrum is None:
            spectrum = _read_uvvis_spectrum_multiwfn(scan_dir)
        if spectrum is None:
            continue

        xs = np.asarray(spectrum[0], dtype=float)
        ys = np.asarray(spectrum[1], dtype=float)
        if xs.size == 0 or ys.size == 0:
            continue

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

        if axis_ref is None:
            axis_ref = xs
        else:
            ys = np.interp(axis_ref, xs, ys)

        if bool(settings.stacked_uvvis_normalize):
            peak = float(np.max(ys)) if ys.size else 0.0
            if peak > 0.0:
                ys = ys / peak

        profiles.append(ys)
        label_values.append(bond_length)
        plotted_entries.append(entry)

    if axis_ref is None or not profiles:
        print("Warning: No spectra available for fallback stacked UV-vis plot.")
        return False

    labels = np.asarray(label_values, dtype=float)
    offsets = labels * float(settings.stacked_uvvis_offset_scale)
    cmap = plt.get_cmap("viridis")
    vmin = float(np.min(labels))
    vmax = float(np.max(labels))
    if np.isclose(vmin, vmax):
        span = max(abs(vmin), 1.0)
        norm = mcolors.Normalize(vmin=vmin - 0.5 * span, vmax=vmax + 0.5 * span)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(labels))

    fig, ax = plt.subplots()
    min_y = np.inf
    max_y = -np.inf
    for idx, profile in enumerate(profiles):
        stacked_profile = profile + offsets[idx]
        min_y = min(min_y, float(np.min(stacked_profile)))
        max_y = max(max_y, float(np.max(stacked_profile)))
        ax.plot(axis_ref, stacked_profile, color=colors[idx])
    fallback_flags = [
        _is_no_overlap_fallback_marker(entry.get("point_kind"), entry.get("point_label"))
        for entry in plotted_entries
    ]
    _style_no_overlap_fallback_stacked_traces(ax, fallback_flags)

    ax.set_xlabel(_format_uvvis_xlabel(settings.units))
    if bool(settings.stacked_uvvis_normalize):
        ax.set_ylabel("Normalized intensity (a.u.)")
    else:
        ax.set_ylabel("Intensity (a.u.)")
    if np.isfinite(min_y) and np.isfinite(max_y):
        ax.set_ylim(min_y, max_y)

    xmin = settings.stacked_uvvis_xmin
    xmax = settings.stacked_uvvis_xmax
    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin, right=xmax)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax)
    ticks = np.unique(labels)
    if ticks.size:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(_format_stack_colorbar_ticks(ticks, str(settings.stacked_axis_label)))
    cbar.set_label(str(settings.stacked_axis_label))

    secondary = ax.twinx()
    secondary.set_yticks([])
    secondary.set_ylabel("")
    secondary.tick_params(right=False, labelright=False)
    secondary.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "stacked_uvvis.png", dpi=300, bbox_inches="tight")
    fig.set_size_inches(12, 4)
    fig.savefig(output_dir / "stacked_uvvis_wide.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_stacked_uvvis(
    entries: list[dict[str, object]],
    output_dir: Path,
    settings: UVVisPlotSettings,
) -> bool:
    selected = _select_latest_entries_by_bond_length(entries)
    if not selected:
        print("Warning: No valid scan entries available for stacked UV-vis plotting.")
        return False

    folders = [str(Path(entry["scan_dir"]).resolve()) for entry in selected]
    labels = [float(entry["bond_length"]) for entry in selected]

    try:
        import pickle
        from matplotlib import pyplot as plt
        from cats.visualise.uvvis import (
            extract_uv_transitions_from_folders,
            plot_uv_spectra_stacked,
        )
    except Exception as exc:
        print(f"Warning: Failed to import stacked UV-vis plotting dependencies: {exc}")
        return False

    cache_path = output_dir / "stacked_uvvis_transitions.pkl"
    folder_signature = _stacked_uvvis_transitions_cache_signature(folders)
    transitions = None
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if (
                payload.get("folders") == folders
                and payload.get("folder_signature") == folder_signature
            ):
                transitions = payload.get("transitions")
        except Exception:
            transitions = None

    if transitions is None:
        try:
            transitions = extract_uv_transitions_from_folders(folders)
            with cache_path.open("wb") as handle:
                pickle.dump(
                    {
                        "folders": folders,
                        "folder_signature": folder_signature,
                        "transitions": transitions,
                    },
                    handle,
                )
        except Exception as exc:
            print(
                "Warning: Failed to prepare transitions for stacked UV-vis plot; "
                f"attempting spectrum-based fallback: {exc}"
            )
            return _plot_stacked_uvvis_from_spectra(selected, output_dir, settings)

    try:
        ax, ax_stack, _ = plot_uv_spectra_stacked(
            folders,
            transitions=transitions,
            labels=labels,
            linewidth_nm=float(settings.stacked_uvvis_linewidth_nm),
            normalize=bool(settings.stacked_uvvis_normalize),
            normalize_factor=1.0,
            offset_scale=float(settings.stacked_uvvis_offset_scale),
            stack_axis_label=str(settings.stacked_axis_label),
        )
        xmin = settings.stacked_uvvis_xmin
        xmax = settings.stacked_uvvis_xmax
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
            ax_stack.set_xlim(left=xmin, right=xmax)
        fallback_flags = [
            _is_no_overlap_fallback_marker(entry.get("point_kind"), entry.get("point_label"))
            for entry in selected
        ]
        _style_no_overlap_fallback_stacked_traces(ax, fallback_flags)
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(output_dir / "stacked_uvvis.png", dpi=300, bbox_inches="tight")
        fig.set_size_inches(12, 4)
        fig.savefig(output_dir / "stacked_uvvis_wide.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as exc:
        print(
            "Warning: Failed to generate transition-based stacked UV-vis plot; "
            f"attempting spectrum-based fallback: {exc}"
        )
        try:
            plt.close("all")
        except Exception:
            pass
        return _plot_stacked_uvvis_from_spectra(selected, output_dir, settings)


def _plot_stacked_ct(
    entries: list[dict[str, object]],
    output_dir: Path,
    settings: UVVisPlotSettings,
) -> bool:
    selected = _select_latest_entries_by_bond_length(entries)
    if not selected:
        print("Warning: No valid scan entries available for stacked CT plotting.")
        return False

    try:
        import matplotlib.colors as mcolors
        import numpy as np
        from matplotlib import pyplot as plt
    except Exception as exc:
        print(f"Warning: Failed to import stacked CT plotting dependencies: {exc}")
        return False

    axis_ref: Optional[np.ndarray] = None
    profiles: list[np.ndarray] = []
    label_values: list[float] = []
    plotted_entries: list[dict[str, object]] = []

    for entry in selected:
        scan_dir = Path(entry["scan_dir"])
        bond_length = float(entry["bond_length"])
        spectrum = _read_ct_transfer_spectrum(scan_dir, settings)
        if spectrum is None:
            continue

        xs = np.asarray(spectrum[0], dtype=float)
        ys = np.asarray(spectrum[1], dtype=float)
        if xs.size == 0 or ys.size == 0:
            continue

        if axis_ref is None:
            axis_ref = xs
        else:
            ys = np.interp(axis_ref, xs, ys)

        if bool(settings.stacked_ct_normalize):
            peak = float(np.max(ys)) if ys.size else 0.0
            if peak > 0.0:
                ys = ys / peak

        profiles.append(ys)
        label_values.append(bond_length)
        plotted_entries.append(entry)

    if axis_ref is None or not profiles:
        print("Warning: No CT spectra available for stacked CT plot.")
        return False

    # Apply the same linewidth concept used by stacked UV-vis and avoid jagged traces.
    if axis_ref.size > 1:
        axis_step = float(np.median(np.diff(axis_ref)))
    else:
        axis_step = 1.0
    if axis_step <= 0.0:
        axis_step = 1.0
    smooth_fwhm = float(settings.stacked_uvvis_linewidth_nm)
    if smooth_fwhm > 0.0:
        profiles = [
            _smooth_profile_fwhm(profile, axis_step=axis_step, fwhm_nm=smooth_fwhm)
            for profile in profiles
        ]

    labels = np.asarray(label_values, dtype=float)
    offsets = labels * float(settings.stacked_ct_offset_scale)
    cmap = plt.get_cmap("viridis")
    vmin = float(np.min(labels))
    vmax = float(np.max(labels))
    if np.isclose(vmin, vmax):
        span = max(abs(vmin), 1.0)
        norm = mcolors.Normalize(vmin=vmin - 0.5 * span, vmax=vmax + 0.5 * span)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(labels))

    fig, ax = plt.subplots()
    min_y = np.inf
    max_y = -np.inf
    for idx, profile in enumerate(profiles):
        stacked_profile = profile + offsets[idx]
        min_y = min(min_y, float(np.min(stacked_profile)))
        max_y = max(max_y, float(np.max(stacked_profile)))
        ax.plot(axis_ref, stacked_profile, color=colors[idx])
    fallback_flags = [
        _is_no_overlap_fallback_marker(entry.get("point_kind"), entry.get("point_label"))
        for entry in plotted_entries
    ]
    _style_no_overlap_fallback_stacked_traces(ax, fallback_flags)

    ax.set_xlabel("Wavelength / nm")
    if bool(settings.stacked_ct_normalize):
        ax.set_ylabel("Normalized CT intensity")
    else:
        ax.set_ylabel("CT intensity (a.u.)")
    if np.isfinite(min_y) and np.isfinite(max_y):
        ax.set_ylim(min_y, max_y)
    xmin, xmax = _resolve_stacked_ct_xlim(settings)
    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin, right=xmax)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax)
    ticks = np.unique(labels)
    if ticks.size:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(_format_stack_colorbar_ticks(ticks, str(settings.stacked_axis_label)))
    cbar.set_label(str(settings.stacked_axis_label))

    secondary = ax.twinx()
    secondary.set_yticks([])
    secondary.set_ylabel("")
    secondary.tick_params(right=False, labelright=False)
    secondary.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "stacked_ct.png", dpi=300, bbox_inches="tight")
    fig.set_size_inches(12, 4)
    fig.savefig(output_dir / "stacked_ct_wide.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_scan_summaries(
    *,
    max_iter: int,
    bond_pair: tuple[int, int] | None,
    scan_log_path: Path,
    output_dir: Path,
    plotter: ScanSummaryPlotter | None = None,
    use_tddft_optimisation: bool = True,
    scan_endpoints_db_path: Path | None = None,
) -> None:
    if bond_pair is None:
        print("Bond scan disabled; skipping reaction coordinate plots.")
        return

    entries = _collect_scan_endpoints(
        max_iter=max_iter,
        scan_log_path=scan_log_path,
        bond_pair=bond_pair,
    )
    if not entries:
        print("No scan endpoints found for plotting.")
        return
    fallback_labels = _collect_no_overlap_fallback_labels(entries)
    if fallback_labels:
        preview = fallback_labels[:6]
        more = len(fallback_labels) - len(preview)
        suffix = f", +{more} more" if more > 0 else ""
        print(
            "Info: No-overlap fallback mode used at "
            + ", ".join(preview)
            + suffix
            + "."
        )

    plotter = plotter or ScanSummaryPlotter()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = _plot_multiwfn_scan_summaries(entries, output_dir, plotter.uvvis_settings)
    if not generated:
        print("Warning: Multiwfn UV/CT plot generation did not produce any files.")
    if plotter.uvvis_settings.generate_stacked_uvvis:
        ok_uv_stack = _plot_stacked_uvvis(entries, output_dir, plotter.uvvis_settings)
        if not ok_uv_stack:
            print("Warning: Failed to generate stacked UV-vis summary plot.")
    if plotter.uvvis_settings.generate_stacked_ct:
        ok_ct_stack = _plot_stacked_ct(entries, output_dir, plotter.uvvis_settings)
        if not ok_ct_stack:
            print("Warning: Failed to generate stacked CT summary plot.")
    _plot_energy_vs_reaction_coordinate(
        entries,
        output_dir,
        bond_pair=bond_pair,
        scan_endpoints_db_path=scan_endpoints_db_path,
    )
    if use_tddft_optimisation:
        _plot_converged_state_wavelengths(
            max_iter=max_iter,
            scan_log_path=scan_log_path,
            output_dir=output_dir,
        )
    else:
        # DFT-only bond scans do not track converged excited-state wavelengths.
        for stale_name in ("converged_state_wavelengths.png", "converged_state_wavelengths.csv"):
            stale_path = output_dir / stale_name
            if not stale_path.exists():
                continue
            try:
                stale_path.unlink()
            except OSError as exc:
                print(f"Warning: Failed to remove stale {stale_name} for DFT-only scan: {exc}")


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
