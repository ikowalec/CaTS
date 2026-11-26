from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ase.constraints import FixBondLength
from ase.io import read, write

from cats.nwchem.run import get_nwchem_command
from cats.qm_backends.workflow import get_workflow_backend


@contextmanager
def working_directory(path: Path | str):
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(target)
    try:
        yield target
    finally:
        os.chdir(previous)


def first_existing_path(*paths: Path | str) -> Path:
    for candidate in paths:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(paths[0])


def copy_path_if_exists(source: Path | str, target: Path | str) -> str | None:
    source_path = Path(source)
    if not source_path.exists():
        return None
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source_path.resolve() == target_path.resolve():
            return str(target_path)
    except OSError:
        pass
    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()
    if source_path.is_dir():
        shutil.copytree(source_path, target_path)
    else:
        shutil.copy2(source_path, target_path)
    return str(target_path)


def _transition_arrays(transitions: Any) -> tuple[Any, Any]:
    import numpy as np

    wavelengths = (
        np.asarray(transitions.wavelengths_nm, dtype=float)
        if transitions.wavelengths_nm is not None
        else 1.0e7 / np.asarray(transitions.wavenumbers_cm, dtype=float)
    )
    intensities = np.asarray(transitions.intensities, dtype=float)
    finite_mask = np.isfinite(wavelengths) & np.isfinite(intensities)
    wavelengths = wavelengths[finite_mask]
    intensities = intensities[finite_mask]
    if wavelengths.size == 0 or intensities.size == 0:
        raise ValueError("No finite TDDFT transitions available for spectrum export.")
    return wavelengths, intensities


def resolve_uv_spectrum_nm_range(
    wavelengths: Any,
    *,
    start_nm: float | None,
    end_nm: float | None,
    linewidth_nm: float,
    fallback_start_nm: float = 300.0,
    fallback_end_nm: float = 800.0,
) -> tuple[float, float, bool]:
    import numpy as np

    width = float(linewidth_nm)
    axis_start = float(start_nm) if start_nm is not None else max(0.0, float(np.nanmin(wavelengths)) - 5.0 * width)
    axis_end = float(end_nm) if end_nm is not None else float(np.nanmax(wavelengths)) + 5.0 * width
    fallback_used = False

    if start_nm is not None and end_nm is not None and axis_end > axis_start:
        in_requested_window = (wavelengths >= axis_start) & (wavelengths <= axis_end)
        if not bool(np.any(in_requested_window)):
            axis_start = float(fallback_start_nm)
            axis_end = float(fallback_end_nm)
            fallback_used = True

    if axis_end <= axis_start:
        axis_end = axis_start + 10.0 * width
    return axis_start, axis_end, fallback_used


def build_uv_spectrum_nm(
    transitions: Any,
    *,
    linewidth_nm: float,
    start_nm: float | None,
    end_nm: float | None,
    step: float,
    normalize: bool,
    normalize_factor: float = 1.0,
) -> tuple[Any, Any, bool]:
    import numpy as np

    wavelengths, intensities = _transition_arrays(transitions)
    width = float(linewidth_nm)
    if width <= 0:
        raise ValueError("Line width must be positive.")
    if step <= 0:
        raise ValueError("Step size must be positive.")
    if normalize_factor <= 0:
        raise ValueError("normalize_factor must be positive.")

    axis_start, axis_end, fallback_used = resolve_uv_spectrum_nm_range(
        wavelengths,
        start_nm=start_nm,
        end_nm=end_nm,
        linewidth_nm=width,
    )
    axis = np.arange(axis_start, axis_end + float(step), float(step), dtype=float)

    sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    profile = np.zeros_like(axis)
    for wavelength, intensity in zip(wavelengths, intensities):
        profile += float(intensity) * np.exp(-0.5 * ((axis - float(wavelength)) / sigma) ** 2)

    if normalize and profile.size:
        peak = float(np.nanmax(profile))
        if peak > 0.0:
            profile = profile / peak * float(normalize_factor)
    return axis, profile, fallback_used


def export_final_tddft_spectrum(
    transition_output_path: Path | str,
    *,
    final_spectrum_dir: Path | str = Path("DFT") / "final_tddft_spectrum",
    backend: Any | None = None,
    transitions: Any | None = None,
) -> dict[str, object]:
    spectrum_dir = Path(final_spectrum_dir)
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    csv_path = spectrum_dir / "final_tddft_spectrum.csv"
    png_path = spectrum_dir / "final_tddft_spectrum.png"
    backend = backend or get_workflow_backend("nwchem")

    if transitions is None:
        transitions = backend.extract_transitions(Path(transition_output_path))

    from cats.orca.tddft.scan import UVVisPlotSettings

    uvvis_settings = UVVisPlotSettings()
    axis, profile, range_fallback_used = build_uv_spectrum_nm(
        transitions,
        linewidth_nm=float(uvvis_settings.stacked_uvvis_linewidth_nm),
        start_nm=uvvis_settings.stacked_uvvis_xmin,
        end_nm=uvvis_settings.stacked_uvvis_xmax,
        step=float(uvvis_settings.step),
        normalize=False,
        normalize_factor=1.0,
    )

    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("wavelength_nm,intensity\n")
        for x_value, y_value in zip(axis, profile):
            handle.write(f"{float(x_value):.8g},{float(y_value):.8g}\n")

    result: dict[str, object] = {
        "directory": str(spectrum_dir),
        "spectrum_csv": str(csv_path),
        "wavelength_range_nm": [float(axis[0]), float(axis[-1])],
        "wavelength_range_fallback": bool(range_fallback_used),
        "intensity_units": "broadened_oscillator_strength",
        "normalization_mode": "raw",
    }
    fig = None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(axis, profile)
        ax.set_xlabel("Wavelength / nm")
        ax.set_ylabel("Broadened oscillator strength")
        ax.set_title("Final TDDFT spectrum")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        result["spectrum_png"] = str(png_path)
    except Exception as exc:
        result["plot_warning"] = str(exc)
    finally:
        if fig is not None:
            plt.close(fig)
    return result


def _resolve_workflow_resources(
    *,
    nodes: int,
    tasks_per_node: int,
    cpus_per_task: int,
) -> tuple[int, int, int, int, int]:
    workflow_nodes = int(os.environ.get("WORKFLOW_NODES", int(nodes)))
    workflow_tasks_per_node = int(os.environ.get("WORKFLOW_TASKS_PER_NODE", int(tasks_per_node)))
    workflow_cpus_per_task = int(os.environ.get("WORKFLOW_CPUS_PER_TASK", int(cpus_per_task)))
    workflow_cpus_per_node = workflow_tasks_per_node * workflow_cpus_per_task
    workflow_total_cpus = workflow_nodes * workflow_cpus_per_node
    if workflow_nodes <= 0 or workflow_cpus_per_node <= 0:
        raise ValueError(
            "Invalid workflow resource mapping: "
            f"nodes={workflow_nodes}, cpus_per_node={workflow_cpus_per_node}"
        )
    print(
        "Workflow/NWCHEM resources: "
        f"nodes={workflow_nodes}, tasks_per_node={workflow_tasks_per_node}, "
        f"cpus_per_task={workflow_cpus_per_task}, cpus_per_node={workflow_cpus_per_node}, "
        f"total_cpus={workflow_total_cpus}"
    )
    return (
        workflow_nodes,
        workflow_tasks_per_node,
        workflow_cpus_per_task,
        workflow_cpus_per_node,
        workflow_total_cpus,
    )


def _resolve_nwchem_command(
    *,
    hpc_name: str,
    workflow_nodes: int,
    workflow_cpus_per_node: int,
    workflow_total_cpus: int,
) -> str:
    command_template = str(os.environ.get("NWCHEM_COMMAND") or "").strip()
    if not command_template:
        hpc_key = str(hpc_name or "").strip().lower()
        try:
            command_template = get_nwchem_command(
                hpc_key,
                workflow_nodes,
                workflow_cpus_per_node,
                workflow_total_cpus,
            )
        except Exception as exc:
            print(f"Warning: Could not build configured NWChem command: {exc}")
            command_template = ""
    if (not command_template) or "None" in command_template:
        command_template = "nwchem PREFIX.nwi > PREFIX.nwo"
    print(f"Using ASE NWChem command template: {command_template}")
    return command_template


def run_nwchem_single_optimisation(
    *,
    input_path: str = "input_structure.extxyz",
    hpc_name: str = "",
    nodes: int = 1,
    tasks_per_node: int = 1,
    cpus_per_task: int = 1,
    backend_settings: dict[str, Any] | None = None,
    basis_keyword: str = "def2-SVP",
    xc_functional: str = "PBE0",
    cam: str | None = None,
    solvent: str | None = None,
    spin_multiplicity: int | None = None,
    bond_pair_indices: tuple[int, int] | list[int] | None = None,
    target_distance: float | None = None,
    bond_fix: float = 0.5,
    final_tddft_spectrum: bool = False,
    final_cluster_tddft_difference_spectrum: bool = False,
    final_tddft_nroots: int = 50,
) -> dict[str, Any]:
    (
        workflow_nodes,
        workflow_tasks_per_node,
        workflow_cpus_per_task,
        workflow_cpus_per_node,
        workflow_total_cpus,
    ) = _resolve_workflow_resources(
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
    )

    try:
        atoms = read(input_path)
    except Exception:
        atoms = read(input_path, format="xyz")

    if bond_pair_indices is not None:
        if target_distance is not None:
            atoms.set_distance(
                int(bond_pair_indices[0]),
                int(bond_pair_indices[1]),
                float(target_distance),
                fix=float(bond_fix),
            )
        atoms.set_constraint(
            FixBondLength(
                int(bond_pair_indices[0]),
                int(bond_pair_indices[1]),
            )
        )

    backend_settings = dict(backend_settings or {})
    write("input_structure_resolved.extxyz", atoms)

    command_template = _resolve_nwchem_command(
        hpc_name=hpc_name,
        workflow_nodes=workflow_nodes,
        workflow_cpus_per_node=workflow_cpus_per_node,
        workflow_total_cpus=workflow_total_cpus,
    )

    workflow_backend_settings = {
        **backend_settings,
        "command_template": command_template,
        "cam": cam,
    }

    from cats.working_directory.tddft_active_learning_workflow import BondScanSettings, run_workflow

    final_atoms = run_workflow(
        atoms,
        n_iter=1,
        bond_scan=BondScanSettings(enabled=False),
        use_tddft_optimisation=False,
        final_tddft_spectrum=bool(final_tddft_spectrum),
        final_cluster_tddft_difference_spectrum=bool(final_cluster_tddft_difference_spectrum),
        final_tddft_nroots=int(final_tddft_nroots),
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=spin_multiplicity,
        nodes=int(workflow_nodes),
        cpus_per_node=int(workflow_cpus_per_node),
        qm_backend="nwchem",
        qm_backend_settings=workflow_backend_settings,
        compute_cluster_fragment_single_point=False,
    )
    write("nwchem_final.extxyz", final_atoms)

    summary_path = Path("reaction_hub_summary.json")
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    summary.setdefault("workflow", "nwchem_single_optimisation")
    summary.setdefault("status", "completed")
    settings = dict(summary.get("settings", {}) or {})
    settings.update(
        {
            "qm_backend": "nwchem",
            "basis_keyword": basis_keyword,
            "xc_functional": xc_functional,
            "solvent": solvent,
            "spin_multiplicity": spin_multiplicity,
            "task": "engrad",
            "final_tddft_spectrum": bool(final_tddft_spectrum),
            "final_cluster_tddft_difference_spectrum": bool(
                final_cluster_tddft_difference_spectrum
            ),
        }
    )
    summary["settings"] = settings
    summary["nwchem_command_template"] = command_template
    summary["final_tddft_spectrum_requested"] = bool(
        final_tddft_spectrum or final_cluster_tddft_difference_spectrum
    )
    summary["final_cluster_tddft_difference_spectrum_requested"] = bool(
        final_cluster_tddft_difference_spectrum
    )
    summary["final_tddft_nroots"] = int(final_tddft_nroots)
    final = dict(summary.get("final", {}) or {})
    output = final.get("output")
    if output:
        summary.setdefault("output", output)
        summary.setdefault("nwchem_out", output)
    if "energy_eV" in final:
        summary.setdefault("final_energy_eV", final.get("energy_eV"))
    if "energy_hartree" in final:
        summary.setdefault("final_energy_hartree", final.get("energy_hartree"))
    tddft_output = Path("DFT") / "final_tddft_spectrum" / "nwchem.nwo"
    if bool(final_tddft_spectrum or final_cluster_tddft_difference_spectrum) and tddft_output.exists():
        summary.setdefault("tddft_output", str(tddft_output))
        summary.setdefault("tddft_transition_output", str(tddft_output))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
