from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.mep.neb import NEB

from cats.multiwfn.iroot_scoring import IRootCandidate
from cats.orca.run.orca_property import get_charges_from_orca_output
from cats.orca.tddft.ct import (
    extract_state_transition_vectors,
    get_ct_info,
    pick_iroot,
    record_geometry_choice,
)
from cats.orca.tddft.io import (
    copy_initial_gbw,
    resolve_restart_gbw_path,
    select_gbw_source,
    validate_orca_output,
)
from cats.working_directory.active_learning_neb_workflow import (
    _copy_with_constraint_set,
    _generate_bond_extension_final,
    _image_count_from_shift,
    _max_atom_shift_A,
    _merged_constraints,
    _normalize_final_image_strategy,
    _validate_endpoints,
    _with_constraints,
)


TARGET_WAVELENGTH_NM = 520.0
TDDFT_SCAN_ROOTS = 50
DEFAULT_BASIS_KEYWORD = "lanl2dz_mixed"
DEFAULT_XC_FUNCTIONAL = "pbe0"
DEFAULT_SOLVENT = "Water"
IROOT_SCAN_LOG_FILENAME = "iroot_mlneb_band_log.csv"
IROOT_SCORE_WEIGHTS = {
    "intensity": 0.1,
    "ct": 0.2,
    "wavelength": 0.7,
    "similarity": 1.0,
}
IROOT_FOLLOWUP_SCORE_WEIGHTS = {
    "intensity": 0.15,
    "ct": 0.20,
    "wavelength": 0.75,
    "similarity": 0.40,
}
IROOT_WAVELENGTH_SIGMA_NM = 10.0
IROOT_SIMILARITY_SIGMA_NM = 15.0
IROOT_SIMILARITY_SIGMA_CT = 0.40
IROOT_MIN_SIMILARITY = 0.10
IROOT_MIN_OSCILLATOR_STRENGTH = 5.0e-4
IROOT_FOLLOWUP_MIN_RELATIVE_INTENSITY = 0.10
IROOT_MIN_TRANSITION_OVERLAP = 0.05
IROOT_MIN_CT_SIMILARITY = 0.15


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_workflow_log(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{_timestamp_utc()}] {str(message).rstrip()}\n")
    except Exception:
        return


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except Exception:
        return kwargs
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _import_mlneb():
    try:
        from catlearn.activelearning.mlneb import MLNEB
    except Exception:
        try:
            from catlearn.optimize.mlneb import MLNEB
        except Exception as exc:
            raise ImportError(
                "TDDFT MLNEB requires catlearn MLNEB. "
                "Install CatLearn in the runtime environment used by this workflow."
            ) from exc
    return MLNEB


def _get_tddft_workflow_helpers():
    from cats.working_directory.tddft_active_learning_workflow import (
        build_fragments,
        get_orca_tddft_calculator,
        _resolve_orca_resources,
    )

    return build_fragments, get_orca_tddft_calculator, _resolve_orca_resources


@dataclass
class ProxyContext:
    target_wavelength_nm: float
    fragments: dict[str, str]
    initial_score_weights: dict[str, float]
    followup_score_weights: dict[str, float]
    min_oscillator_strength: float
    wavelength_sigma_nm: float
    similarity_sigma_nm: float
    similarity_sigma_ct: float
    nroots: int
    min_similarity: float = 0.0
    min_relative_intensity: float = 0.0
    min_transition_overlap: float = 0.0
    min_ct_similarity: float = 0.0


def _max_force_from_cached_atoms(atoms: Atoms) -> float | None:
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return None
    try:
        forces = np.asarray(atoms.get_forces(), dtype=float)
    except Exception:
        return None
    if forces.size == 0:
        return 0.0
    if not np.all(np.isfinite(forces)):
        return None
    return float(np.max(np.linalg.norm(forces, axis=1)))


def _band_max_force(images: list[Atoms]) -> float | None:
    values = [
        value
        for value in (_max_force_from_cached_atoms(image) for image in images)
        if value is not None
    ]
    if not values:
        return None
    return float(max(values))


def _attach_charges_from_orca_output(atoms: Atoms, orca_out: Path, *, label: str) -> None:
    try:
        get_charges_from_orca_output(atoms, orca_out, method="mulliken")
    except Exception as exc:
        print(f"Warning: Failed to recover charges for {label} from {orca_out}: {exc}")


@dataclass
class BandStateRecord:
    cycle: int
    image_index: int
    geometry_step: int
    scan_dir: str
    engrad_dir: str
    iroot: int
    energy_eV: float
    max_force_eV_A: float | None
    wavelength_nm: float | None = None
    oscillator_strength: float | None = None


@dataclass
class BandStateRegistry:
    root: Path
    records: list[BandStateRecord] = field(default_factory=list)
    choices: dict[int, IRootCandidate] = field(default_factory=dict)
    states: dict[int, int] = field(default_factory=dict)
    images: list[Atoms] = field(default_factory=list)
    cycle: int = 0
    geometry_step: int = 0

    def start_cycle(self, cycle: int, images: list[Atoms]) -> None:
        self.cycle = int(cycle)
        self.images = [image.copy() for image in images]
        self.choices = {}
        self.states = {}

    def previous_for(self, image_index: int) -> tuple[IRootCandidate | None, int | None]:
        previous_index = int(image_index) - 1
        if previous_index < 0:
            return None, None
        return self.choices.get(previous_index), self.states.get(previous_index)

    def next_step(self) -> int:
        step = self.geometry_step
        self.geometry_step += 1
        return step

    def record(
        self,
        *,
        image_index: int,
        scan_dir: Path,
        engrad_dir: Path,
        choice: IRootCandidate,
        energy_eV: float,
        max_force_eV_A: float | None,
    ) -> None:
        index = int(image_index)
        self.choices[index] = choice
        self.states[index] = int(choice.state)
        self.records.append(
            BandStateRecord(
                cycle=int(self.cycle),
                image_index=index,
                geometry_step=max(0, self.geometry_step - 1),
                scan_dir=str(scan_dir),
                engrad_dir=str(engrad_dir),
                iroot=int(choice.state),
                energy_eV=float(energy_eV),
                max_force_eV_A=max_force_eV_A,
                wavelength_nm=getattr(choice, "wavelength_nm", None),
                oscillator_strength=getattr(choice, "oscillator_strength", None),
            )
        )

    def write_json(self, path: Path) -> None:
        data = [record.__dict__ for record in self.records]
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


class BandTDDFTProxy(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        *,
        registry: BandStateRegistry,
        image_index: int,
        cpus: int,
        mem: int,
        scan_root: Path,
        engrad_root: Path,
        gbw_source_dir: Path,
        geometry_log_path: Path,
        context: ProxyContext,
        calc_factory=None,
    ):
        super().__init__()
        self.registry = registry
        self.image_index = int(image_index)
        self.cpus = int(cpus)
        self.mem = int(mem)
        self.scan_root = Path(scan_root)
        self.engrad_root = Path(engrad_root)
        self.gbw_source_dir = Path(gbw_source_dir)
        self.geometry_log_path = Path(geometry_log_path)
        self.context = context
        if calc_factory is None:
            _, calc_factory, _ = _get_tddft_workflow_helpers()
        self.calc_factory = calc_factory
        self.directory = self.engrad_root / f"image_{self.image_index:03d}"

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            atoms = self.atoms
        step = self.registry.next_step()
        previous_choice, previous_state = self.registry.previous_for(self.image_index)
        scan_dir = self.scan_root / f"cycle_{self.registry.cycle:03d}" / f"image_{self.image_index:03d}_step_{step:05d}"
        engrad_dir = self.engrad_root / f"cycle_{self.registry.cycle:03d}" / f"image_{self.image_index:03d}_step_{step:05d}"
        scan_dir.mkdir(parents=True, exist_ok=True)
        engrad_dir.mkdir(parents=True, exist_ok=True)

        gbw_source = select_gbw_source(self.gbw_source_dir, [scan_dir, engrad_dir])
        if resolve_restart_gbw_path(gbw_source) is not None:
            copy_initial_gbw(gbw_source, scan_dir)
        scan_calc = self.calc_factory(
            atoms,
            **_filter_supported_kwargs(
                self.calc_factory,
                {
                    "cpus": self.cpus,
                    "mem": self.mem,
                    "task": "energy",
                    "directory": scan_dir,
                    "nroots": self.context.nroots,
                    "use_tda": True,
                },
            ),
        )
        atoms.calc = scan_calc
        atoms.get_potential_energy()

        candidates = get_ct_info(
            scan_dir,
            self.context.fragments,
            min_oscillator_strength=self.context.min_oscillator_strength,
        )
        candidate_states = None
        if (
            candidates is not None
            and hasattr(candidates, "empty")
            and not candidates.empty
            and "state" in candidates.columns
        ):
            candidate_states = set(candidates["state"].astype(int))
            if previous_state is not None:
                candidate_states.add(int(previous_state))
            if previous_choice is not None and getattr(previous_choice, "state", None) is not None:
                candidate_states.add(int(previous_choice.state))
        state_vectors = extract_state_transition_vectors(
            scan_dir / "orca.out",
            state_filter=candidate_states,
        )
        has_previous = previous_choice is not None or previous_state is not None
        score_weights = (
            self.context.followup_score_weights
            if has_previous
            else self.context.initial_score_weights
        )
        choice, _ = pick_iroot(
            candidates,
            target_wavelength_nm=self.context.target_wavelength_nm,
            previous_choice=previous_choice,
            previous_state=previous_state,
            score_weights=score_weights,
            wavelength_sigma_nm=self.context.wavelength_sigma_nm,
            similarity_sigma_nm=self.context.similarity_sigma_nm,
            similarity_sigma_ct=self.context.similarity_sigma_ct,
            min_similarity=self.context.min_similarity if has_previous else 0.0,
            min_relative_intensity=self.context.min_relative_intensity if has_previous else 0.0,
            min_transition_overlap=self.context.min_transition_overlap if has_previous else 0.0,
            min_ct_similarity=self.context.min_ct_similarity if has_previous else 0.0,
            require_transition_overlap=has_previous,
            state_vectors=state_vectors if has_previous else None,
        )
        if choice is None and has_previous:
            choice, _ = pick_iroot(
                candidates,
                target_wavelength_nm=self.context.target_wavelength_nm,
                previous_choice=None,
                previous_state=None,
                score_weights=self.context.initial_score_weights,
                wavelength_sigma_nm=self.context.wavelength_sigma_nm,
                similarity_sigma_nm=self.context.similarity_sigma_nm,
                similarity_sigma_ct=self.context.similarity_sigma_ct,
                min_similarity=0.0,
                min_relative_intensity=0.0,
                min_transition_overlap=0.0,
                min_ct_similarity=0.0,
                require_transition_overlap=False,
                state_vectors=None,
            )
        if choice is None:
            raise RuntimeError(
                "No valid TDDFT IROOT candidates found for "
                f"cycle {self.registry.cycle}, image {self.image_index}."
            )

        record_geometry_choice(
            self.geometry_log_path,
            iteration=int(self.registry.cycle),
            geometry_step=step,
            scan_dir=scan_dir,
            bond_pair=None,
            bond_length=None,
            choice=choice,
            target_wavelength_nm=self.context.target_wavelength_nm,
        )

        copy_initial_gbw(scan_dir, engrad_dir)
        engrad_calc = self.calc_factory(
            atoms,
            **_filter_supported_kwargs(
                self.calc_factory,
                {
                    "cpus": self.cpus,
                    "mem": self.mem,
                    "task": "engrad",
                    "directory": engrad_dir,
                    "nroots": int(choice.state) + 5,
                    "iroot": int(choice.state),
                    "use_tda": True,
                },
            ),
        )
        atoms.calc = engrad_calc
        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        if not np.isfinite(energy) or not np.all(np.isfinite(np.asarray(forces, dtype=float))):
            raise RuntimeError("TDDFT engrad returned non-finite energy or forces.")
        if not validate_orca_output(engrad_dir / "orca.out"):
            raise RuntimeError(f"ORCA engrad output invalid: {engrad_dir / 'orca.out'}")
        _attach_charges_from_orca_output(
            atoms,
            engrad_dir / "orca.out",
            label=f"cycle {self.registry.cycle} image {self.image_index}",
        )
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        max_force = _max_force_from_cached_atoms(atoms)
        self.registry.record(
            image_index=self.image_index,
            scan_dir=scan_dir,
            engrad_dir=engrad_dir,
            choice=choice,
            energy_eV=energy,
            max_force_eV_A=max_force,
        )
        self.results["energy"] = energy
        self.results["forces"] = forces


def _prepare_images(
    *,
    start: Atoms,
    end: Atoms,
    merged_constraints: list[Any],
    atom_shift_per_image_A: float,
    min_images: int,
    max_images: int,
    initial_band_atoms: list[Atoms] | None = None,
) -> tuple[list[Atoms], float]:
    if initial_band_atoms is not None:
        if len(initial_band_atoms) < 2:
            raise ValueError("Initial TDDFT MLNEB band requires at least two images.")
        images: list[Atoms] = []
        expected_numbers = np.asarray(start.get_atomic_numbers(), dtype=int)
        for index, source in enumerate(initial_band_atoms):
            image = source.copy()
            image.calc = None
            image.set_constraint()
            if not np.array_equal(np.asarray(image.get_atomic_numbers(), dtype=int), expected_numbers):
                raise ValueError(
                    "Initial TDDFT MLNEB band atom identities/order differ at "
                    f"image {index}."
                )
            images.append(_copy_with_constraint_set(image, merged_constraints))
        _validate_endpoints(images[0], images[-1])
        return images, _max_atom_shift_A(images[0], images[-1])

    max_shift_A = _max_atom_shift_A(start, end)
    image_count = _image_count_from_shift(
        max_shift_A=max_shift_A,
        atom_shift_per_image_A=atom_shift_per_image_A,
        min_images=min_images,
        max_images=max_images,
    )
    images = [_copy_with_constraint_set(start, merged_constraints)]
    for _ in range(max(0, int(image_count) - 2)):
        images.append(_copy_with_constraint_set(start, merged_constraints))
    images.append(_copy_with_constraint_set(end, merged_constraints))
    neb = NEB(images)
    neb.interpolate(method="idpp", apply_constraint=True)
    return images, max_shift_A


def _write_cycle_endpoints(stage_dir: Path, images: list[Atoms]) -> tuple[Path, Path]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    start_path = stage_dir / "initial.traj"
    end_path = stage_dir / "final.traj"
    write(start_path, images[0], format="traj")
    write(end_path, images[-1], format="traj")
    write(stage_dir / "band_start.extxyz", images, format="extxyz")
    return start_path, end_path


def _refresh_band_calculations(
    *,
    images: list[Atoms],
    registry: BandStateRegistry,
    cpus: int,
    mem: int,
    scan_root: Path,
    engrad_root: Path,
    gbw_source_dir: Path,
    geometry_log_path: Path,
    context: ProxyContext,
) -> None:
    for idx, image in enumerate(images):
        image.calc = BandTDDFTProxy(
            registry=registry,
            image_index=idx,
            cpus=cpus,
            mem=mem,
            scan_root=scan_root,
            engrad_root=engrad_root,
            gbw_source_dir=gbw_source_dir,
            geometry_log_path=geometry_log_path,
            context=context,
        )
        image.get_forces()


def _run_mlneb_cycle(
    *,
    images: list[Atoms],
    cycle_dir: Path,
    proxy_calc: Calculator,
    fmax: float,
    spring_constant: float,
    climb: bool,
    mlneb_steps_per_cycle: int,
    restart: bool,
) -> Any:
    MLNEB = _import_mlneb()
    start_path, end_path = _write_cycle_endpoints(cycle_dir, images)
    kwargs = _filter_supported_kwargs(
        MLNEB,
        {
            "start": str(start_path),
            "end": str(end_path),
            "ase_calc": proxy_calc,
            "n_images": max(1, len(images) - 2),
            "interpolation": images,
            "restart": bool(restart),
            "trajectory": str(cycle_dir / "MLNEB.traj"),
            "full_output": True,
            "k": float(spring_constant),
            "climb": bool(climb),
        },
    )
    dyn = MLNEB(**kwargs)
    run_kwargs = _filter_supported_kwargs(
        dyn.run,
        {
            "fmax": float(fmax),
            "steps": int(mlneb_steps_per_cycle),
        },
    )
    dyn.run(**run_kwargs)
    return dyn


def _extract_mlneb_images(dyn: Any, fallback: list[Atoms]) -> list[Atoms]:
    for attr_name in ("images", "all_images", "neb_images"):
        value = getattr(dyn, attr_name, None)
        if isinstance(value, (list, tuple)) and value:
            try:
                return [image.copy() for image in value]
            except Exception:
                pass
    for attr_name in ("neb", "ase_neb", "mep"):
        value = getattr(dyn, attr_name, None)
        images = getattr(value, "images", None)
        if isinstance(images, (list, tuple)) and images:
            try:
                return [image.copy() for image in images]
            except Exception:
                pass
    return [image.copy() for image in fallback]


def _resolve_context(atoms: Atoms, *, target_wavelength_nm: float, tddft_scan_nroots: int) -> ProxyContext:
    build_fragments, _, _ = _get_tddft_workflow_helpers()
    return ProxyContext(
        target_wavelength_nm=float(target_wavelength_nm),
        fragments=build_fragments(atoms),
        initial_score_weights=dict(IROOT_SCORE_WEIGHTS),
        followup_score_weights=dict(IROOT_FOLLOWUP_SCORE_WEIGHTS),
        min_oscillator_strength=float(IROOT_MIN_OSCILLATOR_STRENGTH),
        wavelength_sigma_nm=float(IROOT_WAVELENGTH_SIGMA_NM),
        similarity_sigma_nm=float(IROOT_SIMILARITY_SIGMA_NM),
        similarity_sigma_ct=float(IROOT_SIMILARITY_SIGMA_CT),
        nroots=int(tddft_scan_nroots),
        min_similarity=float(IROOT_MIN_SIMILARITY),
        min_relative_intensity=float(IROOT_FOLLOWUP_MIN_RELATIVE_INTENSITY),
        min_transition_overlap=float(IROOT_MIN_TRANSITION_OVERLAP),
        min_ct_similarity=float(IROOT_MIN_CT_SIMILARITY),
    )


def run_tddft_mlneb_workflow(
    *,
    initial_atoms: Atoms | None,
    final_atoms: Atoms | None = None,
    fmax: float = 0.05,
    final_image_strategy: str = "explicit_final_image",
    break_bond_i: int | None = None,
    break_bond_j: int | None = None,
    set_bond_distance_A: float | None = None,
    break_bond_extension_A: float | None = None,
    break_bond_fix: float = 0.5,
    atom_shift_per_image_A: float = 0.2,
    min_images: int = 7,
    max_images: int = 21,
    spring_constant: float = 0.1,
    max_steps: int = 300,
    climb: bool = True,
    fixed_symbols: tuple[str, ...] = (),
    catlearn_steps: int = 200,
    catlearn_ml_steps: int = 500,
    catlearn_dtrust: float = 2.0,
    target_wavelength_nm: float = TARGET_WAVELENGTH_NM,
    tddft_scan_nroots: int = TDDFT_SCAN_ROOTS,
    basis_keyword: str = DEFAULT_BASIS_KEYWORD,
    xc_functional: str = DEFAULT_XC_FUNCTIONAL,
    solvent: str | None = DEFAULT_SOLVENT,
    spin_multiplicity: int | None = None,
    nodes: int = 2,
    cpus_per_node: int = 32,
    mlneb_steps_per_cycle: int = 1,
    output_dir: Path | str = Path("tddft_mlneb"),
    initial_band_atoms: list[Atoms] | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    # These CatLearn parameters are accepted for submit compatibility. MLNEB
    # support differs across CatLearn releases, so only supported kwargs are
    # passed at call sites.
    _ = (catlearn_steps, catlearn_ml_steps, catlearn_dtrust)
    _import_mlneb()

    if initial_band_atoms is not None and len(initial_band_atoms) >= 2:
        if initial_atoms is None:
            initial_atoms = initial_band_atoms[0].copy()
        if final_atoms is None:
            final_atoms = initial_band_atoms[-1].copy()

    strategy = _normalize_final_image_strategy(final_image_strategy)
    if strategy == "explicit_final_image":
        if final_atoms is None:
            raise ValueError("Explicit TDDFT MLNEB final-image mode requires final_atoms.")
        _validate_endpoints(initial_atoms, final_atoms)

    run_dir = Path(output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    workflow_log_path = run_dir / "tddft_mlneb.log"
    workflow_log_path.write_text("", encoding="utf-8")
    _append_workflow_log(
        workflow_log_path,
        (
            "Workflow start: TDDFT MLNEB "
            f"(target_wavelength_nm={float(target_wavelength_nm):.6f}, "
            f"nroots={int(tddft_scan_nroots)})."
        ),
    )

    if initial_band_atoms is not None:
        cleaned_band: list[Atoms] = []
        for source in initial_band_atoms:
            image = source.copy()
            image.calc = None
            image.set_constraint()
            cleaned_band.append(image)
        initial_band_atoms = cleaned_band
        initial_atoms = cleaned_band[0]
        final_atoms = cleaned_band[-1]
        strategy = "explicit_final_image"

    final_constraint_source = final_atoms if final_atoms is not None else initial_atoms
    merged_constraints = _merged_constraints(
        initial_atoms=initial_atoms,
        final_atoms=final_constraint_source,
        fixed_symbols=tuple(fixed_symbols or ()),
    )
    start = _with_constraints(initial_atoms, merged_constraints)
    if strategy == "explicit_final_image":
        end = _with_constraints(final_atoms, merged_constraints)
    else:
        target_distance_A = (
            float(set_bond_distance_A)
            if set_bond_distance_A is not None
            else float(break_bond_extension_A) if break_bond_extension_A is not None else 2.5
        )
        end, _, _, _, _ = _generate_bond_extension_final(
            initial_atoms=start,
            bond_i=break_bond_i,
            bond_j=break_bond_j,
            target_distance_A=target_distance_A,
            fix=break_bond_fix,
        )
        end = _with_constraints(end, merged_constraints)

    _validate_endpoints(start, end)
    images, max_shift_A = _prepare_images(
        start=start,
        end=end,
        merged_constraints=merged_constraints,
        atom_shift_per_image_A=atom_shift_per_image_A,
        min_images=min_images,
        max_images=max_images,
        initial_band_atoms=initial_band_atoms,
    )
    _, _, resolve_orca_resources = _get_tddft_workflow_helpers()
    total_cpus, mem = resolve_orca_resources(nodes=nodes, cpus_per_node=cpus_per_node)
    context = _resolve_context(
        start,
        target_wavelength_nm=target_wavelength_nm,
        tddft_scan_nroots=tddft_scan_nroots,
    )
    registry = BandStateRegistry(root=run_dir)
    geometry_log_path = run_dir / IROOT_SCAN_LOG_FILENAME
    scan_root = run_dir / "TDDFT_scans"
    engrad_root = run_dir / "TDDFT_engrad"
    catlearn_root = run_dir / "catlearn_mlneb_cycles"

    max_cycles = max(1, int(max_steps))
    converged = False
    last_fmax: float | None = None
    for cycle in range(max_cycles):
        registry.start_cycle(cycle, images)
        _append_workflow_log(workflow_log_path, f"Cycle {cycle}: refreshing TDDFT band calculations.")
        _refresh_band_calculations(
            images=images,
            registry=registry,
            cpus=total_cpus,
            mem=mem,
            scan_root=scan_root,
            engrad_root=engrad_root,
            gbw_source_dir=engrad_root,
            geometry_log_path=geometry_log_path,
            context=context,
        )
        last_fmax = _band_max_force(images)
        write(run_dir / f"band_cycle_{cycle:03d}.extxyz", images, format="extxyz")
        registry.write_json(run_dir / "band_state_log.json")
        if last_fmax is not None and last_fmax <= float(fmax):
            converged = True
            _append_workflow_log(workflow_log_path, f"Cycle {cycle}: converged with fmax={last_fmax:.6f}.")
            break

        proxy_calc = BandTDDFTProxy(
            registry=registry,
            image_index=0,
            cpus=total_cpus,
            mem=mem,
            scan_root=scan_root,
            engrad_root=engrad_root,
            gbw_source_dir=engrad_root,
            geometry_log_path=geometry_log_path,
            context=context,
        )
        cycle_dir = catlearn_root / f"cycle_{cycle:03d}"
        _append_workflow_log(
            workflow_log_path,
            f"Cycle {cycle}: starting one-step CatLearn MLNEB in {cycle_dir}.",
        )
        cwd = Path.cwd()
        try:
            cycle_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(cycle_dir)
            dyn = _run_mlneb_cycle(
                images=images,
                cycle_dir=cycle_dir,
                proxy_calc=proxy_calc,
                fmax=fmax,
                spring_constant=spring_constant,
                climb=climb,
                mlneb_steps_per_cycle=mlneb_steps_per_cycle,
                restart=cycle > 0,
            )
        finally:
            os.chdir(cwd)
        images = _extract_mlneb_images(dyn, images)
        if len(images) < 2:
            raise RuntimeError("CatLearn MLNEB returned fewer than two images.")
        _validate_endpoints(images[0], images[-1])

    energies: list[float] = []
    for image in images:
        try:
            energies.append(float(image.get_potential_energy()))
        except Exception:
            energies.append(float("nan"))
    finite_energies = np.asarray([value for value in energies if math.isfinite(value)], dtype=float)
    ts_index = int(np.argmax(np.asarray(energies, dtype=float))) if finite_energies.size else 0

    final_band_path = run_dir / "final_band.extxyz"
    final_xyz_path = run_dir / "reaction_hub_final.xyz"
    ts_xyz_path = run_dir / "transition_state.xyz"
    write(final_band_path, images, format="extxyz")
    write(final_xyz_path, images[-1], format="xyz")
    write(ts_xyz_path, images[ts_index], format="xyz")
    registry.write_json(run_dir / "band_state_log.json")

    summary = {
        "mode": "tddft_mlneb",
        "backend": "tddft_mlneb",
        "converged": bool(converged),
        "cycle_count": int(min(max_cycles, (registry.cycle + 1))),
        "fmax_eV_A": float(fmax),
        "last_band_fmax_eV_A": last_fmax,
        "target_wavelength_nm": float(target_wavelength_nm),
        "tddft_scan_nroots": int(tddft_scan_nroots),
        "image_count": int(len(images)),
        "initial_band_source": "manual_geometry_only" if initial_band_atoms is not None else "idpp",
        "max_atom_shift_A": float(max_shift_A),
        "energies_eV": [float(value) if math.isfinite(value) else None for value in energies],
        "ts_image_index": int(ts_index),
        "ts_xyz_path": str(ts_xyz_path),
        "final_band_path": str(final_band_path),
        "final_xyz_path": str(final_xyz_path),
        "workflow_log_path": str(workflow_log_path),
        "band_state_log_path": str(run_dir / "band_state_log.json"),
        "catlearn_cycle_root": str(catlearn_root),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _append_workflow_log(
        workflow_log_path,
        f"Workflow complete: converged={bool(converged)}, cycles={summary['cycle_count']}.",
    )
    return summary
