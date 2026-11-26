from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

from ase.io import read
from ase.units import Hartree

from cats.orca.routines.orca_uv_extractor import extract_uv_transitions
from cats.orca.run.orca_property import get_charges_from_orca_output
from cats.orca.tddft.io import (
    copy_initial_gbw,
    get_last_iroot,
    resolve_restart_gbw_path,
    select_gbw_source,
    validate_orca_output,
)

from .workflow import QMWorkflowBackend, WorkflowBackendCapabilities, WorkflowBackendResources

_ORCA_FINAL_SP_ENERGY_RE = re.compile(
    r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)"
)
_ORCA_TOTAL_ENERGY_RE = re.compile(
    r"Total\s+Energy\s*[:=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)\s*Eh",
    re.IGNORECASE,
)


class OrcaWorkflowBackend(QMWorkflowBackend):
    key = "orca"
    label = "ORCA"
    output_filename = "orca.out"
    gradient_filename = "orca.engrad"
    restart_filenames = ("initial.gbw", "orca.gbw")
    capabilities = WorkflowBackendCapabilities(
        supports_dft_optimisation=True,
        supports_tddft_scans=True,
        supports_tddft_state_following=True,
        supports_tddft_gradients=True,
        supports_ct_vectors=True,
        supports_molden=True,
        supports_active_learning_workflow=True,
    )

    def __init__(
        self,
        *,
        settings: dict[str, Any] | None = None,
        tddft_calculator_factory: Callable[..., Any] | None = None,
        dft_calculator_factory: Callable[..., Any] | None = None,
        dft_force_calculator_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = dict(settings or {})
        self._tddft_calculator_factory = tddft_calculator_factory
        self._dft_calculator_factory = dft_calculator_factory
        self._dft_force_calculator_factory = dft_force_calculator_factory

    def with_calculator_factories(
        self,
        *,
        tddft_calculator_factory: Callable[..., Any],
        dft_calculator_factory: Callable[..., Any],
        dft_force_calculator_factory: Callable[..., Any],
    ) -> "OrcaWorkflowBackend":
        clone = self.__class__(
            settings=self.settings,
            tddft_calculator_factory=tddft_calculator_factory,
            dft_calculator_factory=dft_calculator_factory,
            dft_force_calculator_factory=dft_force_calculator_factory,
        )
        return clone

    def normalize_basis_keyword(self, basis_keyword: str) -> str:
        key = str(basis_keyword).strip().lower()
        aliases = {
            "lanl2dz_mixed": "lanl2dz_mixed",
            "lanl2dz": "lanl2dz_mixed",
            "mixed": "lanl2dz_mixed",
            "mixed_lanl2dz": "lanl2dz_mixed",
            "def2-svp": "def2-svp",
            "def2svp": "def2-svp",
            "def2": "def2-svp",
        }
        resolved = aliases.get(key)
        if resolved is None:
            raise ValueError("Unsupported basis_keyword. Choose 'lanl2dz_mixed' or 'def2-svp'.")
        return resolved

    def normalize_xc_functional(self, xc_functional: str) -> str:
        key = str(xc_functional).strip().lower().replace("_", "-")
        aliases = {
            "pbe0": "PBE0",
            "cam-b3lyp": "CAM-B3LYP",
        }
        resolved = aliases.get(key)
        if resolved is None:
            raise ValueError("Unsupported xc_functional. Choose 'pbe0' or 'cam-b3lyp'.")
        return resolved

    def resolve_solvation_keyword(self, solvent: str | None) -> str | None:
        if solvent is None:
            return None
        text = str(solvent).strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"none", "off", "gas", "gas-phase", "gas_phase", "vacuum"}:
            return None
        if "(" in text and ")" in text:
            return text
        return f"CPCM({text})"

    def resolve_resources(self, *, nodes: int, cpus_per_node: int) -> WorkflowBackendResources:
        if nodes <= 0:
            raise ValueError("nodes must be positive.")
        if cpus_per_node <= 0:
            raise ValueError("cpus_per_node must be positive.")
        total_cpus = int(nodes * cpus_per_node)
        explicit_mem = (
            self.settings.get("memory_per_cpu_mb")
            or self.settings.get("memory_mb")
            or os.environ.get("WORKFLOW_MEM_PER_CPU")
        )
        if explicit_mem:
            mem_per_cpu = int(explicit_mem)
        elif os.environ.get("SLURM_MEM_PER_NODE"):
            try:
                mem_per_cpu = int(int(os.environ["SLURM_MEM_PER_NODE"]) * 0.70 / cpus_per_node)
            except (ValueError, TypeError):
                mem_per_cpu = int(1300 * 128 / cpus_per_node)
        else:
            mem_per_cpu = int(1300 * 128 / cpus_per_node)
        if mem_per_cpu <= 0:
            raise ValueError("Derived memory_per_cpu is non-positive; reduce cpus_per_node.")
        return WorkflowBackendResources(cpus=total_cpus, memory_per_cpu_mb=mem_per_cpu)

    def resolve_restart_path(self, directory: Path) -> Path | None:
        return resolve_restart_gbw_path(directory)

    def select_restart_source(
        self,
        directory: Path,
        fallback_dirs: list[Path] | tuple[Path, ...] | None = None,
    ) -> Path:
        return select_gbw_source(directory, list(fallback_dirs or ()))

    def copy_restart(self, source_dir: Path, target_dir: Path, *, verbose: bool = False) -> bool:
        copy_initial_gbw(source_dir, target_dir, verbose=verbose)
        return self.resolve_restart_path(target_dir) is not None

    def make_tddft_calculator(self, atoms: Any, **kwargs: Any):
        if self._tddft_calculator_factory is None:
            raise RuntimeError("ORCA TDDFT calculator factory was not configured.")
        call_kwargs = dict(kwargs)
        if "use_tda" not in call_kwargs and "use_tda" in self.settings:
            call_kwargs["use_tda"] = bool(self.settings["use_tda"])
        return self._tddft_calculator_factory(atoms, **call_kwargs)

    def make_dft_calculator(self, atoms: Any, **kwargs: Any):
        if self._dft_calculator_factory is None:
            raise RuntimeError("ORCA DFT calculator factory was not configured.")
        return self._dft_calculator_factory(atoms, **kwargs)

    def make_dft_force_calculator(self, atoms: Any, **kwargs: Any):
        if self._dft_force_calculator_factory is None:
            raise RuntimeError("ORCA DFT force calculator factory was not configured.")
        return self._dft_force_calculator_factory(atoms, **kwargs)

    def validate_output(self, output_path: Path, *, warn: bool = True) -> bool:
        return bool(validate_orca_output(output_path, warn=warn))

    def read_output_atoms(self, output_path: Path, *, label: str):
        atoms = read(output_path, format="orca-output")
        try:
            get_charges_from_orca_output(atoms, output_path, method="mulliken")
        except Exception as exc:
            print(f"Warning: Failed to recover charges for {label} from {output_path}: {exc}")
        return atoms

    def update_atoms_from_output(self, atoms, output_path: Path, *, label: str) -> bool:
        from cats.orca.tddft.active_learning import update_atoms_from_orca_output

        return bool(update_atoms_from_orca_output(atoms, output_path, label=label))

    def parse_total_energy_eV(self, output_path: Path) -> float | None:
        final_sp_hartree: float | None = None
        total_hartree: float | None = None
        try:
            with open(output_path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    final_match = _ORCA_FINAL_SP_ENERGY_RE.search(line)
                    if final_match:
                        final_sp_hartree = float(final_match.group(1))
                        continue
                    total_match = _ORCA_TOTAL_ENERGY_RE.search(line)
                    if total_match:
                        total_hartree = float(total_match.group(1))
        except OSError:
            return None

        energy_hartree = final_sp_hartree if final_sp_hartree is not None else total_hartree
        if energy_hartree is None:
            return None
        return float(energy_hartree * Hartree)

    def extract_transitions(self, output_path: Path):
        return extract_uv_transitions(str(output_path))

    def get_followed_state(self, output_path: Path) -> int | None:
        return get_last_iroot(output_path)

    def extract_state_transition_vectors(self, output_path: Path, **kwargs: Any):
        from cats.orca.tddft.ct import extract_state_transition_vectors

        return extract_state_transition_vectors(output_path, **kwargs)

    def get_ct_info(
        self,
        scan_dir: Path,
        fragments: dict[str, str],
        *,
        min_oscillator_strength: float,
    ):
        from cats.orca.tddft.ct import get_ct_info

        return get_ct_info(
            scan_dir,
            fragments,
            min_oscillator_strength=min_oscillator_strength,
        )

    def pick_iroot(self, transition_table, **kwargs: Any):
        from cats.orca.tddft.ct import pick_iroot

        return pick_iroot(transition_table, **kwargs)

    def bond_length_from_output(self, directory: Path, bond_pair: tuple[int, int]) -> float | None:
        from cats.orca.tddft.io import bond_length_from_orca

        return bond_length_from_orca(directory, bond_pair)

    def read_forces_eV_A(self, directory: Path, *, reference_atoms: Any | None = None):
        gradient = self.gradient_path(directory)
        if gradient is None or not gradient.exists():
            return None
        for read_kwargs in (
            {"format": "orca-engrad"},
            {},
        ):
            try:
                atoms = read(gradient, **read_kwargs)
            except Exception:
                continue
            try:
                forces = atoms.get_forces()
                if reference_atoms is None or len(forces) == len(reference_atoms):
                    return forces
            except Exception:
                continue
        return None
