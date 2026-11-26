from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cats.visualise.uvvis_data import TransitionData


@dataclass(frozen=True)
class WorkflowBackendCapabilities:
    supports_dft_optimisation: bool = False
    supports_tddft_scans: bool = False
    supports_tddft_state_following: bool = False
    supports_tddft_gradients: bool = False
    supports_ct_vectors: bool = False
    supports_molden: bool = False
    supports_active_learning_workflow: bool = False
    requires_restart_data: bool = True


@dataclass(frozen=True)
class WorkflowBackendResources:
    cpus: int
    memory_per_cpu_mb: int


class QMWorkflowBackend(ABC):
    """Backend contract used by the TDDFT workflow orchestration layer."""

    key: str
    label: str
    output_filename: str
    gradient_filename: str | None
    restart_filenames: tuple[str, ...]
    capabilities: WorkflowBackendCapabilities

    def normalize_basis_keyword(self, basis_keyword: str) -> str:
        return str(basis_keyword or "").strip()

    def normalize_xc_functional(self, xc_functional: str) -> str:
        return str(xc_functional or "").strip()

    def resolve_solvation_keyword(self, solvent: str | None) -> str | None:
        text = str(solvent or "").strip()
        return text or None

    def resolve_resources(self, *, nodes: int, cpus_per_node: int) -> WorkflowBackendResources:
        if nodes <= 0:
            raise ValueError("nodes must be positive.")
        if cpus_per_node <= 0:
            raise ValueError("cpus_per_node must be positive.")
        return WorkflowBackendResources(cpus=int(nodes * cpus_per_node), memory_per_cpu_mb=0)

    def output_path(self, directory: Path) -> Path:
        return Path(directory) / self.output_filename

    def gradient_path(self, directory: Path) -> Path | None:
        if not self.gradient_filename:
            return None
        return Path(directory) / self.gradient_filename

    def restart_name(self, directory: Path) -> str | None:
        for filename in self.restart_filenames:
            if (Path(directory) / filename).exists():
                return filename
        return None

    def has_restart(self, directory: Path) -> bool:
        return self.resolve_restart_path(directory) is not None

    def resolve_restart_path(self, directory: Path) -> Path | None:
        name = self.restart_name(directory)
        if name is None:
            return None
        return Path(directory) / name

    def select_restart_source(
        self,
        directory: Path,
        fallback_dirs: list[Path] | tuple[Path, ...] | None = None,
    ) -> Path:
        if self.has_restart(directory):
            return Path(directory)
        for fallback in fallback_dirs or ():
            if self.has_restart(fallback):
                return Path(fallback)
        return Path(directory)

    def copy_restart(self, source_dir: Path, target_dir: Path, *, verbose: bool = False) -> bool:
        source = self.resolve_restart_path(source_dir)
        if source is None:
            return False
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if target.resolve() == source.resolve():
            return True
        import shutil

        shutil.copy2(source, target)
        if verbose:
            print(f"Copied restart file {source} -> {target}")
        return True

    def ensure_restart(
        self,
        target_dir: Path,
        *,
        fallback_dirs: list[Path] | tuple[Path, ...] | None = None,
    ) -> bool:
        if self.has_restart(target_dir):
            return True
        source = self.select_restart_source(target_dir, fallback_dirs or ())
        if not self.has_restart(source):
            return False
        return self.copy_restart(source, target_dir, verbose=False)

    def recover_restart_from_seed(self, directory: Path, seed_dir: Path) -> bool:
        if self.has_restart(directory):
            return True
        return self.copy_restart(seed_dir, directory, verbose=False)

    def make_tddft_calculator(self, atoms: Any, **kwargs: Any):
        raise NotImplementedError(f"{self.label} does not provide a TDDFT calculator.")

    def make_dft_calculator(self, atoms: Any, **kwargs: Any):
        raise NotImplementedError(f"{self.label} does not provide a DFT calculator.")

    def make_dft_force_calculator(self, atoms: Any, **kwargs: Any):
        raise NotImplementedError(f"{self.label} does not provide a DFT force calculator.")

    @abstractmethod
    def validate_output(self, output_path: Path, *, warn: bool = True) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_output_atoms(self, output_path: Path, *, label: str):
        raise NotImplementedError

    def update_atoms_from_output(self, atoms, output_path: Path, *, label: str) -> bool:
        parsed = self.read_output_atoms(output_path, label=label)
        atoms.positions = parsed.positions
        atoms.cell = parsed.cell
        atoms.pbc = parsed.pbc
        return True

    @abstractmethod
    def parse_total_energy_eV(self, output_path: Path) -> float | None:
        raise NotImplementedError

    @abstractmethod
    def extract_transitions(self, output_path: Path) -> TransitionData:
        raise NotImplementedError

    def get_followed_state(self, output_path: Path) -> int | None:
        return None

    def has_tddft_state_tag(self, output_path: Path) -> bool:
        if not self.validate_output(output_path, warn=False):
            return False
        return self.get_followed_state(output_path) is not None

    def extract_state_transition_vectors(self, output_path: Path):
        return None

    def get_ct_info(
        self,
        scan_dir: Path,
        fragments: dict[str, str],
        *,
        min_oscillator_strength: float,
    ):
        if not self.capabilities.supports_ct_vectors:
            raise NotImplementedError(f"{self.label} does not support CT-vector extraction.")
        raise NotImplementedError(f"{self.label} does not provide CT-vector extraction.")

    def pick_iroot(self, transition_table, **kwargs: Any):
        if not self.capabilities.supports_ct_vectors:
            raise NotImplementedError(f"{self.label} does not support CT-vector IROOT scoring.")
        raise NotImplementedError(f"{self.label} does not provide IROOT scoring.")

    def bond_length_from_output(self, directory: Path, bond_pair: tuple[int, int]) -> float | None:
        output = self.output_path(directory)
        if not self.validate_output(output, warn=False):
            return None
        try:
            atoms = self.read_output_atoms(output, label="bond length")
            return float(atoms.get_distance(*bond_pair))
        except Exception as exc:
            print(f"Warning: Failed to read bond length from {output}: {exc}")
            return None

    def read_forces_eV_A(self, directory: Path, *, reference_atoms: Any | None = None):
        gradient = self.gradient_path(directory)
        if gradient is None or not gradient.exists():
            return None
        try:
            from ase.io import read

            atoms = read(gradient)
            forces = atoms.get_forces()
            if reference_atoms is None or len(forces) == len(reference_atoms):
                return forces
        except Exception:
            return None
        return None

    def get_metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "output_filename": self.output_filename,
            "gradient_filename": self.gradient_filename,
            "restart_filenames": list(self.restart_filenames),
            "capabilities": self.capabilities.__dict__.copy(),
        }


def get_workflow_backend(
    key: str | QMWorkflowBackend | None = None,
    settings: dict[str, Any] | None = None,
) -> QMWorkflowBackend:
    if isinstance(key, QMWorkflowBackend):
        return key
    normalized = str(key or "orca").strip().lower()
    if normalized == "orca":
        from .orca import OrcaWorkflowBackend

        return OrcaWorkflowBackend(settings=settings or {})
    if normalized == "nwchem":
        from .nwchem import NWChemWorkflowBackend

        return NWChemWorkflowBackend(settings=settings or {})
    raise ValueError(f"Unsupported QM workflow backend '{key}'.")


def list_workflow_backends() -> tuple[dict[str, Any], ...]:
    return (
        get_workflow_backend("orca").get_metadata(),
        get_workflow_backend("nwchem").get_metadata(),
    )
