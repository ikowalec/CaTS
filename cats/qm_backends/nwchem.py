from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ase.units import Hartree

from cats.nwchem.nwchem_uv_extractor import extract_uv_transitions_nwchem

from .workflow import QMWorkflowBackend, WorkflowBackendCapabilities, WorkflowBackendResources

_NWCHEM_TOTAL_ENERGY_RE = re.compile(
    r"Total\s+(?:DFT\s+)?energy\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE,
)
_HSE06_XC = "xpbe96 1.0 xcampbe96 -0.25 cpbe96 1.0 srhfexch 0.25"
_HSE06_CAM = "0.11 cam_alpha 0.0 cam_beta 1.0"


def resolve_nwchem_xc(xc_functional: str) -> tuple[str, str | None]:
    text = str(xc_functional or "PBE0").strip() or "PBE0"
    if text.lower().replace("_", "-") == "hse06":
        return _HSE06_XC, _HSE06_CAM
    return text, None


class NWChemWorkflowBackend(QMWorkflowBackend):
    key = "nwchem"
    label = "NWChem"
    output_filename = "nwchem.nwo"
    gradient_filename = None
    restart_filenames = ("nwchem.db", "nwchem.movecs")
    capabilities = WorkflowBackendCapabilities(
        supports_dft_optimisation=True,
        supports_tddft_scans=True,
        supports_tddft_state_following=False,
        supports_tddft_gradients=False,
        supports_ct_vectors=False,
        supports_molden=True,
        supports_active_learning_workflow=True,
        requires_restart_data=False,
    )

    def __init__(self, *, settings: dict[str, Any] | None = None) -> None:
        self.settings = dict(settings or {})

    def normalize_basis_keyword(self, basis_keyword: str) -> str:
        text = str(basis_keyword or "def2-SVP").strip()
        if not text:
            return "def2-SVP"
        return text

    def normalize_xc_functional(self, xc_functional: str) -> str:
        return resolve_nwchem_xc(xc_functional)[0]

    def resolve_solvation_keyword(self, solvent: str | None) -> str | None:
        from cats.nwchem.run import normalize_nwchem_solvent

        return normalize_nwchem_solvent(solvent)

    def resolve_resources(self, *, nodes: int, cpus_per_node: int) -> WorkflowBackendResources:
        if nodes <= 0:
            raise ValueError("nodes must be positive.")
        if cpus_per_node <= 0:
            raise ValueError("cpus_per_node must be positive.")
        memory = int(self.settings.get("memory_per_cpu_mb") or self.settings.get("memory_mb") or 4000)
        if memory <= 0:
            raise ValueError("NWChem memory_per_cpu_mb must be positive.")
        return WorkflowBackendResources(cpus=int(nodes * cpus_per_node), memory_per_cpu_mb=memory)

    def _command_template(self) -> str:
        command_template = str(
            self.settings.get("command_template")
            or self.settings.get("nwchem_command_template")
            or ""
        ).strip()
        return command_template or "nwchem PREFIX.nwi > PREFIX.nwo"

    def _spin_multiplicity(
        self,
        atoms,
        spin_multiplicity: int | None = None,
        system_charge: int = 0,
    ) -> int:
        if spin_multiplicity is not None:
            return int(spin_multiplicity)
        return int((sum(atoms.get_atomic_numbers()) - int(system_charge)) % 2 + 1)

    def _calculator_kwargs(
        self,
        atoms,
        *,
        task: str,
        directory: Path,
        cpus: int,
        mem: int,
        basis_keyword: str,
        xc_functional: str,
        spin_multiplicity: int | None = None,
        nroots: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        _ = cpus
        xc_functional, default_cam = resolve_nwchem_xc(xc_functional)
        label = str(kwargs.pop("label", "nwchem") or "nwchem")
        dft_settings = dict(self.settings.get("dft", {}) or {})
        dft_settings.update(dict(kwargs.pop("dft_settings", {}) or {}))
        driver_settings = dict(self.settings.get("driver", {}) or {})
        driver_settings.update(dict(kwargs.pop("driver_settings", {}) or {}))
        cam = kwargs.pop("cam", self.settings.get("cam", None))
        system_charge = int(kwargs.pop("system_charge", 0))
        if cam is None:
            cam = default_cam
        calc_kwargs: dict[str, Any] = {
            "nwchem_command": self._command_template(),
            "charge": system_charge,
            "mult": self._spin_multiplicity(atoms, spin_multiplicity, system_charge),
            "memory_per_core": int(mem),
            "basis": basis_keyword,
            "xc": xc_functional,
            "cam": cam,
            "dft_settings": dft_settings,
            "driver_settings": driver_settings,
            "directory": str(directory),
            "label": label,
        }
        uses_def2 = str(basis_keyword).lower().replace("_", "-").startswith("def2")
        if uses_def2 and "Ce" in atoms.get_chemical_symbols():
            calc_kwargs["basispar"] = "bse"
        if "ecp" not in kwargs and uses_def2:
            from ase.data import atomic_numbers

            ecp_symbols = {
                symbol
                for symbol in atoms.get_chemical_symbols()
                if 37 <= atomic_numbers[symbol] <= 57
                or 72 <= atomic_numbers[symbol] <= 86
            }
            if ecp_symbols:
                calc_kwargs["ecp"] = {
                    symbol: "library def2-ecp" for symbol in sorted(ecp_symbols)
                }
        solvent = self.resolve_solvation_keyword(kwargs.pop("solvent", None))
        if solvent is not None:
            calc_kwargs["solvent"] = solvent
        if nroots is not None:
            calc_kwargs["tddft"] = {"nroots": int(nroots)}
        calc_kwargs.update(kwargs)
        calc_kwargs["task"] = task
        return calc_kwargs

    def make_dft_calculator(self, atoms: Any, **kwargs: Any):
        from cats.nwchem.run import get_nwchem_calculator

        task = str(kwargs.pop("task", "energy") or "energy").strip()
        if task.lower() in {"sp", "singlepoint", "single_point"}:
            task = "energy"
        calc_kwargs = self._calculator_kwargs(atoms, task=task, **kwargs)
        return get_nwchem_calculator(**calc_kwargs)

    def make_dft_force_calculator(self, atoms: Any, **kwargs: Any):
        from cats.nwchem.run import get_nwchem_calculator

        task = str(kwargs.pop("task", "engrad") or "engrad").strip()
        if task.lower() != "engrad":
            print(f"Warning: Overriding DFT task '{task}' with 'engrad' for ground-state forces.")
        calc_kwargs = self._calculator_kwargs(atoms, task="gradient", **kwargs)
        return get_nwchem_calculator(**calc_kwargs)

    def make_tddft_calculator(self, atoms: Any, **kwargs: Any):
        from cats.nwchem.run import get_nwchem_calculator

        task = str(kwargs.pop("task", "energy") or "energy").strip()
        if task.lower() in {"sp", "singlepoint", "single_point"}:
            task = "energy"
        nroots = kwargs.pop("nroots", kwargs.pop("tddft_nroots", None))
        calc_kwargs = self._calculator_kwargs(atoms, task=task, nroots=nroots, **kwargs)
        calc_kwargs.setdefault("tddft", {"nroots": int(nroots or 20)})
        return get_nwchem_calculator(**calc_kwargs)

    def render_input_files(self, atoms, params: dict[str, Any] | None = None) -> dict[str, str]:
        params = dict(params or {})
        task = str(params.get("task") or "energy").strip().lower().replace("-", "_")
        if task not in {"energy", "optimize", "tddft", "optimize_tddft"}:
            raise ValueError("Unsupported NWChem task. Choose energy, optimize, tddft, or optimize_tddft.")

        basis = self.normalize_basis_keyword(str(params.get("basis") or params.get("basis_keyword") or "def2-SVP"))
        xc, default_cam = resolve_nwchem_xc(
            str(params.get("xc") or params.get("xc_functional") or "PBE0")
        )
        charge = int(params.get("charge", 0) or 0)
        multiplicity = int(params.get("multiplicity", params.get("spin_multiplicity", 1)) or 1)
        memory_mb = int(params.get("memory_mb", self.settings.get("memory_mb", 4000)) or 4000)
        nroots = int(params.get("nroots", params.get("tddft_nroots", 20)) or 20)
        molden = bool(params.get("molden", True))
        prefix = str(params.get("prefix") or "nwchem").strip() or "nwchem"
        solvent = self.resolve_solvation_keyword(params.get("solvent"))
        cam = params.get("cam", self.settings.get("cam", None))
        if cam is None:
            cam = default_cam

        lines = [
            f'title "{prefix}"',
            "echo",
            "",
            f"memory total {memory_mb} mb",
            f"charge {charge}",
            "",
            self._geometry_block(atoms),
            "",
            "basis",
            f"  * library {basis}",
            "end",
            "",
            "dft",
            f"  xc {xc}",
            *([f"  cam {cam}"] if cam else []),
            f"  mult {multiplicity}",
            "  iterations 800",
            "end",
            "",
        ]
        if solvent is not None:
            lines.extend(["cosmo", f"  solvent {solvent}", "end", ""])
        if task in {"optimize", "optimize_tddft"}:
            lines.extend(["driver", "  maxiter 200", "end", "", "task dft optimize", ""])
        elif task == "energy":
            lines.extend(["task dft energy", ""])

        if task in {"tddft", "optimize_tddft"}:
            lines.extend(
                [
                    "tddft",
                    f"  nroots {nroots}",
                    "end",
                    "",
                    "task tddft energy",
                    "",
                ]
            )

        if molden:
            lines.extend(self._molden_property_lines(prefix))

        return {f"{prefix}.nwi": "\n".join(lines).rstrip() + "\n"}

    def _geometry_block(self, atoms) -> str:
        lines = ['geometry units angstrom nocenter noautoz noautosym']
        for symbol, position in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            x, y, z = (float(position[0]), float(position[1]), float(position[2]))
            lines.append(f"  {symbol:2s} {x:16.10f} {y:16.10f} {z:16.10f}")
        lines.append("end")
        return "\n".join(lines)

    def _molden_property_lines(self, prefix: str) -> list[str]:
        _ = prefix
        return [
            "property",
            "  moldenfile",
            "  molden_norm none",
            "end",
            "",
            "task dft property",
            "",
        ]

    def validate_output(self, output_path: Path, *, warn: bool = True) -> bool:
        path = Path(output_path)
        if not path.exists():
            if warn:
                print(f"Warning: NWChem output does not exist: {path}")
            return False
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        if "current input line" in text.lower() and "error" in text.lower():
            return False
        if "total times" in text.lower() or "task times" in text.lower():
            return True
        return bool(self.parse_total_energy_eV(path) is not None)

    def read_output_atoms(self, output_path: Path, *, label: str):
        from ase.io import read

        try:
            return read(str(output_path), format="nwchem-out")
        except Exception as exc:
            raise RuntimeError(f"Failed to read NWChem output for {label}: {output_path}") from exc

    def read_forces_eV_A(self, directory: Path, *, reference_atoms: Any | None = None):
        output = self.output_path(directory)
        if not output.exists():
            return None
        try:
            atoms = self.read_output_atoms(output, label="forces")
            forces = atoms.get_forces()
            if reference_atoms is None or len(forces) == len(reference_atoms):
                return forces
        except Exception:
            return None
        return None

    def parse_total_energy_eV(self, output_path: Path) -> float | None:
        energy_hartree: float | None = None
        try:
            with open(output_path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    match = _NWCHEM_TOTAL_ENERGY_RE.search(line)
                    if match:
                        energy_hartree = float(match.group(1))
        except OSError:
            return None
        if energy_hartree is None:
            return None
        return float(energy_hartree * Hartree)

    def extract_transitions(self, output_path: Path):
        return extract_uv_transitions_nwchem(str(output_path))
