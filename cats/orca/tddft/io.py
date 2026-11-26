from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Sequence

from ase.io import read


def resolve_restart_gbw_path(directory: str | Path) -> Path | None:
    directory = Path(directory) if isinstance(directory, str) else directory
    output = directory / "orca.out"
    gbw = directory / "orca.gbw"
    if gbw.is_file() and gbw.stat().st_size and orca_completed_successfully(output):
        return gbw
    initial = directory / "initial.gbw"
    if not output.exists() and initial.is_file() and initial.stat().st_size:
        return initial
    return None


def stage_restart_gbw_path(directory: str | Path) -> Path | None:
    directory = Path(directory)
    restart = resolve_restart_gbw_path(directory)
    if restart is None:
        return None
    if restart.name == "orca.gbw":
        staged = directory / "initial.gbw"
        shutil.copy2(restart, staged)
        restart = staged
    return restart.resolve()


def orca_completed_successfully(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            return b"ORCA TERMINATED NORMALLY" in handle.read().upper()
    except OSError:
        return False


def copy_initial_gbw(src_dir: str | Path, dest_dir: str | Path, ensure_dir=True, verbose=True):
    src_dir = Path(src_dir) if isinstance(src_dir, str) else src_dir
    dest_dir = Path(dest_dir) if isinstance(dest_dir, str) else dest_dir
    src_file = resolve_restart_gbw_path(src_dir)
    if src_file is None:
        raise FileNotFoundError(
            "Missing ORCA GBW restart file in "
            f"{src_dir}; expected orca.gbw or initial.gbw."
        )
    if ensure_dir:
        dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_file, dest_dir / "initial.gbw")
    if verbose:
        print(f"Copied {src_file.name} from {src_dir} to {dest_dir} as initial.gbw")


def select_gbw_source(primary: Path, fallbacks: Sequence[Path] | None = None) -> Path:
    if resolve_restart_gbw_path(primary) is not None:
        return primary
    if fallbacks:
        for candidate in fallbacks:
            if resolve_restart_gbw_path(candidate) is not None:
                return candidate
    return primary


def is_valid_trainingset(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        if path.stat().st_size == 0:
            return False
        read(path, index=0)
    except Exception:
        return False
    return True


def is_valid_trajectory(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        if path.stat().st_size == 0:
            return False
        read(path, index=0)
    except Exception:
        return False
    return True


def get_last_iroot(path: str | Path) -> int | None:
    path = Path(path)
    if not path.exists():
        return None
    last = None
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                s = line.strip()
                if s.startswith("IROOT"):
                    parts = s.split()
                    if len(parts) >= 2:
                        try:
                            last = int(parts[1])
                        except ValueError:
                            pass
    except OSError:
        return None
    return last


def validate_orca_output(orca_out: Path, *, warn: bool = True) -> bool:
    if not orca_out.exists():
        return False
    try:
        read(orca_out, format="orca-output")
    except Exception as exc:
        if warn:
            print(f"Warning: Failed to read {orca_out}: {exc}")
        return False
    return True


def bond_length_from_orca(scan_dir: Path, bond_pair: tuple[int, int]) -> Optional[float]:
    orca_out = scan_dir / "orca.out"
    if not validate_orca_output(orca_out):
        return None
    try:
        atoms = read(orca_out, format="orca-output")
        return float(atoms.get_distance(*bond_pair))
    except Exception as exc:
        print(f"Warning: Failed to read bond length from {orca_out}: {exc}")
        return None


def energy_from_orca(scan_dir: Path) -> Optional[float]:
    orca_out = scan_dir / "orca.out"
    if not validate_orca_output(orca_out):
        return None
    try:
        atoms = read(orca_out, format="orca-output")
        return float(atoms.get_potential_energy())
    except Exception as exc:
        print(f"Warning: Failed to read energy from {orca_out}: {exc}")
        return None
