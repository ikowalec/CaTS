from __future__ import annotations

import hashlib
import json
import os
import tempfile
from io import StringIO
from pathlib import Path

from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read, write

_INDEXED_STRUCTURE_SUFFIXES = {".traj", ".db"}
_BATCH_STRUCTURE_SUFFIXES = {".traj", ".db"}


def _uploaded_filename_forces_vasp_format(filename: str) -> bool:
    basename = Path(str(filename or "")).name.lower()
    return "poscar" in basename or "contcar" in basename


def _coerce_atoms_result(result):
    if hasattr(result, "get_chemical_formula"):
        return result
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        if hasattr(first, "get_chemical_formula"):
            return first
    raise ValueError("Geometry source did not return an ASE Atoms object.")


def _coerce_atoms_list_result(result):
    if hasattr(result, "get_chemical_formula"):
        return [result]
    if isinstance(result, (list, tuple)):
        atoms_list = [entry for entry in result if hasattr(entry, "get_chemical_formula")]
        if atoms_list:
            return atoms_list
    raise ValueError("Geometry source did not return a list of ASE Atoms objects.")


def atoms_to_xyz(atoms) -> str:
    buf = StringIO()
    write(buf, atoms, format="xyz")
    return buf.getvalue()


def atoms_to_extxyz(atoms) -> str:
    buf = StringIO()
    write(buf, atoms, format="extxyz")
    return buf.getvalue()


def _finalize_metadata(atoms, metadata: dict) -> dict:
    symbols = [str(symbol) for symbol in atoms.get_chemical_symbols()]
    positions = [[round(float(x), 8), round(float(y), 8), round(float(z), 8)] for x, y, z in atoms.get_positions()]
    cell = [
        [round(float(value), 8) for value in row]
        for row in atoms.get_cell().array.tolist()
    ]
    pbc = [bool(flag) for flag in atoms.get_pbc().tolist()]
    digest_payload = json.dumps(
        {"symbols": symbols, "positions_A": positions},
        sort_keys=True,
        separators=(",", ":"),
    )
    data = dict(metadata)
    data["atom_count"] = len(atoms)
    data["formula"] = atoms.get_chemical_formula()
    data["cell_A"] = cell
    data["pbc"] = pbc
    data["is_periodic"] = bool(any(pbc))
    data["structure_sha256"] = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    return data


def structure_path_to_atoms(
    structure_path: str | Path,
    *,
    source: str = "existing_run",
    source_run_id: int | None = None,
    source_artifact_id: int | None = None,
    source_label: str = "",
):
    path = Path(structure_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Source structure file is not available locally: {path}")
    suffix = path.suffix.lower() or ".xyz"
    index = -1
    atoms = _coerce_atoms_result(read(str(path), index=index))
    metadata = _finalize_metadata(
        atoms,
        {
            "source": source,
            "source_path": str(path),
            "source_suffix": suffix,
            "frame_index": index if suffix in _INDEXED_STRUCTURE_SUFFIXES else None,
            "source_run_id": source_run_id,
            "source_artifact_id": source_artifact_id,
            "source_label": source_label,
        },
    )
    return atoms, metadata


def structure_path_to_atoms_frames(
    structure_path: str | Path,
    *,
    source: str = "existing_run",
    source_run_id: int | None = None,
    source_artifact_id: int | None = None,
    source_label: str = "",
):
    path = Path(structure_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Source structure file is not available locally: {path}")
    suffix = path.suffix.lower() or ".xyz"
    if suffix not in _BATCH_STRUCTURE_SUFFIXES:
        raise ValueError("Batch frame extraction requires a .traj or .db file.")

    frames = _coerce_atoms_list_result(read(str(path), index=":"))
    total = len(frames)
    if total == 0:
        raise ValueError(f"Source structure file contains no frames: {path}")

    result = []
    for frame_index, atoms in enumerate(frames):
        metadata = _finalize_metadata(
            atoms,
            {
                "source": source,
                "source_path": str(path),
                "source_suffix": suffix,
                "frame_index": frame_index,
                "frame_count": total,
                "source_run_id": source_run_id,
                "source_artifact_id": source_artifact_id,
                "source_label": source_label,
            },
        )
        result.append((atoms, metadata))
    return result


def structure_file_to_atoms(uploaded_file, *, traj_frame: int | None):
    if uploaded_file is None:
        raise ValueError("No structure file provided.")
    suffix = Path(uploaded_file.name).suffix.lower() or ".xyz"
    temp_path: str | None = None
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            for chunk in uploaded_file.chunks():
                handle.write(chunk)
            temp_path = handle.name

        index = (traj_frame if traj_frame is not None else -1) if suffix in _INDEXED_STRUCTURE_SUFFIXES else -1
        read_kwargs: dict[str, object] = {"index": index}
        if _uploaded_filename_forces_vasp_format(uploaded_file.name):
            read_kwargs["format"] = "vasp"
        atoms = read(temp_path, **read_kwargs)
        metadata = _finalize_metadata(
            atoms,
            {
                "source": "upload_file",
                "source_filename": uploaded_file.name,
                "source_suffix": suffix,
                "frame_index": index,
                "source_read_format": str(read_kwargs.get("format") or ""),
            },
        )
        return atoms, metadata
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def structure_file_to_atoms_frames(uploaded_file):
    if uploaded_file is None:
        raise ValueError("No structure file provided.")
    suffix = Path(uploaded_file.name).suffix.lower() or ".xyz"
    if suffix not in _BATCH_STRUCTURE_SUFFIXES:
        raise ValueError("Batch frame submission requires a .traj or .db file.")

    temp_path: str | None = None
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            for chunk in uploaded_file.chunks():
                handle.write(chunk)
            temp_path = handle.name

        frames = _coerce_atoms_list_result(read(temp_path, index=":"))
        total = len(frames)
        if total == 0:
            raise ValueError("Structure file contains no frames.")

        result = []
        for frame_index, atoms in enumerate(frames):
            metadata = _finalize_metadata(
                atoms,
                {
                    "source": "upload_file",
                    "source_filename": uploaded_file.name,
                    "source_suffix": suffix,
                    "frame_index": frame_index,
                    "frame_count": total,
                },
            )
            result.append((atoms, metadata))
        return result
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def structure_file_to_xyz_frames(uploaded_file) -> list[tuple[str, dict]]:
    frames = structure_file_to_atoms_frames(uploaded_file)
    return [(atoms_to_xyz(atoms), metadata) for atoms, metadata in frames]


def ase_molecule_to_atoms(formula: str):
    from ase.build import molecule

    formula_clean = str(formula or "").strip()
    if not formula_clean:
        raise ValueError("ASE molecule formula/name is empty.")
    atom_symbol = _single_atom_symbol(formula_clean)
    if atom_symbol is not None:
        atoms = Atoms(symbols=[atom_symbol], positions=[(0.0, 0.0, 0.0)])
    else:
        try:
            atoms = _coerce_atoms_result(molecule(formula_clean))
        except Exception as exc:
            raise ValueError(f"Unable to build ASE molecule '{formula_clean}': {exc}") from exc
    metadata = _finalize_metadata(
        atoms,
        {
            "source": "ase_molecule",
            "ase_molecule_formula": formula_clean,
        },
    )
    return atoms, metadata


def _single_atom_symbol(value: str) -> str | None:
    token = str(value or "").strip()
    if not token:
        return None
    if len(token) > 2:
        return None
    if not token[0].isupper():
        return None
    if len(token) == 2 and not token[1].islower():
        return None
    if token not in atomic_numbers:
        return None
    return token


def pubchem_to_atoms(*, cid: int | None, query: str | None):
    from ase.data.pubchem import pubchem_atoms_search

    query_text = str(query or "").strip()
    search_errors: list[str] = []

    if cid is not None:
        try:
            atoms = _coerce_atoms_result(pubchem_atoms_search(cid=int(cid)))
            metadata = _finalize_metadata(
                atoms,
                {
                    "source": "pubchem",
                    "pubchem_cid": int(cid),
                },
            )
            return atoms, metadata
        except Exception as exc:
            search_errors.append(f"CID search failed: {exc}")

    if query_text:
        # Try the common keyword signatures first, then positional fallback.
        for kwargs, label in (
            ({"name": query_text}, "name"),
            ({"formula": query_text}, "formula"),
        ):
            try:
                atoms = _coerce_atoms_result(pubchem_atoms_search(**kwargs))
                metadata = _finalize_metadata(
                    atoms,
                    {
                        "source": "pubchem",
                        "pubchem_query": query_text,
                        "pubchem_query_mode": label,
                    },
                )
                return atoms, metadata
            except TypeError:
                continue
            except Exception as exc:
                search_errors.append(f"{label} search failed: {exc}")

        try:
            atoms = _coerce_atoms_result(pubchem_atoms_search(query_text))
            metadata = _finalize_metadata(
                atoms,
                {
                    "source": "pubchem",
                    "pubchem_query": query_text,
                    "pubchem_query_mode": "positional",
                },
            )
            return atoms, metadata
        except Exception as exc:
            search_errors.append(f"positional search failed: {exc}")

        if query_text.isdigit() and cid is None:
            try:
                numeric_cid = int(query_text)
                atoms = _coerce_atoms_result(pubchem_atoms_search(cid=numeric_cid))
                metadata = _finalize_metadata(
                    atoms,
                    {
                        "source": "pubchem",
                        "pubchem_cid": numeric_cid,
                    },
                )
                return atoms, metadata
            except Exception as exc:
                search_errors.append(f"numeric CID search failed: {exc}")

    detail = "; ".join(search_errors) if search_errors else "No search input provided."
    raise ValueError(f"Unable to resolve geometry from PubChem. {detail}")


def structure_input_to_atoms(
    *,
    geometry_source: str,
    structure_file=None,
    traj_frame: int | None = None,
    ase_molecule_formula: str = "",
    pubchem_cid: int | None = None,
    pubchem_query: str = "",
):
    source = str(geometry_source or "").strip()
    if source == "upload_file":
        return structure_file_to_atoms(structure_file, traj_frame=traj_frame)
    if source == "ase_molecule":
        return ase_molecule_to_atoms(ase_molecule_formula)
    if source == "pubchem":
        return pubchem_to_atoms(cid=pubchem_cid, query=pubchem_query)
    raise ValueError(f"Unsupported geometry source: {source!r}")


def structure_input_to_xyz(
    *,
    geometry_source: str,
    structure_file=None,
    traj_frame: int | None = None,
    ase_molecule_formula: str = "",
    pubchem_cid: int | None = None,
    pubchem_query: str = "",
) -> tuple[str, dict]:
    atoms, metadata = structure_input_to_atoms(
        geometry_source=geometry_source,
        structure_file=structure_file,
        traj_frame=traj_frame,
        ase_molecule_formula=ase_molecule_formula,
        pubchem_cid=pubchem_cid,
        pubchem_query=pubchem_query,
    )
    return atoms_to_xyz(atoms), metadata


def structure_file_to_xyz(uploaded_file, *, traj_frame: int | None) -> tuple[str, dict]:
    atoms, metadata = structure_file_to_atoms(uploaded_file, traj_frame=traj_frame)
    return atoms_to_xyz(atoms), metadata


def structure_path_to_xyz_frames(
    structure_path: str | Path,
    *,
    source: str = "existing_run",
    source_run_id: int | None = None,
    source_artifact_id: int | None = None,
    source_label: str = "",
) -> list[tuple[str, dict]]:
    frames = structure_path_to_atoms_frames(
        structure_path,
        source=source,
        source_run_id=source_run_id,
        source_artifact_id=source_artifact_id,
        source_label=source_label,
    )
    return [(atoms_to_xyz(atoms), metadata) for atoms, metadata in frames]
