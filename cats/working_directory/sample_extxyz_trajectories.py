from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from ase import Atoms
from ase.io import iread, write

from evaluated_traj_to_xyz import (
    DEFAULT_SIMILARITY_THRESHOLD,
    selected_indices,
    trajectory_reference_energy_per_atom,
)


def trajectories(
    images: Iterable[Atoms], frame_key: str = "source_frame", start_frame: int = 0
) -> Iterator[list[Atoms]]:
    current: list[Atoms] = []
    seen_start = False
    for atoms in images:
        value = atoms.info.get(frame_key)
        try:
            frame = None if value is None else int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-integer {frame_key!r}: {value!r}") from exc
        if current and frame == start_frame and seen_start:
            yield current
            current = []
        current.append(atoms)
        seen_start |= frame == start_frame
    if current:
        yield current


SUPPORTED_EXTENSIONS: set[str] = {".xyz", ".extxyz", ".traj", ".db"}


def _read_images(path: Path) -> list[Atoms]:
    suffix = path.suffix.lower()
    if suffix == ".traj":
        from ase.io.trajectory import Trajectory

        return [a.copy() for a in Trajectory(str(path), mode="r")]
    if suffix == ".db":
        import ase.db

        return [row.toatoms() for row in ase.db.connect(str(path)).select()]
    try:
        return list(iread(str(path), format="extxyz"))
    except Exception:
        return list(iread(str(path)))


def extxyz_files(inputs: Iterable[Path], output_path: Path) -> list[Path]:
    output_path = output_path.resolve()
    files = {
        candidate.resolve()
        for path in inputs
        for candidate in ([path] if path.is_file() else path.rglob("*"))
        if candidate.is_file()
        and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        and candidate.resolve() != output_path
    }
    return sorted(files)


def sample(
    input_paths: Iterable[Path],
    output_path: Path,
    audit_path: Path,
    frame_key: str,
    start_frame: int,
    similarity_threshold: float | str | None = "dynamic",
):
    rows = []
    written = 0
    trajectory_count = 0
    files = extxyz_files(input_paths, output_path)
    for input_path in files:
        for file_trajectory, images in enumerate(
            trajectories(_read_images(input_path), frame_key, start_frame), start=1
        ):
            trajectory_count += 1
            reference = trajectory_reference_energy_per_atom(images)
            indices, rejected = selected_indices(
                images, reference, similarity_threshold=similarity_threshold
            )
            selected = [images[index] for index in indices]
            if selected:
                write(output_path, selected, format="extxyz", append=written > 0)
                written += len(selected)
            rows.append(
                (
                    input_path,
                    file_trajectory,
                    len(images),
                    len(selected),
                    " ".join(map(str, indices)),
                    ";".join(rejected),
                )
            )

    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "trajectory", "input_frames", "kept_frames", "kept_indices", "rejected"]
        )
        writer.writerows(rows)
    return len(files), trajectory_count, written


def _self_check() -> None:
    images = []
    for frame in (0, 1, 2, 0, 1, 2, 0):
        atoms = Atoms("H", positions=[[frame, 0, 0]])
        atoms.info["source_frame"] = frame
        images.append(atoms)
    assert [len(group) for group in trajectories(images)] == [3, 3, 1]
    assert [len(group) for group in trajectories([Atoms("H"), Atoms("H")])] == [2]


def main() -> None:
    if "--self-check" in sys.argv:
        _self_check()
        print("self-check passed")
        return
    parser = argparse.ArgumentParser(
        description="Sample concatenated extxyz trajectories using ReactionHub training-set filters."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="extxyz files and/or directories")
    parser.add_argument("-o", "--output", type=Path, default=Path("sampled.extxyz"))
    parser.add_argument("--audit", type=Path, default=Path("sampled_audit.csv"))
    parser.add_argument("--frame-key", default="source_frame")
    parser.add_argument("--start-frame", type=int, default=0)
    def _parse_threshold(val: str) -> float | str:
        s = val.strip().lower()
        if s in ("dynamic", "auto", "none"):
            return "dynamic"
        return float(val)

    parser.add_argument(
        "--similarity-threshold",
        type=_parse_threshold,
        default="dynamic",
        help="Max SOAP REMatch similarity cutoff for intermediate frame deduplication, or 'dynamic' (default: dynamic)",
    )
    args = parser.parse_args()

    files_seen, trajectories_seen, frames_written = sample(
        args.inputs,
        args.output,
        args.audit,
        args.frame_key,
        args.start_frame,
        similarity_threshold=args.similarity_threshold,
    )
    if not frames_written:
        raise SystemExit(
            f"No sensible frames found in {trajectories_seen} trajectories from {files_seen} files"
        )
    print(
        f"Wrote {frames_written} frames from {trajectories_seen} trajectories in {files_seen} files "
        f"to {args.output}; audit: {args.audit}"
    )


if __name__ == "__main__":
    main()
