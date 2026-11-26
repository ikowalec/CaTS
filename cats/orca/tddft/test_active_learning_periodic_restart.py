import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil

from ase import Atoms
from ase.io.trajectory import Trajectory


def _load_restart_helpers():
    source_path = Path(__file__).with_name("active_learning.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "_archive_localal_state",
        "_training_frame_count",
        "_read_recent_training_frames",
        "_write_training_frames",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "Path": Path,
        "Trajectory": Trajectory,
        "datetime": datetime,
        "timezone": timezone,
        "shutil": shutil,
        "_LOCALAL_STATE_FILENAMES": (
            "predicted.traj",
            "predicted_evaluated.traj",
            "converged.traj",
            "initial_struc.traj",
            "ml_summary.txt",
            "ml_time.txt",
        ),
    }
    exec(compile(ast.Module(functions, []), source_path, "exec"), namespace)
    return namespace


def test_periodic_restart_archives_state_and_keeps_recent_frames(tmp_path):
    helpers = _load_restart_helpers()
    training_path = tmp_path / "evaluated.traj"
    assert helpers["_training_frame_count"](training_path) == 0
    frames = [Atoms("H", positions=[[float(index), 0.0, 0.0]]) for index in range(5)]
    helpers["_write_training_frames"](training_path, frames)
    (tmp_path / "ml_summary.txt").write_text("old state", encoding="utf-8")

    retained = helpers["_read_recent_training_frames"](training_path, 3)
    archive_dir = helpers["_archive_localal_state"](
        tmp_path,
        training_path,
        reason="periodic",
    )
    helpers["_write_training_frames"](training_path, retained)

    assert helpers["_training_frame_count"](training_path) == 3
    assert [frame.positions[0, 0] for frame in retained] == [2.0, 3.0, 4.0]
    assert helpers["_training_frame_count"](archive_dir / "evaluated.traj") == 5
    assert (archive_dir / "ml_summary.txt").read_text(encoding="utf-8") == "old state"
