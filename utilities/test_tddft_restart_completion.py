import ast
from pathlib import Path


def _load_count_completed_iterations():
    source_path = (
        Path(__file__).parents[1]
        / "cats"
        / "working_directory"
        / "tddft_active_learning_workflow.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "count_completed_iterations"
    )
    namespace = {"QMWorkflowBackend": object}
    exec(compile(ast.Module([function], []), source_path, "exec"), namespace)
    return namespace


def test_valid_output_without_completion_marker_is_not_complete():
    namespace = _load_count_completed_iterations()
    complete = False

    class FakePath:
        def __init__(self, name):
            self.name = name

        def __truediv__(self, name):
            return FakePath(name)

        def exists(self):
            return complete if self.name == "iteration_complete" else True

    opt_dir = FakePath("iter_1")

    class Backend:
        def output_path(self, directory):
            return directory / "orca.out"

        def validate_output(self, path, warn=False):
            return path.exists()

    namespace.update(
        {
            "SHARED_TRAINING_DIR": FakePath("shared"),
            "IROOT_SCAN_LOG_FILENAME": "scan.csv",
            "ITERATION_COMPLETE_FILENAME": "iteration_complete",
            "_workflow_backend_or_default": lambda backend: backend,
            "load_scan_log": lambda path: {},
            "_resolve_opt_dir_for_mode": lambda i, **kwargs: opt_dir,
            "_has_tddft_state_tag": lambda *args, **kwargs: False,
            "_tddft_scan_geom_dirs": lambda i: [],
        }
    )
    function = namespace["count_completed_iterations"]

    assert function(1, workflow_backend=Backend()) == 0
    complete = True
    assert function(1, workflow_backend=Backend()) == 1
