from __future__ import annotations

import runpy
import sys

from similarity.workflows.mace_pool import (
    initialize_mpi_worker_pool,
    shutdown_mpi_workers,
)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m similarity.workflows.mace_mpi_entry SCRIPT [ARGS...]")
    if not initialize_mpi_worker_pool():
        return

    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        shutdown_mpi_workers()


if __name__ == "__main__":
    main()
