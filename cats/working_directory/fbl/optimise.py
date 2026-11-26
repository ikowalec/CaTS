import sys

from cats.working_directory.fbl import optimise_helpers as helpers


def main(argv: list[str]) -> int:
    
    output_path = helpers.run_optimisation(
        task_id=int(argv[1]),
        trajectory_file=argv[2],
        output_path=argv[3],
    )

    print(f"Job complete: results written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
