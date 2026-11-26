import argparse
import os
import sys

# Make local project imports work when executed from the repo root.
sys.path.append(rf"{os.getcwd()}")

from similarity.rematch.memory_rematch import get_unique_db


def main():
    parser = argparse.ArgumentParser(
        description="Prune similar structures from an ASE database using SOAP REMatch."
    )
    parser.add_argument(
        "--db-path",
        default="scratch/optimized_structures.db", #"similarity/data/prescreened_structures.db",
        help="Input ASE database containing candidate structures.",
    )
    parser.add_argument(
        "--db-out-path",
        default="scratch/CuTiO2_unique.db",
        help="Output ASE database that will store unique structures.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.9950,
        help="SOAP REMatch similarity threshold above which structures are treated as duplicates.",
    )
    parser.add_argument(
        "--kernel-alpha",
        "--rematch-alpha",
        dest="kernel_alpha",
        type=float,
        default=0.1,
        help="Non-zero positive REMatch alpha weighting for local-environment matching.",
    )
    parser.add_argument(
        "--kernel-gamma",
        type=float,
        default=1.0,
        help="REMatch kernel gamma parameter.",
    )
    parser.add_argument(
        "--kernel-threshold",
        type=float,
        default=1e-6,
        help="REMatch kernel convergence threshold.",
    )
    parser.add_argument(
        "--support-species",
        default=["Zn", "Ce", "Ti", "O"],
        help="List of support species to consider.",
    )
    parser.add_argument(
        "--slab-shave-distance",
        type=float,
        default=3,
        help="Provide the z-coordinate for slab atoms to be retained. e.g. 3.0 is about 2 layers, -1 is cluster-only.",
    )

    args = parser.parse_args()

    get_unique_db(
        db_path=args.db_path,
        db_out_path=args.db_out_path,
        similarity_threshold=args.similarity_threshold,
        kernel_gamma=args.kernel_gamma,
        kernel_alpha=args.kernel_alpha,
        kernel_threshold=args.kernel_threshold,
        support_species=["Zn", "Ce", "Ti", "O"],
        slab_shave_distance=args.slab_shave_distance,
    )

if __name__ == "__main__":
    import time

    start = time.time()
    main()
    print(f"Finished after {time.time()-start} s.")

