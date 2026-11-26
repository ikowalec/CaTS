#!/usr/bin/env python3
"""
Concurrent ORCA launcher using your cats helpers + ASE.

Focus of this helper:
- Evenly slice the Slurm-assigned CPUs across concurrent ORCA workers.
- Pin each worker (and therefore its MPI ranks) via ``taskset -c`` so CPU ranges never overlap.
- Keep the rest of the cats/ASE workflow untouched.
"""

from __future__ import annotations
import os, sys, tempfile, argparse
from pathlib import Path
from typing import List, Tuple, Any, Dict
import pandas as pd
from multiprocessing import Process, Queue

# --- your imports ---
from ase.optimize.bfgs import BFGS
from cats.orca.run.orca_calculator import get_orca_calculator, get_orcablocks, get_orca_input_header
from cats.orca.run.orca_restart import restart_optimisation, check_forces

ORCA_HPC = (os.environ.get("CATS_ORCA_HPC") or os.environ.get("CPLAS_ORCA_HPC", "falcon")).lower()
ORCA_VERSION = os.environ.get("CATS_ORCA_VERSION") or os.environ.get("CPLAS_ORCA_VERSION", "6_1_0")

# =====================  CPU HELPERS  =====================

def allowed_cpus() -> List[int]:
    """CPUs allowed by Slurm cgroup/affinity."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        import multiprocessing as mp
        return list(range(mp.cpu_count()))

def cpu_slices(cpus: List[int], k: int) -> List[List[int]]:
    """
    Split the allowed CPUs into ``k`` contiguous-ish slices.

    The allocation is even (difference at most 1 core) and deterministic so
    workers launched in successive batches re-use the same CPU lists.
    """
    if k < 1:
        raise ValueError("concurrency must be >= 1")
    cpus = sorted(cpus)
    total = len(cpus)
    if total < k:
        raise ValueError(f"Only {total} CPUs available for {k} slices")

    base, rem = divmod(total, k)
    slices: List[List[int]] = []
    start = 0
    for i in range(k):
        need = base + (1 if i < rem else 0)
        end = start + need
        slices.append(cpus[start:end])
        start = end
    return slices

# =====================  ENV SETUP PER WORKER  =====================

def set_worker_env(cpus: List[int]):
    """Threading & OpenMPI hygiene inside each worker, plus expose CPU list."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    # ORCA's OpenMPI shouldn't rebind outside our cpuset
    os.environ["OMPI_MCA_hwloc_base_binding_policy"] = "none"
    # Also tell OMPI the exact allowed CPU set (string like "0,2,4,6")
    os.environ["OMPI_MCA_hwloc_base_cpu_list"] = ",".join(map(str, cpus))
    # Favor on-node transports (harmless if unsupported)
    os.environ.setdefault("OMPI_MCA_btl", "vader,self")
    os.environ.setdefault("OMPI_MCA_pml", "ob1")

def prepare_instance_dirs(root: Path, idx: int):
    """Private scratch + MPI session dirs for a worker. Returns (workdir)."""
    work = (Path.cwd() / f"pack_work_{idx:03d}").absolute()
    tmpd = (root / f"idx{idx}" / "tmp")
    scrd = (root / f"idx{idx}" / "scr")
    ompi_old = (root / f"idx{idx}" / "ompi")
    ompi_new = (root / f"idx{idx}" / "prte")
    for p in (work, tmpd, scrd, ompi_old, ompi_new):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpd)
    os.environ["ORCA_SCRDIR"] = str(scrd)
    # per-instance MPI session dirs
    os.environ["OMPI_MCA_orte_tmpdir_base"] = str(ompi_old)
    os.environ["OMPI_MCA_prte_tmpdir_base"] = str(ompi_new)
    return work

# =====================  YOUR DOMAIN LOGIC  =====================

def build_samples() -> List[Tuple[str, Any]]:
    """Example sample set; replace with your real dataset."""
    from ase.build import molecule
    atoms = molecule("CO")
    return [(f"CO_{i}", atoms.copy()) for i in range(4)]

def make_orca_inputs(nprocs: int, mem_per_cpu_mb: int) -> Tuple[str, str]:
    """Return (orcasimpleinput, orcablocks) customized per worker."""
    orcasimpleinput = get_orca_input_header(
        scf_procedure="TRAH",
        xc="CAM-B3LYP",  # or "CAM-B3LYP"
        basis_set="6-31g(d,p)",
        integration_grid="DefGrid2",
        scf_convergence_criteria="TightSCF",
        output_level="NormalPrint",
        task="OPT",
        force_convergence_criteria="TightOpt",
        scf_convergence_strategy="SlowConv",
    )
    blocks_settings = [
        "%GEOM", "Convergence Normal", "Trust 0.5", "END",
        # add %BASIS overrides if needed
    ]
    # IMPORTANT: pass per-worker nprocs so %pal matches the CPU slice
    orcablocks = get_orcablocks(blocks_settings, nprocs, mem_per_cpu_mb)
    return orcasimpleinput, orcablocks

def routine(atoms, directory: str, orcasimpleinput: str, orcablocks: str,
            cpus: List[int]) -> float:
    """Your original QM routine; we only add the profile command wrapper."""
    atoms, results = restart_optimisation(atoms, directory, keep_temp=False)
    converged = check_forces(results, fmax=0.05)
    
    mult = 2 if (sum(atoms.get_atomic_numbers()) % 2) else 1

    if not converged:
        atoms.calc = get_orca_calculator(
            version=ORCA_VERSION,
            hpc=ORCA_HPC,
            orcasimpleinput=orcasimpleinput,
            orcablocks=orcablocks,
            charge=0,
            mult=mult,
            directory=directory,
        )
        # Wrap ORCA with taskset for explicit CPU pinning
        cpulist_str = ",".join(map(str, cpus))
        orca_cmd = " ".join(str(atoms.calc.profile.command).split())
        atoms.calc.profile.command = f"taskset -c {cpulist_str} {orca_cmd}"

        opt = BFGS(atoms, trajectory=f"{directory}/opt.traj", logfile=f"{directory}/opt.log")
        opt.run(fmax=0.05)
        e_final = atoms.get_potential_energy()
    else:
        e_final = results["energy"]
    return e_final

# =====================  WORKER & DRIVER  =====================

def worker(idx: int,
           sample: Tuple[str, Any],
           cpus: List[int],
           mem_per_cpu_mb: int,
           result_q: Queue):
    """One concurrent ORCA job."""
    name, atoms = sample
    try:
        if not cpus:
            raise RuntimeError("Empty CPU slice")

        # Pin this worker so ORCA's MPI ranks inherit it
        os.sched_setaffinity(0, set(cpus))
        set_worker_env(cpus)

        # private scratch for this worker
        jobid = os.environ.get("SLURM_JOB_ID", "nojob")
        scratch_root = (
            os.environ.get("TEMP_DIRECTORY")
            or os.environ.get("SLURM_TMPDIR")
            or tempfile.gettempdir()
        )
        base = Path(scratch_root)
        root = base / f"orca-pack-{jobid}"
        workdir = prepare_instance_dirs(root, idx)

        # per-worker input (%pal nprocs)
        nprocs = len(cpus)
        orcasimpleinput, orcablocks = make_orca_inputs(nprocs, mem_per_cpu_mb)

        # Run your routine in its own directory (name)
        directory = name
        Path(directory).mkdir(exist_ok=True)
        e_final = routine(atoms.copy(), directory, orcasimpleinput, orcablocks, cpus)

        slurm_task = os.environ.get("SLURM_PROCID") or os.getpid()
        result_q.put({
            "SLURM_TASK_ID": slurm_task,
            "structure_index": idx,
            "name": name,
            "energy": e_final,
            "nprocs": nprocs,
            "cpus": ",".join(map(str, cpus)),
            "workdir": str(workdir),
        })
    except Exception as e:
        result_q.put({
            "structure_index": idx,
            "name": name,
            "error": repr(e),
        })

def main():
    ap = argparse.ArgumentParser(description="Run multiple ORCA instances concurrently with taskset pinning.")
    ap.add_argument("--concurrency", "-K", type=int, default=None,
                    help="How many calculations to run at once. Default=min(#samples, #CPUs).")
    ap.add_argument("--mem-per-cpu", type=int, default=3000,
                    help="Memory per CPU in MB passed into orcablocks.")
    args = ap.parse_args()

    samples = build_samples()
    cpus_all = allowed_cpus()
    K = args.concurrency or min(len(samples), len(cpus_all))
    if K < 1:
        print("Nothing to do.", file=sys.stderr)
        return

    # run in batches of size K
    df_name = "configuration_info.csv"
    result_rows: List[Dict[str, Any]] = []

    for start in range(0, len(samples), K):
        batch = samples[start:start+K]
        cpu_pairs = cpu_slices(cpus_all, len(batch))
        q: Queue = Queue()
        procs: List[Process] = []

        for i, sample in enumerate(batch):
            cpulist = cpu_pairs[i]
            p = Process(target=worker,
                        args=(start + i, sample, cpulist, args.mem_per_cpu, q))
            p.start()
            procs.append(p)
            print(f"[launch] {sample[0]}  CPUs={cpulist}  nprocs={len(cpulist)}", flush=True)

        # collect results
        for _ in procs:
            row = q.get()
            result_rows.append(row)

        # join and check
        for p in procs:
            p.join()
            if p.exitcode != 0:
                print(f"[warn] worker exit code {p.exitcode}", file=sys.stderr)

        # write incremental CSV (parent-only; avoids race conditions)
        df = pd.DataFrame(result_rows)
        if Path(df_name).exists():
            df.to_csv(df_name, mode="a", index=False, header=False)
        else:
            df.to_csv(df_name, index=False)
        result_rows.clear()

    print("Job complete")

if __name__ == "__main__":
    main()


'''
Example Slurm
-------------
#!/bin/bash
#SBATCH -N 1 -c 40 -t 04:00:00 -J orca-concurrent
module purge
module load python  # or your venv
python concurrent_orca.py --concurrency 4 --mem-per-cpu 3000
'''
