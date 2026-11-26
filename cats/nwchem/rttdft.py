from __future__ import annotations

from pathlib import Path
from ase import Atoms

def nwchem_geometry_block(atoms: Atoms, name: str = "system") -> str:
    # NWChem RT-TDDFT example requires a named geometry. 
    lines = [f'geometry "{name}" units angstrom nocenter noautoz noautosym']
    for sym, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        lines.append(f"  {sym:2s} {x:14.8f} {y:14.8f} {z:14.8f}")
    lines.append("end")
    return "\n".join(lines)

def get_multiplicity(atoms):
    electrons = sum(atoms.get_atomic_numbers())
    if electrons % 2:
        return 2
    else:
        return 1

def write_ground_state(atoms: Atoms, *, basis: str, movecs: str) -> str:
    mult = get_multiplicity(atoms)
    return f"""
title "GS for RT-TDDFT"
echo

start gs

{nwchem_geometry_block(atoms, name="system")}
set geometry "system"

basis
  * library {basis}
end

dft
  xc xpbe96 1.0 xcampbe96 -0.25 cpbe96 1.0 srhfexch 0.25
  cam 0.11 cam_alpha 0.0 cam_beta 1.0
  mult {mult}
  vectors output "{movecs}"
end

task dft energy
"""

def write_kick(atoms: Atoms, *, basis: str, movecs: str,
               pol: str, tag: str, tmax: float = 200.0, dt: float = 0.2,
               emax: float = 1.0e-4) -> str:
    # Mirrors the official “kick” concept (delta field, x/y/z polarization, tag).
    mult = get_multiplicity(atoms)
    return f"""
title "RT-TDDFT kick {pol}"
echo

start {tag}

{nwchem_geometry_block(atoms, name="system")}
set geometry "system"

basis
  * library {basis}
end

dft
  xc xpbe96 1.0 xcampbe96 -0.25 cpbe96 1.0 srhfexch 0.25
  cam 0.11 cam_alpha 0.0 cam_beta 1.0
  mult {mult}
end

unset rt_tddft:*
rt_tddft
  tmax {tmax}
  dt {dt}

  tag "{tag}"
  load vectors "{movecs}"

  # Avoid “no print options” crash reports by being explicit.
  print dipole field energy s2 charge

  field "kick"
    type delta
    polarization {pol}
    max {emax}
  end

  excite "system" with "kick"
end

task dft rt_tddft
"""

import os

def run_nwchem(input_text: str, workdir: Path, name: str, nwchem_cmd: str = "nwchem") -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    input_file = f"{name}.nwi"
    out = f"{name}.nwo"
    inp = workdir / input_file
    inp.write_text(input_text)
    final_command = f"cd {workdir} && {nwchem_cmd} {input_file} > {out}"
    print(final_command)
    os.system(final_command)

# ---- Example usage ----
from ase.io import read
atoms = read("atoms.traj")

work = Path("Cu13")
basis = "def2-svp"
movecs = "gs.movecs"

# 1) Ground state
from cats.nwchem.run import get_nwchem_command
nodes = 1
nodes_gs = 3
cores_per_node = 128
cores = nodes * cores_per_node
memory_per_core = 1500 # mb

nwchem_cmd_gs = " ".join(get_nwchem_command("falcon", nodes_gs, cores_per_node, cores).split()[:-3])
nwchem_cmd = " ".join(get_nwchem_command("falcon", nodes, cores_per_node, cores).split()[:-3])

run_nwchem(write_ground_state(atoms, basis=basis, movecs=movecs), work, "00_gs", nwchem_cmd_gs)

def job_wrapper(pol):
    tag = f"10_kick_{pol}"
    run_nwchem(write_kick(atoms, basis=basis, movecs=movecs, pol=pol, tag=tag), work, pol, nwchem_cmd)
    return None

# 2–4) Kicks
pols = ["x", "y", "z"]

from multiprocessing import Pool

with Pool(processes=3) as pool:
        pool.starmap(job_wrapper, pols)
