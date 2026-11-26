"""Transition State Ground-State Frequencies & Excited-State Gradient Analysis.

Performs:
1. Ground-state (S0) NumFreq using repository ORCA settings profile (CAM-B3LYP / 6-31g(d,p) + LANL2DZ ECP on metal, RIJCOSX).
2. Excited-state (S_iroot) EnGrad using matching TDDFT settings.
3. Projects F_S* onto v_TS (and MEP tangent tau) to evaluate whether excited state promotes bond breaking.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase.io import read
from ase.units import Bohr, Hartree

from cats.orca.run.orca_calculator import get_orca_input_header, get_orcablocks, get_orca_path

logger = logging.getLogger(__name__)


def _build_metal_basis_blocks(atoms) -> list[str]:
    atomic_numbers = set(atoms.get_atomic_numbers())
    basis_lines: list[str] = []
    for atomic_number in (79, 29):
        if atomic_number in atomic_numbers:
            basis_lines.extend(
                [
                    f'newGTO {atomic_number} "LANL2DZ" end',
                    f'newECP {atomic_number} "HayWadt" end',
                ]
            )
    if not basis_lines:
        return []
    return ["%BASIS", *basis_lines, "END"]


def parse_orca_engrad(engrad_path: Path | str) -> Tuple[float, np.ndarray]:
    """Parse orca.engrad file returning (energy_hartree, forces_eV_A).
    
    In orca.engrad:
    Forces F = -grad(E). Gradient is in Hartree/Bohr.
    Forces in eV/A = -grad * (Hartree / Bohr).
    """
    path = Path(engrad_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing engrad file: {path}")

    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    
    n_atoms = None
    energy_hartree = None
    gradient_vals = []
    
    mode = None
    for line in lines:
        if line.startswith("#"):
            if "Number of atoms" in line:
                mode = "natoms"
            elif "current total energy" in line:
                mode = "energy"
            elif "current gradient" in line:
                mode = "gradient"
            elif line.strip() == "#":
                pass
            else:
                mode = None
            continue
            
        if mode == "natoms" and n_atoms is None:
            n_atoms = int(line)
        elif mode == "energy" and energy_hartree is None:
            energy_hartree = float(line)
        elif mode == "gradient":
            gradient_vals.append(float(line))

    if n_atoms is None or energy_hartree is None or len(gradient_vals) != n_atoms * 3:
        raise ValueError(f"Malformed orca.engrad file {path}: natoms={n_atoms}, grad_len={len(gradient_vals)}")

    grad_matrix = np.array(gradient_vals).reshape((n_atoms, 3))
    # Forces in eV / Angstrom: F = -dE/dR
    forces_eV_A = -grad_matrix * (Hartree / Bohr)
    return energy_hartree, forces_eV_A


def parse_orca_hess(hess_path: Path | str) -> Dict[str, Any]:
    """Parse orca.hess file returning frequencies and normal mode eigenvectors."""
    path = Path(hess_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing hess file: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    
    # 1. Parse frequencies
    freq_match = re.search(r"\$vibrational_frequencies\s+(\d+)\s+(.*?)(?=\n\$|\Z)", content, re.DOTALL)
    if not freq_match:
        raise ValueError(f"Could not find $vibrational_frequencies in {path}")
    
    n_freqs = int(freq_match.group(1))
    freq_lines = freq_match.group(2).strip().splitlines()
    frequencies_cm1 = []
    for line in freq_lines:
        parts = line.split()
        if len(parts) >= 2:
            frequencies_cm1.append(float(parts[1]))

    # 2. Parse normal modes
    modes_match = re.search(r"\$normal_modes\s+(\d+)\s+(\d+)\s+(.*?)(?=\n\$|\Z)", content, re.DOTALL)
    if not modes_match:
        raise ValueError(f"Could not find $normal_modes in {path}")
        
    n_rows = int(modes_match.group(1))
    n_cols = int(modes_match.group(2))
    
    # Block matrix parser for ORCA normal modes
    mode_blocks = modes_match.group(3).strip().splitlines()
    normal_modes = np.zeros((n_rows, n_cols), dtype=float)
    
    col_indices = []
    for line in mode_blocks:
        parts = line.split()
        if not parts:
            continue
        # Header line with column numbers: e.g. "0 1 2 3 4"
        if all(p.isdigit() for p in parts):
            col_indices = [int(p) for p in parts]
            continue
        # Data line: row_idx val0 val1 ...
        if col_indices and parts[0].isdigit():
            row_idx = int(parts[0])
            for i, val in enumerate(parts[1:]):
                col_idx = col_indices[i]
                normal_modes[row_idx, col_idx] = float(val)

    n_atoms = n_rows // 3
    # Find imaginary frequencies (conventionally negative in ORCA)
    imaginary_indices = [i for i, f in enumerate(frequencies_cm1) if f < -1.0 or (f < 0 and abs(f) > 1e-2)]
    
    return {
        "frequencies_cm1": frequencies_cm1,
        "normal_modes": normal_modes,
        "n_atoms": n_atoms,
        "imaginary_indices": imaginary_indices,
    }


def _resolve_spin_multiplicity(atoms, spin_multiplicity: int | None = None, system_charge: int = 0) -> int:
    if spin_multiplicity is None:
        return (sum(atoms.get_atomic_numbers()) - int(system_charge)) % 2 + 1
    return int(spin_multiplicity)


def relax_ts_endpoints(atoms, hessian, calculator, output_dir, *, displacement_A=0.1,
                       fmax=0.02, steps=300, reaction_bond=None, product_bond_change=-1):
    """Relax both signs of one Cartesian normal mode; retain every evaluated optimiser frame."""
    from ase.io import write
    from ase.optimize import BFGS

    output_dir = Path(output_dir)
    indices = [i for i, frequency in enumerate(hessian["frequencies_cm1"]) if frequency < -10.0]
    if len(indices) != 1:
        return {"status": "blocked", "reason": "Requires exactly one frequency below -10 cm^-1", "mode_indices": indices}
    mode = np.asarray(hessian["normal_modes"], dtype=float)[:, indices[0]].reshape((-1, 3))
    if mode.shape != atoms.positions.shape or not np.isfinite(mode).all() or np.linalg.norm(mode) < 1e-12:
        raise ValueError("Invalid Cartesian imaginary mode")
    mode /= np.linalg.norm(mode)
    labels = ("minus", "plus")
    if reaction_bond is not None:
        i, j = reaction_bond
        derivative = np.dot(atoms.positions[j] - atoms.positions[i], mode[j] - mode[i])
        if abs(derivative) < 1e-10:
            raise ValueError("Imaginary mode has no signed component along reaction_bond")
        if derivative * product_bond_change < 0:
            mode *= -1
        labels = ("reactant", "product")
    centre = atoms.copy()
    centre.set_constraint()
    centre.calc = calculator(output_dir / "ts_energy")
    centre.get_forces()
    centre_energy = float(centre.get_potential_energy())
    branches, frames = {}, {}
    for sign, label in zip((-1, 1), labels):
        directory = output_dir / label
        directory.mkdir(parents=True, exist_ok=True)
        displaced = centre.copy()
        displaced.positions += sign * displacement_A * mode
        displaced.calc = calculator(directory / "orca")
        # Bound full-configuration displacement as well as each atomic step.
        with BFGS(displaced, trajectory=str(directory / "optimisation.traj"),
                  logfile=str(directory / "optimisation.log"),
                  maxstep=min(0.05, 0.25 / np.sqrt(len(atoms)))) as optimizer:
            converged = bool(optimizer.run(fmax=fmax, steps=steps))
        frames[label] = read(str(directory / "optimisation.traj"), ":")
        write(str(directory / "optimisation.extxyz"), frames[label])
        final = frames[label][-1]
        branches[label] = {
            "converged": converged, "frames": len(frames[label]), "sign": sign,
            "energy_eV": float(final.get_potential_energy()),
            "fmax_eV_A": float(np.linalg.norm(final.get_forces(), axis=1).max()),
            "trajectory": f"{label}/optimisation.extxyz",
        }
    separation = float(np.linalg.norm(frames[labels[0]][-1].positions - frames[labels[1]][-1].positions))
    valid = all(b["converged"] and b["energy_eV"] < centre_energy for b in branches.values()) and separation > 0.1
    path = list(reversed(frames[labels[0]])) + [centre] + frames[labels[1]]
    for index, image in enumerate(path):
        image.info.update(path_index=index, path_type="ts_linked_optimisation", is_ts=index == len(frames[labels[0]]))
    write(str(output_dir / "ts_linked_path.extxyz"), path)
    return {
        "status": "completed" if valid else "incomplete", "branches": branches,
        "path": "ts_linked_path.extxyz", "ts_index": len(frames[labels[0]]),
        "ts_energy_eV": centre_energy, "endpoint_separation_A": separation,
        "mode_index": indices[0], "frequency_cm1": float(hessian["frequencies_cm1"][indices[0]]),
        "signed_normalised_cartesian_mode": mode.tolist(), "displacement_A": displacement_A,
        "path_type": "optimisation_descent_not_irc_or_converged_neb",
        "minimum_frequencies_verified": False,
    }


def run_ts_vibrations_gradient(
    structure_path: str | Path,
    *,
    output_dir: str | Path = ".",
    iroot: int = 31,
    nroots: int = 40,
    spin_multiplicity: int | None = None,
    system_charge: int = 0,
    xc_functional: str = "CAM-B3LYP",
    basis_set: str = "6-31g(d,p)",
    integration_grid: str = "DefGrid2",
    scf_convergence: str = "TightSCF",
    nprocs: int = 48,
    memory_per_cpu_mb: int = 10500,
    hpc: str = "falcon",
    ground_state_only: bool = False,
    read_initial_wavefunction: bool = False,
    relax_endpoints: bool = False,
    displacement_A: float = 0.1,
    relaxation_fmax: float = 0.02,
    relaxation_steps: int = 300,
    reaction_bond: list[int] | None = None,
    product_bond_change: int = -1,
) -> Dict[str, Any]:
    """Execute S0 NumFreq and S_iroot EnGrad, followed by normal mode force projection."""
    if relax_endpoints and not ground_state_only:
        raise ValueError("Endpoint relaxation requires ground_state_only")
    if not 0 < displacement_A <= 0.3 or not 0 < relaxation_fmax <= 0.1 or not 1 <= relaxation_steps <= 2000:
        raise ValueError("Invalid downhill displacement, force target or step limit")
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    struct = Path(structure_path).resolve()
    if not struct.is_file():
        raise FileNotFoundError(f"Structure not found: {struct}")
        
    atoms = read(str(struct))
    n_atoms = len(atoms)
    if reaction_bond is not None and (len(reaction_bond) != 2 or any(type(i) is not int or not 0 <= i < n_atoms for i in reaction_bond) or reaction_bond[0] == reaction_bond[1]):
        raise ValueError("reaction_bond requires two distinct zero-based atom indices")
    if product_bond_change not in (-1, 1):
        raise ValueError("product_bond_change must be -1 or +1")
    mult = _resolve_spin_multiplicity(atoms, spin_multiplicity, system_charge)
    
    # Save clean xyz for ORCA
    xyz_filename = "coord.xyz"
    atoms.write(str(out_dir / xyz_filename))
    
    orca_cmd = get_orca_path(hpc=hpc) or shutil.which("orca") or "orca"
    logger.info("Using ORCA executable: %s", orca_cmd)
    
    metal_basis = _build_metal_basis_blocks(atoms)
    
    # -------------------------------------------------------------
    # 1. Ground State NumFreq (S0)
    # -------------------------------------------------------------
    numfreq_dir = out_dir / "s0_numfreq"
    numfreq_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / xyz_filename, numfreq_dir / xyz_filename)
    
    simple_numfreq = get_orca_input_header(
        xc=xc_functional,
        cosx="RIJCOSX",
        basis_set=basis_set,
        integration_grid=integration_grid,
        scf_convergence_criteria=scf_convergence,
        output_level="NormalPrint",
        task="NumFreq",
        scf_convergence_strategy="SlowConv",
    )
    blocks_numfreq = get_orcablocks(metal_basis, number_of_cpus=nprocs, memory_per_cpu=memory_per_cpu_mb)
    simple_numfreq_header = f"! {simple_numfreq.lstrip('!')}"
    if read_initial_wavefunction:
        seed = numfreq_dir / "initial.gbw"
        if not seed.is_file() or seed.stat().st_size == 0:
            raise FileNotFoundError("Missing staged NumFreq wavefunction")
        simple_numfreq_header += ' MOREAD\n%moinp "initial.gbw"'
    numfreq_inp_content = f"{simple_numfreq_header}\n{blocks_numfreq}\n* xyzfile {system_charge} {mult} {xyz_filename}\n"
    
    numfreq_inp = numfreq_dir / "numfreq.inp"
    numfreq_inp.write_text(numfreq_inp_content, encoding="utf-8")
    logger.info("Starting Ground-State (S0) NumFreq in %s...", numfreq_dir)
    
    numfreq_out = numfreq_dir / "numfreq.out"
    with open(numfreq_out, "w", encoding="utf-8") as out_fh:
        proc = subprocess.run([orca_cmd, "numfreq.inp"], cwd=numfreq_dir, stdout=out_fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"S0 NumFreq failed with exit code {proc.returncode}. See {numfreq_out}")

    if ground_state_only:
        hess_data = parse_orca_hess(numfreq_dir / "numfreq.hess")
        results = {
            "status": "completed", "ground_state_only": True,
            "all_frequencies_cm1": [float(f) for f in hess_data["frequencies_cm1"]],
            "imaginary_count": len(hess_data["imaginary_indices"]),
            "imaginary_indices": list(hess_data["imaginary_indices"]),
            "normal_modes": hess_data["normal_modes"].tolist(),
        }
        if relax_endpoints:
            from cats.orca.run.orca_calculator import get_orca_calculator

            simple = get_orca_input_header(
                xc=xc_functional, cosx="RIJCOSX", basis_set=basis_set,
                integration_grid=integration_grid, scf_convergence_criteria=scf_convergence,
                output_level="NormalPrint", task="EnGrad", scf_convergence_strategy="SlowConv",
            )
            def calculator(directory):
                return get_orca_calculator(
                    orcasimpleinput=simple.lstrip("! "), orcablocks=blocks_numfreq,
                    directory=str(directory), hpc=hpc, charge=system_charge, mult=mult,
                )

            results["linked_path"] = relax_ts_endpoints(
                atoms, hess_data, calculator, out_dir, displacement_A=displacement_A,
                fmax=relaxation_fmax, steps=relaxation_steps,
                reaction_bond=reaction_bond, product_bond_change=product_bond_change,
            )
            results["status"] = results["linked_path"]["status"]
        (out_dir / "ts_vibrations_gradient_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out_dir / "reaction_hub_summary.json").write_text(json.dumps({
            "status": results["status"], "run_type": "ts_vibrations_gradient",
            "tags": ["ts_vibrations"], "results": results,
            "artifacts": sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()),
        }, indent=2), encoding="utf-8")
        return results

    # -------------------------------------------------------------
    # 2. Excited State TDDFT EnGrad (S_iroot)
    # -------------------------------------------------------------
    engrad_dir = out_dir / f"s{iroot}_engrad"
    engrad_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / xyz_filename, engrad_dir / xyz_filename)
    
    simple_engrad = get_orca_input_header(
        xc=xc_functional,
        cosx="RIJCOSX",
        basis_set=basis_set,
        integration_grid=integration_grid,
        scf_convergence_criteria=scf_convergence,
        output_level="NormalPrint",
        task="EnGrad",
        scf_convergence_strategy="SlowConv",
    )
    tddft_blocks = [
        *metal_basis,
        "%TDDFT",
        "MaxIter 999",
        f"NRoots {nroots}",
        f"IRoot {iroot}",
        "FollowIRoot TRUE",
        "END",
    ]
    blocks_engrad = get_orcablocks(tddft_blocks, number_of_cpus=nprocs, memory_per_cpu=memory_per_cpu_mb)
    simple_engrad_header = f"! {simple_engrad.lstrip('!')}"
    engrad_inp_content = f"{simple_engrad_header}\n{blocks_engrad}\n* xyzfile {system_charge} {mult} {xyz_filename}\n"
    
    engrad_inp = engrad_dir / "engrad.inp"
    engrad_inp.write_text(engrad_inp_content, encoding="utf-8")
    logger.info("Starting Excited-State S%d EnGrad in %s...", iroot, engrad_dir)
    
    engrad_out = engrad_dir / "engrad.out"
    with open(engrad_out, "w", encoding="utf-8") as out_fh:
        proc_eng = subprocess.run([orca_cmd, "engrad.inp"], cwd=engrad_dir, stdout=out_fh, stderr=subprocess.STDOUT)
    if proc_eng.returncode != 0:
        raise RuntimeError(f"S{iroot} EnGrad failed with exit code {proc_eng.returncode}. See {engrad_out}")

    # -------------------------------------------------------------
    # 3. Parse and Project Forces
    # -------------------------------------------------------------
    hess_data = parse_orca_hess(numfreq_dir / "numfreq.hess")
    e_hartree, forces_s_iroot = parse_orca_engrad(engrad_dir / "engrad.engrad")
    
    frequencies = hess_data["frequencies_cm1"]
    imag_indices = hess_data["imaginary_indices"]
    
    ts_imag_freq = None
    v_ts = None
    if imag_indices:
        # Lowest frequency mode is transition vector
        ts_mode_idx = imag_indices[0]
        ts_imag_freq = frequencies[ts_mode_idx]
        v_ts_flat = hess_data["normal_modes"][:, ts_mode_idx]
        v_ts = v_ts_flat.reshape((n_atoms, 3))
        v_ts = v_ts / np.linalg.norm(v_ts)
    else:
        ts_mode_idx = 0
        ts_imag_freq = frequencies[0]
        v_ts_flat = hess_data["normal_modes"][:, 0]
        v_ts = v_ts_flat.reshape((n_atoms, 3))
        v_ts = v_ts / np.linalg.norm(v_ts)

    # Align direction of v_ts with forward bond breaking
    # Dynamically find C and adsorbed/reacting O
    c_indices = [i for i, a in enumerate(atoms) if a.symbol == "C"]
    o_indices = [i for i, a in enumerate(atoms) if a.symbol == "O"]
    if c_indices and o_indices:
        c_idx = c_indices[0]
        if len(o_indices) > 1:
            # Forming/adsorbed oxygen has larger distance to C at TS than intact CO bond
            o_sorted = sorted(o_indices, key=lambda idx: np.linalg.norm(atoms.positions[c_idx] - atoms.positions[idx]))
            o_ads_idx = o_sorted[-1]
        else:
            o_ads_idx = o_indices[0]
        r_co_bond = atoms.positions[c_idx] - atoms.positions[o_ads_idx]
        r_co_unit = r_co_bond / np.linalg.norm(r_co_bond)
        rel_vel = v_ts[c_idx] - v_ts[o_ads_idx]
        # Align v_ts with bond breaking: positive projection means C-O distance increases
        if np.dot(rel_vel, r_co_unit) < 0:
            v_ts = -v_ts
    elif n_atoms == 2:
        # Diatomic (e.g. O2 dissociation)
        r_unit = (atoms.positions[1] - atoms.positions[0]) / np.linalg.norm(atoms.positions[1] - atoms.positions[0])
        rel_vel = v_ts[1] - v_ts[0]
        if np.dot(rel_vel, r_unit) < 0:
            v_ts = -v_ts

    # Force projection: Pi = dot(F, v_ts)
    force_projection = float(np.sum(forces_s_iroot * v_ts))
    force_norm = float(np.linalg.norm(forces_s_iroot))
    favours_bond_breaking = bool(force_projection > 0)
    
    results = {
        "status": "completed",
        "iroot": int(iroot),
        "ts_imaginary_frequency_cm1": float(ts_imag_freq) if ts_imag_freq is not None else None,
        "all_frequencies_cm1": [float(f) for f in frequencies],
        "imaginary_count": len(imag_indices),
        "force_norm_eV_A": force_norm,
        "force_projection_eV_A": force_projection,
        "favours_bond_breaking": favours_bond_breaking,
        "atom_forces_eV_A": forces_s_iroot.tolist(),
        "ts_mode_vector": v_ts.tolist(),
    }
    
    res_path = out_dir / "ts_vibrations_gradient_results.json"
    res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    summary = {
        "status": "completed",
        "run_type": "ts_vibrations_gradient",
        "tags": ["ts_vibrations", "excited_state_gradient"],
        "results": results,
    }
    (out_dir / "reaction_hub_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("TS Analysis completed: imag_freq=%.1f cm^-1, proj=%.4f eV/A (favours_breaking=%s)",
                ts_imag_freq, force_projection, favours_bond_breaking)
    return results
