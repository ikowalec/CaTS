"""Orbital-resolved Mulliken composition; never ground-state atomic charges."""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
from pathlib import Path

from cats.multiwfn.tools import run_multiwfn, save_routine_config, set_multiwfn_path


def mulliken_orbital_composition(path, *, spin="default", role="hole", output_prefix="nto_hole", runner=run_multiwfn):
    """Analyse the leading hole or electron in an exported NTO wavefunction.

    Retain every signed AO contribution (compthres=-1), validate basis coverage
    and the reported orbital index, then aggregate by atom and shell angular
    momentum. Percentages describe one normalised orbital, not electron charges.
    """
    source = Path(path).resolve()
    text = source.read_text()
    if spin not in {"default", "alpha", "beta"}:
        raise ValueError("spin must be default, alpha, or beta")
    if role not in {"hole", "electron"}:
        raise ValueError("role must be hole or electron")
    blocks = re.split(r"(?m)^\s*Index=", text)[1:]
    orbitals = []
    for block in blocks:
        index = int(block.split()[0])
        kind = int(re.search(r"(?m)^Type=\s*(\d+)", block)[1])
        occ = float(re.search(r"(?m)^Occ=\s*(\S+)", block)[1])
        if (occ > 0 if role == "hole" else occ == 0) and (kind == {"alpha": 1, "beta": 2}.get(spin, kind)):
            orbitals.append((index, kind))
    if not orbitals:
        raise ValueError("No orbital for requested spin and role")
    # Default is restricted, or alpha for an unrestricted NTO wavefunction.
    if spin == "default":
        kind = min(kind for _, kind in orbitals)
        orbitals = [(i, k) for i, k in orbitals if k == kind]
    index, kind = (max if role == "hole" else min)(orbitals)
    nbasis = int(re.search(r"(?m)^Nbasis=\s*(\d+)", text)[1])
    same_spin = [block for block in blocks if int(re.search(r"(?m)^Type=\s*(\d+)", block)[1]) == kind]
    spin_local_index = next(i for i, block in enumerate(same_spin, 1) if int(block.split()[0]) == index)
    # Multiwfn pads unrestricted blocks to Nbasis on loading, unlike file indices.
    native_index = spin_local_index + (nbasis if kind == 2 else 0)
    selector = {("hole",0):"h",("electron",0):"l",("hole",1):"ha",
                ("electron",1):"la",("hole",2):"hb",("electron",2):"lb"}[role,kind]
    shell_types = list(map(int, text.split('$Shell types', 1)[1].split('$', 1)[0].split()))
    shell_centres = list(map(int, text.split('$Shell centers', 1)[1].split('$', 1)[0].split()))
    config, log = Path(f'{output_prefix}_config.txt'), Path(f'{output_prefix}_log.txt')
    save_routine_config([str(source), '8', '1', selector, '0', '-10', 'q'], str(config))
    ini = Path('settings.ini')
    original = ini.read_bytes() if ini.exists() else None
    try:
        if original is None:
            set_multiwfn_path()
            install = Path(os.environ['Multiwfnpath']) / 'settings.ini'
            shutil.copy2(install, ini)
        settings = ini.read_text()
        settings, count = re.subn(r'(?m)^\s*compthres\s*=\s*[^\n]*', 'compthres= -1.0', settings)
        if count != 1:
            raise ValueError('Expected exactly one compthres setting')
        ini.write_text(settings)
        runner(str(config), str(log))
    finally:
        if original is None:
            ini.unlink(missing_ok=True)
        else:
            ini.write_bytes(original)
    report = log.read_text(errors='replace')
    if not re.search(r'Orbital:\s*'+str(native_index)+r'\b', report):
        raise ValueError('Multiwfn did not report the requested orbital')
    expected_type = {0:'Alpha&Beta',1:'Alpha',2:'Beta'}[kind]
    if not re.search(r'Orbital:\s*'+str(native_index)+r'\b[^\n]*Type:\s*'+re.escape(expected_type)+r'\s*(?:\n|$)', report):
        raise ValueError('Multiwfn did not report the requested spin')
    number = r'[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?'
    rows = re.findall(r'^\s*(\d+)\s+(\S+)\s+(\d+)\(([A-Za-z]+)\s*\)\s+(\d+)\s+('+number+r')\s*%\s+('+number+r')\s*%\s+('+number+r')\s*%', report, re.M)
    if len(rows) != nbasis or {int(row[0]) for row in rows} != set(range(1, nbasis+1)):
        raise ValueError(f'Incomplete AO composition: {len(rows)} rows, expected {nbasis}; retained raw log')
    atoms, basis_rows = {}, []
    for basis, label, atom, symbol, shell, local, cross, total in rows:
        atom, shell = int(atom), int(shell)
        if shell_centres[shell-1] != atom:
            raise ValueError('AO shell/atom lineage mismatch')
        angular = abs(shell_types[shell-1])
        channel = ['s','p','d','f','g','h'][angular]
        values = atoms.setdefault(str(atom), {'symbol':symbol, 's':0., 'p':0., 'd':0.})
        values[channel] = values.get(channel, 0.) + float(total)
        basis_rows.append({'basis_index':int(basis), 'atom_index':atom, 'shell_index':shell,
                           'basis_label':label, 'angular_channel':channel,
                           'local_pct':float(local),'cross_pct':float(cross),'total_pct':float(total)})
    total = sum(row['total_pct'] for row in basis_rows)
    if abs(total-100.) > 0.05:
        raise ValueError(f'Mulliken orbital composition sums to {total}%, not 100%')
    result = {'method':'Mulliken orbital composition (Multiwfn 8-1)', 'source_wavefunction':str(source),
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'orbital_index':index, 'orbital_type':kind, 'spin':spin,
              'native_orbital_index':native_index,'spin_local_orbital_index':spin_local_index,
              'multiwfn_selector':selector,'role':role,
              'units':'percent of one normalised orbital; signed Mulliken contributions, not atomic charges',
              'total_pct':total, 'atoms':atoms,'basis_rows':basis_rows,'raw_log':str(log),
              'config':str(config),'composition_threshold':-1.0}
    Path(f'{output_prefix}.json').write_text(json.dumps(result, indent=2))
    return result
