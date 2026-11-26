from __future__ import annotations

import re
from pathlib import Path


def get_charges(atoms, path_to_property, method="loewdin"):

    if method == "loewdin":
        with open(path_to_property, 'r') as f:
            lines = f.readlines()
            start_index = None
            data_lines = [] 

            for i, line in enumerate(lines):
                if "$Loewdin_Population_Analysis" in line:
                    start_index = i
                    continue
                
                if start_index is not None:
                    # skip 9 + n atoms lines after the header
                    if i < start_index + len(atoms) + 10:
                        continue
            
                    # append n lines based on number of atoms
                    if len(data_lines) < len(atoms):
                        data_lines.append(line.strip())
                    else:
                        break
        
        assert len(data_lines) == len(atoms), "Number of data lines does not match number of atoms"
        print(data_lines)
        charges = [line.split()[1] for line in data_lines]
        charges = [float(charge) for charge in charges]

        atoms.set_initial_charges(charges)
        
        return atoms


def _charge_header_for_method(method: str) -> str:
    method_lc = method.strip().lower()
    if method_lc == "loewdin":
        return "LOEWDIN ATOMIC CHARGES"
    if method_lc == "mulliken":
        return "MULLIKEN ATOMIC CHARGES"
    raise ValueError(f"Unsupported charge method '{method}'. Choose 'loewdin' or 'mulliken'.")


def _parse_orca_output_charges(path_to_output: str | Path, method: str = "mulliken") -> list[float]:
    """
    Parse atomic charges from an ORCA output file.

    Uses the last charge block for the selected method to reflect the final
    reported population analysis in the output.
    """

    header = _charge_header_for_method(method)
    # ORCA prints either "Cu:" or "O :" depending on formatting width.
    # Accept optional whitespace between element symbol and ":".
    charge_line = re.compile(
        r"^\s*(\d+)\s+([A-Za-z]{1,3})\s*:\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)"
    )

    path = Path(path_to_output)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header_indices = [idx for idx, line in enumerate(lines) if header in line]
    if not header_indices:
        raise ValueError(f"Charge section '{header}' not found in {path}.")

    start = header_indices[-1] + 1
    parsed_rows: list[tuple[int, float]] = []
    in_block = False
    for line in lines[start:]:
        match = charge_line.match(line)
        if match:
            in_block = True
            atom_index = int(match.group(1))
            charge = float(match.group(3))
            parsed_rows.append((atom_index, charge))
            continue
        if in_block:
            break

    if not parsed_rows:
        raise ValueError(f"No charge data parsed from '{header}' in {path}.")
    parsed_rows.sort(key=lambda row: row[0])
    return [charge for _, charge in parsed_rows]


def get_charges_from_orca_output(atoms, path_to_output: str | Path, method: str = "mulliken"):
    """
    Attach charges parsed from ORCA output to an ASE Atoms object.

    Falls back from Loewdin to Mulliken when Loewdin is unavailable.
    """

    methods = [method]
    if method.strip().lower() == "loewdin":
        methods.append("mulliken")

    last_error: Exception | None = None
    for candidate in methods:
        try:
            charges = _parse_orca_output_charges(path_to_output, method=candidate)
            if len(charges) != len(atoms):
                raise ValueError(
                    f"Parsed {len(charges)} charges from {path_to_output}, expected {len(atoms)}."
                )
            atoms.set_initial_charges(charges)
            atoms.info["charge_method"] = candidate
            return atoms
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    return atoms
    

def test_get_charges():
    
    from ase.visualize import view
    from ase.io import read

    atoms = read(r"<repo-root>\.data\Au63_O2_chemisorbed\evaluated.traj")
    path_to_property = r"<repo-root>\.data\Au63_O2_chemisorbed\orca.property.txt"

    atoms = get_charges(atoms, path_to_property, method="loewdin")

    view(atoms)

if __name__ == "__main__":
    test_get_charges()
