from cover_np_functions import get_outer_shell_indices
from ase.io import read

# Au63
# atoms = read(r"<research-data-root>\2024\10.October\orca.xyz")
# Au55
atoms = read(
    r"<research-data-root>\2024\7.July\Au55\orca_trj.xyz",
    format="xyz",
)
outer_indices = get_outer_shell_indices(atoms)
print(outer_indices)
