import sys, os
sys.path.append(os.getcwd())

from similarity.rematch.memory_rematch import suggest_similarity_threshold
from ase.io import read

db_path = "similarity/data/structures.db"
atoms_list = read(db_path, index=":")  # Load all structures from the database

import random
random.shuffle(atoms_list)  # Shuffle in-place; random.shuffle returns None

# Run a couple of times to see how the suggested threshold changes with different random samples

df = suggest_similarity_threshold(
    atoms_list=atoms_list,
    reference_row_id=1,
    support_species=["Zn", "Ce", "Ti", "O"],
    slab_shave_distance=None,
    kernel_gamma=1.0,
    kernel_alpha=1.0,
    kernel_threshold=1e-6,
    exclude_self=True,
)

threshold = df.attrs["suggested_threshold"]
print(f"Suggested similarity threshold: {threshold}")
print(df.head())

from ase.visualize import view

top_ids = df.head(n=20)["row_id"].astype(int).tolist()
top_atoms = [atoms_list[row_id - 1] for row_id in top_ids]  # row_id is 1-based
view(top_atoms)
