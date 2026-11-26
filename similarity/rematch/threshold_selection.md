# REMatch Threshold Selection

This folder includes a helper to estimate a sensible **maximum similarity threshold** for duplicate filtering:

- Function: `suggest_similarity_threshold`
- Location: `similarity/rematch/memory_rematch.py`

## Purpose

Given one reference structure, compare it to all structures in an ASE database, inspect the similarity spread, and suggest a threshold above which structures are likely too similar.

## Function Signature

```python
suggest_similarity_threshold(
    db_path=None,
    atoms_list=None,
    reference_row_id=1,
    support_species=["Zn", "Ce", "Ti", "O"],
    slab_shave_distance=None,
    kernel_gamma=1.0,
    kernel_alpha=1.0,
    kernel_threshold=1e-6,
    exclude_self=True,
)
```

Use exactly one input source:
- `db_path` for ASE database on disk
- `atoms_list` for an in-memory sequence of ASE `Atoms`
- `slab_shave_distance` is optional (`None` disables shaving)

## Returned Data

The function returns a `pandas.DataFrame` sorted by descending similarity with columns:

- `row_id`
- `similarity`

Metadata is stored in `df.attrs`:

- `reference_row_id`: the chosen reference structure id
- `suggested_threshold`: proposed max similarity threshold
- `similarity_spread`: summary stats (`min`, `max`, `mean`, `median`, `std`, `p90`, `p95`, `p99`)
- `largest_gap`: index and gap size used by the elbow heuristic

## How The Suggestion Is Computed

1. Similarities to all structures are sorted descending.
2. The largest drop (gap) in the upper part of that ranked list is found.
3. Suggested threshold = midpoint across that largest gap.
4. If no clear gap is present (`gap < 1e-3`), fallback to `p95`.

## Example

```python
from similarity.rematch.memory_rematch import suggest_similarity_threshold, get_unique_db

df = suggest_similarity_threshold(
    db_path="scratch/optimized_structures.db",
    reference_row_id=1,
    slab_shave_distance=None,
    kernel_alpha=0.1,
)

threshold = df.attrs["suggested_threshold"]
print("Suggested threshold:", threshold)
print("Spread:", df.attrs["similarity_spread"])
print(df.head(10))

# Optional: use the suggestion for pruning
get_unique_db(
    db_path="scratch/optimized_structures.db",
    db_out_path="scratch/unique_structures.db",
    similarity_threshold=threshold,
    kernel_alpha=0.1,
    slab_shave_distance=None,
)
```

Example with in-memory structures:

```python
from ase.db import connect
from similarity.rematch.memory_rematch import suggest_similarity_threshold

with connect("scratch/optimized_structures.db") as db:
    atoms_list = [row.toatoms() for row in db.select()]

df = suggest_similarity_threshold(
    atoms_list=atoms_list,
    reference_row_id=1,  # 1-based index in the provided list
)
print(df.attrs["suggested_threshold"])
```

## Practical Note

For robust threshold choice, run the helper for several reference structures (not just one), then choose a conservative value near the low end of suggested thresholds.

For actual de-duplication once a threshold is chosen, use:

- `deduplicate_atoms_by_similarity(...)` in `similarity/rematch/memory_rematch.py`
