from ase.db import connect
from sklearn.preprocessing import normalize
from similarity.rematch.get_rematch import get_cached_rematch_kernel
from similarity.run.evaluate import shave_slab
from ase import Atoms
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import time
from pathlib import Path
from collections.abc import Sequence as SequenceABC
from similarity.rematch.prescreen import split_by_1st_layer
from tqdm import tqdm


def _load_atoms_from_db(db_path, support_species, slab_shave_distance):
    row_ids = []
    atoms_list = []
    with connect(db_path) as db:
        for row in db.select():
            atoms = row.toatoms()
            if slab_shave_distance is not None:
                keep_mask = shave_slab(
                    atoms,
                    threshold=slab_shave_distance,
                    fix=support_species,
                )[0]
                atoms = atoms[keep_mask]
            row_ids.append(int(row.id))
            atoms_list.append(atoms)
    return row_ids, atoms_list


def _load_atoms_from_list(atoms_iterable, support_species, slab_shave_distance):
    row_ids = []
    atoms_list = []
    for idx, atoms in enumerate(atoms_iterable, start=1):
        if not isinstance(atoms, Atoms):
            raise TypeError(
                "atoms_list must contain ASE Atoms objects, "
                f"got {type(atoms).__name__} at position {idx}."
            )
        if slab_shave_distance is not None:
            keep_mask = shave_slab(
                atoms,
                threshold=slab_shave_distance,
                fix=support_species,
            )[0]
            atoms = atoms[keep_mask]
        row_ids.append(idx)
        atoms_list.append(atoms)
    return row_ids, atoms_list


def _load_atoms(source, support_species, slab_shave_distance):
    if isinstance(source, Atoms):
        return _load_atoms_from_list([source], support_species, slab_shave_distance)
    if isinstance(source, (str, Path)):
        return _load_atoms_from_db(source, support_species, slab_shave_distance)
    if isinstance(source, SequenceABC):
        return _load_atoms_from_list(source, support_species, slab_shave_distance)
    raise TypeError(
        "source must be a database path (str/Path) or a sequence of ASE Atoms objects."
    )


def _atoms_is_periodic(atoms: Atoms) -> bool:
    return bool(any(bool(flag) for flag in atoms.get_pbc()))


def _build_shared_soap_descriptor(atoms_list):
    from dscribe.descriptors import SOAP

    species = sorted(
        {
            symbol
            for atoms in atoms_list
            for symbol in atoms.get_chemical_symbols()
        }
    )
    if not species:
        raise ValueError("Cannot build SOAP descriptor: no species found in input structures.")

    periodic = any(_atoms_is_periodic(atoms) for atoms in atoms_list)
    r_cut = 6.0
    n_max = 8
    l_max = 6

    return SOAP(
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
    )


def get_soap(atoms, soap_descriptor=None):
    assert isinstance(atoms, Atoms), f"Must pass an Atoms object to get_soap, not {type(atoms)}."
    soap = soap_descriptor
    if soap is None:
        soap = _build_shared_soap_descriptor([atoms])
    return soap.create(atoms)


def suggest_similarity_threshold(
    db_path=None,
    atoms_list=None,
    reference_row_id=1,
    support_species=["Zn", "Ce", "Ti", "O"],
    slab_shave_distance=None,
    kernel_gamma=1.0,
    kernel_alpha=1.0,
    kernel_threshold=1e-6,
    exclude_self=True,
):
    """
    Compare one reference structure against all others and suggest a REMatch
    similarity threshold for duplicate filtering.

    Input source:
        - db_path: ASE database path
        - atoms_list: sequence of ASE Atoms objects

    Provide exactly one of db_path or atoms_list.

    Returns:
        pandas.DataFrame sorted by descending similarity with columns:
            - row_id
            - similarity

        Metadata is stored in df.attrs:
            - reference_row_id
            - suggested_threshold
            - similarity_spread
            - largest_gap
    """
    if db_path is None and atoms_list is None:
        raise ValueError(
            "No input provided. Pass db_path or atoms_list. "
            "If you shuffled atoms_list, note random.shuffle(...) returns None."
        )
    if db_path is not None and atoms_list is not None:
        raise ValueError("Provide only one input source: db_path or atoms_list.")

    source = atoms_list if atoms_list is not None else db_path
    row_ids, loaded_atoms = _load_atoms(
        source,
        support_species=support_species,
        slab_shave_distance=slab_shave_distance,
    )
    if not loaded_atoms:
        raise ValueError("No structures found in input source.")
    if reference_row_id not in row_ids:
        raise ValueError(
            f"Reference row id {reference_row_id} not found. Available ids: "
            f"{row_ids[:10]}{'...' if len(row_ids) > 10 else ''}"
        )

    soap_descriptor = _build_shared_soap_descriptor(loaded_atoms)
    soap_list = np.array(
        [normalize(get_soap(atoms, soap_descriptor=soap_descriptor)) for atoms in loaded_atoms],
        dtype=object,
    )
    rematch_kernel = get_cached_rematch_kernel(
        gamma=kernel_gamma, alpha=kernel_alpha, threshold=kernel_threshold
    )

    ref_pos = row_ids.index(reference_row_id)
    ref_soap = soap_list[ref_pos]

    similarities = []
    for row_id, soap in zip(row_ids, soap_list):
        sim = float(rematch_kernel.create([ref_soap, soap])[0, 1])
        if exclude_self and row_id == reference_row_id:
            continue
        similarities.append({"row_id": int(row_id), "similarity": sim})

    if not similarities:
        raise ValueError("No comparison structures available after filtering.")

    similarities_df = pd.DataFrame(similarities).sort_values(
        by="similarity",
        ascending=False,
        kind="mergesort",
    )
    similarities_df = similarities_df.reset_index(drop=True)
    values = similarities_df["similarity"].to_numpy(dtype=float)
    spread = {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }

    if len(values) == 1:
        suggested_threshold = float(values[0])
        largest_gap = {"index": 0, "gap": 0.0}
    else:
        search_end = min(len(values) - 1, max(3, int(np.ceil(0.5 * len(values)))))
        diffs = values[:search_end] - values[1 : search_end + 1]
        best_idx = int(np.argmax(diffs))
        best_gap = float(diffs[best_idx])
        suggested_threshold = float(0.5 * (values[best_idx] + values[best_idx + 1]))
        # Fallback for smooth distributions with no clear elbow.
        if best_gap < 1e-3:
            suggested_threshold = float(np.percentile(values, 95))
        largest_gap = {"index": best_idx, "gap": best_gap}
    similarities_df.attrs["reference_row_id"] = int(reference_row_id)
    similarities_df.attrs["suggested_threshold"] = float(suggested_threshold)
    similarities_df.attrs["similarity_spread"] = spread
    similarities_df.attrs["largest_gap"] = largest_gap
    return similarities_df


def deduplicate_atoms_by_similarity(
    atoms_list,
    similarity_threshold,
    support_species=["Zn", "Ce", "Ti", "O"],
    slab_shave_distance=None,
    kernel_gamma=1.0,
    kernel_alpha=1.0,
    kernel_threshold=1e-6,
    exclude_self=True,
):
    """
    Greedy de-duplication of a sequence of Atoms using REMatch similarity.

    Structures are processed in input order; a structure is kept only if its
    max similarity to already-kept structures is below `similarity_threshold`.

    Returns:
        unique_atoms, report_df
            - unique_atoms: list[Atoms] retained after filtering
            - report_df: pandas.DataFrame with per-structure decisions
    """
    _ = exclude_self  # kept for keyword compatibility with threshold suggestion API
    input_atoms = list(atoms_list)
    if not input_atoms:
        report_df = pd.DataFrame(
            columns=["row_id", "kept", "max_similarity_to_kept", "closest_kept_row_id"]
        )
        report_df.attrs["similarity_threshold"] = float(similarity_threshold)
        report_df.attrs["n_input"] = 0
        report_df.attrs["n_unique"] = 0
        return [], report_df

    _, processed_atoms = _load_atoms_from_list(
        input_atoms,
        support_species=support_species,
        slab_shave_distance=slab_shave_distance,
    )
    soap_descriptor = _build_shared_soap_descriptor(processed_atoms)
    soap_list = np.array(
        [normalize(get_soap(atoms, soap_descriptor=soap_descriptor)) for atoms in processed_atoms],
        dtype=object,
    )
    rematch_kernel = get_cached_rematch_kernel(
        gamma=kernel_gamma, alpha=kernel_alpha, threshold=kernel_threshold
    )

    kept_positions = []
    rows = []
    for pos, soap in enumerate(soap_list):
        row_id = pos + 1  # 1-based to match suggest_similarity_threshold
        max_similarity = None
        closest_kept_row_id = None

        for kept_pos in kept_positions:
            sim = float(rematch_kernel.create([soap, soap_list[kept_pos]])[0, 1])
            if max_similarity is None or sim > max_similarity:
                max_similarity = sim
                closest_kept_row_id = kept_pos + 1

        is_unique = max_similarity is None or max_similarity < similarity_threshold
        if is_unique:
            kept_positions.append(pos)

        rows.append(
            {
                "row_id": row_id,
                "kept": bool(is_unique),
                "max_similarity_to_kept": (
                    np.nan if max_similarity is None else float(max_similarity)
                ),
                "closest_kept_row_id": (
                    np.nan if closest_kept_row_id is None else int(closest_kept_row_id)
                ),
            }
        )

    report_df = pd.DataFrame(rows)
    unique_atoms = [input_atoms[pos] for pos in kept_positions]
    report_df.attrs["similarity_threshold"] = float(similarity_threshold)
    report_df.attrs["n_input"] = len(input_atoms)
    report_df.attrs["n_unique"] = len(unique_atoms)
    return unique_atoms, report_df


def filter_similar_structures(db_path,
                              support_species=["Zn", "Ce", "Ti", "O"],
                              slab_shave_distance=None,
                              similarity_threshold=0.9999, 
                              kernel_gamma=1.0, 
                              kernel_alpha=1.0, 
                              kernel_threshold=1e-6):
    # First pass: load and normalize SOAP descriptors (O(n * descriptor_size) memory).
    # Shaved typically 1 MB per structure
    soap_list = []
    _, atoms_list = _load_atoms_from_db(
        db_path,
        support_species=support_species,
        slab_shave_distance=slab_shave_distance,
    )
    
    soap_descriptor = _build_shared_soap_descriptor(atoms_list)

    def process_atoms_to_soap(atoms):
        '''Helper function to save memory in Parallel() output list'''
        return normalize(get_soap(atoms, soap_descriptor=soap_descriptor))


    soap_list = np.array(list(map(process_atoms_to_soap, atoms_list)), dtype=object)

    #soap_list = Parallel(n_jobs=n_jobs, backend=backend)(delayed(process_atoms_to_soap)(atoms) for atoms in atoms_list)
    #soap_list = np.array(soap_list)

    # clear memory
    del atoms_list

    print("SOAP memory usage is", soap_list.nbytes/1e6, "Mb.")

    rematch_kernel = get_cached_rematch_kernel(
        gamma=kernel_gamma, alpha=kernel_alpha, threshold=kernel_threshold
    )

    # Split the datasets to more manageable subsets 
    data_subsets = split_by_1st_layer(connect(db_path), layer_tolerance=1.0)
    # initialise the container for row ids to retain
    keep_indices = []
    
    for _,subset in enumerate(tqdm(data_subsets)):

        active_indices = np.array(data_subsets[subset])
        print(f"Starting subset {_} of {len(data_subsets)}, with size {len(active_indices)}.")
        s = time.time() # keep track of the time

        while active_indices.size != 0:

            # compare current_soap to the active subset
            targets = [soap_list[int(idx - 1)] for idx in active_indices]

            current_index = active_indices[0]
            current_soap = targets[0]
            
            def get_soap_score(soap):
                return round(rematch_kernel.create([current_soap, soap])[0,1], 8)

            scores = np.array(list(map(get_soap_score, targets)))

            # Find the indices to keep
            active_indices = active_indices[scores < similarity_threshold]
            keep_indices.append(int(current_index))
            
        # Print the timings
        print(f"{time.time()-s} s per subset.")
   
    return keep_indices


def get_unique_db(
    db_path,
    db_out_path,
    slab_shave_distance=None,
    similarity_threshold=0.9999,
    support_species=["Zn", "Ce", "Ti", "O"],
    kernel_gamma=1.0,
    kernel_alpha=1.0,
    kernel_threshold=1e-6,
    rematch_alpha=None,
):
    '''Take a database of Atoms objects and save unique structures
    in a new database'''
    if rematch_alpha is not None:
        # Backward-compatible alias.
        kernel_alpha = rematch_alpha

    keep_indices = filter_similar_structures(
        db_path,
        support_species=support_species,
        slab_shave_distance=slab_shave_distance,
        similarity_threshold=similarity_threshold,
        kernel_gamma=kernel_gamma,
        kernel_alpha=kernel_alpha,
        kernel_threshold=kernel_threshold,
    )
    print(f"Keeping {len(keep_indices)} unique structures.")
    print(keep_indices)
    

    '''MAKE SURE TO SAVE THE KEEP INDICES SOMEWHERE SAFE'''
    with connect(db_path) as db:
        with connect(db_out_path) as db_out:
            for i in keep_indices:
                row = db.get(i)  # ASE DB indices are 1-based
                db_out.write(
                    row.toatoms(),
                    data=dict(row.data),
                    key_value_pairs=dict(row.key_value_pairs),
                )

