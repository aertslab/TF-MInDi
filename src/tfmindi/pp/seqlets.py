"""Seqlet extraction and motif similarity preprocessing functions for TF-MInDi."""

from __future__ import annotations

import math
from typing import Any

import numba
import numpy as np
import pandas as pd
from anndata import AnnData
from memelite import tomtom
from scipy import sparse
from tqdm import tqdm


def get_example_idx(adata: AnnData, seqlet_idx: int) -> int:
    """
    Get the index for an example associated with a seqlet.

    Parameters
    ----------
    adata
        AnnData object containing seqlet data with unique examples storage
    seqlet_idx
        Index of the seqlet (row index in adata.obs)

    Returns
    -------
    index integer value

    """
    if "unique_examples" not in adata.uns:
        raise ValueError("No unique_examples found in adata.uns. Use the new storage format.")
    if "example_oh_idx" not in adata.obs.columns:
        raise ValueError("No example_oh_idx found in adata.obs. Use the new storage format.")

    example_idx = int(adata.obs["example_oh_idx"].iloc[seqlet_idx])

    return example_idx


def get_example_oh(adata: AnnData, seqlet_idx: int) -> np.ndarray:
    """
    Get the one-hot sequence for an example associated with a seqlet.

    Parameters
    ----------
    adata
        AnnData object containing seqlet data with unique examples storage
    seqlet_idx
        Index of the seqlet (row index in adata.obs)

    Returns
    -------
    One-hot sequence array with shape (4, sequence_length)
    """
    if "unique_examples" not in adata.uns:
        raise ValueError("No unique_examples found in adata.uns. Use the new storage format.")
    if "example_oh_idx" not in adata.obs.columns:
        raise ValueError("No example_oh_idx found in adata.obs. Use the new storage format.")

    example_idx = get_example_idx(adata, seqlet_idx)
    return adata.uns["unique_examples"]["oh"][example_idx]


def get_example_contrib(adata: AnnData, seqlet_idx: int) -> np.ndarray:
    """
    Get the contribution scores for an example associated with a seqlet.

    Parameters
    ----------
    adata
        AnnData object containing seqlet data with unique examples storage
    seqlet_idx
        Index of the seqlet (row index in adata.obs)

    Returns
    -------
    Contribution scores array with shape (4, sequence_length)
    """
    if "unique_examples" not in adata.uns:
        raise ValueError("No unique_examples found in adata.uns. Use the new storage format.")
    if "example_contrib_idx" not in adata.obs.columns:
        raise ValueError("No example_contrib_idx found in adata.obs. Use the new storage format.")

    example_idx = get_example_idx(adata, seqlet_idx)
    return adata.uns["unique_examples"]["contrib"][example_idx]


def extract_seqlets(
    contrib: np.ndarray,
    oh: np.ndarray,
    threshold: float = 0.05,
    additional_flanks: int = 3,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """
    Extract, scale, and process seqlets from saliency maps using Tangermeme.

    Seqlets are normalized based on their maximum absolute contribution value.

    Parameters
    ----------
    contrib
        Contribution scores array with shape (n_examples, 4, length)
    oh
        One-hot encoded sequences array with shape (n_examples, 4, length)
    threshold
        Importance threshold for seqlet extraction (default: 0.05)
    additional_flanks
        Additional flanking bases to include around seqlets (default: 3)

    Returns
    -------
    - DataFrame with seqlet coordinates [example_idx, start, end, chrom, g_start, g_end]
    - List of processed seqlet contribution matrices

    Examples
    --------
    >>> seqlets_df, seqlet_matrices = extract_seqlets(contrib, oh, threshold=0.05)
    >>> print(seqlets_df.columns.tolist())
    ['example_idx', 'start', 'end', 'attribution', 'p-value']
    >>> print(len(seqlet_matrices))
    1250
    """
    assert contrib.shape == oh.shape, "Contribution and one-hot arrays must have the same shape"
    seqlets_df = recursive_seqlets(
        (contrib * oh).sum(1),
        threshold=threshold,
        additional_flanks=additional_flanks,
    )

    # extract and normalize contribution scores
    seqlet_matrices = []

    for _, (ex_idx, start, end) in tqdm(
        seqlets_df[["example_idx", "start", "end"]].iterrows(),
        total=len(seqlets_df),
        desc="Processing seqlets",
    ):
        # Extract contribution scores and one-hot sequences for this seqlet
        X = contrib[ex_idx, :, start:end]  # (4, seqlet_length)
        O = oh[ex_idx, :, start:end]  # (4, seqlet_length)

        # Normalize contributions by maximum absolute value
        if abs(X).max() > 0:
            X = X / abs(X).max()

        seqlet_contrib_actual = X * O

        # Apply sign correction based on mean contribution
        unsigned_contrib = np.sign(seqlet_contrib_actual.mean()) * X

        seqlet_matrices.append(unsigned_contrib)

    return seqlets_df, seqlet_matrices


def calculate_motif_similarity(
    seqlets: list[np.ndarray],
    known_motifs: list[np.ndarray] | dict[tuple[str, str], np.ndarray],
    chunk_size: int | None = None,
    n_nearest: int | None = None,
    threshold: float | None = None,
    **kwargs,
) -> sparse.csr_array:
    """
    Calculate TomTom similarity and convert to log-space for clustering.

    Parameters
    ----------
    seqlets
        List of seqlet contribution matrices, each with shape (4, length)
    known_motifs
        List of known motif PPM matrices, each with shape (4, length)
        or a dictionary of motif PPMs, each with shape (4, length)
    chunk_size
        If provided, process seqlets in chunks of this size to manage memory usage.
        If None, process all seqlets at once (original behavior).
    n_nearest
        If provided, only keep the n most similar motifs for each seqlet.
        This creates naturally sparse matrices and reduces memory usage.
        If None, computes similarities to all motifs (with optional thresholding).
    threshold
        Similarity threshold for sparsity when n_nearest is None.
        Values below threshold are clipped to zero. Default 0.05.
        Ignored when n_nearest is specified.
    **kwargs
        Additional arguments for memelite's TomTom

    Returns
    -------
    Sparse log-transformed similarity array with shape (n_seqlets, n_motifs).
    When n_nearest is used, only the top-k similarities per seqlet are stored.
    When threshold is used, values below threshold are clipped to zero.

    Examples
    --------
    >>> _, seqlet_matrices = tfmindi.pp.extract_seqlets(contrib, oh)
    >>> # Memory-efficient: only keep top 50 similarities per seqlet
    >>> similarity_matrix = calculate_motif_similarity(seqlet_matrices, known_motifs, n_nearest=50)
    >>> print(similarity_matrix.shape)
    (1250, 3989)
    >>> # Traditional approach with thresholding
    >>> similarity_matrix = calculate_motif_similarity(seqlet_matrices, known_motifs, threshold=0.1)
    >>> # For large datasets, use chunking with n_nearest
    >>> similarity_matrix = calculate_motif_similarity(seqlet_matrices, known_motifs, chunk_size=10000, n_nearest=50)
    """
    if isinstance(known_motifs, dict):
        known_motifs = list(known_motifs.values())

    n_seqlets = len(seqlets)
    n_motifs = len(known_motifs)

    # Set default threshold if not using n_nearest
    if threshold is None and n_nearest is None:
        threshold = 0.05

    # If no chunking requested or dataset is small
    if chunk_size is None or len(seqlets) <= chunk_size:
        if n_nearest is not None:
            # Use n_nearest approach for memory efficiency
            sim, _, _, _, _, idxs = tomtom(Qs=seqlets, Ts=known_motifs, n_nearest=n_nearest, **kwargs)
            l_sim = np.nan_to_num(-np.log10(sim + 1e-10)).astype(np.float32)

            # Build sparse matrix directly from n_nearest results
            row_indices = []
            col_indices = []
            data_values = []

            for i in range(n_seqlets):
                for j in range(min(n_nearest, l_sim.shape[1])):
                    if l_sim[i, j] > 0:  # Only store positive similarities
                        row_indices.append(i)
                        col_indices.append(idxs[i, j])
                        data_values.append(l_sim[i, j])

            return sparse.csr_array(
                (data_values, (row_indices, col_indices)),
                shape=(n_seqlets, n_motifs),
                dtype=np.float32,
            )
        else:
            # Traditional full matrix approach with thresholding
            sim, _, _, _, _ = tomtom(Qs=seqlets, Ts=known_motifs, **kwargs)
            l_sim = np.nan_to_num(-np.log10(sim + 1e-10))

            # Handle empty arrays
            if l_sim.size == 0:
                return sparse.csr_array(l_sim)

            # Clip values below threshold to zero and create sparse array
            l_sim[l_sim < threshold] = 0
            return sparse.csr_array(l_sim.astype(np.float32))

    # Chunked processing - build final sparse matrix directly from coordinates
    # Collect coordinates and data for final sparse matrix construction
    row_indices = []
    col_indices = []
    data_values = []

    for i in tqdm(range(0, len(seqlets), chunk_size), desc="Processing chunks"):
        end_idx = min(i + chunk_size, len(seqlets))
        chunk = seqlets[i:end_idx]

        if n_nearest is not None:
            # Use n_nearest approach for memory efficiency
            sim_chunk, _, _, _, _, idxs_chunk = tomtom(Qs=chunk, Ts=known_motifs, n_nearest=n_nearest, **kwargs)
            l_sim_chunk = np.nan_to_num(-np.log10(sim_chunk + 1e-10)).astype(np.float32)

            # Build sparse coordinates from n_nearest results
            for local_i in range(len(chunk)):
                global_i = i + local_i
                for j in range(min(n_nearest, l_sim_chunk.shape[1])):
                    if l_sim_chunk[local_i, j] > 0:  # Only store positive similarities
                        row_indices.append(global_i)
                        col_indices.append(idxs_chunk[local_i, j])
                        data_values.append(l_sim_chunk[local_i, j])
        else:
            # Traditional thresholding approach
            sim_chunk, _, _, _, _ = tomtom(Qs=chunk, Ts=known_motifs, **kwargs)
            l_sim_chunk = np.nan_to_num(-np.log10(sim_chunk + 1e-10)).astype(np.float32)

            # Find non-zero entries above threshold
            mask = l_sim_chunk >= threshold
            if mask.any():
                chunk_rows, chunk_cols = np.where(mask)
                chunk_data = l_sim_chunk[mask]

                # Adjust row indices for global matrix position
                global_rows = chunk_rows + i

                # Accumulate coordinates and data
                row_indices.extend(global_rows)
                col_indices.extend(chunk_cols)
                data_values.extend(chunk_data)

        del sim_chunk, l_sim_chunk, chunk

    # Handle empty result
    if len(data_values) == 0:
        return sparse.csr_array((n_seqlets, n_motifs), dtype=np.float32)

    # Build final sparse matrix directly
    return sparse.csr_array(
        (data_values, (row_indices, col_indices)),
        shape=(n_seqlets, n_motifs),
        dtype=np.float32,
    )


def create_seqlet_adata(
    similarity_matrix: sparse.csr_array,
    seqlet_metadata: pd.DataFrame,
    seqlet_matrices: list[np.ndarray[Any, np.dtype[np.floating]]] | None = None,
    oh_sequences: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    contrib_scores: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    motif_names: list[str] | list[tuple[str, str]] | None = None,
    motif_collection: dict[tuple[str, str], np.ndarray[Any, np.dtype[np.floating]]]
    | list[np.ndarray[Any, np.dtype[np.floating]]]
    | None = None,
    motif_annotations: pd.DataFrame | None = None,
    motif_to_dbd: dict[str, str] | None = None,
    dtype: type[np.floating] = np.float32,
) -> AnnData:
    """
    Create comprehensive AnnData object storing all seqlet data for analysis pipeline.

    Parameters
    ----------
    similarity_matrix
        Sparse log-transformed similarity array with shape (n_seqlets, n_motifs)
    seqlet_metadata
        DataFrame with seqlet coordinates and metadata
    seqlet_matrices
        List of seqlet contribution matrices, each with shape (4, length)
    oh_sequences
        One-hot sequences for each seqlet region with shape (n_examples, 4, total_length)
    contrib_scores
        Raw contribution scores for each seqlet region with shape (n_examples, 4, total_length)
    motif_names
        List of motif names corresponding to similarity matrix columns
    motif_collection
        Dictionary or list of motif PPM matrices, each with shape (4, length)
    motif_annotations
        DataFrame with motif annotations containing TF names and other metadata
    motif_to_dbd
        Dictionary mapping motif names to DNA-binding domain annotations
    dtype
        Data type for numerical arrays to optimize memory usage (default: np.float32)

    Returns
    -------
    AnnData object with all data needed for downstream analysis

    Data Storage:

    - .X: Sparse log-transformed motif similarity array (n_seqlets × n_motifs)
    - .obs: Seqlet metadata and variable-length arrays stored per seqlet

      - Standard metadata: coordinates, attribution, p-values
      - .obs["seqlet_matrix"]: Individual seqlet contribution matrices
      - .obs["seqlet_oh"]: Individual seqlet one-hot sequences
    - .obs: Additional seqlet mapping indices
      - .obs["example_oh_idx"]: Index into unique examples for one-hot sequences
      - .obs["example_contrib_idx"]: Index into unique examples for contribution scores
    - .uns: Memory-efficient storage for unique examples
      - .uns["unique_examples"]["oh"]: Unique example one-hot sequences (n_unique_examples × 4 × length)
      - .uns["unique_examples"]["contrib"]: Unique example contribution scores (n_unique_examples × 4 × length)
    - .var: Motif names and annotations
      - .var["motif_ppm"]: Individual motif PPM matrices
      - .var["dbd"]: DNA-binding domain annotations
      - .var["direct_annot"]: Direct TF annotations
      - Other annotation columns from motif_annotations DataFrame

    Examples
    --------
    >>> seqlets_df, seqlet_matrices = tm.pp.extract_seqlets(contrib, oh)
    >>> similarity_matrix = tm.pp.calculate_motif_similarity(seqlet_matrices, motifs)
    >>> adata = tm.pp.create_seqlet_adata(
    ...     similarity_matrix,
    ...     seqlets_df,
    ...     seqlet_matrices=seqlet_matrices,
    ...     oh_sequences=oh,
    ...     contrib_scores=contrib,
    ...     motif_collection=motifs,
    ...     motif_annotations=annotations,
    ...     motif_to_dbd=motif_to_dbd_dict,
    ... )
    >>> print(adata.shape)
    (295, 17995)
    """
    # Validate inputs
    n_seqlets = similarity_matrix.shape[0]  # type: ignore
    if n_seqlets != len(seqlet_metadata):
        raise ValueError(
            f"Number of seqlets in similarity matrix ({n_seqlets}) "
            f"does not match seqlet metadata ({len(seqlet_metadata)})"
        )

    if seqlet_matrices is not None and len(seqlet_matrices) != n_seqlets:
        raise ValueError(
            f"Number of seqlet matrices ({len(seqlet_matrices)}) does not match number of seqlets ({n_seqlets})"
        )

    # Create AnnData object with proper string indices
    obs_df = seqlet_metadata.copy()
    obs_df.index = obs_df.index.astype(str)

    # Create var DataFrame for motifs
    n_motifs = similarity_matrix.shape[1]  # type: ignore
    if motif_names is not None:
        if len(motif_names) != n_motifs:
            raise ValueError(
                f"Number of motif names ({len(motif_names)}) "
                f"does not match number of motifs in similarity matrix ({n_motifs})"
            )
        var_df = pd.DataFrame(index=[fn_name[1] if isinstance(fn_name, tuple) else fn_name for fn_name in motif_names])
    else:
        var_df = pd.DataFrame(index=[f"motif_{i}" for i in range(n_motifs)])

    # Store motif PPMs in .var if provided
    if motif_collection is not None:
        if isinstance(motif_collection, dict):
            motif_ppms = list(motif_collection.values())
            if motif_names is None:
                motif_names = list(motif_collection.keys())
                var_df = pd.DataFrame(
                    index=[fn_name[1] if isinstance(fn_name, tuple) else fn_name for fn_name in motif_names]
                )
        else:
            motif_ppms = motif_collection

        if len(motif_ppms) != n_motifs:
            raise ValueError(
                f"Number of motif PPMs ({len(motif_ppms)}) "
                f"does not match number of motifs in similarity matrix ({n_motifs})"
            )

        # Apply dtype conversion to motif PPMs for memory optimization
        motif_ppms_typed = [ppm.astype(dtype) for ppm in motif_ppms]
        var_df["motif_ppm"] = motif_ppms_typed

    # Store motif annotations in .var if provided
    if motif_annotations is not None and motif_names is not None:
        # Add annotations for motifs that are present in the similarity matrix
        for fn_name in motif_names:
            file_name = fn_name[0] if isinstance(fn_name, tuple) else fn_name
            name = fn_name[1] if isinstance(fn_name, tuple) else fn_name
            if file_name in motif_annotations.index:
                # Add all annotation columns for this motif
                for col in motif_annotations.columns:
                    if col not in var_df.columns:
                        var_df[col] = None  # Initialize column
                    var_df.loc[name, col] = motif_annotations.loc[file_name, col]

    # Store DNA-binding domain annotations if provided
    if motif_to_dbd is not None and motif_names is not None:
        var_df["dbd"] = None  # Initialize column
        for fn_name in motif_names:
            file_name = fn_name[0] if isinstance(fn_name, tuple) else fn_name
            name = fn_name[1] if isinstance(fn_name, tuple) else fn_name
            if file_name in motif_to_dbd:
                var_df.loc[name, "dbd"] = motif_to_dbd[file_name]

    # Convert sparse array data to specified dtype for memory optimization
    if hasattr(similarity_matrix, "astype"):
        # Modern sparse arrays have astype method
        similarity_matrix_typed = similarity_matrix.astype(dtype)
    else:
        # Fallback for older sparse matrices
        similarity_matrix_typed = similarity_matrix.copy()
        similarity_matrix_typed.data = similarity_matrix_typed.data.astype(dtype)

    adata = AnnData(
        X=similarity_matrix_typed,
        obs=obs_df,
        var=var_df,
    )

    # Store seqlet-level data in .obs columns (variable length, must stay in .obs)
    if seqlet_matrices is not None and len(seqlet_matrices) > 0:
        # Apply dtype conversion and store seqlet matrices
        seqlet_matrices_typed = [matrix.astype(dtype) for matrix in seqlet_matrices]
        adata.obs["seqlet_matrix"] = seqlet_matrices_typed

        # Process seqlet sequences and store unique examples
        if (oh_sequences is not None or contrib_scores is not None) and n_seqlets > 0:
            # Get unique example indices and create mapping
            unique_example_indices = seqlet_metadata["example_idx"].unique()
            example_idx_to_pos = {idx: pos for pos, idx in enumerate(unique_example_indices)}

            adata.uns["unique_examples"] = {}
            seqlet_oh_sequences = [] if oh_sequences is not None else None
            seqlet_to_example_pos_oh = [] if oh_sequences is not None else None
            seqlet_to_example_pos_contrib = [] if contrib_scores is not None else None

            for _, row in seqlet_metadata.iterrows():
                ex_idx = int(row["example_idx"])

                # Extract seqlet OH sequences if needed
                if oh_sequences is not None:
                    start = int(row["start"])
                    end = int(row["end"])
                    seqlet_oh = oh_sequences[ex_idx, :, start:end].astype(dtype)
                    seqlet_oh_sequences.append(seqlet_oh)
                    seqlet_to_example_pos_oh.append(example_idx_to_pos[ex_idx])

                # Create contrib mapping if needed
                if contrib_scores is not None:
                    seqlet_to_example_pos_contrib.append(example_idx_to_pos[ex_idx])

            # Store results
            if oh_sequences is not None:
                adata.obs["seqlet_oh"] = seqlet_oh_sequences
                unique_oh_sequences = oh_sequences[unique_example_indices].astype(dtype)
                adata.uns["unique_examples"]["oh"] = unique_oh_sequences
                adata.obs["example_oh_idx"] = seqlet_to_example_pos_oh

            if contrib_scores is not None:
                unique_contrib_scores = contrib_scores[unique_example_indices].astype(dtype)
                adata.uns["unique_examples"]["contrib"] = unique_contrib_scores
                adata.obs["example_contrib_idx"] = seqlet_to_example_pos_contrib

    return adata


def recursive_seqlets(X, threshold=0.01, min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0, n_bins=1000):
    """Call seqlets using the recursive seqlet algorithm.

    THIS FUNCTION IS A DIRECT COPY FROM THE TANGERMEME REPOSITORY FROM JACOB SCHREIBER.
    We do a direct copy here since we only need this function and we want to avoid the heavy torch installation.

    This algorithm identifies spans of high attribution characters, called
    seqlets, using a simple approach derived from the Tomtom/FIMO algorithms.
    First, distributions of attribution sums are created for all potential
    seqlet lengths by discretizing the sum, with one set of distributions for
    positive attribution values and one for negative attribution values. Then,
    CDFs are calculated for each distribution (or, more specifically, 1-CDFs).
    Finally, p-values are calculated via lookup to these 1-CDFs for all
    potential CDFs, yielding a (n_positions, n_lengths) matrix of p-values.

    This algorithm then identifies seqlets by defining them to have a key
    property: all internal spans of a seqlet must also have been called a
    seqlet. This means that all spans from `min_seqlet_len` to `max_seqlet_len`,
    starting at any position in the seqlet, and fully contained by the borders,
    must have a p-value below the threshold. Functionally, this means finding
    entries where the upper left triangle rooted in it is comprised entirely of
    values below the threshold. Graphically, for a candidate seqlet starting at
    X and ending at Y to be called a seqlet, all the values within the bounds
    (in addition to X) must also have a p-value below the threshold.


                                                    min_seqlet_len
                                --------
    . . . . . . . | . . . . / . . . . . . . .
    . . . . . . . | . . . / . . . . . . . . .
    . . . . . . . | . . / . . . . . . . . . .
    . . . . . . . | . / . . . . . . . . . . .
    . . . . . . . | / . . . . . . . . . . . .
    . . . . . . . X . . . . . . . . Y . . . .
    . . . . . . . . . . . . . . . . . . . . .
    . . . . . . . . . . . . . . . . . . . . .


    The seqlets identified by this approach will usually be much smaller than
    those identified by the TF-MoDISco approach, including sometimes missing
    important characters on the flanks. You can set `additional_flanks` to
    a higher value if you want to include additional positions on either side.
    Importantly, the initial seqlet calls cannot overlap, but these additional
    characters are not considered when making that determination. This means
    that seqlets may appear to overlap when `additional_flanks` is set to a
    higher value.


    Parameters
    ----------
    X: np.ndarray, shape=(-1, length)
            Attributions for each position in each example. The identity of the
            characters is not relevant for seqlet calling, so this should be the
            "projected" attributions, i.e., the attribution of the observed
            characters.

    threshold: float, optional
            The p-value threshold for calling seqlets. All positions within the
            triangle (as detailed above) must be below this threshold. Default is
            0.01.

    min_seqlet_len: int, optional
            The minimum length that a seqlet must be, and the minimal length of
            span that must be identified as a seqlet in the recursive property.
            Default is 4.

    max_seqlet_len: int, optional
            The maximum length that a seqlet can be. Default is 25.

    additional_flanks: int, optional
            An additional value to subtract from the start, and to add to the end,
            of all called seqlets. Does not affect the called seqlets.
    n_bins: int, optional
        The number of bins to use when estimating the PDFs and CDFs. Default is
        1000.


    Returns
    -------
    seqlets: pandas.DataFrame, shape=(-1, 5)
            A BED-formatted dataframe containing the called seqlets, ranked from
            lowest p-value to higher p-value. The returned p-value is the p-value
            of the (location, length) span and is not influenced by the other
            values within the triangle.
    """
    columns = ["example_idx", "start", "end", "attribution", "p-value"]
    seqlets = _recursive_seqlets(X, threshold, min_seqlet_len, max_seqlet_len, additional_flanks, n_bins)
    seqlets = pd.DataFrame(seqlets, columns=columns)
    return seqlets.sort_values("p-value").reset_index(drop=True)


@numba.njit
def _recursive_seqlets(X, threshold=0.01, min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0, n_bins=1000):
    """Call seqlets recursively using the Tangermeme algorithm.

    This algorithm has four steps.

    (1) Convert attribution scores into integer bins and calculate a histogram
    (2) Convert these histograms into null distributions across lengths
    (3) Use the null distributions to calculate p-values for each possible length
    (4) Decode this matrix of p-values to find the longest seqlets
    """
    n, l = X.shape
    m = n * l

    ###
    # Step 1: Calculate a histogram of binned scores
    ###

    xmax, xmin = X.max(), X.min()
    bin_width = (xmax - xmin) / (n_bins - 1)

    f = np.zeros(n_bins, dtype=np.float64)

    for i in range(n):
        for j in range(l):
            x_bin = math.floor((X[i, j] - xmin) / bin_width)
            f[x_bin] += 1

    f = f / m

    ###
    # Step 2: Calculate null distributions across lengths
    ###

    scores = np.zeros((max_seqlet_len + 1, n_bins * max_seqlet_len), dtype=np.float64)
    scores[1, :n_bins] = f

    rcdfs = np.zeros_like(scores)
    rcdfs[:, 0] = 1.0

    for seqlet_len in range(2, max_seqlet_len + 1):
        for i in range(n_bins * (seqlet_len - 1)):
            for j in range(n_bins):
                scores[seqlet_len, i + j] += scores[seqlet_len - 1, i] * f[j]

        for i in range(1, n_bins * seqlet_len):
            rcdfs[seqlet_len, i] = max(rcdfs[seqlet_len, i - 1] - scores[seqlet_len, i], 0)

    ###
    # Step 3: Calculate p-values given these 1-CDFs
    ###

    X_csum = np.zeros((n, l + 1))
    for i in range(n):
        for j in range(l):
            X_csum[i, j + 1] = X_csum[i, j] + X[i, j]

    ###
    # Step 4: Decode p-values into seqlets
    ###

    seqlets = []

    for i in range(n):
        p_value = np.ones((max_seqlet_len + 1, l), dtype=np.float64)
        p_value[:min_seqlet_len] = 0
        p_value[:, -min_seqlet_len] = 1

        for seqlet_len in range(min_seqlet_len, max_seqlet_len + 1):
            for k in range(l - seqlet_len + 1):
                x_ = X_csum[i, k + seqlet_len] - X_csum[i, k]
                x_ = math.floor((x_ - xmin * seqlet_len) / bin_width)

                p_value[seqlet_len, k] = max(rcdfs[seqlet_len, x_], p_value[seqlet_len - 1, k])

        # Iteratively identify spans, from longest to shortest, that satisfy the
        # recursive p-value threshold.
        for j in range(max_seqlet_len - min_seqlet_len + 1):
            seqlet_len = max_seqlet_len - j

            while True:
                start = p_value[seqlet_len].argmin()
                p = p_value[seqlet_len, start]
                p_value[seqlet_len, start] = 1

                if p >= threshold:
                    break

                for k in range(1, seqlet_len):
                    if p_value[seqlet_len - k, start + k] >= threshold:
                        break

                else:
                    for end in range(start, min(start + seqlet_len, l - 1)):
                        p_value[:, end] = 1

                    end = min(start + seqlet_len + additional_flanks, l - 1)
                    start = max(start - additional_flanks, 0)
                    attr = X_csum[i, end] - X_csum[i, start]
                    seqlets.append((i, start, end, attr, p))

    return seqlets
