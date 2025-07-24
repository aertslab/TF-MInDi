"""Seqlet extraction and motif similarity preprocessing functions for TF-MInDi."""

from __future__ import annotations

import math

import numba
import numpy as np
import pandas as pd
from anndata import AnnData
from memelite import tomtom
from tqdm import tqdm


def extract_seqlets(
    contrib: np.ndarray, oh: np.ndarray, threshold: float = 0.05, additional_flanks: int = 3
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """
    Extract, scale, and process seqlets from saliency maps using Tangermeme.

    Seqlets are normalized based on their maximum absolute contribution value.

    Parameters
    ----------
    contrib
        Contribution scores array with shape (n_examples, length, 4)
    oh
        One-hot encoded sequences array with shape (n_examples, length, 4)
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
        seqlets_df[["example_idx", "start", "end"]].iterrows(), total=len(seqlets_df), desc="Processing seqlets"
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
    known_motifs: list[np.ndarray] | dict[str, np.ndarray],
    chunk_size: int | None = None,
    **kwargs,
) -> np.ndarray:
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
    **kwargs
        Additional arguments for memelite's TomTom (e.g., `n_nearest`)

    Returns
    -------
    Log-transformed similarity matrix with shape (n_seqlets, n_motifs)

    Examples
    --------
    >>> _, seqlet_matrices = tfmindi.pp.extract_seqlets(contrib, oh)
    >>> similarity_matrix = calculate_motif_similarity(seqlet_matrices, known_motifs)
    >>> print(similarity_matrix.shape)
    (1250, 3989)
    >>> # For large datasets, use chunking
    >>> similarity_matrix = calculate_motif_similarity(seqlet_matrices, known_motifs, chunk_size=10000)
    """
    if isinstance(known_motifs, dict):
        known_motifs = list(known_motifs.values())

    # If no chunking requested or dataset is small
    if chunk_size is None or len(seqlets) <= chunk_size:
        sim, _, _, _, _ = tomtom(Qs=seqlets, Ts=known_motifs, **kwargs)
        l_sim = np.nan_to_num(-np.log10(sim + 1e-10))

        # Handle empty arrays
        if l_sim.size == 0:
            return l_sim

        l_sim_sparse = np.clip(l_sim, 0.05, l_sim.max())
        return l_sim_sparse

    # Chunked processing
    all_similarities = []

    for i in tqdm(range(0, len(seqlets), chunk_size), desc="Processing chunks"):
        end_idx = min(i + chunk_size, len(seqlets))
        chunk = seqlets[i:end_idx]

        # Process this chunk
        sim_chunk, _, _, _, _ = tomtom(Qs=chunk, Ts=known_motifs, **kwargs)
        l_sim_chunk = np.nan_to_num(-np.log10(sim_chunk + 1e-10))

        all_similarities.append(l_sim_chunk)

    # Combine all chunks
    l_sim = np.vstack(all_similarities)

    # Handle empty arrays
    if l_sim.size == 0:
        return l_sim

    l_sim_sparse = np.clip(l_sim, 0.05, l_sim.max())
    return l_sim_sparse


def create_seqlet_adata(
    similarity_matrix: np.ndarray,
    seqlet_metadata: pd.DataFrame,
    seqlet_matrices: list[np.ndarray] | None = None,
    oh_sequences: np.ndarray | None = None,
    contrib_scores: np.ndarray | None = None,
    motif_names: list[str] | None = None,
    motif_collection: dict[str, np.ndarray] | list[np.ndarray] | None = None,
    motif_annotations: pd.DataFrame | None = None,
    motif_to_dbd: dict[str, str] | None = None,
) -> AnnData:
    """
    Create comprehensive AnnData object storing all seqlet data for analysis pipeline.

    Parameters
    ----------
    similarity_matrix
        Log-transformed similarity matrix with shape (n_seqlets, n_motifs)
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

    Returns
    -------
    AnnData object with all data needed for downstream analysis

    Data Storage:
    - .X: Log-transformed motif similarity matrix (n_seqlets × n_motifs)
    - .obs: Seqlet metadata and variable-length arrays stored per seqlet
      - Standard metadata: coordinates, attribution, p-values
      - .obs["seqlet_matrix"]: Individual seqlet contribution matrices
      - .obs["seqlet_oh"]: Individual seqlet one-hot sequences
      - .obs["example_oh"]: Full example one-hot sequences per seqlet
      - .obs["example_contrib"]: Full example contribution scores per seqlet
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
    n_seqlets = similarity_matrix.shape[0]
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
    n_motifs = similarity_matrix.shape[1]
    if motif_names is not None:
        if len(motif_names) != n_motifs:
            raise ValueError(
                f"Number of motif names ({len(motif_names)}) "
                f"does not match number of motifs in similarity matrix ({n_motifs})"
            )
        var_df = pd.DataFrame(index=motif_names)
    else:
        var_df = pd.DataFrame(index=[f"motif_{i}" for i in range(n_motifs)])

    # Store motif PPMs in .var if provided
    if motif_collection is not None:
        if isinstance(motif_collection, dict):
            motif_ppms = list(motif_collection.values())
            if motif_names is None:
                motif_names = list(motif_collection.keys())
                var_df = pd.DataFrame(index=motif_names)
        else:
            motif_ppms = motif_collection

        if len(motif_ppms) != n_motifs:
            raise ValueError(
                f"Number of motif PPMs ({len(motif_ppms)}) "
                f"does not match number of motifs in similarity matrix ({n_motifs})"
            )

        var_df["motif_ppm"] = motif_ppms

    # Store motif annotations in .var if provided
    if motif_annotations is not None and motif_names is not None:
        # Add annotations for motifs that are present in the similarity matrix
        for motif_name in motif_names:
            if motif_name in motif_annotations.index:
                # Add all annotation columns for this motif
                for col in motif_annotations.columns:
                    if col not in var_df.columns:
                        var_df[col] = None  # Initialize column
                    var_df.loc[motif_name, col] = motif_annotations.loc[motif_name, col]

    # Store DNA-binding domain annotations if provided
    if motif_to_dbd is not None and motif_names is not None:
        var_df["dbd"] = None  # Initialize column
        for motif_name in motif_names:
            if motif_name in motif_to_dbd:
                var_df.loc[motif_name, "dbd"] = motif_to_dbd[motif_name]

    adata = AnnData(
        X=similarity_matrix,
        obs=obs_df,
        var=var_df,
    )

    # Store seqlet-level data in .obs columns
    if seqlet_matrices is not None:
        adata.obs["seqlet_matrix"] = seqlet_matrices

        # Also store seqlet one-hot sequences extracted from the full sequences
        if oh_sequences is not None:
            seqlet_oh_sequences = []
            for _, row in seqlet_metadata.iterrows():
                ex_idx = int(row["example_idx"])
                start = int(row["start"])
                end = int(row["end"])
                seqlet_oh = oh_sequences[ex_idx, :, start:end]
                seqlet_oh_sequences.append(seqlet_oh)
            adata.obs["seqlet_oh"] = seqlet_oh_sequences

    # Store example-level data in .obs columns (mapped to each seqlet)
    if oh_sequences is not None:
        example_oh_per_seqlet = []
        for _, row in seqlet_metadata.iterrows():
            ex_idx = int(row["example_idx"])
            example_oh_per_seqlet.append(oh_sequences[ex_idx])
        adata.obs["example_oh"] = example_oh_per_seqlet

    if contrib_scores is not None:
        example_contrib_per_seqlet = []
        for _, row in seqlet_metadata.iterrows():
            ex_idx = int(row["example_idx"])
            example_contrib_per_seqlet.append(contrib_scores[ex_idx])
        adata.obs["example_contrib"] = example_contrib_per_seqlet

    return adata


@numba.njit
def _recursive_seqlets(X, threshold=0.01, min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0):
    """An internal function implementing the recursive seqlet algorithm."""
    n, l = X.shape

    # Convert observed attributions into cumsums for fast span calculation
    X_csum = np.zeros((n + 1, l))
    for i in range(n):
        for j in range(l):
            X_csum[i, j + 1] = X_csum[i, j] + X[i, j]

    xmins = np.empty(max_seqlet_len + 1, dtype=np.float64)
    xmaxs = np.empty(max_seqlet_len + 1, dtype=np.float64)
    X_cdfs = np.zeros((2, max_seqlet_len + 1, 1000), dtype=np.float64)

    # Construct background distributions
    for j in range(min_seqlet_len, max_seqlet_len + 1):
        xmin, xmax = 0.0, 0.0
        n_pos, n_neg = 0.0, 0.0

        # For a given span size, find the minimum and maximum values and the count
        # of the number of attribution values in each.
        for i in range(n):
            for k in range(l - j + 1):
                x_ = X_csum[i, k + j] - X_csum[i, k]

                if x_ > 0:
                    xmax = max(x_, xmax)
                    n_pos += 1.0
                else:
                    xmin = min(x_, xmin)
                    n_neg += 1.0

        xmins[j] = xmin
        xmaxs[j] = xmax
        p_pos, p_neg = 1 / n_pos, 1 / n_neg

        # Now go through again and bin the attribution value, recording the count of
        # the number of occurences, pre-divided by the total count to get probs.
        for i in range(n):
            for k in range(l - j + 1):
                x_ = X_csum[i, k + j] - X_csum[i, k]

                if x_ > 0:
                    x_int = math.floor(999 * x_ / xmax)
                    X_cdfs[0, j, x_int] += p_pos
                else:
                    x_int = math.floor(999 * x_ / xmin)
                    X_cdfs[1, j, x_int] += p_neg

        # Convert these PDFs into 1 - CDFs.
        for i in range(1, 1001):
            if i < 1000:
                X_cdfs[0, j, i] += X_cdfs[0, j, i - 1]
                X_cdfs[1, j, i] += X_cdfs[1, j, i - 1]

            X_cdfs[0, j, i - 1] = 1 - X_cdfs[0, j, i - 1]
            X_cdfs[1, j, i - 1] = 1 - X_cdfs[1, j, i - 1]

    ###

    p_value = np.ones((max_seqlet_len + 1, l), dtype=np.float64)
    seqlets = []

    # Calculate p-values for each seqlet span and keep only the maximum p-value of
    # spans starting here thus-far. Because of the recursive property, if a span
    # has a high p-value, definitionally none of the other spans including it can.
    for i in range(n):
        for j in range(min_seqlet_len, max_seqlet_len + 1):
            for k in range(l - j + 1):
                x_ = X_csum[i, k + j] - X_csum[i, k]

                if x_ > 0:
                    x_int = math.floor(999 * x_ / xmaxs[j])
                    p_value[j, k] = X_cdfs[0, j, x_int]
                else:
                    x_int = math.floor(999 * x_ / xmins[j])
                    p_value[j, k] = X_cdfs[1, j, x_int]

                if j > min_seqlet_len:
                    if p_value[j - 1, k] >= threshold:
                        p_value[j, k] = threshold
                    # p_value[j, k] = max(p_value[j-1, k], p_value[j, k])

        # Iteratively identify spans, from longest to shortest, that satisfy the
        # recursive p-value threshold.
        for j in range(max_seqlet_len - min_seqlet_len + 1):
            j = max_seqlet_len - j

            while True:
                start = p_value[j].argmin()
                p = p_value[j, start]
                p_value[j, start] = 1

                if p >= threshold:
                    break

                for k in range(j - min_seqlet_len):
                    if p_value[j - k, start + k + 1] >= threshold:
                        break

                else:
                    for end in range(start, min(start + j, l - 1)):
                        p_value[:, end] = 1

                    end = min(start + j + additional_flanks, l - 1)
                    start = max(start - additional_flanks, 0)
                    attr = X_csum[i, end] - X_csum[i, start]
                    seqlets.append((i, start, end, attr, p))

    return seqlets


def recursive_seqlets(X, threshold=0.01, min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0):
    """A seqlet caller implementing the recursive seqlet algorithm.

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
    X: numpy.ndarray, shape=(-1, length)
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


    Returns
    -------
    seqlets: pandas.DataFrame, shape=(-1, 5)
            A BED-formatted dataframe containing the called seqlets, ranked from
            lowest p-value to higher p-value. The returned p-value is the p-value
            of the (location, length) span and is not influenced by the other
            values within the triangle.
    """
    columns = ["example_idx", "start", "end", "attribution", "p-value"]
    seqlets = _recursive_seqlets(X, threshold, min_seqlet_len, max_seqlet_len, additional_flanks)
    seqlets = pd.DataFrame(seqlets, columns=columns)
    return seqlets.sort_values("p-value").reset_index(drop=True)
