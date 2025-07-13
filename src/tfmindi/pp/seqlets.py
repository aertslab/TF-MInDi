"""Seqlet extraction and motif similarity preprocessing functions for TF-MInDi."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from memelite import tomtom
from tangermeme.seqlet import recursive_seqlets
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
    seqlets: list[np.ndarray], known_motifs: list[np.ndarray] | dict[str, np.ndarray]
) -> np.ndarray:
    """
    Calculate TomTom similarity and convert to log-space for clustering.

    Parameters
    ----------
    seqlets
        List of seqlet contribution matrices, each with shape (4, length)
    known_motifs
        List of known motif PWM matrices, each with shape (4, length)
        or a dictionary of motifs, each with shape (4, length)

    Returns
    -------
    Log-transformed similarity matrix with shape (n_seqlets, n_motifs)

    Examples
    --------
    >>> _, seqlet_matrices = tfmindi.pp.extract_seqlets(contrib, oh)
    >>> similarity_matrix = calculate_motif_similarity(seqlet_matrices, known_motifs)))
    >>> print(similarity_matrix.shape)
    (1250, 3989)
    """
    if isinstance(known_motifs, dict):
        known_motifs = list(known_motifs.values())
    sim, _, _, _, _ = tomtom(Qs=seqlets, Ts=known_motifs)

    l_sim = np.nan_to_num(-np.log10(sim + 1e-10))

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
        Dictionary or list of motif PWM matrices, each with shape (4, length)
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
      - .var["motif_pwm"]: Individual motif PWM matrices
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

    # Store motif PWMs in .var if provided
    if motif_collection is not None:
        if isinstance(motif_collection, dict):
            motif_pwms = list(motif_collection.values())
            if motif_names is None:
                motif_names = list(motif_collection.keys())
                var_df = pd.DataFrame(index=motif_names)
        else:
            motif_pwms = motif_collection

        if len(motif_pwms) != n_motifs:
            raise ValueError(
                f"Number of motif PWMs ({len(motif_pwms)}) "
                f"does not match number of motifs in similarity matrix ({n_motifs})"
            )

        var_df["motif_pwm"] = motif_pwms

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
