"""Seqlet extraction and motif similarity preprocessing functions for TF-MInDi."""

from __future__ import annotations

import numpy as np
import pandas as pd
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
