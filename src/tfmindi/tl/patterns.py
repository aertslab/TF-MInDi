"""Pattern creation and alignment tools for seqlet clusters."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from anndata import AnnData
from memelite import tomtom

from tfmindi.types import Pattern, Seqlet


def create_patterns(adata: AnnData, max_n: int | None = None) -> dict[str, Pattern]:
    """
    Generate aligned PWM patterns from seqlet clusters using stored data.

    This function performs the following steps for each cluster:
    1. Extract seqlets belonging to that cluster
    2. Use TomTom to align seqlets within the cluster
    3. Find consensus root seqlet (lowest mean similarity)
    4. Apply strand and offset corrections using stored sequence data
    5. Generate Pattern object with PWM, contribution scores, and seqlet instances

    Parameters
    ----------
    adata
        AnnData object with cluster assignments and stored seqlet data.
        Must contain:
        - adata.obs["leiden"]: Cluster assignments
        - adata.obs["seqlet_matrix"]: Individual seqlet contribution matrices
        - adata.obs["example_oh"]: Full example one-hot sequences per seqlet
        - adata.obs["example_contrib"]: Full example contribution scores per seqlet
    max_n
        Maximum number of seqlets to use per cluster for pattern creation.
        If None, all seqlets in each cluster are used. If an integer is provided,
        seqlets are randomly subsampled to speed up pattern creation.
        Default is None.

    Returns
    -------
    Dictionary mapping cluster IDs to Pattern objects

    Examples
    --------
    >>> import tfmindi as tm
    >>> # adata with clustering results
    >>> patterns = tm.tl.create_patterns(adata)
    >>> print(f"Found {len(patterns)} patterns")
    >>> # Use subsampling to speed up pattern creation
    >>> patterns_fast = tm.tl.create_patterns(adata, max_n=300)
    >>> pattern_0 = patterns["0"]
    >>> print(f"Pattern 0 has {pattern_0.n_seqlets} seqlets")
    >>> print(f"Pattern 0 PWM shape: {pattern_0.ppm.shape}")
    """
    # Check required data is present
    required_obs_cols = ["leiden", "seqlet_matrix", "example_oh", "example_contrib"]
    missing_cols = [col for col in required_obs_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in adata.obs: {missing_cols}")

    patterns = {}
    clusters = adata.obs["leiden"].unique()

    print(f"Creating patterns for {len(clusters)} clusters...")

    for cluster in clusters:
        cluster_str = str(cluster)

        cluster_mask = adata.obs["leiden"] == cluster
        cluster_indices = adata.obs.index[cluster_mask].tolist()

        if len(cluster_indices) < 2:
            print(f"Skipping cluster {cluster_str} with only {len(cluster_indices)} seqlets")
            continue

        # Subsample seqlets to speed up pattern creation
        if max_n is not None and len(cluster_indices) > max_n:
            rng = random.Random(123)
            cluster_indices = rng.sample(cluster_indices, max_n)

        cluster_seqlet_matrices = [adata.obs.loc[idx, "seqlet_matrix"] for idx in cluster_indices]

        # Perform TomTom alignment within cluster
        sim_matrix, _, offsets, _, strands = tomtom(Qs=cluster_seqlet_matrices, Ts=cluster_seqlet_matrices)

        # Find root seqlet (lowest mean similarity to others)
        root_idx = sim_matrix.mean(axis=0).argmin()
        root_strands = strands[root_idx, :]
        root_offsets = offsets[root_idx, :]

        cluster_metadata = adata.obs.loc[cluster_indices].copy()

        pattern = _create_pattern_from_cluster(
            cluster_indices=cluster_indices,
            cluster_metadata=cluster_metadata,
            adata=adata,
            strands=root_strands,
            offsets=root_offsets,
            cluster_id=cluster_str,
        )
        patterns[cluster_str] = pattern
    return patterns


def _create_pattern_from_cluster(
    cluster_indices: list[str],
    cluster_metadata: pd.DataFrame,
    adata: AnnData,
    strands: np.ndarray,
    offsets: np.ndarray,
    cluster_id: str,
) -> Pattern:
    """Create a Pattern object from aligned cluster data."""
    n_seqlets = len(cluster_indices)

    # Calculate maximum seqlet length for padding
    seqlet_lengths = [
        int(cluster_metadata.loc[idx, "end"]) - int(cluster_metadata.loc[idx, "start"])  # type: ignore
        for idx in cluster_indices  # type: ignore
    ]
    max_length = max(seqlet_lengths)

    seqlets = []
    seqlet_instances = np.zeros((n_seqlets, max_length, 4))
    seqlet_contribs = np.zeros((n_seqlets, max_length, 4))

    for i, idx in enumerate(cluster_indices):
        start = int(cluster_metadata.loc[idx, "start"])  # type: ignore
        end = int(cluster_metadata.loc[idx, "end"])  # type: ignore

        # Get full example sequences and contributions
        example_oh = np.array(adata.obs.loc[idx, "example_oh"])  # Shape: (4, seq_length)
        example_contrib = np.array(adata.obs.loc[idx, "example_contrib"])  # Shape: (4, seq_length)

        # Calculate alignment coordinates
        strand = bool(strands[i])
        offset = int(offsets[i])
        offset = offset * -1 if strand else offset

        if not strand:
            aligned_start = start + offset
            aligned_end = start + offset + max_length
        else:
            aligned_start = end + offset - max_length
            aligned_end = end + offset

        # Check bounds
        if aligned_start < 0 or aligned_end > example_oh.shape[1]:
            print(f"Warning: Seqlet {idx} exceeds sequence bounds, skipping alignment")
            # Use original seqlet without alignment
            seqlet_length = end - start
            padded_oh = np.zeros((4, max_length))
            padded_contrib = np.zeros((4, max_length))
            padded_oh[:, :seqlet_length] = example_oh[:, start:end]
            padded_contrib[:, :seqlet_length] = example_contrib[:, start:end]
            instance = padded_oh.T
            contrib = padded_contrib.T
        else:
            # Extract aligned region
            instance = example_oh[:, aligned_start:aligned_end].T  # Shape: (max_length, 4)
            contrib = example_contrib[:, aligned_start:aligned_end].T  # Shape: (max_length, 4)

        # Apply strand correction if needed
        if strand:
            instance = instance[::-1, ::-1]  # Reverse complement
            contrib = contrib[::-1, ::-1]

        seqlet_instances[i] = instance
        seqlet_contribs[i] = contrib

        seqlet = Seqlet(
            seq_instance=instance,
            start=start,
            end=end,
            region_one_hot=example_oh,
            is_revcomp=strand,
            contrib_scores=instance * contrib,  # Masked by actual sequence
            hypothetical_contrib_scores=contrib,  # Raw contribution scores
        )
        seqlets.append(seqlet)

    # Calculate consensus PWM and contribution scores with proper normalization
    ppm = seqlet_instances.mean(axis=0)  # Shape: (max_length, 4)

    # Normalize PWM so each position sums to 1
    position_sums = ppm.sum(axis=1, keepdims=True)
    # For positions with all zeros (alignment gaps), use uniform distribution
    uniform_prob = 0.25
    ppm = np.where(position_sums == 0, uniform_prob, ppm / np.maximum(position_sums, 1e-10))

    mean_contrib_scores = (seqlet_instances * seqlet_contribs).mean(axis=0)
    mean_hypothetical_contrib = seqlet_contribs.mean(axis=0)

    return Pattern(
        ppm=ppm,
        contrib_scores=mean_contrib_scores,
        hypothetical_contrib_scores=mean_hypothetical_contrib,
        seqlets=seqlets,
        cluster_id=cluster_id,
        n_seqlets=n_seqlets,
    )
