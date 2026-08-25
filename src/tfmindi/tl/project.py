"""Tools to project seqlets in reference dataset and annotate seqlets."""

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData  # type: ignore
from pynndescent import NNDescent  # type: ignore
from scipy import sparse
from tfmindi.backends import run_accelerated, to_numpy
from tfmindi.datasets import MotifCollectionData


def _project_in_reference(X_sim: np.ndarray | sparse.csr_array, ref_pcs: np.ndarray) -> np.ndarray:
    """Project seqlet similarities onto the reference principal components.

    Row-centering a sparse matrix would destroy its sparsity, but the projection of a
    centered matrix has a rank-1 identity, ``(X - mu.1^T) P == X P - outer(mu, P.sum(0))``,
    so the similarity matrix is never densified.

    Parameters
    ----------
    X_sim
        Seqlet x reference-motif similarities, sparse or dense.
    ref_pcs
        Reference principal component loadings, shape (n_motifs, n_PCs).

    Returns
    -------
    Dense projected coordinates, shape (n_seqlets, n_PCs).
    """
    if sparse.issparse(X_sim):
        row_means = np.asarray(X_sim.mean(axis=1)).ravel()
        return X_sim @ ref_pcs - np.outer(row_means, ref_pcs.sum(axis=0))
    X_sim_center = X_sim - X_sim.mean(1)[:, None]
    return X_sim_center @ ref_pcs


# Device-resident bytes of similarity data one projection chunk may hold, which sets the peak
# device footprint of the whole prediction: measured 4.5 GB here, 14.8 GB at 2 GiB and 61 GB
# unbounded, at 583k seqlets. Throughput is flat from 64 MiB to 8 GiB (23-31 s), so this buys
# portability to small cards for nothing, and it keeps a chunk's nonzero count far below the
# int32 range cuSPARSE indexes with.
_PROJECTION_CHUNK_BYTES = 512 << 20


def _row_chunk_bounds(indptr: np.ndarray, max_nnz: int) -> list[int]:
    """Split CSR rows into groups holding at most ``max_nnz`` nonzeros each.

    Parameters
    ----------
    indptr
        Row pointer of the matrix being split.
    max_nnz
        Largest number of nonzeros a single group may hold. A row exceeding it on its own
        becomes a group of one rather than being split further.

    Returns
    -------
    Row boundaries, starting at 0 and ending at the row count, so consecutive pairs are the
    ``[start, stop)`` of each group.
    """
    # int64 throughout: a row pointer narrow enough for int32 still has to be compared
    # against an offset that a generous byte budget pushes past the int32 range.
    offsets = np.asarray(indptr, dtype=np.int64)
    n_rows = len(offsets) - 1
    bounds = [0]
    while bounds[-1] < n_rows:
        start = bounds[-1]
        stop = int(np.searchsorted(offsets, offsets[start] + max_nnz, side="right")) - 1
        bounds.append(min(max(stop, start + 1), n_rows))
    return bounds


def _project_in_reference_gpu(
    X_sim: np.ndarray | sparse.csr_array,
    ref_pcs: np.ndarray,
    chunk_bytes: int = _PROJECTION_CHUNK_BYTES,
) -> Any:
    """Project seqlet similarities onto the reference PCs on the GPU.

    Uses the same rank-1 centering identity as :func:`_project_in_reference`, and leaves
    the result on the device so the nearest-neighbour query that follows does not have to
    ship the projected coordinates back and forth.

    Rows go over in chunks rather than all at once, which bounds the device footprint by
    ``chunk_bytes`` instead of by the size of the whole similarity matrix -- tens of
    gigabytes at genome scale, with a nonzero count that passes the int32 range cuSPARSE
    indexes with beyond ~2.1e9 nonzeros. Without chunking the projection cannot run at all
    at that size.

    Each projected row is a function of its own input row alone, but the result is *not*
    bit-identical across chunk sizes: cuSPARSE picks its multiplication strategy from the
    block's shape, so the summation order inside a row shifts. The measured spread is
    ~5e-5 on coordinates reaching 72 -- half the 1e-4 gap between this float32 projection and
    the float64 CPU one -- and the labels the nearest-neighbour vote assigns are unaffected:
    identical to the unchunked result on every chunk size from 8 MiB to 8 GiB, and identical
    across chunk sizes over all 583k seqlets of a genome-scale run.

    Parameters
    ----------
    X_sim
        Seqlet x reference-motif similarities, sparse or dense.
    ref_pcs
        Reference principal component loadings, shape (n_motifs, n_PCs).
    chunk_bytes
        Device-resident bytes of similarity data per chunk.

    Returns
    -------
    Device-resident projected coordinates, shape (n_seqlets, n_PCs).
    """
    import cupy as cp  # type: ignore
    import cupyx.scipy.sparse as cusparse  # type: ignore

    pcs = cp.asarray(ref_pcs, dtype=cp.float32)
    pcs_colsum = pcs.sum(axis=0)
    n_rows = X_sim.shape[0]
    out = cp.empty((n_rows, pcs.shape[1]), dtype=cp.float32)

    if sparse.issparse(X_sim):
        X_csr = X_sim if X_sim.format == "csr" else X_sim.tocsr()
        # 8 bytes per nonzero: float32 values alongside int32 column indices.
        bounds = _row_chunk_bounds(X_csr.indptr, max(chunk_bytes // 8, 1))
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            block = X_csr[start:stop]
            X_gpu = cusparse.csr_matrix(
                (
                    cp.asarray(block.data, dtype=cp.float32),
                    # Per-chunk indices always fit int32, which is what cuSPARSE wants even
                    # when the full matrix needed int64.
                    cp.asarray(block.indices, dtype=cp.int32),
                    cp.asarray(block.indptr, dtype=cp.int32),
                ),
                shape=block.shape,
            )
            row_means = X_gpu.mean(axis=1).reshape(-1)
            out[start:stop] = X_gpu @ pcs - cp.outer(row_means, pcs_colsum)
            del X_gpu, block, row_means
        return out

    rows_per_chunk = max(chunk_bytes // (4 * max(X_sim.shape[1], 1)), 1)
    for start in range(0, n_rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_rows)
        X_gpu = cp.asarray(X_sim[start:stop], dtype=cp.float32)
        out[start:stop] = (X_gpu - X_gpu.mean(1)[:, None]) @ pcs
        del X_gpu
    return out


def _build_index(
    pca_ref: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    **kwargs,
) -> NNDescent:
    index = NNDescent(pca_ref, metric=metric, n_neighbors=n_neighbors, **kwargs)
    index.prepare()
    return index


def _tally_votes(neighbor_label_ids: Any, w: Any, n_classes: int, xp: Any) -> Any:
    """Accumulate each query's neighbour votes into an (n_query, n_classes) table.

    The two array modules want opposite formulations, so this is the one place the CPU and
    GPU vote reductions differ.

    Parameters
    ----------
    neighbor_label_ids
        Label ids of each query's neighbours, shape (n_query, n_neighbors).
    w
        Per-neighbour weights of the same shape, or None to count each neighbour once.
    n_classes
        Number of distinct reference labels, i.e. the width of the table.
    xp
        Array module to compute with: :mod:`numpy` or :mod:`cupy`.

    Returns
    -------
    Vote mass per query and class, shape (n_query, n_classes).
    """
    n_query = neighbor_label_ids.shape[0]

    if xp is np:
        # Offsetting each row into its own slice of one flat bincount is by far the fastest
        # numpy formulation: np.add.at on a 2D table is orders of magnitude slower, and
        # scanning the neighbour array once per class allocates a boolean temporary per class.
        flat_bins = (np.arange(n_query, dtype=np.int64)[:, None] * n_classes + neighbor_label_ids).ravel()
        return np.bincount(flat_bins, weights=None if w is None else w.ravel(), minlength=n_query * n_classes).reshape(
            n_query, n_classes
        )

    # The same flat bincount is pathological on the GPU: cupy routes it through CUB's
    # histogram, whose temporary storage scales with the bin count, so a 10^8-bin table asks
    # for hundreds of GB and fails. Scattering into a preallocated table costs only the table.
    import cupyx  # type: ignore

    votes = xp.zeros((n_query, n_classes), dtype=xp.int32 if w is None else xp.float64)
    cupyx.scatter_add(votes, (xp.arange(n_query)[:, None], neighbor_label_ids), 1 if w is None else w)
    return votes


def _take_top_k(votes: Any, k: int, xp: Any) -> tuple[Any, Any]:
    """Pull the k highest-voted classes per query out of the vote table.

    Repeated argmax with in-place knockout, rather than ``argpartition``: partitioning
    allocates a second full-width index table, in int64, so it would cost more than the
    vote table itself -- the one allocation :func:`_vote_labels` is written to avoid
    moving or duplicating. Here the only extra allocations are ``(n_query,)`` per round.

    Consumes ``votes``: knocked-out entries are overwritten with -1. Vote mass is
    non-negative under both weightings, so -1 can never win a later round.

    Parameters
    ----------
    votes
        Vote mass per query and class, shape (n_query, n_classes). Modified in place.
    k
        Number of classes to pull, in descending vote order.
    xp
        Array module to compute with: :mod:`numpy` or :mod:`cupy`.

    Returns
    -------
    Tuple of class indices and their vote mass, both shape (n_query, k), still device-resident.
    """
    rows = xp.arange(votes.shape[0])
    ids, vals = [], []
    for round_ in range(k):
        idx = votes.argmax(axis=1)
        ids.append(idx)
        vals.append(votes[rows, idx])
        if round_ + 1 < k:
            votes[rows, idx] = -1
    return xp.stack(ids, axis=1), xp.stack(vals, axis=1)


def _vote_labels(
    neighbor_label_ids: Any,
    neighbor_distances: Any,
    unique_labels: np.ndarray,
    n_neighbors: int,
    weights: Literal["uniform", "distance"],
    xp: Any,
    top_k: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce each query's neighbour labels to a winning label and a confidence score.

    Written against the array module ``xp`` so the CPU and GPU paths apply the same vote
    weighting, the same argmax tie-breaking and the same score definition, rather than each
    relying on its own KNN library's notion of a class probability.

    Parameters
    ----------
    neighbor_label_ids
        Label ids of each query's neighbours, shape (n_query, n_neighbors).
    neighbor_distances
        Matching neighbour distances, only read when ``weights="distance"``.
    unique_labels
        Sorted unique labels, indexed by the ids in ``neighbor_label_ids``. Host array.
    n_neighbors
        Number of neighbours per query, used to normalise uniform votes.
    weights
        ``"uniform"`` counts each neighbour once; ``"distance"`` weights by 1/distance.
    xp
        Array module to compute with: :mod:`numpy` or :mod:`cupy`.
    top_k
        How many runners-up to return alongside the winner, for callers that need to see a
        split vote rather than only its argmax. Costs ``(n_query, top_k)`` on the host,
        against ``(n_query, n_classes)`` for the table itself.

    Returns
    -------
    Tuple of the winning label per query, its normalised score, the top-k label *ids* and
    their normalised scores, all as host arrays.
    """
    n_classes = len(unique_labels)

    if weights == "uniform":
        w = None
        norm: float | Any = float(n_neighbors)
    else:
        # weight by 1/distance; clamp zeros to avoid division by zero
        w = 1.0 / xp.where(neighbor_distances == 0, 1e-10, neighbor_distances)
        norm = w.sum(axis=1)

    votes = _tally_votes(neighbor_label_ids, w, n_classes, xp)

    # Normalize after the reduction, not over the whole n_query x n_classes table: dividing
    # by a positive constant per row leaves argmax unchanged, and the table is the largest
    # allocation here (tens of GB at genome scale). Only the reduced (n_query, top_k) arrays
    # cross back to the host, so on GPU the table never travels over PCIe.
    top_ids, top_vals = _take_top_k(votes, min(top_k, n_classes), xp)
    norm_col = norm if np.isscalar(norm) else norm.reshape(-1, 1)

    top_cluster_ids = to_numpy(top_ids)
    top_scores = to_numpy(top_vals / norm_col)
    predicted_label = unique_labels[top_cluster_ids[:, 0]]
    prediction_score = top_scores[:, 0]
    return predicted_label, prediction_score, top_cluster_ids, top_scores


def _predict_label_cpu(
    X_pca_proj: np.ndarray,
    ref_pca: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    weights: Literal["uniform", "distance"] = "uniform",
    top_k: int = 1,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("building index ...")
    index = _build_index(pca_ref=ref_pca, n_neighbors=n_neighbors, metric=metric, **kwargs)

    print("predicting ...")
    neighbor_indices, neighbor_distances = index.query(X_pca_proj, k=n_neighbors)

    unique_labels, label_ids = np.unique(labels, return_inverse=True)
    return _vote_labels(
        label_ids[neighbor_indices], neighbor_distances, unique_labels, n_neighbors, weights, np, top_k
    )


def _predict_label_gpu(
    X_pca_proj: Any,
    ref_pca: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    weights: Literal["uniform", "distance"] = "uniform",
    top_k: int = 1,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import cupy as cp  # type: ignore
    from cuml.neighbors import NearestNeighbors  # type: ignore

    # A cached projection comes back from obsm as a host array; asarray is a no-op when the
    # projection was just computed and is still on the device.
    X_pca_proj = cp.asarray(X_pca_proj, dtype=cp.float32)

    print("building index ...")
    # Fitting with a device array makes cuML mirror it on output, so the neighbour arrays
    # stay on the GPU and feed the vote reduction without a host round-trip.
    index = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, **kwargs)
    index.fit(cp.asarray(ref_pca, dtype=cp.float32))

    print("predicting ...")
    # cuML returns (distances, indices); pynndescent returns them the other way round.
    neighbor_distances, neighbor_indices = index.kneighbors(X_pca_proj)

    unique_labels, label_ids = np.unique(labels, return_inverse=True)
    neighbor_label_ids = cp.asarray(label_ids)[neighbor_indices]
    return _vote_labels(neighbor_label_ids, neighbor_distances, unique_labels, n_neighbors, weights, cp, top_k)


# The reference components are fitted on complete similarity profiles, so a profile that has been
# pruned -- `calculate_motif_similarity(..., n_nearest=k)`, or a stricter `threshold` -- projects
# from a small fraction of the loadings and collapses toward the densest part of reference space,
# i.e. the largest cluster. The default threshold leaves ~66% of each row, so this only fires on
# deliberate pruning. Measured at 5707 reference motifs: n_nearest=100 moved the top family from
# 30% to 76% of seqlets and agreed with the unpruned answer for 23% of them, while the KNN
# confidence score *rose*, so nothing downstream reveals it.
_MIN_PROFILE_DENSITY = 0.5


def _warn_if_profiles_pruned(X_sim: np.ndarray | sparse.csr_array) -> None:
    """Warn when the similarity profiles are too sparse for the reference projection to hold.

    Parameters
    ----------
    X_sim
        Seqlet x reference-motif similarities about to be projected.

    Returns
    -------
    None
        Emits a :class:`UserWarning` when the stored fraction is below
        :data:`_MIN_PROFILE_DENSITY`.
    """
    if not sparse.issparse(X_sim):
        return
    n_rows, n_cols = X_sim.shape
    density = X_sim.nnz / (n_rows * n_cols) if n_rows and n_cols else 1.0
    if density >= _MIN_PROFILE_DENSITY:
        return
    warnings.warn(
        f"adata.X stores only {X_sim.nnz / max(n_rows, 1):.0f} of {n_cols} motif similarities per "
        f"seqlet ({density:.1%} of the full profile). The reference components were fitted on "
        "complete profiles, so predictions will be biased toward the largest reference cluster. "
        "Recompute calculate_motif_similarity without n_nearest and with the default threshold; "
        "use chunk_size to bound memory, which does not change the result.",
        UserWarning,
        stacklevel=3,
    )


def _co_significant_families(
    group: pd.DataFrame,
    annotation_col: str,
    pval_col: str,
    max_pval: float,
    max_families: int,
) -> pd.Series:
    """Reduce one reference cluster's family annotations to the ones worth reporting.

    Keeps every family independently significant at ``max_pval``, up to ``max_families``,
    joined with "+", rather than only the single most extreme one. A reference cluster is
    not always one family: cluster 86 of the v11 collection is AP-2 at p=7e-232 *and* COE
    at p=3e-66, and keeping only the minimum discarded COE even for seqlets that landed in
    the right cluster. Falls back to the single best family when nothing clears
    ``max_pval``, so every cluster still gets an answer.
    """
    g = group.sort_values(pval_col)
    keep = g[g[pval_col] <= max_pval].head(max_families)
    if keep.empty:
        keep = g.head(1)
    return pd.Series(
        {
            annotation_col: "+".join(keep[annotation_col].astype(str)),
            pval_col: keep[pval_col].iloc[0],
        }
    )


def _build_cluster_to_best_fam(
    cluster_tf_family_annot: pd.DataFrame,
    annotation_col: str,
    pval_col: str,
    max_pval: float,
    max_families: int,
) -> pd.DataFrame:
    return cluster_tf_family_annot.groupby("cluster", observed=True)[[annotation_col, pval_col]].apply(
        lambda g: _co_significant_families(g, annotation_col, pval_col, max_pval, max_families)
    )


def _family_membership_matrix(
    ref_metadata: pd.DataFrame,
    motif_names: list,
    family_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """One-hot ``(n_families, n_motifs)`` family membership plus the ordered family names.

    Motifs with no family are dropped rather than collected into a "nan" family.
    """
    fam = ref_metadata.loc[motif_names, family_col].fillna("").astype(str).to_numpy()
    keep = (fam != "nan") & (fam != "None") & (fam != "") & (fam != "NaN")
    if not keep.any():
        raise ValueError(f"No reference motif has a usable value in metadata column '{family_col}'.")
    families, codes = np.unique(fam[keep], return_inverse=True)
    membership = np.zeros((len(families), len(motif_names)))
    membership[codes, np.nonzero(keep)[0]] = 1
    return membership, families


# Below this many seqlets the query is too small to describe what a typical similarity looks
# like, and the correction weakens: see the `rerank` parameter.
_MIN_BACKGROUND_SEQLETS = 5_000


def _score_families_against_background(
    X_sim: np.ndarray | sparse.csr_array,
    family_membership: np.ndarray,
    family_names: np.ndarray,
    n_quantiles: int = 101,
    block: int = 256,
) -> np.ndarray:
    """Score each seqlet against each TF family, correcting for motif promiscuity.

    A raw similarity says nothing on its own: a short, low-information reference motif
    scores high against almost any seqlet, so "high" is its normal state. Converting each
    similarity to a percentile of that motif's own distribution removes exactly that, since
    a motif that is high for everyone puts every seqlet near its median. Family scores are
    the mean percentile over that family's motifs.

    Background and percentile are computed together, one block of motif columns at a time.
    Per-column quantiles are independent, so a block's quantiles equal the full matrix's,
    and nothing the size of ``X_sim`` is ever allocated -- neither a dense copy of a sparse
    input, nor the full percentile matrix, nor a full quantile grid.

    Note this shares a failure mode with :func:`_warn_if_profiles_pruned`: on pruned
    profiles the implicit zeros dominate each column, so the percentiles describe the
    pruning rather than the biology.

    Parameters
    ----------
    X_sim
        Seqlet x reference-motif similarities, sparse or dense.
    family_membership
        One-hot ``(n_families, n_motifs)`` membership from :func:`_family_membership_matrix`.
    family_names
        Family names indexing ``family_membership``'s rows.
    n_quantiles
        Background resolution. 101 gives whole percentiles.
    block
        Motif columns per pass. Sets peak memory: one dense ``(n_seqlets, block)`` array
        plus the sort ``np.quantile`` makes of it.

    Returns
    -------
    Mean background percentile per seqlet and family, shape (n_seqlets, n_families), 0-100.
    """
    n_obs, n_motifs = X_sim.shape
    # CSC so a column block is contiguous; CSR would gather across every row for each block.
    columns = X_sim.tocsc() if sparse.issparse(X_sim) else X_sim
    quantile_levels = np.linspace(0.0, 1.0, n_quantiles)
    fam_sum = np.zeros((n_obs, len(family_names)), dtype=np.float64)

    for start in range(0, n_motifs, block):
        stop = min(start + block, n_motifs)
        chunk = columns[:, start:stop]
        chunk = chunk.toarray() if sparse.issparse(chunk) else np.asarray(chunk)
        chunk = chunk.astype(np.float64, copy=False)

        grid = np.quantile(chunk, quantile_levels, axis=0)
        pct = np.empty_like(chunk)
        for j in range(stop - start):
            pct[:, j] = np.searchsorted(grid[:, j], chunk[:, j])
        # searchsorted returns n_quantiles for anything above the top cut-point.
        pct = np.clip(pct / (n_quantiles - 1) * 100.0, 0.0, 100.0)

        fam_sum += pct @ family_membership[:, start:stop].T

    return fam_sum / family_membership.sum(axis=1)[None, :]


def _rerank_families_with_background(
    top_cluster_ids: np.ndarray,
    top_scores: np.ndarray,
    unique_labels: np.ndarray,
    cluster_to_best_fam: pd.DataFrame,
    fam_scores: np.ndarray,
    family_names: np.ndarray,
    annotation_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick each seqlet's best-scoring family from among the ones its own vote shortlisted.

    The shortlist is the union of families reachable from the seqlet's top-voted reference
    clusters, each of which may name several co-significant families. This only ever
    reorders that shortlist -- it cannot introduce a family the vote found implausible, and
    a one-entry shortlist passes through untouched.

    Vectorised over seqlets. The per-seqlet loop this replaces cost ~30 s per 100k seqlets
    and broke ties by Python set iteration order, which is not stable across runs; ties now
    break by family name.

    Parameters
    ----------
    top_cluster_ids
        Indices into ``unique_labels`` of each seqlet's top-voted clusters, (n_seqlets, k).
    top_scores
        Matching vote shares. A zero means the seqlet had fewer than k voted clusters.
    unique_labels
        Sorted unique reference cluster labels.
    cluster_to_best_fam
        Cluster -> family frame from :func:`_build_cluster_to_best_fam`.
    fam_scores
        Background-percentile scores from :func:`_score_families_against_background`.
    family_names
        Family names indexing ``fam_scores``' columns.
    annotation_col
        Family column of ``cluster_to_best_fam``.

    Returns
    -------
    Tuple of the reranked family per seqlet and its background percentile, or
    "undetermined"/NaN where the shortlist held no family present in ``fam_scores``.
    """
    n_obs, k = top_cluster_ids.shape
    fam_index = {name: i for i, name in enumerate(family_names)}

    # (n_clusters, n_families): which families each reference cluster can vouch for.
    cluster_fam_mask = np.zeros((len(unique_labels), len(family_names)), dtype=bool)
    for ci, cl in enumerate(unique_labels):
        try:
            fam_str = cluster_to_best_fam.loc[int(cl)][annotation_col]
        except (KeyError, ValueError, TypeError):
            continue
        for fam in str(fam_str).split("+"):
            j = fam_index.get(fam)
            if j is not None:
                cluster_fam_mask[ci, j] = True

    candidate_mask = np.zeros((n_obs, len(family_names)), dtype=bool)
    for t in range(k):
        candidate_mask |= cluster_fam_mask[top_cluster_ids[:, t]] & (top_scores[:, t] > 0)[:, None]

    masked = np.where(candidate_mask, fam_scores, -np.inf)
    best = masked.argmax(axis=1)
    has_candidate = candidate_mask.any(axis=1)
    return (
        np.where(has_candidate, family_names[best], "undetermined"),
        np.where(has_candidate, masked[np.arange(n_obs), best], np.nan),
    )


def predict_tf_family_seqlets(
    adata: AnnData,
    motif_collection: MotifCollectionData,
    cluster_resolution: str | float | int,
    n_motifs_per_reference_cluster: int | str,
    key_added: str = "predicted",
    annotation_col: str = "family",
    pval_col: str = "pval_adj",
    recompute: bool = False,
    max_pval: float = 1e-6,
    max_families: int = 3,
    motif_family_col: str = "tf_family",
    rerank: bool = True,
    rerank_top_k: int = 3,
    **kwargs,
):
    """
    Predict TF family for each seqlet by projecting into a reference motif PCA space.

    Seqlets in ``adata`` are projected onto the principal components of the
    reference motif collection and classified with a k-nearest-neighbours
    classifier trained on the reference cluster labels. The predicted cluster
    and its family annotation are stored in ``adata.obs``, alongside a
    promiscuity-corrected reranking of that family call (see ``rerank``).

    Parameters
    ----------
    adata
        AnnData object whose ``var_names`` include all reference motif names
        and whose ``X`` holds motif similarity scores (seqlets × motifs).
    motif_collection
        Motif collection data object providing PCA embeddings, cluster
        annotations, and metadata.
    cluster_resolution
        Leiden clustering resolution to use for label look-up and family
        annotation. Converted to ``str`` internally if an int or float is given.
    n_motifs_per_reference_cluster
        Number of representative motifs per reference cluster that defines
        which PCA embedding and motif subset to use.
    key_added
        Prefix for the keys written to ``adata.obs``.
    recompute
        If False (default), reuse a reference projection already cached in
        ``obsm["X_ref_proj_{n_motifs_per_reference_cluster}"]`` rather than rebuilding it from
        ``.X``. If True, always rebuild it.
    max_pval
        Significance a family must reach to be reported for a reference cluster. Every
        family clearing it is kept, not just the most extreme one, so a cluster that is
        genuinely a mix -- AP-2 and COE, say -- reports both.
    max_families
        Cap on families reported per reference cluster, joined with "+". ``1`` restores the
        previous single-best behaviour exactly.
    motif_family_col
        Column of ``motif_collection.metadata`` holding each reference MOTIF's family.
        Distinct from ``annotation_col``, which is the CLUSTER-level annotation table's
        family column. Reranking needs the two to share a vocabulary, and raises if they
        do not rather than returning "undetermined" everywhere.
    rerank
        Whether to add the promiscuity-corrected family call described in Notes. On by
        default. The background it corrects against is measured from this call's own
        seqlets, so run on the largest, most heterogeneous set available -- ideally the
        whole dataset at once. On a small or single-family subset the seqlets become their
        own background, a genuinely enriched motif reads as typical, and the correction
        weakens; a warning is issued below 5000 seqlets. Reranked calls are therefore
        comparable only across runs over the same seqlet set. Skipped with a warning when
        a cached projection is reused and ``.X`` no longer holds the reference motifs.
    rerank_top_k
        How many of a seqlet's top-voted reference clusters to draw candidate families from.
    **kwargs
        Additional keyword arguments forwarded to the NNDescent index
        (e.g. ``n_neighbors``, ``metric``, ``weights``).

    Returns
    -------
    None
        Results are written directly to ``adata.obs``:

        - ``{key_added}_{cluster_resolution}_predicted_cluster`` — predicted cluster label.
        - ``{key_added}_{cluster_resolution}_predicted_cluster_score`` — KNN confidence score.
        - ``{key_added}_{cluster_resolution}_predicted_family`` — TF family annotation(s) for
          that cluster, possibly several joined with "+" (see ``max_pval``).
        - ``{key_added}_{cluster_resolution}_predicted_family_reranked`` — promiscuity-
          corrected family call, and ``..._reranked_score``, its mean background percentile
          (0-100). Written unless ``rerank`` is False or was skipped.

        and to ``adata.obsm``:

        - ``X_ref_proj_{n_motifs_per_reference_cluster}`` — seqlet coordinates in the reference
          PCA space. Resolution-independent, so re-annotating at another resolution reuses it
          and needs no similarity matrix at all.

    Raises
    ------
    ValueError
        If any reference motif name is missing from ``adata.var_names``, if the metadata
        column for the requested resolution is absent, if no reference motif carries a
        usable ``motif_family_col`` value, or if the cluster-level and motif-level family
        vocabularies do not overlap.

    Notes
    -----
    A k-NN vote over raw similarity can be tipped toward the wrong family by "sticky"
    reference motifs -- short, low-information PWMs that score high against almost any
    seqlet. They inflate similarity, and so the PCA geometry and the vote, for everyone
    rather than only for seqlets that genuinely match them. This showed up as clusters
    confidently and wrongly labelled a common zinc-finger family when their attributed
    motifs were plainly the AP-2/EBF dyad.

    Reranking corrects for it without touching the k-NN or PCA machinery: it re-scores the
    shortlist of families the vote already found plausible, using each similarity converted
    to a percentile of that motif's own background rather than its raw value. A motif that
    is high for everyone has a high background, so it stops winning on loudness alone. Both
    the original and reranked calls are written, so the two can always be compared.
    """
    if isinstance(cluster_resolution, float | int):
        cluster_resolution = str(cluster_resolution)

    cluster_tf_family_annot = motif_collection.get_cluster_annotation(cluster_resolution)

    cluster_to_best_fam = _build_cluster_to_best_fam(
        cluster_tf_family_annot,
        annotation_col=annotation_col,
        pval_col=pval_col,
        max_pval=max_pval,
        max_families=max_families,
    )

    ref_metadata = motif_collection.metadata
    pca_data = motif_collection.get_pca_data(n_motifs_per_cluster=n_motifs_per_reference_cluster)

    metadata_col = f"leiden_{cluster_resolution}"

    if metadata_col not in ref_metadata.columns:
        raise ValueError(f"{metadata_col} not in motif collection metadata.")
    # Reuse ref_metadata rather than re-reading it from the archive.
    labels = ref_metadata.loc[pca_data.obs_names, metadata_col].values

    # The projection depends only on the reference budget, never on the resolution, so it is
    # cached under the budget and reused across resolutions. That is also what lets a caller
    # prune or drop `.X` after the first call: the full similarity profile is needed to build
    # this array, but nothing downstream re-reads it.
    proj_key = f"X_ref_proj_{n_motifs_per_reference_cluster}"
    # Held so reranking can reuse it instead of slicing `.X` a second time; None means the
    # cached projection was used and the similarity matrix was never touched.
    X_sim = None
    if proj_key in adata.obsm and not recompute:
        print(f"Reusing existing reference projection in obsm['{proj_key}'] ...")
        X_pca_proj = adata.obsm[proj_key]
    else:
        # Only the rebuild path touches the similarity matrix, so a cached projection stays
        # usable after `.X` has been pruned down or replaced entirely.
        ref_motif_names = motif_collection.get_motif_names(n_motifs_per_reference_cluster)
        if not all(motif_name in adata.var_names for motif_name in ref_motif_names):
            raise ValueError("Not all reference motifs in adata.")

        # Left sparse on purpose: both projection helpers center without densifying.
        X_sim = adata[:, ref_motif_names].X  # type: ignore
        _warn_if_profiles_pruned(X_sim)
        X_pca_proj = run_accelerated(
            "reference projection",
            lambda: _project_in_reference_gpu(X_sim=X_sim, ref_pcs=pca_data.pcs),
            lambda: _project_in_reference(X_sim=X_sim, ref_pcs=pca_data.pcs),
        )
        adata.obsm[proj_key] = to_numpy(X_pca_proj)

    predict_kwargs = dict(
        X_pca_proj=X_pca_proj,
        ref_pca=pca_data.pca,
        labels=labels,
        top_k=rerank_top_k if rerank else 1,
        **kwargs,
    )
    pred_label, pred_score, top_cluster_ids, top_scores = run_accelerated(
        "KNN family prediction",
        lambda: _predict_label_gpu(**predict_kwargs),  # type: ignore[arg-type]
        lambda: _predict_label_cpu(**predict_kwargs),  # type: ignore[arg-type]
    )
    # Build the cluster -> family lookup once. A scalar .loc per seqlet constructs a new
    # Series each time, which dominates runtime at millions of seqlets.
    cluster_to_fam_name = cluster_to_best_fam[annotation_col].to_dict()
    # Format one string per *distinct* cluster, not per seqlet.
    clusters, inverse = np.unique(pred_label, return_inverse=True)
    fam_per_cluster = np.array(
        [
            f"{cl}|{cluster_to_fam_name[int(cl)]}" if int(cl) in cluster_to_fam_name else "undetermined"
            for cl in clusters
        ]
    )
    pred_fam = fam_per_cluster[inverse]

    key_cluster = f"{key_added}_{cluster_resolution}_predicted_cluster"
    key_score = f"{key_added}_{cluster_resolution}_predicted_cluster_score"
    key_fam = f"{key_added}_{cluster_resolution}_predicted_family"

    adata.obs[key_cluster] = pred_label
    adata.obs[key_score] = pred_score
    adata.obs[key_fam] = pred_fam  # type: ignore

    added_keys = [key_cluster, key_score, key_fam]

    if rerank:
        ref_motif_names = motif_collection.get_motif_names(n_motifs_per_reference_cluster)
        if X_sim is None:
            # Cached projection: `.X` was never read, and may since have been pruned or
            # dropped. Degrade to the unreranked answer rather than failing the whole call.
            if all(motif_name in adata.var_names for motif_name in ref_motif_names):
                X_sim = adata[:, ref_motif_names].X  # type: ignore
            else:
                warnings.warn(
                    f"Reusing the cached projection in obsm['{proj_key}'], but adata.var no longer "
                    "holds the reference motifs, so the background needed for reranking cannot be "
                    "measured. Writing the unreranked call only; pass recompute=True against a "
                    "full similarity matrix to rerank.",
                    UserWarning,
                    stacklevel=2,
                )
                rerank = False

    if rerank:
        family_membership, family_names = _family_membership_matrix(
            ref_metadata, ref_motif_names, family_col=motif_family_col
        )
        annotated = {f for s in cluster_to_best_fam[annotation_col].astype(str) for f in s.split("+") if f}
        if not (annotated & set(family_names)):
            raise ValueError(
                f"Cluster-level families (from '{annotation_col}') and motif-level families (from "
                f"'{motif_family_col}') share no names, so every seqlet would be 'undetermined'. "
                f"Examples: {sorted(annotated)[:3]} vs {sorted(family_names)[:3]}."
            )
        if adata.n_obs < _MIN_BACKGROUND_SEQLETS:
            warnings.warn(
                f"The motif background is measured from only {adata.n_obs} seqlets. On a small or "
                "single-family subset the seqlets become their own background, so a genuinely "
                "enriched motif reads as typical and the correction weakens. Run on the full "
                "seqlet set in one call, or treat the reranked call as unreliable here.",
                UserWarning,
                stacklevel=2,
            )

        print("scoring families against motif background ...")
        fam_scores = _score_families_against_background(X_sim, family_membership, family_names)
        reranked_family, reranked_score = _rerank_families_with_background(
            top_cluster_ids=top_cluster_ids,
            top_scores=top_scores,
            unique_labels=np.unique(labels),
            cluster_to_best_fam=cluster_to_best_fam,
            fam_scores=fam_scores,
            family_names=family_names,
            annotation_col=annotation_col,
        )

        key_reranked = f"{key_added}_{cluster_resolution}_predicted_family_reranked"
        key_reranked_score = f"{key_added}_{cluster_resolution}_predicted_family_reranked_score"
        adata.obs[key_reranked] = reranked_family
        adata.obs[key_reranked_score] = reranked_score
        added_keys += [key_reranked, key_reranked_score]

    print("Added following keys to adata.obs: ")
    for key in added_keys:
        print(f"\t{key}")