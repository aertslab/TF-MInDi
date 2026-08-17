"""Tools to project seqlets in reference dataset and annotate seqlets."""

from typing import Any, Literal

import numpy as np
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


def _project_in_reference_gpu(X_sim: np.ndarray | sparse.csr_array, ref_pcs: np.ndarray) -> Any:
    """Project seqlet similarities onto the reference PCs on the GPU.

    Uses the same rank-1 centering identity as :func:`_project_in_reference`, and leaves
    the result on the device so the nearest-neighbour query that follows does not have to
    ship the projected coordinates back and forth.

    Parameters
    ----------
    X_sim
        Seqlet x reference-motif similarities, sparse or dense.
    ref_pcs
        Reference principal component loadings, shape (n_motifs, n_PCs).

    Returns
    -------
    Device-resident projected coordinates, shape (n_seqlets, n_PCs).
    """
    import cupy as cp  # type: ignore
    import cupyx.scipy.sparse as cusparse  # type: ignore

    pcs = cp.asarray(ref_pcs, dtype=cp.float32)
    if sparse.issparse(X_sim):
        # cupyx takes scipy matrices, not the newer array types.
        X_gpu = cusparse.csr_matrix(sparse.csr_matrix(X_sim).astype(np.float32))
        row_means = X_gpu.mean(axis=1).reshape(-1)
        return X_gpu @ pcs - cp.outer(row_means, pcs.sum(axis=0))
    X_gpu = cp.asarray(X_sim, dtype=cp.float32)
    return (X_gpu - X_gpu.mean(1)[:, None]) @ pcs


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
        return np.bincount(
            flat_bins, weights=None if w is None else w.ravel(), minlength=n_query * n_classes
        ).reshape(n_query, n_classes)

    # The same flat bincount is pathological on the GPU: cupy routes it through CUB's
    # histogram, whose temporary storage scales with the bin count, so a 10^8-bin table asks
    # for hundreds of GB and fails. Scattering into a preallocated table costs only the table.
    import cupyx  # type: ignore

    votes = xp.zeros((n_query, n_classes), dtype=xp.int32 if w is None else xp.float64)
    cupyx.scatter_add(votes, (xp.arange(n_query)[:, None], neighbor_label_ids), 1 if w is None else w)
    return votes


def _vote_labels(
    neighbor_label_ids: Any,
    neighbor_distances: Any,
    unique_labels: np.ndarray,
    n_neighbors: int,
    weights: Literal["uniform", "distance"],
    xp: Any,
) -> tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    Tuple of the winning label per query and its normalised score, both as host arrays.
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
    # allocation here (tens of GB at genome scale). Only the two reduced (n_query,) vectors
    # cross back to the host, so on GPU the table never travels over PCIe.
    predicted_label = unique_labels[to_numpy(votes.argmax(axis=1))]
    prediction_score = to_numpy(votes.max(axis=1) / norm)
    return predicted_label, prediction_score


def _predict_label_cpu(
    X_sim: np.ndarray,
    ref_pcs: np.ndarray,
    ref_pca: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    weights: Literal["uniform", "distance"] = "uniform",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    X_pca_proj = _project_in_reference(X_sim=X_sim, ref_pcs=ref_pcs)

    print("building index ...")
    index = _build_index(pca_ref=ref_pca, n_neighbors=n_neighbors, metric=metric, **kwargs)

    print("predicting ...")
    neighbor_indices, neighbor_distances = index.query(X_pca_proj, k=n_neighbors)

    unique_labels, label_ids = np.unique(labels, return_inverse=True)
    return _vote_labels(
        label_ids[neighbor_indices], neighbor_distances, unique_labels, n_neighbors, weights, np
    )


def _predict_label_gpu(
    X_sim: np.ndarray,
    ref_pcs: np.ndarray,
    ref_pca: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    weights: Literal["uniform", "distance"] = "uniform",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    import cupy as cp  # type: ignore
    from cuml.neighbors import NearestNeighbors  # type: ignore

    X_pca_proj = _project_in_reference_gpu(X_sim=X_sim, ref_pcs=ref_pcs)

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
    return _vote_labels(neighbor_label_ids, neighbor_distances, unique_labels, n_neighbors, weights, cp)


def predict_tf_family_seqlets(
    adata: AnnData,
    motif_collection: MotifCollectionData,
    cluster_resolution: str | float | int,
    n_motifs_per_reference_cluster: int | str,
    key_added: str = "predicted",
    annotation_col: str = "family",
    pval_col: str = "pval_adj",
    **kwargs,
):
    """
    Predict TF family for each seqlet by projecting into a reference motif PCA space.

    Seqlets in ``adata`` are projected onto the principal components of the
    reference motif collection and classified with a k-nearest-neighbours
    classifier trained on the reference cluster labels. The predicted cluster
    and its best-scoring family annotation are stored in ``adata.obs``.

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
        Prefix for the three keys written to ``adata.obs``.
    **kwargs
        Additional keyword arguments forwarded to the NNDescent index
        (e.g. ``n_neighbors``, ``metric``, ``weights``).

    Returns
    -------
    None
        Results are written directly to ``adata.obs``:

        - ``{key_added}_{cluster_resolution}_predicted_cluster`` — predicted cluster label.
        - ``{key_added}_{cluster_resolution}_predicted_cluster_score`` — KNN confidence score.
        - ``{key_added}_{cluster_resolution}_predicted_family`` — best-scoring TF family annotation.

    Raises
    ------
    ValueError
        If any reference motif name is missing from ``adata.var_names``, or if
        the metadata column for the requested resolution is absent.
    """
    ref_motif_names = motif_collection.get_motif_names(n_motifs_per_reference_cluster)
    if not all(motif_name in adata.var_names for motif_name in ref_motif_names):
        raise ValueError("Not all reference motifs in adata.")

    if isinstance(cluster_resolution, float | int):
        cluster_resolution = str(cluster_resolution)

    cluster_tf_family_annot = motif_collection.get_cluster_annotation(cluster_resolution)

    cluster_to_best_fam = cluster_tf_family_annot.groupby("cluster", observed=True)[[annotation_col, pval_col]].apply(
        lambda x: x.iloc[np.argmin(x[pval_col])]
    )

    ref_metadata = motif_collection.metadata
    pca_data = motif_collection.get_pca_data(n_motifs_per_cluster=n_motifs_per_reference_cluster)

    metadata_col = f"leiden_{cluster_resolution}"

    if metadata_col not in ref_metadata.columns:
        raise ValueError(f"{metadata_col} not in motif collection metadata.")
    # Reuse ref_metadata rather than re-reading it from the archive.
    labels = ref_metadata.loc[pca_data.obs_names, metadata_col].values

    # Left sparse on purpose: both projection helpers center without densifying.
    X_sim = adata[:, ref_motif_names].X  # type: ignore

    predict_kwargs = dict(
        X_sim=X_sim,
        ref_pcs=pca_data.pcs,
        ref_pca=pca_data.pca,
        labels=labels,
        **kwargs,
    )
    pred_label, pred_score = run_accelerated(
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

    print("Added following keys to adata.obs: ")
    print(f"\t{key_cluster}")
    print(f"\t{key_score}")
    print(f"\t{key_fam}")
