"""Tools to project seqlets in reference dataset and annotate seqlets."""

import itertools
from typing import Literal

import numpy as np
from anndata import AnnData  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from tqdm import tqdm  # type: ignore

from tfmindi.datasets import MotifCollectionData


def _project_in_reference(X_sim: np.ndarray, ref_pcs: np.ndarray) -> np.ndarray:
    X_sim_center = X_sim - X_sim.mean(1)[:, None]
    X_pca_proj = X_sim_center @ ref_pcs
    return X_pca_proj


def _fit_knn(
    pca_ref: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    weights: Literal["uniform", "distance"] = "distance",
    **kwargs,
) -> KNeighborsClassifier:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights, **kwargs)
    knn.fit(pca_ref, labels)
    return knn


def _predict_label(
    X_sim: np.ndarray,
    ref_pcs: np.ndarray,
    ref_pca: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 500,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    weights: Literal["uniform", "distance"] = "distance",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    # TODO: improve efficiency (FAISS?)
    X_pca_proj = _project_in_reference(X_sim=X_sim, ref_pcs=ref_pcs)
    knn = _fit_knn(pca_ref=ref_pca, labels=labels, n_neighbors=n_neighbors, metric=metric, weights=weights, **kwargs)

    predicted_label = []
    prediction_score = []
    for batch_idc in tqdm(
        itertools.batched(np.arange(X_pca_proj.shape[0]), batch_size), total=X_pca_proj.shape[0] // batch_size
    ):
        X_pca_proj_batch = X_pca_proj[np.array(batch_idc),]
        predicted_label.extend(knn.predict(X_pca_proj_batch))
        prediction_score.extend(knn.predict_proba(X_pca_proj_batch).max(axis=1))  # type: ignore

    return (np.array(predicted_label), np.array(prediction_score))


def predict_tf_family_seqlets(
    adata: AnnData,
    motif_collection: MotifCollectionData,
    cluster_resolution: str | float | int,
    n_motifs_per_reference_cluster: int | str,
    key_added: str = "predicted",
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
        Additional keyword arguments forwarded to the KNN classifier
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

    cluster_to_best_fam = cluster_tf_family_annot.groupby("cluster", observed=True)[["annotation", "pval"]].apply(
        lambda x: x.iloc[np.argmin(x.pval)]
    )

    ref_metadata = motif_collection.metadata
    pca_data = motif_collection.get_pca_data(n_motifs_per_cluster=n_motifs_per_reference_cluster)

    metadata_col = f"leiden_{cluster_resolution}"

    if metadata_col not in ref_metadata.columns:
        raise ValueError(f"{metadata_col} not in motif collection metadata.")
    labels = motif_collection.metadata[metadata_col].values

    X_sim = adata[:, ref_motif_names].X  # type: ignore
    if hasattr(X_sim, "toarray"):
        X_sim = X_sim.toarray()  # type: ignore

    pred_label, pred_score = _predict_label(
        X_sim=X_sim,  # type: ignore
        ref_pcs=pca_data.pcs,
        ref_pca=pca_data.pca,
        labels=labels,  # type: ignore
        **kwargs,
    )
    pred_fam = [cluster_to_best_fam.loc[int(cl)]["annotation"] for cl in pred_label]

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
