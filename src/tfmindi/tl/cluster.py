"""Clustering and dimensionality reduction tools for seqlet analysis."""

from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData


def cluster_seqlets(adata: AnnData, resolution: float = 3.0) -> None:
    """
    Perform complete clustering workflow including dimensionality reduction, clustering, and functional annotation.

    This function performs the following steps:
    1. PCA on similarity matrix
    2. Compute neighborhood graph
    3. Generate t-SNE embedding
    4. Leiden clustering at specified resolution
    5. Calculate mean contribution scores from stored seqlet matrices
    6. Assign DBD annotations based on top motif similarity per seqlet
    7. Map leiden clusters to consensus DBD annotations

    Parameters
    ----------
    adata
        AnnData object with similarity matrix in .X and seqlet data in .obs.
        Expects .obs to contain seqlet matrices and .var to contain motif annotations.
    resolution
        Clustering resolution for Leiden algorithm (default: 3.0)

    Returns
    -------
    Modifies adata in-place with cluster assignments and annotations:
    - adata.obsm["X_pca"]: PCA coordinates
    - adata.obsm["X_tsne"]: t-SNE coordinates
    - adata.obs["leiden"]: Cluster assignments
    - adata.obs["mean_contrib"]: Mean contribution scores per seqlet
    - adata.obs["seqlet_dbd"]: DBD annotations per seqlet
    - adata.obs["cluster_dbd"]: Consensus DBD annotations per cluster

    Examples
    --------
    >>> import tfmindi as tm
    >>> # adata created with tm.pp.create_seqlet_adata()
    >>> tm.tl.cluster_seqlets(adata, resolution=3.0)
    >>> print(adata.obs["leiden"].value_counts())
    >>> print(adata.obs["cluster_dbd"].value_counts())
    """
    if adata.X is None:
        raise ValueError("adata.X is None. Similarity matrix is required for motif assignment.")
    print("Computing PCA...")
    sc.tl.pca(adata)

    print("Computing neighborhood graph...")
    sc.pp.neighbors(adata)

    print("Computing t-SNE embedding...")
    sc.tl.tsne(adata)

    print(f"Performing Leiden clustering with resolution {resolution}...")
    sc.tl.leiden(adata, flavor="igraph", resolution=resolution)

    if "seqlet_matrix" in adata.obs.columns:
        mean_contribs = []
        for seqlet_matrix in adata.obs["seqlet_matrix"]:
            mean_contrib = np.abs(seqlet_matrix).mean()
            mean_contribs.append(mean_contrib)
        adata.obs["mean_contrib"] = mean_contribs
    else:
        print("Warning: No seqlet matrices found in adata.obs['seqlet_matrix']")
        adata.obs["mean_contrib"] = np.nan

    if "dbd" in adata.var.columns:
        seqlet_dbds = []
        for i in range(adata.n_obs):
            # Find motif with highest similarity for this seqlet
            top_motif_idx = adata.X[i, :].argmax()  # type: ignore
            top_motif_name = adata.var.index[top_motif_idx]  # type: ignore
            # Get DBD annotation for this motif
            dbd = adata.var.loc[top_motif_name, "dbd"]
            seqlet_dbds.append(dbd)
        adata.obs["seqlet_dbd"] = seqlet_dbds
    else:
        print("Warning: No DBD annotations found in adata.var['dbd']")
        adata.obs["seqlet_dbd"] = np.nan

    if "seqlet_dbd" in adata.obs.columns and "leiden" in adata.obs.columns:
        cluster_dbds = []
        # Group by cluster and find consensus DBD
        cluster_dbd_mapping = (
            adata.obs[["leiden", "seqlet_dbd"]]
            .dropna()
            .groupby("leiden", observed=True)["seqlet_dbd"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
            .to_dict()
        )

        for cluster in adata.obs["leiden"]:
            consensus_dbd = cluster_dbd_mapping.get(cluster, np.nan)
            cluster_dbds.append(consensus_dbd)

        adata.obs["cluster_dbd"] = cluster_dbds
    else:
        print("Warning: Cannot compute consensus DBD annotations")
        adata.obs["cluster_dbd"] = np.nan

    print(f"Clustering complete. Found {adata.obs['leiden'].nunique()} clusters.")
    print(f"DBD annotation coverage: {adata.obs['cluster_dbd'].notna().sum()}/{adata.n_obs} seqlets")
