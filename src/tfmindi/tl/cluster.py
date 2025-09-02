"""Clustering and dimensionality reduction tools for seqlet analysis."""

from __future__ import annotations

import warnings

import numpy as np
import scanpy as sc
from anndata import AnnData

from tfmindi.backends import get_backend, is_gpu_available


def cluster_seqlets(adata: AnnData, resolution: float = 3.0) -> None:
    """
    Perform complete clustering workflow including dimensionality reduction, clustering, and functional annotation.

    This function performs the following steps:
    1. PCA on similarity matrix (GPU-accelerated if available)
    2. Compute neighborhood graph (GPU-accelerated if available)
    3. Generate t-SNE embedding (GPU-accelerated if available)
    4. Leiden clustering at specified resolution (GPU-accelerated if available)
    5. Calculate mean contribution scores from stored seqlet matrices
    6. Assign DBD annotations based on top motif similarity per seqlet
    7. Map leiden clusters to consensus DBD annotations

    GPU Acceleration:
    When tfmindi[gpu] is installed and CUDA is available, this function automatically uses
    RAPIDS-accelerated implementations. The API remains identical between CPU and GPU versions.

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

    # Determine if we should use GPU at runtime
    _using_gpu = get_backend() == "gpu" and is_gpu_available()
    if _using_gpu:
        import rapids_singlecell as rsc  # type: ignore
    backend_info = "GPU-accelerated" if _using_gpu else "CPU"
    print(f"Using {backend_info} backend for clustering operations...")

    print("Computing PCA...")
    if _using_gpu:
        try:
            rsc.pp.pca(adata)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"GPU PCA failed: {e}. Falling back to CPU.", UserWarning, stacklevel=2)
            sc.tl.pca(adata, svd_solver="covariance_eigh")
    else:
        sc.tl.pca(adata, svd_solver="covariance_eigh")

    print("Computing neighborhood graph...")
    if _using_gpu:
        try:
            rsc.pp.neighbors(adata, use_rep="X_pca")
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"GPU neighbors failed: {e}. Falling back to CPU.", UserWarning, stacklevel=2)
            sc.pp.neighbors(adata, use_rep="X_pca")
    else:
        sc.pp.neighbors(adata, use_rep="X_pca")

    print("Computing t-SNE embedding...")
    if _using_gpu:
        try:
            rsc.tl.tsne(adata, use_rep="X_pca")
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"GPU t-SNE failed: {e}. Falling back to CPU.", UserWarning, stacklevel=2)
            sc.tl.tsne(adata, use_rep="X_pca")
    else:
        sc.tl.tsne(adata, use_rep="X_pca")

    print(f"Performing Leiden clustering with resolution {resolution}...")
    if _using_gpu:
        try:
            rsc.tl.leiden(adata, resolution=resolution)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"GPU Leiden clustering failed: {e}. Falling back to CPU.", UserWarning, stacklevel=2)
            sc.tl.leiden(adata, flavor="igraph", resolution=resolution)
    else:
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
        # find top motif for all seqlets at once
        # For sparse matrices, argmax along axis=1 gives the column index of max value in each row
        from scipy import sparse

        if sparse.issparse(adata.X):
            # argmax on sparse matrix can return 2D array, ensure 1D
            top_motif_indices = np.asarray(adata.X.argmax(axis=1)).flatten()
        else:
            top_motif_indices = adata.X.argmax(axis=1)

        top_motif_names = adata.var.index[top_motif_indices]
        seqlet_dbds = [adata.var.loc[motif_name, "dbd"] for motif_name in top_motif_names]
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
