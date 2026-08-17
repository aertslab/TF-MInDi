"""Clustering and dimensionality reduction tools for seqlet analysis."""

from __future__ import annotations

import scanpy as sc
from anndata import AnnData

from tfmindi.backends import rapids_singlecell, run_accelerated, using_gpu


def embed_and_cluster(
    adata: AnnData, resolution: float = 3.0, pca_svd_solver: str | None = None, *, recompute: bool = False
) -> None:
    """
    Embed seqlets with PCA and t-SNE and cluster them with Leiden.

    Runs PCA on the similarity matrix, builds a neighbourhood graph, computes a t-SNE
    embedding and clusters at the requested resolution. Seqlet annotation is a separate
    step -- see :func:`tfmindi.tl.predict_tf_family_seqlets`.

    Performance Optimization:
    By default, PCA, neighborhood graph, and t-SNE computations are reused if already present
    in the AnnData object. This allows fast re-clustering with different resolutions without
    recomputing expensive preprocessing steps.

    GPU Acceleration:
    When tfmindi[gpu] is installed and CUDA is available, this function automatically uses
    RAPIDS-accelerated implementations. The API remains identical between CPU and GPU versions.

    Parameters
    ----------
    adata
        AnnData object with the seqlet x motif similarity matrix in .X.
    resolution
        Clustering resolution for Leiden algorithm (default: 3.0)
    pca_svd_solver
        svd_solver used for calculating pca see: https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html#scanpy.pp.pca (default: None, i.e. choose automatically).
    recompute
        If False (default), reuse existing PCA and neighborhood graph computations if available.
        If True, always recompute PCA, neighbors, and t-SNE from scratch.

    Returns
    -------
    Modifies adata in-place:

    - ``adata.obsm["X_pca"]``: PCA coordinates
    - ``adata.obsm["X_tsne"]``: t-SNE coordinates
    - ``adata.obs["leiden"]``: cluster assignments
    - ``adata.uns["leiden_colors"]``: cluster palette

    Examples
    --------
    >>> import tfmindi as tm
    >>> # adata created with tm.pp.create_seqlet_adata()
    >>>
    >>> # Initial clustering - computes PCA, neighbors, t-SNE, and clustering
    >>> tm.tl.embed_and_cluster(adata, resolution=3.0)
    >>> print(f"Found {adata.obs['leiden'].nunique()} clusters")
    >>>
    >>> # Fast re-clustering with different resolution - reuses PCA, neighbors, t-SNE
    >>> tm.tl.embed_and_cluster(adata, resolution=5.0)
    >>>
    >>> # Force recomputation of all steps
    >>> tm.tl.embed_and_cluster(adata, resolution=3.0, recompute=True)
    """
    if adata.X is None:
        raise ValueError("adata.X is None. Similarity matrix is required for clustering.")

    backend_info = "GPU-accelerated" if using_gpu() else "CPU"
    print(f"Using {backend_info} backend for clustering operations...")

    # Check if PCA already exists and we don't need to recompute
    if "X_pca" in adata.obsm and not recompute:
        print("Reusing existing PCA...")
    else:
        print("Computing PCA...")
        run_accelerated(
            "PCA",
            lambda: rapids_singlecell().pp.pca(adata),
            lambda: sc.tl.pca(adata, svd_solver=pca_svd_solver),
        )

    # Check if neighborhood graph already exists and we don't need to recompute
    if "connectivities" in adata.obsp and "distances" in adata.obsp and not recompute:
        print("Reusing existing neighborhood graph...")
    else:
        print("Computing neighborhood graph...")
        run_accelerated(
            "neighbors",
            lambda: rapids_singlecell().pp.neighbors(adata, use_rep="X_pca"),
            lambda: sc.pp.neighbors(adata, use_rep="X_pca"),
        )

    # Check if t-SNE already exists and we don't need to recompute
    if "X_tsne" in adata.obsm and not recompute:
        print("Reusing existing t-SNE embedding...")
    else:
        print("Computing t-SNE embedding...")
        run_accelerated(
            "t-SNE",
            lambda: rapids_singlecell().tl.tsne(adata, use_rep="X_pca"),
            lambda: sc.tl.tsne(adata, use_rep="X_pca"),
        )

    print(f"Performing Leiden clustering with resolution {resolution}...")
    run_accelerated(
        "Leiden clustering",
        lambda: rapids_singlecell().tl.leiden(adata, resolution=resolution),
        lambda: sc.tl.leiden(adata, flavor="igraph", resolution=resolution),
    )

    print(f"Clustering complete. Found {adata.obs['leiden'].nunique()} clusters.")

    from tfmindi.pl._utils import ensure_colors

    ensure_colors(adata, "leiden", cmap="tab20")
