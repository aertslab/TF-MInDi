"""Clustering and dimensionality reduction tools for seqlet analysis."""

from __future__ import annotations

import scanpy as sc
from anndata import AnnData
from scipy import sparse

from tfmindi.backends import rapids_singlecell, run_accelerated, to_numpy, using_gpu

# cuSPARSE's SpMM raises CUSPARSE_STATUS_INTERNAL_ERROR once a matrix carries more than
# int32-many nonzeros, and that product is what rapids_singlecell's sparse PCA uses to project
# the data onto its components. The boundary is exact: 2,142,480,245 nonzeros work,
# 2,152,479,962 do not. Revisit when cupy ships 64-bit sparse indices for SpMM -- the
# rapids_singlecell kernels around it are already index-templated.
_CUSPARSE_MAX_NNZ = 2**31 - 1

# Rows per block once the matrix has to be split. At genome scale a 50k-row block holds ~0.12x
# the nonzero ceiling, so the split stays valid for matrices far denser than the ones we see.
_PCA_BLOCK_ROWS = 50_000


def _pca_blocked_gpu(adata: AnnData) -> None:
    """
    Run the GPU sparse PCA over Dask row-blocks, for matrices cuSPARSE cannot take whole.

    rapids_singlecell's Dask path accumulates the gram matrix per block and projects per block,
    so no single block ever reaches the nonzero ceiling that defeats the direct call. This is the
    route rapids_singlecell documents for such matrices; only the block construction is ours.

    Parameters
    ----------
    adata
        AnnData whose sparse ``.X`` exceeds :data:`_CUSPARSE_MAX_NNZ` nonzeros.

    Returns
    -------
    None
        Writes ``obsm["X_pca"]``, ``varm["PCs"]`` and ``uns["pca"]``, matching what
        rapids_singlecell writes when it is handed the AnnData directly.
    """
    import cupy as cp  # type: ignore
    import dask  # type: ignore
    import dask.array as da  # type: ignore
    from cupyx.scipy.sparse import csr_matrix as csr_gpu  # type: ignore

    X = adata.X if adata.X.format == "csr" else adata.X.tocsr()  # type: ignore[union-attr]
    n_rows, n_cols = X.shape

    def _block(start: int, stop: int):
        """Move one row block to the device, narrowing its indices to what cuSPARSE wants."""
        block = X[start:stop]
        return csr_gpu(
            (
                cp.asarray(block.data, dtype=cp.float32),
                # A block's indices always fit int32 even when the whole matrix needs int64.
                cp.asarray(block.indices, dtype=cp.int32),
                cp.asarray(block.indptr, dtype=cp.int32),
            ),
            shape=block.shape,
        )

    bounds = [*range(0, n_rows, _PCA_BLOCK_ROWS), n_rows]
    meta = csr_gpu(cp.zeros((0, 0), dtype=cp.float32))
    blocks = [
        da.from_delayed(dask.delayed(_block)(start, stop), shape=(stop - start, n_cols), dtype="float32", meta=meta)
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
    ]

    # Single-threaded so only one block is resident on the device at a time, and scoped to this
    # call so a caller's own dask configuration is left as they set it.
    with dask.config.set(scheduler="single-threaded"):
        X_pca, components, variance_ratio, variance = rapids_singlecell().pp.pca(
            da.concatenate(blocks, axis=0), return_info=True
        )
        X_pca = to_numpy(X_pca.compute() if hasattr(X_pca, "compute") else X_pca)

    adata.obsm["X_pca"] = X_pca
    adata.varm["PCs"] = to_numpy(components).T
    adata.uns["pca"] = {
        "params": {"zero_center": True, "use_highly_variable": False, "mask_var": None},
        "variance": to_numpy(variance),
        "variance_ratio": to_numpy(variance_ratio),
    }


def _pca_gpu(adata: AnnData) -> None:
    """
    Run the GPU PCA, splitting the matrix only when it is too large to pass whole.

    Parameters
    ----------
    adata
        AnnData to run PCA on, modified in place.

    Returns
    -------
    None
        Writes the same keys as :func:`rapids_singlecell.pp.pca` either way.
    """
    # One attribute read on the ordinary path: matrices below the ceiling go straight through,
    # exactly as before.
    if sparse.issparse(adata.X) and adata.X.nnz > _CUSPARSE_MAX_NNZ:  # type: ignore[union-attr]
        _pca_blocked_gpu(adata)
    else:
        rapids_singlecell().pp.pca(adata)


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
            lambda: _pca_gpu(adata),
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
