"""Clustering and dimensionality reduction tools for seqlet analysis."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import binom

from tfmindi.backends import get_backend, is_gpu_available


def cluster_seqlets(
    adata: AnnData,
    resolution: float = 3.0,
    pca_svd_solver: str | None = None,
    *,
    reduction: str = "pca",
    vae_kwargs: dict | None = None,
    recompute: bool = False,
) -> None:
    """
    Perform complete clustering workflow including dimensionality reduction, clustering, and functional annotation.

    This function performs the following steps:
    1. Dimensionality reduction on similarity matrix - skipped if already present (PCA or VAE, see ``reduction``)
    2. Compute neighborhood graph (GPU-accelerated if available) - skipped if already present
    3. Generate t-SNE embedding (GPU-accelerated if available) - skipped if already present
    4. Leiden clustering at specified resolution (GPU-accelerated if available) - always computed
    5. Calculate mean contribution scores from stored seqlet matrices
    6. Assign DBD annotations based on top motif similarity per seqlet
    7. Map leiden clusters to consensus DBD annotations

    Performance Optimization:
    By default, dimensionality reduction, neighborhood graph, and t-SNE computations are reused
    if already present in the AnnData object. This allows fast re-clustering with different
    resolutions without recomputing expensive preprocessing steps.

    GPU Acceleration:
    When tfmindi[gpu] is installed and CUDA is available, this function automatically uses
    RAPIDS-accelerated implementations for PCA, neighbors, t-SNE, and Leiden.
    The API remains identical between CPU and GPU versions.
    Note: GPU acceleration applies to PCA only; VAE training uses PyTorch's own device handling.

    Parameters
    ----------
    adata
        AnnData object with similarity matrix in .X and seqlet data in .obs.
        Expects .obs to contain seqlet matrices and .var to contain motif annotations.
    resolution
        Clustering resolution for Leiden algorithm (default: 3.0)
    pca_svd_solver
        svd_solver used for calculating pca see: https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html#scanpy.pp.pca (default: None, i.e. choose automatically).
        Ignored when ``reduction="vae"``.
    reduction
        Dimensionality reduction method to use before building the neighbour graph.
        Must be one of:

        - ``"pca"`` *(default)* - linear PCA via scanpy, optionally GPU-accelerated.
          Fast and parameter-free; recommended for most use cases.
        - ``"vae"`` - trains a beta-VAE (non-linear) and stores posterior means in
          ``adata.obsm["X_vae"]``.  Captures non-linear structure that PCA can miss,
          useful when clusters are poorly separated with PCA.
          Requires PyTorch (``pip install torch``).

        The ``recompute`` flag applies to both methods: if the target key
        (``X_pca`` or ``X_vae``) is already present in ``adata.obsm`` and
        ``recompute=False``, the reduction step is skipped.
    vae_kwargs
        Extra keyword arguments forwarded to
        :func:`tfmindi.tl.vae.fit_vae_latents`.  Only used when
        ``reduction="vae"``.  Ignored otherwise.

        Commonly tuned options::

            vae_kwargs = dict(
                latent_dim  = 10,    # bottleneck size; analogous to n_comps for PCA
                hidden      = 512,   # MLP hidden layer width
                n_layers    = 2,     # encoder/decoder depth
                epochs      = 50,    # training epochs; increase to 100-200 for large data
                beta        = 0.1,   # KL weight; higher = more regularised latent space
                batch_size  = 4096,  # reduce if GPU runs out of memory
                num_workers = 0,     # DataLoader workers; 0 is safest in notebooks
                device      = "auto",  # "cpu", "cuda", or "auto"
            )

    recompute
        If False (default), reuse existing dimensionality reduction and neighborhood
        graph computations if available.
        If True, always recompute from scratch.

    Returns
    -------
    Modifies adata in-place with cluster assignments and annotations:
    - adata.obsm["X_pca"]: PCA coordinates (when ``reduction="pca"``)
    - adata.obsm["X_vae"]: VAE latent coordinates (when ``reduction="vae"``)
    - adata.obsm["X_tsne"]: t-SNE coordinates
    - adata.obs["leiden"]: Cluster assignments
    - adata.obs["mean_contrib"]: Mean contribution scores per seqlet
    - adata.obs["seqlet_dbd"]: DBD annotations per seqlet
    - adata.obs["cluster_dbd"]: Consensus DBD annotations per cluster

    Examples
    --------
    >>> import tfmindi as tm
    >>> # adata created with tm.pp.create_seqlet_adata()
    >>>
    >>> # Initial clustering - computes PCA, neighbors, t-SNE, and clustering
    >>> tm.tl.cluster_seqlets(adata, resolution=3.0)
    >>> print(f"Found {adata.obs['leiden'].nunique()} clusters")
    >>>
    >>> # Fast re-clustering with different resolution - reuses PCA, neighbors, t-SNE
    >>> tm.tl.cluster_seqlets(adata, resolution=5.0)
    >>> print(f"Found {adata.obs['leiden'].nunique()} clusters")
    >>>
    >>> # Force recomputation of all steps
    >>> tm.tl.cluster_seqlets(adata, resolution=3.0, recompute=True)
    >>>
    >>> # VAE-based clustering (non-linear; requires PyTorch)
    >>> tm.tl.cluster_seqlets(
    ...     adata,
    ...     reduction="vae",
    ...     vae_kwargs=dict(latent_dim=10, epochs=100, beta=0.5),
    ...     resolution=3.0,
    ... )
    >>> print(f"Found {adata.obs['leiden'].nunique()} clusters")
    """
    if adata.X is None:
        raise ValueError("adata.X is None. Similarity matrix is required for motif assignment.")

    _valid_reductions = ("pca", "vae")
    if reduction not in _valid_reductions:
        raise ValueError(f"reduction must be one of {_valid_reductions!r}, got {reduction!r}.")

    # Determine if we should use GPU at runtime
    _using_gpu = get_backend() == "gpu" and is_gpu_available()
    if _using_gpu:
        import rapids_singlecell as rsc  # type: ignore
    backend_info = "GPU-accelerated" if _using_gpu else "CPU"
    print(f"Using {backend_info} backend for clustering operations...")

    # ------------------------------------------------------------------
    # Dimensionality reduction: PCA (default) or VAE
    # ------------------------------------------------------------------
    if reduction == "pca":
        # Check if PCA already exists and we don't need to recompute
        if "X_pca" in adata.obsm and not recompute:
            print("Reusing existing PCA...")
        else:
            print("Computing PCA...")
            if _using_gpu:
                try:
                    rsc.pp.pca(adata)
                except Exception as e:  # noqa: BLE001
                    warnings.warn(f"GPU PCA failed: {e}. Falling back to CPU.", UserWarning, stacklevel=2)
                    sc.tl.pca(adata, svd_solver=pca_svd_solver)
            else:
                sc.tl.pca(adata, svd_solver=pca_svd_solver)
        _reduction_rep = "X_pca"

    else:  # reduction == "vae"
        if "X_vae" in adata.obsm and not recompute:
            print("Reusing existing VAE embedding...")
        else:
            # Lazy import: torch is only required when reduction="vae"
            from tfmindi.tl.vae import fit_vae_latents  # noqa: PLC0415

            _kw: dict = dict(
                latent_dim=10,
                hidden=256,
                n_layers=2,
                beta=0.05,
                lr=1e-3,
                epochs=100,
                batch_size=256,
                free_bits=0.5,
                patience=5,
                num_workers=0,
                max_grad_norm=5.0,
                use_amp=True,
                dropout=0.1,
                n_sample_stats=20_000,
                device="auto",
                verbose=True,
            )
            if vae_kwargs:
                _kw.update(vae_kwargs)

            print(
                f"Training VAE (latent_dim={_kw['latent_dim']}, epochs={_kw['epochs']}, "
                f"beta={_kw['beta']}, device={_kw['device']})..."
            )
            adata.obsm["X_vae"] = fit_vae_latents(adata.X, **_kw)
        _reduction_rep = "X_vae"

    # ------------------------------------------------------------------
    # Neighbourhood graph (uses whichever reduction was computed above)
    # ------------------------------------------------------------------

    # Check if neighborhood graph already exists and we don't need to recompute
    if "connectivities" in adata.obsp and "distances" in adata.obsp and not recompute:
        print("Reusing existing neighborhood graph...")
    else:
        print(f"Computing neighborhood graph (use_rep='{_reduction_rep}')...")
        if _using_gpu:
            try:
                rsc.pp.neighbors(adata, use_rep=_reduction_rep)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"GPU neighbors failed: {e}. Falling back to CPU.", UserWarning, stacklevel=2)
                sc.pp.neighbors(adata, use_rep=_reduction_rep)
        else:
            sc.pp.neighbors(adata, use_rep=_reduction_rep)

    # Check if t-SNE already exists and we don't need to recompute
    if "X_tsne" in adata.obsm and not recompute:
        print("Reusing existing t-SNE embedding...")
    else:
        print(f"Computing t-SNE embedding (use_rep='{_reduction_rep}')...")
        if _using_gpu:
            try:
                rsc.tl.tsne(adata, use_rep=_reduction_rep)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"GPU t-SNE failed: {e}. Falling back to CPU.", UserWarning, stacklevel=2)
                sc.tl.tsne(adata, use_rep=_reduction_rep)
        else:
            sc.tl.tsne(adata, use_rep=_reduction_rep)

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
        # Annotate clusters based on TF-family enrichment:
        #
        # Background: Each seqlet has been pre-labeled with its best matching TF-family.
        #
        # Goal: For each cluster, determine which TF-families are statistically enriched
        # (appear more frequently than expected by chance).
        #
        # Statistical Framework:
        # - Null hypothesis: Seqlets in a cluster are randomly sampled from the overall
        #   distribution of TF-family annotations
        # - For each TF-family in a cluster, we observe k occurrences out of N seqlets
        # - We compare this to the expected frequency based on the background probability p
        #   (the fraction of all seqlets annotated to that TF-family)
        #
        # Model: Binomial distribution with parameters:
        # - n = N (cluster size, number of seqlets in the cluster)
        # - p = background probability of a TF-family
        # - k = observed count of that TF-family in the cluster
        #
        # Test: One-tailed binomial test asking "Is k significantly greater than expected?"
        # This gives us a p-value for enrichment: P(X >= k | n, p)

        # background probability.
        dbd_to_probability = adata.var["dbd"].value_counts(normalize=True, dropna=False).to_dict()

        def get_dbd_min_pval(df: pd.Series) -> str:
            """
            Get the dbd with the lowest p-value according to binomial distribution.

            Parameters
            ----------
            df: pandas series of value counts sorted descending
            """
            N = sum(df)  # number of samples drawn (i.e. number of seqlets per cluster)
            min_pval = np.inf
            best_dbd = df.head(1).index[0]  # take most often occuring annotation by default.
            for dbd, k in df.to_dict().items():
                #  k = n_success
                #  N = number of draws
                #  dbd_to_p = prob of sucess
                p_value = binom.sf(k - 1, N, dbd_to_probability[dbd])
                if p_value < min_pval:
                    min_pval = p_value
                    best_dbd = dbd
            return best_dbd

        cluster_dbds = []
        # Group by cluster and find consensus DBD
        cluster_dbd_mapping = (
            adata.obs[["leiden", "seqlet_dbd"]]
            .dropna()
            .groupby("leiden", observed=True)["seqlet_dbd"]
            .agg(lambda seqlet_dbd_per_cluster: get_dbd_min_pval(seqlet_dbd_per_cluster.value_counts(dropna=False)))
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

    # Generate consistent colors for clustering results
    from tfmindi.pl._utils import ensure_colors

    # Generate colors for leiden clusters
    if "leiden" in adata.obs.columns:
        ensure_colors(adata, "leiden", cmap="tab20")

    # Generate colors for cluster DBD annotations
    if "cluster_dbd" in adata.obs.columns:
        ensure_colors(adata, "cluster_dbd", cmap="tab10")