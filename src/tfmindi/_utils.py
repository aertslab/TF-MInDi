"""Small helpers shared across the pp/tl/pl layers."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from tfmindi.backends import run_accelerated, to_numpy


def accelerated_tsne(X: np.ndarray, **kwargs) -> np.ndarray:
    """Compute a t-SNE embedding, on the GPU when that backend is active.

    t-SNE is the one step the package runs on several different matrices (seqlets,
    regions, region-topic weights), so the cuML/scikit-learn dispatch lives here rather
    than being spelled out at each call site.

    Parameters
    ----------
    X
        Feature matrix to embed, shape (n_samples, n_features).
    **kwargs
        Passed to the t-SNE implementation. ``n_jobs`` is a scikit-learn threading knob
        with no cuML counterpart and is dropped on the GPU path.

    Returns
    -------
    Host array of embedded coordinates, shape (n_samples, n_components).
    """

    def _gpu() -> np.ndarray:
        from cuml.manifold import TSNE as cuTSNE  # type: ignore

        gpu_kwargs = {k: v for k, v in kwargs.items() if k != "n_jobs"}
        return to_numpy(cuTSNE(**gpu_kwargs).fit_transform(X))

    def _cpu() -> np.ndarray:
        from sklearn.manifold import TSNE

        return TSNE(**kwargs).fit_transform(X)

    return run_accelerated("t-SNE", _gpu, _cpu)


def resolve_annotation_col(adata: AnnData, annotation_col: str | None) -> str:
    """Validate the seqlet annotation column a function was asked to use.

    The column written by :func:`tfmindi.tl.predict_tf_family_seqlets` carries the clustering
    resolution in its name, so there is no default that is correct for every object; callers
    must name it. This raises with the candidate columns rather than letting a stale name
    silently annotate with the wrong thing.

    Parameters
    ----------
    adata
        Seqlet AnnData.
    annotation_col
        Column name in ``adata.obs``, or None.

    Returns
    -------
    The validated column name.

    Raises
    ------
    ValueError
        If ``annotation_col`` is None or absent from ``adata.obs``.
    """
    candidates = [col for col in adata.obs.columns if col.endswith("_predicted_family")]
    hint = f" Available: {candidates}." if candidates else " Run tfmindi.tl.predict_tf_family_seqlets() first."
    if annotation_col is None:
        raise ValueError(f"annotation_col is required.{hint}")
    if annotation_col not in adata.obs.columns:
        raise ValueError(f"Column '{annotation_col}' not found in adata.obs.{hint}")
    return annotation_col
