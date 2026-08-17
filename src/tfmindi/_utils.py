"""Small helpers shared across the pp/tl/pl layers."""

from __future__ import annotations

from anndata import AnnData


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
