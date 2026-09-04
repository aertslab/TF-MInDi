"""Tools to extract cell type-level enhancer codes."""

from collections.abc import Callable, Hashable
from typing import Any

import anndata as ad
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from tfmindi.datasets import MotifCollectionData


def get_code_table(
    metadata: pd.DataFrame,
    seqlet_annotation_key: str,
    cell_type_key: str,
    aggregate_by: str,
    aggregation_func: Callable | str = "mean",
    cell_type_order: list[str] | None = None,
    order_diagonal: bool = False,
    normalize_per_aggregate_by: bool = False,
) -> pd.DataFrame:
    """Pivot per-seqlet metadata into a cell type x annotation enhancer code table.

    Rows are grouped by ``cell_type_key`` and ``seqlet_annotation_key``, the ``aggregate_by``
    column is reduced within each group with ``aggregation_func``, and the result is pivoted
    so cell types become columns and annotations become the row index.

    Parameters
    ----------
    metadata
        Per-seqlet metadata table containing ``seqlet_annotation_key``, ``cell_type_key`` and
        ``aggregate_by`` as columns.
    seqlet_annotation_key
        Column holding the per-seqlet annotation (e.g. predicted TF family) that becomes the
        row index of the returned table.
    cell_type_key
        Column holding the per-seqlet cell type that becomes the columns of the returned
        table.
    aggregate_by
        Column reduced within each (cell type, annotation) group, e.g. a region identifier or
        score column.
    aggregation_func
        Aggregation applied to ``aggregate_by`` within each group, passed through to
        :meth:`pandas.core.groupby.DataFrameGroupBy.agg` (e.g. ``"mean"``, ``"count"``, or a
        custom callable).
    cell_type_order
        Explicit column order for the returned table. Every entry must be present in
        ``metadata[cell_type_key]``; cell types not listed are dropped from the result.
    order_diagonal
        If True, reorder rows so each annotation is placed at the position of the cell type
        (from ``cell_type_order``) at which it reaches its maximum value, giving the table a
        block-diagonal layout when plotted as a heatmap. Requires ``cell_type_order``.
    normalize_per_aggregate_by
        If True, divide each cell type's column by the number of distinct ``aggregate_by``
        values recorded for that cell type in ``metadata``, turning per-group aggregates into
        a value relative to that cell type's total. For example, counting the average number of
        seqlets annotated to a given family across regions of a cell type.
        Only meaningful when ``aggregation_func`` produces a count-like quantity (e.g. ``"count"`` or ``"nunique"``)
        -- dividing a ``"mean"`` or other non-count aggregate by this total does not have a coherent
        interpretation.

    Returns
    -------
    Cell type (columns) x annotation (rows) table of aggregated values, with missing
    combinations filled with 0.

    Raises
    ------
    ValueError
        If ``seqlet_annotation_key``, ``cell_type_key`` or ``aggregate_by`` is not a column in
        ``metadata``; if ``cell_type_order`` contains cell types absent from
        ``metadata[cell_type_key]``; or if ``order_diagonal`` is set without
        ``cell_type_order``.

    Examples
    --------
    >>> import tfmindi as tm
    >>> adata = tm.load_h5ad("seqlet_adata.h5ad")
    >>> metadata = adata.obs.copy()
    >>> # average contribution score per predicted TF family and cell type
    >>> code_table_attr = tm.tl.get_code_table(
    ...     metadata=metadata,
    ...     seqlet_annotation_key="predicted_5.0_predicted_family_no_cl",
    ...     cell_type_key="cell_type",
    ...     aggregate_by="attribution",
    ...     cell_type_order=cell_type_order,
    ...     order_diagonal=True,
    ...     aggregation_func="mean",
    ... )
    >>> code_table_attr.iloc[:5, :2].round(3)
    cell_type                             neu. prog. ventr. (32)  neu. prog. ventr. (33)
    predicted_5.0_predicted_family_no_cl
    AP-2, Homeobox                                          0.042                   0.015
    ETS, GCM                                                0.108                   0.000
    GCM, Homeobox                                           0.027                  -0.035
    MBD                                                     0.444                   0.000
    Pou, T-box                                              0.032                   0.002
    >>> # average seqlet count per region and per predicted TF family and cell types
    >>> code_table_count = get_code_table(
     ...     metadata=metadata,
     ...     seqlet_annotation_key="predicted_5.0_predicted_family_no_cl",
     ...     cell_type_key="cell_type",
     ...     aggregate_by="example_idx",
     ...     cell_type_order=cell_type_order,
     ...     order_diagonal=True,
     ...     aggregation_func="count",
     ...     normalize_per_aggregate_by=True
     ... )
    >>> code_table_count.iloc[:5, :2].round(3)
    cell_type                             neu. prog. ventr. (32)  neu. prog. ventr. (33)
    predicted_5.0_predicted_family_no_cl
    NF-YA                                                  1.146                   0.004
    NF-YB                                                  0.047                   0.000
    Nrf1                                                   0.268                   0.000
    TF_bZIP                                                0.532                   0.056
    THAP                                                   0.064                   0.000
    """
    # Make sure all keys are in metadata
    _keys_not_found: list[str] = []
    for _key in [seqlet_annotation_key, aggregate_by, cell_type_key]:
        if _key not in metadata.columns:
            _keys_not_found.append(_key)

    if len(_keys_not_found) > 0:
        raise ValueError("The following keys are not in the metadata DataFrame:\n\t" + "\n\t".join(_keys_not_found))

    # make sure all cell types are in the dataframe, if cell_type_order is not None
    if cell_type_order is not None:
        _cell_types_in_metadata = set(metadata[cell_type_key])
        _cell_types_not_found = set(cell_type_order) - _cell_types_in_metadata
        if len(_cell_types_not_found) > 0:
            raise ValueError(
                "The following cell types (in `cell_type_order`) were not found in the metadata DataFrame:\n\t"
                + "\n\t".join(_cell_types_not_found)
            )

    # make sure that cell_type order is given when order_diagonal is not False
    if order_diagonal and cell_type_order is None:
        raise ValueError("When `order_diagonal` is set, `cell_type_order` has to be provided as well.")

    code_table = (
        metadata.groupby([cell_type_key, seqlet_annotation_key], as_index=False, observed=True)[aggregate_by]
        .agg(aggregation_func)
        .pivot(columns=cell_type_key, index=seqlet_annotation_key, values=aggregate_by)
        .fillna(0)
    )

    if normalize_per_aggregate_by:
        n_per_aggregate_by = metadata.groupby(cell_type_key, observed=True)[aggregate_by].nunique()
        code_table = code_table.div(n_per_aggregate_by, axis=1)

    if cell_type_order is not None:
        code_table = code_table[cell_type_order]

    if order_diagonal and cell_type_order is not None:
        seqlet_annot_order = code_table.idxmax(axis=1).map(cell_type_order.index).sort_values(kind="stable").index
        code_table = code_table.loc[seqlet_annot_order]

    return code_table


def _pearson(x: Any, y: Any) -> float:
    """Pearson correlation coefficient of two equal-length vectors.

    Parameters
    ----------
    x
        First vector.
    y
        Second vector.

    Returns
    -------
    The correlation coefficient, NaN if either vector is constant.
    """
    return pearsonr(x, y).correlation


def correlate_tf_expression(
    seqlet_adata: ad.AnnData,
    gene_expression_adata: ad.AnnData,
    family_to_tf: dict[str, set[str]],
    motif_collection: MotifCollectionData,
    motif_collection_cluster_resolution: str,
    seqlet_annotation_key: str,
    cell_type_key: str,
    aggregate_by: str = "attribution",
    aggregation_func: Callable | str = "mean",
    normalize_per_aggregate_by: bool = False,
    motif_collection_cluster_family_assignment_pvalue_threshold: float = 1e-6,
    motif_collection_cluster_family_assignment_pvalue_key: str = "pval_adj",
    motif_collection_cluster_family_assignment_family_key: str = "family",
    motif_collection_cluster_family_assignment_cluster_key: str = "cluster",
    correlation_func: Callable[[Any, Any], float] = _pearson,
    gene_expression_layer: str | None = None,
) -> pd.DataFrame:
    """
    Correlate enhancer code values with TF expression across cell types.

    Every motif collection cluster in the code table is scored against the TFs assigned to it,
    through the cluster's family annotation and ``family_to_tf``: the cluster's row of the code
    table (e.g. mean attribution per cell type) is correlated with the TF's mean expression per
    cell type. A high coefficient marks a TF whose expression tracks the contribution pattern of
    the motifs in that cluster, i.e. a candidate driver of that part of the enhancer code.

    Only the cell types shared by both AnnData objects are used.

    Parameters
    ----------
    seqlet_adata
        Seqlet AnnData whose ``.obs`` holds ``seqlet_annotation_key``, ``cell_type_key`` and
        ``aggregate_by``.
    gene_expression_adata
        Gene expression AnnData with cells in ``.obs`` and genes in ``.var``. ``var_names`` have
        to be unique and hold the TF names used in ``family_to_tf``.
    family_to_tf
        Mapping of TF family name to the TFs belonging to that family.
    motif_collection
        Motif collection providing the cluster-to-family annotation.
    motif_collection_cluster_resolution
        Leiden resolution of the motif collection whose cluster-to-family annotation is used,
        e.g. ``"5.0"``. Converted to ``str`` internally if a number is given.
    seqlet_annotation_key
        Column in ``seqlet_adata.obs`` holding the per-seqlet motif collection cluster, i.e. the
        ``{key_added}_{cluster_resolution}_predicted_cluster`` column written by
        :func:`tfmindi.tl.predict_tf_family_seqlets`. Its values have to be the cluster ids of
        the motif collection, not the family names.
    cell_type_key
        Column holding the cell type, present in both ``.obs``.
    aggregate_by
        Column of ``seqlet_adata.obs`` aggregated into the code table values.
    aggregation_func
        Aggregation applied to ``aggregate_by`` within each (cell type, cluster) group.
    normalize_per_aggregate_by
        Whether to normalise the code table per cell type, see :func:`tfmindi.tl.get_code_table`.
    motif_collection_cluster_family_assignment_pvalue_threshold
        Family annotations with a p-value above this threshold are ignored, so only confidently
        assigned families contribute TFs.
    motif_collection_cluster_family_assignment_pvalue_key
        Column holding the family assignment p-value in the motif collection annotation.
    motif_collection_cluster_family_assignment_family_key
        Column holding the family name in the motif collection annotation.
    motif_collection_cluster_family_assignment_cluster_key
        Column holding the cluster id in the motif collection annotation.
    correlation_func
        Function of two equal-length vectors returning a single coefficient. Defaults to the
        Pearson correlation.
    gene_expression_layer
        Layer of ``gene_expression_adata`` to average. None (default) uses ``.X``.

    Returns
    -------
    One row per (cluster, TF) pair, with the columns ``motif_collection_cluster``,
    ``transcription_factor`` and ``correlation_coef``. Clusters whose code table row is constant
    across cell types give a NaN coefficient.

    Raises
    ------
    ValueError
        If ``cell_type_key`` is missing from either object, ``gene_expression_layer`` is not a
        layer of ``gene_expression_adata``, ``gene_expression_adata.var_names`` is not unique,
        fewer than two cell types are shared, ``motif_collection_cluster_resolution`` is not in
        the motif collection, the cluster annotation lacks one of the configured columns, no code
        table row matches a motif collection cluster, or none of the assigned TFs is in
        ``gene_expression_adata.var_names``.

    Examples
    --------
    >>> import tfmindi as tm
    >>> # seqlets annotated with tm.tl.predict_tf_family_seqlets() at resolution 5.0
    >>> correlations = tm.tl.correlate_tf_expression(
    ...     seqlet_adata=adata,
    ...     gene_expression_adata=rna_adata,
    ...     family_to_tf=family_to_tf,
    ...     motif_collection=motif_collection,
    ...     motif_collection_cluster_resolution="5.0",
    ...     seqlet_annotation_key="predicted_5.0_predicted_cluster",
    ...     cell_type_key="cell_type",
    ... )
    >>> correlations.sort_values("correlation_coef", ascending=False).head()
            motif_collection_cluster transcription_factor  correlation_coef
    30682                       122                 NOTO          1.000000
    30695                       122                PROP1          1.000000
    27580                        92               TWIST2          0.981527
    26750                        90               ZNF735          0.980361
    1072                          7                  ERG          0.969545
    """
    if cell_type_key not in seqlet_adata.obs.columns:
        raise ValueError(f"`cell_type_key` ({cell_type_key}) not found in `seqlet_adata.obs`")
    if cell_type_key not in gene_expression_adata.obs.columns:
        raise ValueError(f"`cell_type_key` ({cell_type_key}) not found in `gene_expression_adata.obs`")

    if gene_expression_layer is not None and gene_expression_layer not in gene_expression_adata.layers:
        raise ValueError(f"`gene_expression_layer` ({gene_expression_layer}) not in `gene_expression_adata.layers`")

    if not gene_expression_adata.var_names.is_unique:
        raise ValueError(
            "`gene_expression_adata.var_names` has to be unique, duplicate gene names make the "
            "per-TF expression look-up ambiguous."
        )

    overlapping_cell_types = sorted(
        set(seqlet_adata.obs[cell_type_key]) & set(gene_expression_adata.obs[cell_type_key])
    )
    if len(overlapping_cell_types) == 0:
        raise ValueError(
            "No cell types overlap between `seqlet_adata.obs[cell_type_key]` and `gene_expression_adata.obs[cell_type_key]`"
        )
    if len(overlapping_cell_types) < 2:
        raise ValueError(
            f"Only one cell type ({overlapping_cell_types[0]}) overlaps between `seqlet_adata.obs[cell_type_key]` "
            "and `gene_expression_adata.obs[cell_type_key]`, a correlation needs at least two."
        )

    if isinstance(motif_collection_cluster_resolution, float | int):
        motif_collection_cluster_resolution = str(motif_collection_cluster_resolution)
    if motif_collection_cluster_resolution not in motif_collection._cluster_to_annot_file.keys():
        raise ValueError(
            f"The `motif_collection_cluster_resolution` ({motif_collection_cluster_resolution}) is not present in the motif_collection."
            + " Available resolutions are:\n\t"
            + "\n\t".join(motif_collection._cluster_to_annot_file.keys())
        )

    # get code table
    seqlet_metadata = seqlet_adata.obs
    assert isinstance(seqlet_metadata, pd.DataFrame)
    code_table = get_code_table(
        metadata=seqlet_metadata,
        seqlet_annotation_key=seqlet_annotation_key,
        cell_type_key=cell_type_key,
        aggregate_by=aggregate_by,
        aggregation_func=aggregation_func,
        normalize_per_aggregate_by=normalize_per_aggregate_by,
    )

    # get cluster to tf families
    motif_collection_metadata = motif_collection.get_cluster_annotation(motif_collection_cluster_resolution).copy()
    _annotation_keys_not_found = [
        _key
        for _key in [
            motif_collection_cluster_family_assignment_cluster_key,
            motif_collection_cluster_family_assignment_family_key,
            motif_collection_cluster_family_assignment_pvalue_key,
        ]
        if _key not in motif_collection_metadata.columns
    ]
    if len(_annotation_keys_not_found) > 0:
        raise ValueError(
            "The following keys are not in the motif collection cluster annotation:\n\t"
            + "\n\t".join(_annotation_keys_not_found)
        )

    motif_collection_metadata = motif_collection_metadata.loc[
        motif_collection_metadata[motif_collection_cluster_family_assignment_pvalue_key]
        <= motif_collection_cluster_family_assignment_pvalue_threshold
    ]
    cluster_to_families: dict[Hashable, set[str]] = (
        motif_collection_metadata.groupby(motif_collection_cluster_family_assignment_cluster_key, observed=True)[
            motif_collection_cluster_family_assignment_family_key
        ]
        .apply(set)
        .to_dict()
    )

    cluster_to_tfs: dict[int | str, set[str]] = {}
    for cl, fams in cluster_to_families.items():
        assert isinstance(cl, int | str)
        tfs = {tf for fam in fams if fam in family_to_tf for tf in family_to_tf[fam]}
        if len(tfs) > 0:
            cluster_to_tfs[cl] = tfs

    clusters_w_tfs_in_code_table = sorted(set(cluster_to_tfs.keys()) & set(code_table.index))
    if len(clusters_w_tfs_in_code_table) == 0:
        raise ValueError(
            f"None of the values in `seqlet_adata.obs['{seqlet_annotation_key}']` matches a motif collection "
            f"cluster with an annotated family at resolution {motif_collection_cluster_resolution}. "
            "`seqlet_annotation_key` has to hold motif collection cluster ids, e.g. the "
            "`{key_added}_{cluster_resolution}_predicted_cluster` column written by "
            "`tfmindi.tl.predict_tf_family_seqlets`."
        )

    tfs_to_correlate = sorted(
        {tf for cluster in clusters_w_tfs_in_code_table for tf in cluster_to_tfs[cluster]}
        & set(gene_expression_adata.var_names)
    )
    if len(tfs_to_correlate) == 0:
        raise ValueError(
            "None of the TFs assigned to the clusters in the code table is present in "
            "`gene_expression_adata.var_names`, check that `family_to_tf` uses the same gene names."
        )

    tf_expression = gene_expression_adata[:, tfs_to_correlate]
    avg_expression = (
        tf_expression.to_df(layer=gene_expression_layer).groupby(tf_expression.obs[cell_type_key], observed=True).mean()
    )

    # get matching cell types across dataframes
    code_table = code_table[overlapping_cell_types]
    avg_expression = avg_expression.loc[overlapping_cell_types]

    l_cluster_tf_corr: list[tuple[int | str, str, float]] = []
    for cluster in tqdm(clusters_w_tfs_in_code_table, desc="annotating clusters"):
        for tf in sorted(cluster_to_tfs[cluster]):
            if tf not in avg_expression.columns:
                continue
            l_cluster_tf_corr.append(
                (
                    cluster,
                    tf,
                    correlation_func(code_table.loc[cluster], avg_expression[tf]),
                )
            )

    return pd.DataFrame(
        l_cluster_tf_corr, columns=["motif_collection_cluster", "transcription_factor", "correlation_coef"]
    )
