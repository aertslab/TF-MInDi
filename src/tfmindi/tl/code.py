"""Tools to extract cell type-level enhancer codes."""

from collections.abc import Callable

import pandas as pd


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
