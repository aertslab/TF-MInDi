"""Plotting functions to visualize cell type-specific enhancer codes."""

from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from tfmindi.datasets import MotifCollectionData
from tfmindi.pl._utils import render_plot
from tfmindi.tl.code import correlate_tf_expression, get_code_table


def _dotplot(
    size_table: pd.DataFrame,
    color_table: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    min_size_value: float | None = None,
    min_color_value: float | None = None,
    max_dotsize: float | None = None,
    max_marker_area: float = 300,
    n_size_legend: int = 4,
    cmap: str = "Reds",
    vmin: float | None = None,
    vmax: float | None = None,
    edge_color: str = "black",
    linewidth: float = 1.0,
    size_legend_title: str = "size",
    colorbar_label: str = "value",
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """Draw two identically indexed tables as one dot plot, sizes from one, colors from the other.

    Parameters
    ----------
    size_table
        Row x column table whose values set the dot sizes.
    color_table
        Table with the same index and columns as ``size_table`` whose values set the dot colors.
    xlabel
        Label of the x-axis, whose ticks are the table columns.
    ylabel
        Label of the y-axis, whose ticks are the table rows.
    min_size_value
        Drop rows whose largest size value is below this threshold. None keeps every row.
    min_color_value
        Drop rows whose largest color value is below this threshold. None keeps every row.
    max_dotsize
        Size value at which a dot reaches ``max_marker_area``; larger values are clipped to this
        cap. Defaults to the largest plotted size value, i.e. no clipping.
    max_marker_area
        Marker area, in points^2, of a dot at ``max_dotsize``.
    n_size_legend
        Number of entries in the dot-size legend.
    cmap
        Colormap for the color values.
    vmin
        Minimum of the color scale. None scales to the data.
    vmax
        Maximum of the color scale. None scales to the data.
    edge_color
        Edge color of the dots.
    linewidth
        Edge line width of the dots.
    size_legend_title
        Title of the dot-size legend.
    colorbar_label
        Label of the color bar.
    **kwargs
        Additional arguments passed to render_plot().

    Returns
    -------
    Figure with the dot plot, or None if show=True.
    """
    if min_size_value is not None:
        keep = size_table.max(axis=1) >= min_size_value
        size_table, color_table = size_table.loc[keep], color_table.loc[keep]
    if min_color_value is not None:
        keep = color_table.max(axis=1) >= min_color_value
        size_table, color_table = size_table.loc[keep], color_table.loc[keep]

    columns = list(size_table.columns)
    rows = list(size_table.index)
    n_x, n_y = len(columns), len(rows)

    size = size_table.to_numpy()
    color = color_table.to_numpy()

    if max_dotsize is None:
        max_dotsize = float(size.max()) if size.size else 1.0
    s = np.clip(size, 0, max_dotsize) / max_dotsize * max_marker_area

    x, y = np.meshgrid(np.arange(n_x), np.arange(n_y))

    figsize = (
        kwargs.get("width", 0.35 * n_x + 3),
        kwargs.get("height", 0.25 * n_y + 3),
    )
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        x.ravel(),
        y.ravel(),
        s=s.ravel(),
        c=color.ravel(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor=edge_color,
        linewidth=linewidth,
    )

    ax.set_xticks(np.arange(n_x))
    ax.set_xticklabels(columns)
    ax.set_yticks(np.arange(n_y))
    ax.set_yticklabels(rows)
    ax.set_xlim(-0.5, n_x - 0.5)
    ax.set_ylim(-0.5, n_y - 0.5)
    ax.invert_yaxis()
    # Set here rather than through render_plot, which labels every axes of the figure and would
    # overwrite the color bar label with the y-axis label.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(colorbar_label)

    positive_sizes = size[size > 0]
    if positive_sizes.size:
        # Near-identical values round to the same label, which would repeat the same dot in the
        # legend, e.g. when a single row is plotted.
        seen_labels: set[str] = set()
        for val in np.linspace(positive_sizes.min(), max_dotsize, n_size_legend):
            label = f"{val:.2f}" if val < max_dotsize else f"≥{val:.2f}"
            if label in seen_labels:
                continue
            seen_labels.add(label)
            ax.scatter(
                [],
                [],
                s=val / max_dotsize * max_marker_area,
                c="none",
                edgecolor=edge_color,
                linewidth=0.3,
                label=label,
            )
        ax.legend(
            title=size_legend_title,
            labelspacing=1.5,
            borderpad=1.2,
            loc="upper left",
            bbox_to_anchor=(1.15, 1),
            frameon=False,
        )

    render_kwargs = {
        "width": figsize[0],
        "height": figsize[1],
        # render_plot resets every tick label rotation, so the default has to be set here.
        "x_label_rotation": 90,
        **kwargs,
    }
    return render_plot(fig, **render_kwargs)


def code_table_dotplot(
    adata: AnnData,
    seqlet_annotation_key: str,
    cell_type_key: str,
    size_by: str,
    color_by: str,
    size_aggregation_func: Callable | str = "count",
    color_aggregation_func: Callable | str = "mean",
    normalize_size_per_aggregate_by: bool = True,
    normalize_color_per_aggregate_by: bool = False,
    cell_type_order: list[str] | None = None,
    order_diagonal_by: Literal["size", "color"] | None = None,
    min_size_value: float | None = None,
    min_color_value: float | None = None,
    max_dotsize: float | None = None,
    max_marker_area: float = 300,
    n_size_legend: int = 4,
    cmap: str = "Reds",
    vmin: float | None = None,
    vmax: float | None = None,
    edge_color: str = "black",
    linewidth: float = 1.0,
    size_legend_title: str = "size",
    colorbar_label: str = "value",
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Dot plot of a cell type-level enhancer code, with one value as dot size and another as color.

    Two code tables are computed from ``adata.obs`` with :func:`tfmindi.tl.get_code_table`, one
    for the quantity shown as dot size (e.g. average seqlet count per region) and one for the
    quantity shown as color (e.g. average contribution). Both tables are ordered identically,
    so a row is the same annotation in both encodings.

    Parameters
    ----------
    adata
        AnnData object with seqlet data. ``adata.obs`` must contain ``seqlet_annotation_key``,
        ``cell_type_key``, ``size_by`` and ``color_by``.
    seqlet_annotation_key
        Column in ``adata.obs`` holding the per-seqlet annotation (e.g. predicted TF family)
        shown along the y-axis.
    cell_type_key
        Column in ``adata.obs`` holding the per-seqlet cell type shown along the x-axis.
    size_by
        Column in ``adata.obs`` aggregated into the dot sizes, e.g. a region identifier
        counted per (cell type, annotation) group.
    color_by
        Column in ``adata.obs`` aggregated into the dot colors, e.g. per-seqlet attribution.
    size_aggregation_func
        Aggregation applied to ``size_by`` within each group.
    color_aggregation_func
        Aggregation applied to ``color_by`` within each group.
    normalize_size_per_aggregate_by
        Whether to divide each cell type's dot sizes by its number of distinct ``size_by``
        values, turning counts into a per-region average.
    normalize_color_per_aggregate_by
        Same normalisation for the color values. Only meaningful for count-like
        ``color_aggregation_func``.
    cell_type_order
        Explicit x-axis order. Required when ``order_diagonal_by`` is set.
    order_diagonal_by
        Which table decides the y-axis order: ``"size"`` or ``"color"`` places every annotation
        at the position of the cell type where that table peaks, giving the plot a diagonal
        layout. None (default) keeps the alphabetical annotation order.
    min_size_value
        Drop annotations whose largest size value across cell types is below this threshold,
        e.g. TF families that never reach a minimal average count. None keeps every annotation.
    min_color_value
        Same filter on the color values. Combined with ``min_size_value`` an annotation has to
        pass both thresholds to be plotted.
    max_dotsize
        Size value at which a dot reaches ``max_marker_area``; larger values are clipped to
        this cap so a few outliers do not shrink every other dot. Defaults to the largest
        plotted size value, i.e. no clipping.
    max_marker_area
        Marker area, in points^2, of a dot at ``max_dotsize``.
    n_size_legend
        Number of entries in the dot-size legend.
    cmap
        Colormap for the color values.
    vmin
        Minimum of the color scale. None (default) scales to the data.
    vmax
        Maximum of the color scale. None (default) scales to the data.
    edge_color
        Edge color of the dots.
    linewidth
        Edge line width of the dots.
    size_legend_title
        Title of the dot-size legend.
    colorbar_label
        Label of the color bar.
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include width, height, title, show, save_path, dpi.

    Returns
    -------
    Figure with the enhancer code dot plot, or None if show=True.

    Examples
    --------
    >>> import tfmindi as tm
    >>> # dot size: average number of seqlets per region, dot color: mean attribution
    >>> fig = tm.pl.code_table_dotplot(
    ...     adata,
    ...     seqlet_annotation_key="predicted_5.0_predicted_family_no_cl",
    ...     cell_type_key="cell_type",
    ...     size_by="region_name",
    ...     color_by="attribution",
    ...     cell_type_order=cell_type_order,
    ...     order_diagonal_by="color",
    ...     min_size_value=0.05,
    ...     size_legend_title="avg # seqlets per region",
    ...     colorbar_label="mean attribution",
    ...     show=False,
    ... )
    >>> # cap outlier counts and fix the color scale
    >>> fig = tm.pl.code_table_dotplot(
    ...     adata,
    ...     seqlet_annotation_key="predicted_5.0_predicted_family_no_cl",
    ...     cell_type_key="cell_type",
    ...     size_by="region_name",
    ...     color_by="attribution",
    ...     cell_type_order=cell_type_order,
    ...     max_dotsize=2,
    ...     vmin=0,
    ...     vmax=0.3,
    ... )
    """
    metadata = adata.obs
    assert isinstance(metadata, pd.DataFrame)

    size_table = get_code_table(
        metadata=metadata,
        seqlet_annotation_key=seqlet_annotation_key,
        cell_type_key=cell_type_key,
        aggregate_by=size_by,
        aggregation_func=size_aggregation_func,
        cell_type_order=cell_type_order,
        order_diagonal=order_diagonal_by == "size",
        normalize_per_aggregate_by=normalize_size_per_aggregate_by,
    )
    color_table = get_code_table(
        metadata=metadata,
        seqlet_annotation_key=seqlet_annotation_key,
        cell_type_key=cell_type_key,
        aggregate_by=color_by,
        aggregation_func=color_aggregation_func,
        cell_type_order=cell_type_order,
        order_diagonal=order_diagonal_by == "color",
        normalize_per_aggregate_by=normalize_color_per_aggregate_by,
    )

    # Each table diagonalises on its own values, so the two only share a row order when it is
    # taken from one of them and imposed on the other.
    if order_diagonal_by == "size":
        color_table = color_table.loc[size_table.index]
    elif order_diagonal_by == "color":
        size_table = size_table.loc[color_table.index]

    return _dotplot(
        size_table=size_table,
        color_table=color_table,
        xlabel=cell_type_key,
        ylabel=seqlet_annotation_key,
        min_size_value=min_size_value,
        min_color_value=min_color_value,
        max_dotsize=max_dotsize,
        max_marker_area=max_marker_area,
        n_size_legend=n_size_legend,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edge_color=edge_color,
        linewidth=linewidth,
        size_legend_title=size_legend_title,
        colorbar_label=colorbar_label,
        **kwargs,
    )


def code_table_tf_expression_dotplot(
    seqlet_adata: AnnData,
    gene_expression_adata: AnnData,
    family_to_tf: dict[str, set[str]],
    motif_collection: MotifCollectionData,
    motif_collection_cluster_resolution: str,
    seqlet_annotation_key: str,
    cell_type_key: str,
    min_correlation_coef: float = 0.4,
    tfs: list[str] | None = None,
    motif_collection_clusters: list[int | str] | None = None,
    color_by: str = "example_idx",
    color_aggregation_func: Callable | str = "count",
    normalize_color_per_aggregate_by: bool = True,
    aggregate_by: str = "attribution",
    aggregation_func: Callable | str = "mean",
    normalize_per_aggregate_by: bool = False,
    gene_expression_layer: str | None = None,
    motif_collection_cluster_family_assignment_pvalue_threshold: float = 1e-6,
    motif_collection_cluster_family_assignment_pvalue_key: str = "pval_adj",
    motif_collection_cluster_family_assignment_family_key: str = "family",
    motif_collection_cluster_family_assignment_cluster_key: str = "cluster",
    cell_type_order: list[str] | None = None,
    order_diagonal_by: Literal["size", "color"] | None = None,
    min_size_value: float | None = None,
    min_color_value: float | None = None,
    max_dotsize: float | None = None,
    max_marker_area: float = 300,
    n_size_legend: int = 4,
    cmap: str = "Reds",
    vmin: float | None = None,
    vmax: float | None = None,
    edge_color: str = "black",
    linewidth: float = 1.0,
    size_legend_title: str = "mean TF expression",
    colorbar_label: str = "value",
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Dot plot of an enhancer code against TF expression, one row per (cluster, TF) pair.

    :func:`tfmindi.tl.correlate_tf_expression` supplies the pairs: every motif collection cluster
    is paired with the TFs of its annotated families, and only pairs whose correlation across cell
    types reaches ``min_correlation_coef`` are plotted. A row's dot sizes are the TF's mean
    expression per cell type, its dot colors the cluster's code table row computed from the seqlet
    data (e.g. average number of seqlets per region), so a row shows whether TF expression and
    motif activity peak in the same cell types.

    Parameters
    ----------
    seqlet_adata
        Seqlet AnnData whose ``.obs`` holds ``seqlet_annotation_key``, ``cell_type_key``,
        ``color_by`` and ``aggregate_by``.
    gene_expression_adata
        Gene expression AnnData with cells in ``.obs`` and genes in ``.var``.
    family_to_tf
        Mapping of TF family name to the TFs belonging to that family.
    motif_collection
        Motif collection providing the cluster-to-family annotation.
    motif_collection_cluster_resolution
        Leiden resolution of the motif collection annotation to use, e.g. ``"5.0"``.
    seqlet_annotation_key
        Column in ``seqlet_adata.obs`` holding the per-seqlet motif collection cluster, i.e. the
        ``{key_added}_{cluster_resolution}_predicted_cluster`` column written by
        :func:`tfmindi.tl.predict_tf_family_seqlets`.
    cell_type_key
        Column holding the cell type, present in both ``.obs``.
    min_correlation_coef
        Only (cluster, TF) pairs whose correlation coefficient is at least this value are plotted.
    tfs
        Restrict the plot to these TFs. None (default) keeps every TF that passes the correlation
        threshold.
    motif_collection_clusters
        Restrict the plot to these motif collection clusters. None (default) keeps every cluster
        that passes the correlation threshold.
    color_by
        Column of ``seqlet_adata.obs`` aggregated into the dot colors, e.g. a region identifier
        counted per (cell type, cluster) group.
    color_aggregation_func
        Aggregation applied to ``color_by`` within each group.
    normalize_color_per_aggregate_by
        Whether to divide each cell type's dot colors by its number of distinct ``color_by``
        values, turning counts into a per-region average.
    aggregate_by
        Column of ``seqlet_adata.obs`` correlated with TF expression to select the pairs.
    aggregation_func
        Aggregation applied to ``aggregate_by`` for the correlation.
    normalize_per_aggregate_by
        Whether to normalise the correlated code table per cell type.
    gene_expression_layer
        Layer of ``gene_expression_adata`` to average. None (default) uses ``.X``.
    motif_collection_cluster_family_assignment_pvalue_threshold
        Family annotations with a p-value above this threshold are ignored.
    motif_collection_cluster_family_assignment_pvalue_key
        Column holding the family assignment p-value in the motif collection annotation.
    motif_collection_cluster_family_assignment_family_key
        Column holding the family name in the motif collection annotation.
    motif_collection_cluster_family_assignment_cluster_key
        Column holding the cluster id in the motif collection annotation.
    cell_type_order
        Explicit x-axis order. Required when ``order_diagonal_by`` is set. Restricted to the cell
        types shared by both AnnData objects.
    order_diagonal_by
        Which values decide the y-axis order: ``"size"`` (TF expression) or ``"color"`` (the code
        table) places every pair at the position of the cell type where that quantity peaks,
        giving the plot a diagonal layout. None (default) keeps the pairs ordered by cluster and
        TF name.
    min_size_value
        Drop pairs whose largest expression value across cell types is below this threshold.
    min_color_value
        Drop pairs whose largest code table value across cell types is below this threshold.
    max_dotsize
        Expression value at which a dot reaches ``max_marker_area``; larger values are clipped to
        this cap. Defaults to the largest plotted expression value, i.e. no clipping.
    max_marker_area
        Marker area, in points^2, of a dot at ``max_dotsize``.
    n_size_legend
        Number of entries in the dot-size legend.
    cmap
        Colormap for the code table values.
    vmin
        Minimum of the color scale. None (default) scales to the data.
    vmax
        Maximum of the color scale. None (default) scales to the data.
    edge_color
        Edge color of the dots.
    linewidth
        Edge line width of the dots.
    size_legend_title
        Title of the dot-size legend.
    colorbar_label
        Label of the color bar. Worth setting to what ``color_by`` and ``color_aggregation_func``
        actually compute, e.g. ``"avg # seqlets per region"``.
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include width, height, title, show, save_path, dpi.

    Returns
    -------
    Figure with the code table vs TF expression dot plot, or None if show=True.

    Raises
    ------
    ValueError
        If no (cluster, TF) pair survives ``min_correlation_coef`` and the ``tfs`` /
        ``motif_collection_clusters`` subsets, or if ``order_diagonal_by`` is set without
        ``cell_type_order``. Input validation of the two AnnData objects is done by
        :func:`tfmindi.tl.correlate_tf_expression`.

    Examples
    --------
    >>> import tfmindi as tm
    >>> # dot size: mean TF expression, dot color: average number of seqlets per region
    >>> fig = tm.pl.code_table_tf_expression_dotplot(
    ...     seqlet_adata=adata,
    ...     gene_expression_adata=rna_adata,
    ...     family_to_tf=family_to_tf,
    ...     motif_collection=motif_collection,
    ...     motif_collection_cluster_resolution="5.0",
    ...     seqlet_annotation_key="predicted_5.0_predicted_cluster",
    ...     cell_type_key="cell_type",
    ...     min_correlation_coef=0.8,
    ...     cell_type_order=cell_type_order,
    ...     order_diagonal_by="size",
    ...     colorbar_label="avg # seqlets per region",
    ...     show=False,
    ... )
    >>> # zoom in on a few TFs of interest
    >>> fig = tm.pl.code_table_tf_expression_dotplot(
    ...     seqlet_adata=adata,
    ...     gene_expression_adata=rna_adata,
    ...     family_to_tf=family_to_tf,
    ...     motif_collection=motif_collection,
    ...     motif_collection_cluster_resolution="5.0",
    ...     seqlet_annotation_key="predicted_5.0_predicted_cluster",
    ...     cell_type_key="cell_type",
    ...     tfs=["SOX2", "PAX6", "NEUROD2"],
    ... )
    """
    if order_diagonal_by is not None and cell_type_order is None:
        raise ValueError("When `order_diagonal_by` is set, `cell_type_order` has to be provided as well.")

    correlations = correlate_tf_expression(
        seqlet_adata=seqlet_adata,
        gene_expression_adata=gene_expression_adata,
        family_to_tf=family_to_tf,
        motif_collection=motif_collection,
        motif_collection_cluster_resolution=motif_collection_cluster_resolution,
        seqlet_annotation_key=seqlet_annotation_key,
        cell_type_key=cell_type_key,
        aggregate_by=aggregate_by,
        aggregation_func=aggregation_func,
        normalize_per_aggregate_by=normalize_per_aggregate_by,
        motif_collection_cluster_family_assignment_pvalue_threshold=motif_collection_cluster_family_assignment_pvalue_threshold,
        motif_collection_cluster_family_assignment_pvalue_key=motif_collection_cluster_family_assignment_pvalue_key,
        motif_collection_cluster_family_assignment_family_key=motif_collection_cluster_family_assignment_family_key,
        motif_collection_cluster_family_assignment_cluster_key=motif_collection_cluster_family_assignment_cluster_key,
        gene_expression_layer=gene_expression_layer,
    )

    # NaN coefficients (a constant code table row or a TF that is not expressed anywhere) fail the
    # comparison, so they drop out here rather than being drawn as an unranked row.
    pairs = correlations.loc[correlations["correlation_coef"] >= min_correlation_coef]
    if tfs is not None:
        pairs = pairs.loc[pairs["transcription_factor"].isin(tfs)]
    if motif_collection_clusters is not None:
        pairs = pairs.loc[pairs["motif_collection_cluster"].isin(motif_collection_clusters)]
    if len(pairs) == 0:
        raise ValueError(
            f"No (cluster, TF) pair reaches a correlation coefficient of {min_correlation_coef}"
            + (" for the requested TFs" if tfs is not None else "")
            + (" and clusters" if tfs is not None and motif_collection_clusters is not None else "")
            + (" for the requested clusters" if tfs is None and motif_collection_clusters is not None else "")
            + ". Lower `min_correlation_coef` or widen the subset."
        )
    pairs = pairs.sort_values(["motif_collection_cluster", "transcription_factor"], kind="stable")

    seqlet_metadata = seqlet_adata.obs
    assert isinstance(seqlet_metadata, pd.DataFrame)
    code_table = get_code_table(
        metadata=seqlet_metadata,
        seqlet_annotation_key=seqlet_annotation_key,
        cell_type_key=cell_type_key,
        aggregate_by=color_by,
        aggregation_func=color_aggregation_func,
        normalize_per_aggregate_by=normalize_color_per_aggregate_by,
    )

    tf_expression = gene_expression_adata[:, sorted(set(pairs["transcription_factor"]))]
    avg_expression = (
        tf_expression.to_df(layer=gene_expression_layer).groupby(tf_expression.obs[cell_type_key], observed=True).mean()
    )

    if cell_type_order is not None:
        _cell_types_not_found = set(cell_type_order) - (set(code_table.columns) & set(avg_expression.index))
        if len(_cell_types_not_found) > 0:
            raise ValueError(
                "The following cell types (in `cell_type_order`) are not in both `seqlet_adata` and "
                "`gene_expression_adata`:\n\t" + "\n\t".join(str(_ct) for _ct in _cell_types_not_found)
            )
        cell_types = cell_type_order
    else:
        cell_types = sorted(set(code_table.columns) & set(avg_expression.index))
    code_table = code_table[cell_types]
    avg_expression = avg_expression.loc[cell_types]

    # One row per pair: the TF's expression profile as dot sizes, the cluster's code table row as
    # dot colors.
    row_labels = [
        f"{cluster} | {tf}"
        for cluster, tf in zip(pairs["motif_collection_cluster"], pairs["transcription_factor"], strict=True)
    ]
    size_table = pd.DataFrame(
        [avg_expression[tf].to_numpy() for tf in pairs["transcription_factor"]],
        index=row_labels,
        columns=cell_types,
    )
    color_table = pd.DataFrame(
        [code_table.loc[cluster].to_numpy() for cluster in pairs["motif_collection_cluster"]],
        index=row_labels,
        columns=cell_types,
    )

    if order_diagonal_by is not None and cell_type_order is not None:
        table_to_order_by = size_table if order_diagonal_by == "size" else color_table
        row_order = table_to_order_by.idxmax(axis=1).map(cell_type_order.index).sort_values(kind="stable").index
        size_table, color_table = size_table.loc[row_order], color_table.loc[row_order]

    return _dotplot(
        size_table=size_table,
        color_table=color_table,
        xlabel=cell_type_key,
        ylabel="motif collection cluster | TF",
        min_size_value=min_size_value,
        min_color_value=min_color_value,
        max_dotsize=max_dotsize,
        max_marker_area=max_marker_area,
        n_size_legend=n_size_legend,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edge_color=edge_color,
        linewidth=linewidth,
        size_legend_title=size_legend_title,
        colorbar_label=colorbar_label,
        **kwargs,
    )
