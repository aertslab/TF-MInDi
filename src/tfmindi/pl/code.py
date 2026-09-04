"""Plotting functions to visualize cell type-specific enhancer codes."""

from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from tfmindi.pl._utils import render_plot
from tfmindi.tl.code import get_code_table


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

    if min_size_value is not None:
        keep = size_table.max(axis=1) >= min_size_value
        size_table, color_table = size_table.loc[keep], color_table.loc[keep]
    if min_color_value is not None:
        keep = color_table.max(axis=1) >= min_color_value
        size_table, color_table = size_table.loc[keep], color_table.loc[keep]

    cell_types = list(size_table.columns)
    annotations = list(size_table.index)
    n_x, n_y = len(cell_types), len(annotations)

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
    ax.set_xticklabels(cell_types)
    ax.set_yticks(np.arange(n_y))
    ax.set_yticklabels(annotations)
    ax.set_xlim(-0.5, n_x - 0.5)
    ax.set_ylim(-0.5, n_y - 0.5)
    ax.invert_yaxis()

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(colorbar_label)

    positive_sizes = size[size > 0]
    if positive_sizes.size:
        for val in np.linspace(positive_sizes.min(), max_dotsize, n_size_legend):
            label = f"{val:.2f}" if val < max_dotsize else f"≥{val:.2f}"
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
        "xlabel": cell_type_key,
        "ylabel": seqlet_annotation_key,
        # render_plot resets every tick label rotation, so the default has to be set here.
        "x_label_rotation": 90,
        **kwargs,
    }
    return render_plot(fig, **render_kwargs)
