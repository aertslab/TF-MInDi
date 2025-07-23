"""DNA-binding domain heatmap visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData

from tfmindi.pl._utils import render_plot


def dbd_topic_heatmap(
    topic_matrix: pd.DataFrame,
    annotation_matrix: pd.DataFrame,
    cmap: str = "viridis",
    vmax: float | None = None,
    vmin: float | None = None,
    row_cluster: bool = False,
    col_cluster: bool = False,
    linewidths: float = 0.5,
    edgecolor: str = "white",
    xticklabels: bool = True,
    yticklabels: bool = True,
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Show enrichment of DNA-binding domains across topics using a clustered heatmap.

    This function creates a heatmap visualization showing the relationship between
    topics and DNA-binding domains (DBDs) or clusters. It supports both topic-cluster
    matrices and topic-DBD matrices with customizable clustering and styling options.

    Parameters
    ----------
    topic_matrix
        Topic-cluster or topic-DBD matrix (topics × clusters/DBDs).
        Rows represent topics, columns represent clusters or DBDs.
    annotation_matrix
        Annotation matrix used for ordering (typically DBD annotations).
        Should have same column structure as topic_matrix.
    cmap
        Colormap name for the heatmap (default: "viridis").
        Any valid matplotlib colormap (e.g., "plasma", "inferno", "Blues").
    vmax
        Maximum value for color scaling. If None, uses data maximum.
    vmin
        Minimum value for color scaling. If None, uses data minimum.
    row_cluster
        Whether to cluster rows (topics) hierarchically.
    col_cluster
        Whether to cluster columns (DBDs/clusters) hierarchically.
    linewidths
        Width of lines separating heatmap cells.
    edgecolor
        Color of lines separating heatmap cells.
    xticklabels
        Whether to show x-axis tick labels.
    yticklabels
        Whether to show y-axis tick labels.
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include width, height, title, show, save_path, dpi.

    Returns
    -------
    matplotlib.Figure or None
        Figure with clustered heatmap, or None if show=True.

    Examples
    --------
    >>> import tfmindi as tm
    >>> # After topic modeling and getting topic-cluster matrix
    >>> topic_cluster = tm.tl.get_topic_cluster_matrix(lda_model, count_table)
    >>> topic_dbd = tm.tl.get_topic_dbd_matrix(lda_model, count_table, cluster_to_dbd)
    >>> # Create heatmap showing topic-DBD relationships
    >>> fig = tm.pl.dbd_heatmap(topic_dbd, topic_cluster, title="Topic-DBD Enrichment")
    >>> # Custom styling with clustering
    >>> tm.pl.dbd_heatmap(
    ...     topic_dbd,
    ...     topic_cluster,
    ...     width=12,
    ...     height=8,
    ...     cmap="plasma",
    ...     row_cluster=True,
    ...     col_cluster=True,
    ...     vmax=0.05,
    ...     save_path="topic_dbd_heatmap.png",
    ... )
    """
    # Validate inputs
    if not isinstance(topic_matrix, pd.DataFrame):
        raise TypeError("topic_matrix must be a pandas DataFrame")
    if not isinstance(annotation_matrix, pd.DataFrame):
        raise TypeError("annotation_matrix must be a pandas DataFrame")

    if topic_matrix.empty:
        raise ValueError("topic_matrix cannot be empty")
    if annotation_matrix.empty:
        raise ValueError("annotation_matrix cannot be empty")

    # Create ordering based on annotation matrix
    # Find the topic with maximum value for each column (DBD/cluster)
    max_topics = annotation_matrix.T.idxmax()

    # Get the order of topics by their maximum values
    topic_order = []
    for topic in max_topics:
        if topic not in topic_order:
            topic_order.append(topic)

    # Add any remaining topics that weren't captured
    for topic in topic_matrix.index:
        if topic not in topic_order:
            topic_order.append(topic)

    # Create column ordering based on which topic is maximum for each column
    col_order_df = pd.DataFrame(max_topics).reset_index()
    col_order_df.columns = ["column", "max_topic"]
    col_order_df["order"] = [topic_order.index(x) for x in col_order_df["max_topic"]]
    sorted_columns = list(col_order_df.sort_values("order")["column"])

    # Ensure all columns from topic_matrix are included
    remaining_cols = [col for col in topic_matrix.columns if col not in sorted_columns]
    sorted_columns.extend(remaining_cols)

    # Reorder matrices
    ordered_topic_matrix = topic_matrix.loc[topic_order, sorted_columns]

    # Extract figure size from kwargs or use defaults
    figsize = (kwargs.get("width", 8), kwargs.get("height", 10))

    # Create the clustered heatmap
    cluster_fig = sns.clustermap(
        ordered_topic_matrix.T,  # Transpose to have topics as columns
        figsize=figsize,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        linewidths=linewidths,
        edgecolor=edgecolor,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
    )

    # Extract the figure object from the ClusterGrid
    fig = cluster_fig.fig

    # Apply render_plot styling
    render_kwargs = {
        "title": "DBD-Topic Enrichment",
        **kwargs,
    }

    return render_plot(fig, **render_kwargs)


def dbd_heatmap(
    adata: AnnData,
    dbd_column: str = "cluster_dbd",
    cell_type_column: str = "cell_type",
    cmap: str = "Spectral_r",
    row_cluster: bool = True,
    col_cluster: bool = True,
    drop_na: bool = True,
    linewidths: float = 0.01,
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Create a clustered heatmap showing seqlet counts per cell type and DNA-binding domain.

    Creates a cross-tabulation of cell types vs DBD annotations and visualizes it as a
    clustered heatmap, similar to the analysis in the original paper.

    Parameters
    ----------
    adata
        AnnData object with seqlet data.
        Must contain specified dbd_column and cell_type_column in adata.obs.
    dbd_column
        Column name in adata.obs containing DNA-binding domain annotations.
    cell_type_column
        Column name in adata.obs containing cell type annotations.
    cmap
        Colormap for the heatmap.
    row_cluster
        Whether to perform hierarchical clustering on the rows.
    col_cluster
        Whether to perform hierarchical clustering on the columns.
    drop_na
        Whether to drop columns/rows with NaN values.
    linewidths
        Width of lines separating cells in the heatmap.
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include width, height, title, show, save_path, dpi.

    Returns
    -------
    Figure with clustered heatmap, or None if show=True.

    Examples
    --------
    >>> import tfmindi as tm
    >>> # After creating AnnData with cell type mapping
    >>> cell_type_mapping = {0: "Neuron", 1: "Astrocyte", 2: "Microglia"}
    >>> adata = tm.pp.create_seqlet_adata(..., cell_type_mapping=cell_type_mapping)
    >>> # Create heatmap
    >>> fig = tm.pl.plot_dbd_heatmap(adata, show=False)
    >>> # Custom styling
    >>> tm.pl.plot_dbd_heatmap(adata, width=12, height=8, title="DBD Counts per Cell Type")
    """
    if dbd_column not in adata.obs.columns:
        raise ValueError(f"Column '{dbd_column}' not found in adata.obs")
    if cell_type_column not in adata.obs.columns:
        raise ValueError(f"Column '{cell_type_column}' not found in adata.obs")

    crosstab = pd.crosstab(adata.obs[cell_type_column].values, adata.obs[dbd_column].values)

    # Drop NaN columns if requested
    if drop_na:
        if "nan" in crosstab.columns:
            crosstab = crosstab.drop("nan", axis=1)
        crosstab = crosstab.dropna(axis=1, how="all")

    # Order columns by descending average values
    column_means = crosstab.mean(axis=0).sort_values(ascending=False)
    crosstab = crosstab[column_means.index]

    figsize = (
        kwargs.get("width", max(8, len(crosstab.columns) * 0.8)),
        kwargs.get("height", max(6, len(crosstab.index) * 0.4)),
    )

    cluster_grid = sns.clustermap(
        crosstab,
        cmap=cmap,
        xticklabels=True,
        yticklabels=True,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        figsize=figsize,
        linecolor="black",
        linewidths=linewidths,
        robust=True,
        cbar_kws={"shrink": 0.5, "aspect": 50, "fraction": 0.02},
    )

    fig = cluster_grid.fig

    # Remove axis labels
    cluster_grid.ax_heatmap.set_xlabel("")
    cluster_grid.ax_heatmap.set_ylabel("")

    # Make colorbar longer, narrower, and add black border
    cluster_grid.ax_cbar.set_position([0.1, 0.1, 0.02, 0.15])

    # Add black border around colorbar
    for spine in cluster_grid.ax_cbar.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)

    render_kwargs = {
        "title": "DBD Counts per Cell Type",
        **kwargs,
    }

    return render_plot(fig, **render_kwargs)
