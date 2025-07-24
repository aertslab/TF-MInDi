"""DNA-binding domain heatmap visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData

from tfmindi.pl._utils import render_plot


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
