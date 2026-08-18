"""t-SNE visualization functions with sequence logos."""

from __future__ import annotations

import distinctipy as dp
import logomaker
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData

from tfmindi.pl._utils import get_point_colors, render_plot
from tfmindi.types import Pattern
def tsne(
    adata: AnnData,
    color_by: str = "leiden",
    obsm_key: str = 'X_tsne',
    alpha: float = 1.0,
    dot_size: float | None = None,
    show_legend: bool = True,
    plot_labels: bool = False,
    palette: dict | None = None,
    cmap: str = "tab20",
    fig_edge: int = 6,
    clean: bool = True,
    title: str | None = None,
    **kwargs,
) -> dict | plt.Figure | None:
    """
    Visualize an AnnData object in t-SNE space as a scatter plot.

    Works for both seqlet-level AnnData (obsm key 'X_tsne') and region-level
    AnnData (obsm key 'TSNE'). The correct key is auto-detected if obsm_key
    is not provided.

    Parameters
    ----------
    adata : AnnData
        AnnData object with t-SNE coordinates. Must contain either
        obsm['X_tsne'] (seqlet-level) or obsm['TSNE'] (region-level),
        or a custom key passed via obsm_key.
    color_by : str, optional
        Column in adata.obs to use for colouring points. Default is 'leiden'.
    obsm_key : str or None, optional
        Key in adata.obsm containing the 2D embedding. If None, auto-detects
        'X_tsne' or 'TSNE'. Default is None.
    alpha : float, optional
        Transparency of scatter points. Default is 0.2.
    s : float or None, optional
        Marker size. If None, defaults to fig_edge**2 / 5. Default is None.
    show_legend : bool, optional
        Whether to display a legend outside the plot. Default is True.
    plot_labels : bool, optional
        Whether to overlay cluster name labels at centroid positions with
        a white background box. Default is False.
    palette : dict or None, optional
        Mapping of category label -> colour. If None, auto-generated.
        Default is None.
    cmap : str, optional
        Colormap name for categorical data. Default is 'tab20'.
    fig_edge : int, optional
        Width and height of the figure in inches. Default is 6.
    clean : bool, optional
        If True, removes axes borders, ticks, and axis labels. Default is False.
    title : str or None, optional
        Plot title. If None, defaults to "{obsm_key} embedding with {color_by} colouring".
    **kwargs
        Additional arguments passed to render_plot().

    Returns
    -------
    dict
        The palette used for colouring, mapping category label -> colour.
    """
    
    if dot_size is None:
        dot_size = fig_edge ** 2 / 5

    # --- build plot df ------------------------------------------------------
    coords = adata.obsm[obsm_key]
    df = pd.DataFrame({
        "x":       coords[:, 0],
        "y":       coords[:, 1],
        "cluster": adata.obs[color_by].astype(str),
    })
    centroids = df.groupby("cluster")[["x", "y"]].mean()

    # --- palette ------------------------------------------------------------
    if not palette:
        palette = dict(zip(
            set(adata.obs[color_by].astype(str)),
            dp.get_colors(adata.obs[color_by].nunique()),
            strict=False,
        ))

    # --- scatter ------------------------------------------------------------
    plt.figure(figsize=(fig_edge, fig_edge))
    g = sns.scatterplot(
        x=df["x"], y=df["y"],
        s=dot_size, hue=df["cluster"],
        palette=palette,
        alpha=alpha,
        ec=None,
        legend=show_legend,
    )

    # --- legend -------------------------------------------------------------
    if show_legend:
        handles, labels = g.get_legend_handles_labels()
        label_handle = dict(zip(labels, handles, strict=False))
        ordered_labels = [l for l in palette if l in label_handle]
        g.legend(
            [label_handle[l] for l in ordered_labels],
            ordered_labels,
            loc="upper left", bbox_to_anchor=(1, 1),
            markerscale=12 / dot_size ** 0.5,
            ncol=len(set(df["cluster"])) // (4 * fig_edge) + 1,
            title=color_by,
        )

    # --- centroid labels ----------------------------------------------------
    if plot_labels:
        for cluster, (x, y) in centroids.iterrows():
            plt.text(x, y, cluster, fontsize=7, weight="bold",
                     ha="center", va="center", color="black",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5})

    # --- clean up -----------------------------------------------------------
    if clean:
        sns.despine(bottom=True, left=True)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

    plt.title(title if title else f"{obsm_key} embedding coloured by {color_by}")
    plt.show()

    return palette


# def tsne(
#     adata: AnnData,
#     color_by: str = "leiden",
#     alpha: float = 0.2,
#     s: float = 2,
#     show_legend: bool = True,
#     cmap: str = "tab20",
#     **kwargs,
# ) -> plt.Figure | None:  # type: ignore[return]
#     """
#     Visualize seqlet clusters in t-SNE space as a scatter plot.

#     Fast, lightweight function for data exploration without sequence logos.
#     Ideal for quickly examining cluster structure and testing different coloring schemes.

#     Parameters
#     ----------
#     adata
#         AnnData object with t-SNE coordinates and cluster assignments.
#         Must contain adata.obsm["X_tsne"].
#     color_by
#         Column in adata.obs to use for coloring points (default: "leiden").
#     alpha
#         Transparency of scatter points.
#     s
#         Size of scatter points.
#     show_legend
#         Whether to show the legend (default: True).
#     cmap
#         Colormap name for categorical data (default: "tab20").
#         Any valid matplotlib colormap name (e.g., "viridis", "plasma", "Set1").
#     **kwargs
#         Additional arguments passed to render_plot() for styling and display options.
#         Common options include width, height, title, xlabel, ylabel, show, save_path.

#     Returns
#     -------
#     matplotlib.Figure or None
#         Figure with t-SNE scatter plot, or None if show=True.

#     Examples
#     --------
#     >>> import tfmindi as tm
#     >>> # Basic t-SNE plot colored by clusters
#     >>> fig = tm.pl.tsne(adata, color_by="leiden")
#     >>> # Color by DNA-binding domain annotations
#     >>> tm.pl.tsne(adata, color_by="cluster_dbd", width=8, height=6)
#     >>> # Custom styling
#     >>> tm.pl.tsne(adata, color_by="leiden", alpha=0.8, s=30, title="Seqlet Clusters", show_legend=False)
#     """
#     # Check required data
#     if "X_tsne" not in adata.obsm:
#         raise ValueError("t-SNE coordinates not found. Run tm.tl.cluster_seqlets() first.")

#     # Get t-SNE coordinates
#     tsne_coords = adata.obsm["X_tsne"]
#     x_coords = tsne_coords[:, 0]
#     y_coords = tsne_coords[:, 1]

#     # Get colors using stored color management
#     point_colors, color_map = get_point_colors(adata, color_by, cmap)

#     # Filter color map to only include values present in current data
#     if color_map is not None:
#         color_map = {k: color_map[k] for k in adata.obs[color_by].unique()}

#     # Create scatter plot
#     fig, ax = plt.subplots(figsize=(8, 6))
#     scatter = ax.scatter(x_coords, y_coords, c=point_colors, alpha=alpha, s=s)

#     if show_legend:
#         if color_map is not None:
#             # Discrete/categorical legend
#             from matplotlib.lines import Line2D

#             legend_elements = [
#                 Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=str(val))
#                 for val, color in color_map.items()
#             ]
#             ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left", title=color_by)
#         else:
#             # Continuous colorbar legend
#             cbar = plt.colorbar(scatter, ax=ax)
#             cbar.set_label(color_by)

#     render_kwargs = {
#         "xlabel": "t-SNE 1",
#         "ylabel": "t-SNE 2",
#         "title": "Seqlet Clusters",
#         **kwargs,  # Allow user kwargs to override defaults
#     }

#     return render_plot(fig, **render_kwargs)


def tsne_logos(
    adata: AnnData,
    patterns: dict[str, Pattern] | None = None,
    color_by: str = "cluster_dbd",
    patterns_by: str = "leiden",
    logo_width: float = 1.0,
    logo_height: float = 0.8,
    alpha: float = 0.2,
    s: float = 2,
    ic_threshold: float = 0.2,
    min_nucleotides: int = 4,
    min_seqlets: int = 10,
    show_cluster_labels: bool = True,
    show_legend: bool = True,
    gray_background: bool = True,
    cmap: str = "tab20",
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Visualize seqlet clusters in t-SNE space with optional sequence logos at centroids.

    This function can display both basic t-SNE scatter plots and enhanced plots with
    sequence logos. When show_logos=False, it delegates to the basic tsne() function
    for fast exploration. When show_logos=True, it renders sequence logos at cluster
    centroids for publication-quality visualization.

    Parameters
    ----------
    adata
        AnnData object with t-SNE coordinates and cluster assignments.
        Must contain adata.obsm["X_tsne"].
    patterns
        Dictionary mapping cluster IDs to Pattern objects with PWMs.
        Required when show_logos=True, optional when show_logos=False.
    color_by
        Column in adata.obs to use for coloring points.
    patterns_by
        Categorical columns in adata.obs on which patterns were generated.
    logo_width
        Width of sequence logos relative to plot coordinates.
    logo_height
        Height of sequence logos relative to plot coordinates.
    alpha
        Transparency of scatter points.
    s
        Size of scatter points.
    ic_threshold
        Information content threshold for logo trimming.
    min_nucleotides
        Minimum number of nucleotides required after trimming to show logo.
        Patterns with fewer nucleotides will be skipped to avoid showing noisy logos.
    min_seqlets
        Minimum number of seqlets required in a cluster to display its logo.
        Clusters with fewer seqlets will be skipped to avoid showing weak motifs from small clusters.
    show_cluster_labels
        Whether to show cluster ID labels.
    show_legend
        Whether to show the legend (default: True).
    gray_background
        Whether to use gray background for all scatter points to improve logo readability.
        When True, all points are colored gray and no legend is shown.
    cmap
        Colormap name for categorical data (default: "tab20").
        Any valid matplotlib colormap name (e.g., "viridis", "plasma", "Set1").
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include width, height, title, xlabel, ylabel, show, save_path.

    Returns
    -------
    matplotlib.Figure or None
        Figure with t-SNE plot and optional sequence logos, or None if show=True.

    Examples
    --------
    >>> import tfmindi as tm
    >>> # After clustering and pattern creation - full plot with logos
    >>> tm.pl.tsne_logos(adata, patterns, color_by="cluster_dbd", width=12, height=10)
    >>>
    >>> # Publication plot with custom styling
    >>> tm.pl.tsne_logos(adata, patterns, width=30, height=30, title="My Plot", show=False, save_path="plot.png")
    """
    # Check required data for logo plotting
    if "X_tsne" not in adata.obsm:
        raise ValueError("t-SNE coordinates not found. Run tm.tl.cluster_seqlets() first.")
    if patterns is None:
        raise ValueError("patterns parameter is required when show_logos=True.")
    if patterns_by not in adata.obs:
        raise ValueError(f"patterns_by={patterns_by} not in adata.obs")
    if len(set(adata.obs[patterns_by]) & set(patterns.keys())) == 0:
        raise ValueError(f"adata.obs['{patterns_by}'] does not match with patterns.")

    # Get t-SNE coordinates
    tsne_coords = adata.obsm["X_tsne"]
    x_coords = tsne_coords[:, 0]
    y_coords = tsne_coords[:, 1]

    # Handle gray background option
    if gray_background:
        point_colors = "gray"
        color_map = None
    else:
        point_colors, color_map = get_point_colors(adata, color_by, cmap)
        color_column = color_by

        # Filter color map to only include values present in current data
        if color_map is not None:
            color_map = {k: color_map[k] for k in adata.obs[color_column].unique()}

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(x_coords, y_coords, c=point_colors, alpha=alpha, s=s)

    # Add logos at cluster centroids
    cluster_coords = {}
    for cluster_id in adata.obs[patterns_by].unique():
        if cluster_id in patterns:
            cluster_mask = adata.obs[patterns_by] == cluster_id
            cluster_size = cluster_mask.sum()

            # Skip clusters with too few seqlets
            if cluster_size < min_seqlets:
                continue

            cluster_x = x_coords[cluster_mask].mean()
            cluster_y = y_coords[cluster_mask].mean()
            cluster_coords[cluster_id] = (cluster_x, cluster_y)

            pattern = patterns[cluster_id]
            _add_logo_to_plot(
                ax,
                pattern,
                cluster_x,
                cluster_y,
                logo_width,
                logo_height,
                ic_threshold,
                min_nucleotides,
                cluster_id,
                show_cluster_labels,
            )

    # Add legend if requested (skip when using gray background)
    if show_legend and not gray_background:
        if color_map is not None:
            # Discrete/categorical legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=str(val))
                for val, color in color_map.items()
            ]
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left", title=color_by)
        else:
            # Continuous colorbar legend
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_by)

    # Set default values for render_plot, but allow kwargs to override
    render_kwargs = {
        "xlabel": "t-SNE 1",
        "ylabel": "t-SNE 2",
        "title": "Seqlet Clusters",
        **kwargs,  # Allow user kwargs to override defaults
    }

    return render_plot(fig, **render_kwargs)


def _add_logo_to_plot(
    ax: plt.Axes,  # type: ignore[no-untyped-def]
    pattern: Pattern,
    x: float,
    y: float,
    width: float,
    height: float,
    ic_threshold: float,
    min_nucleotides: int,
    cluster_id: str,
    show_label: bool,
) -> None:
    """Add a sequence logo to the plot at specified coordinates."""
    # Trim pattern based on information content
    ic = pattern.ic()
    start_idx, end_idx = pattern.ic_trim(ic_threshold)
    if start_idx == end_idx:  # If no high-IC region found, use full pattern
        start_idx, end_idx = 0, len(pattern.ppm)

    # Extract trimmed PWM
    trimmed_ppm = pattern.ppm[start_idx:end_idx]
    trimmed_ic = ic[start_idx:end_idx]

    # Skip if pattern is too short or empty
    if len(trimmed_ppm) == 0 or len(trimmed_ppm) < min_nucleotides:
        return

    # Create PWM DataFrame for logomaker
    pwm_df = pd.DataFrame(trimmed_ppm * trimmed_ic[:, None], columns=["A", "C", "G", "T"])
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(ax, width=width, height=height, bbox_to_anchor=(x, y), bbox_transform=ax.transData, loc="center")
    _ = logomaker.Logo(pwm_df, ax=axins, color_scheme="classic")
    axins.set_axis_off()
    if show_label:
        ax.text(x, y - height / 2 - 0.1, f"{cluster_id}", ha="center", va="top", fontsize=8, fontweight="bold")


# def region_tsne(
#     region_adata: AnnData,
#     obsm_key: str = 'TSNE',
#     color_by: str = 'cell_type',
#     title: str | None = None,
#     plot_legend: bool = False,
#     plot_labels: bool = True,
#     palette: dict | None = None,
#     fig_edge: int = 6,
#     dot_size: int = None,
#     clean: bool = True,
# ) -> plt.Figure | None:
#     """
#     Plot a 2D embedding (t-SNE or other) of region_adata coloured by a categorical variable.

#     Produces a scatter plot of the 2D embedding stored in obsm[obsm_key], with points
#     coloured by obs[color_by]. Cluster labels can be overlaid at centroid positions
#     and/or a legend can be shown. A palette is auto-generated if not provided and
#     returned for reuse across plots.

#     Parameters
#     ----------
#     region_adata : AnnData
#         Region-level AnnData object. Must contain a 2D embedding in obsm[obsm_key]
#         and a categorical column obs[color_by].
#     obsm_key : str, optional
#         Key in region_adata.obsm containing the 2D embedding coordinates.
#         Default is 'TSNE'.
#     color_by : str, optional
#         Column in region_adata.obs used to colour points and compute centroids.
#         Default is 'cell_type'.
#     title : str or None, optional
#         Plot title. If None, defaults to "{obsm_key} embedding with {color_by} colouring".
#     plot_legend : bool, optional
#         Whether to display a legend. The legend is ordered to match palette key order
#         and placed outside the plot to the right. Default is False.
#     plot_labels : bool, optional
#         Whether to overlay cluster name labels at centroid positions, with a white
#         background box for readability. Default is True.
#     palette : dict or None, optional
#         Mapping of category label -> colour. If None, a palette is auto-generated
#         using dp.get_colors(). The palette used is always returned. Default is None.
#     fig_edge : int, optional
#         Width and height of the figure in inches. Default is 6.
#     dot_size : int or None, optional
#         Marker size for scatter points. If None, defaults to fig_edge**2 / 5.
#     clean : bool, optional
#         If True, removes axes borders, ticks, and axis labels for a cleaner look.
#         Default is True.

#     Returns
#     -------
#     dict
#         The palette used for colouring, mapping category label -> colour.
#         Useful for reusing consistent colours across multiple plots.
#     """
#     if not dot_size:
#         dot_size = fig_edge**2 / 5

#     df = pd.DataFrame({
#         "x": region_adata.obsm[obsm_key][:,0],
#         "y": region_adata.obsm[obsm_key][:,1],
#         "cluster": region_adata.obs[color_by].astype(str)
#     })
#     centroids = df.groupby("cluster")[["x","y"]].mean()

#     # Create a palette if none is defined
#     if not palette:
#         palette = dict(zip(set(region_adata.obs[color_by]),dp.get_colors(len(set(region_adata.obs[color_by]))), strict=False))

#     # Plot scatterplot
#     plt.figure(figsize=(fig_edge,fig_edge))
#     g = sns.scatterplot(x=df["x"], y=df["y"], s=dot_size, hue= df["cluster"], palette=palette, ec=None, legend=plot_legend)

#     # Plot with legend or plot names on top of TSNE
#     if plot_legend:
#         handles, labels = g.get_legend_handles_labels()
#         label_handle = dict(zip(labels, handles, strict=False))
#         g.legend([label_handle[label] for label in [label for label in list(palette.keys()) if label in labels]],
#                         [label for label in list(palette.keys()) if label in labels],
#                 loc='upper left', bbox_to_anchor=(1, 1), markerscale=12 / dot_size**0.5, ncol=len(set(df["cluster"])) // (4 * fig_edge) + 1)

#     if plot_labels:
#         for cluster, (x, y) in centroids.iterrows():
#             plt.text(x, y, cluster, fontsize=7, weight="bold",
#                     ha="center", va="center", color="black",
#                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5})

#     # Remove ugly borders and useless ticks
#     if clean:
#         sns.despine(bottom=True, left=True)
#         plt.xticks([])
#         plt.yticks([])
#         plt.xlabel('')
#         plt.ylabel('')

#     # Add a title
#     plt.title(title) if title else plt.title(f"{obsm_key} embedding with {color_by} colouring")
#     plt.show()

#     return palette
