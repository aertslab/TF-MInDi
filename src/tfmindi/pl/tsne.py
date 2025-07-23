"""t-SNE visualization functions with sequence logos."""

from __future__ import annotations

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from tfmindi.pl._utils import render_plot
from tfmindi.types import Pattern


def tsne(
    adata: AnnData,
    color_by: str = "leiden",
    alpha: float = 0.2,
    s: float = 2,
    show_legend: bool = True,
    cmap: str = "tab20",
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Visualize seqlet clusters in t-SNE space as a scatter plot.

    Fast, lightweight function for data exploration without sequence logos.
    Ideal for quickly examining cluster structure and testing different coloring schemes.

    Parameters
    ----------
    adata
        AnnData object with t-SNE coordinates and cluster assignments.
        Must contain adata.obsm["X_tsne"] and adata.obs["leiden"].
    color_by
        Column in adata.obs to use for coloring points (default: "leiden").
    alpha
        Transparency of scatter points.
    s
        Size of scatter points.
    show_legend
        Whether to show the legend (default: True).
    cmap
        Colormap name for categorical data (default: "tab20").
        Any valid matplotlib colormap name (e.g., "viridis", "plasma", "Set1").
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include width, height, title, xlabel, ylabel, show, save_path.

    Returns
    -------
    matplotlib.Figure or None
        Figure with t-SNE scatter plot, or None if show=True.

    Examples
    --------
    >>> import tfmindi as tm
    >>> # Basic t-SNE plot colored by clusters
    >>> fig = tm.pl.tsne(adata, color_by="leiden")
    >>> # Color by DNA-binding domain annotations
    >>> tm.pl.tsne(adata, color_by="cluster_dbd", width=8, height=6)
    >>> # Custom styling
    >>> tm.pl.tsne(adata, color_by="leiden", alpha=0.8, s=30, title="Seqlet Clusters", show_legend=False)
    """
    # Check required data
    if "X_tsne" not in adata.obsm:
        raise ValueError("t-SNE coordinates not found. Run tm.tl.cluster_seqlets() first.")
    if "leiden" not in adata.obs:
        raise ValueError("Cluster assignments not found. Run tm.tl.cluster_seqlets() first.")

    # Get t-SNE coordinates
    tsne_coords = adata.obsm["X_tsne"]
    x_coords = tsne_coords[:, 0]
    y_coords = tsne_coords[:, 1]

    # Determine colors and color mapping
    if color_by in adata.obs.columns:
        color_values = adata.obs[color_by]
        if color_values.dtype == "category" or color_values.dtype == object:
            # Handle categorical data with NaN values
            if color_values.dtype == "category":
                if "Unknown" not in color_values.cat.categories:
                    color_values = color_values.cat.add_categories(["Unknown"])
                color_values = color_values.fillna("Unknown")
            else:
                color_values = color_values.fillna("Unknown")

            unique_values = color_values.unique()
            colormap = plt.get_cmap(cmap)
            colors = colormap(np.linspace(0, 1, len(unique_values)))
            color_map = dict(zip(unique_values, colors, strict=False))
            if "Unknown" in color_map:
                color_map["Unknown"] = "lightgray"
            point_colors = [color_map[val] for val in color_values]
        else:
            point_colors = color_values
            color_map = None
    else:
        raise ValueError(f"Column '{color_by}' not found in adata.obs")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x_coords, y_coords, c=point_colors, alpha=alpha, s=s)

    if show_legend:
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

    render_kwargs = {
        "xlabel": "t-SNE 1",
        "ylabel": "t-SNE 2",
        "title": "Seqlet Clusters",
        **kwargs,  # Allow user kwargs to override defaults
    }

    return render_plot(fig, **render_kwargs)


def tsne_logos(
    adata: AnnData,
    patterns: dict[str, Pattern] | None = None,
    color_by: str = "cluster_dbd",
    logo_width: float = 1.0,
    logo_height: float = 0.8,
    alpha: float = 0.2,
    s: float = 2,
    ic_threshold: float = 0.2,
    min_nucleotides: int = 4,
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
        Must contain adata.obsm["X_tsne"] and adata.obs["leiden"].
    patterns
        Dictionary mapping cluster IDs to Pattern objects with PWMs.
        Required when show_logos=True, optional when show_logos=False.
    color_by
        Column in adata.obs to use for coloring points.
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
    if "leiden" not in adata.obs:
        raise ValueError("Cluster assignments not found. Run tm.tl.cluster_seqlets() first.")
    if patterns is None:
        raise ValueError("patterns parameter is required when show_logos=True.")

    # Get t-SNE coordinates
    tsne_coords = adata.obsm["X_tsne"]
    x_coords = tsne_coords[:, 0]
    y_coords = tsne_coords[:, 1]

    # Handle gray background option
    if gray_background:
        point_colors = "gray"
        color_map = None
    elif color_by in adata.obs.columns:
        color_values = adata.obs[color_by]
        if color_values.dtype == "category" or color_values.dtype == object:
            # Handle categorical data with NaN values
            if color_values.dtype == "category":
                # Add "Unknown" to categories if not already present
                if "Unknown" not in color_values.cat.categories:
                    color_values = color_values.cat.add_categories(["Unknown"])
                # Convert NaN values to "Unknown"
                color_values = color_values.fillna("Unknown")
            else:
                # For object dtype, convert NaN to "Unknown"
                color_values = color_values.fillna("Unknown")

            unique_values = color_values.unique()
            colormap = plt.get_cmap(cmap)
            colors = colormap(np.linspace(0, 1, len(unique_values)))
            color_map = dict(zip(unique_values, colors, strict=False))
            # Ensure "Unknown" gets a specific color (lightgray)
            if "Unknown" in color_map:
                color_map["Unknown"] = "lightgray"
            point_colors = [color_map[val] for val in color_values]
        else:
            point_colors = color_values
            color_map = None
    else:
        clusters = adata.obs["leiden"]
        unique_clusters = clusters.unique()
        colormap = plt.get_cmap(cmap)
        colors = colormap(np.linspace(0, 1, len(unique_clusters)))
        color_map = dict(zip(unique_clusters, colors, strict=False))
        point_colors = [color_map[cluster] for cluster in clusters]

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(x_coords, y_coords, c=point_colors, alpha=alpha, s=s)

    # Add logos at cluster centroids
    cluster_coords = {}
    for cluster_id in adata.obs["leiden"].unique():
        if cluster_id in patterns:
            cluster_mask = adata.obs["leiden"] == cluster_id
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
