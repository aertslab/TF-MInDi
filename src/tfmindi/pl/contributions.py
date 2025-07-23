"""Saliency visualization functions."""

from __future__ import annotations

import logomaker
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from tfmindi.pl._utils import render_plot


def region_contributions(
    adata: AnnData,
    example_idx: int | None = None,
    region_name: str | None = None,
    overlap_threshold=25,  # Base pairs - consider labels overlapping if within this distance
    show_unannotated: bool = False,
    cmap: str = "tab20",
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Visualize contribution scores for a full genomic region with annotated seqlet regions.

    Creates a two-panel plot showing contribution scores as sequence logos for the entire
    region on top, and DNA-binding domain annotations for detected seqlets overlaid below.

    Parameters
    ----------
    adata
        AnnData object with seqlet data and region information.
        Must contain adata.obs columns 'example_oh' and 'example_contrib' with
        one-hot sequences and contribution scores for each region.
    example_idx
        Index of the example/region to visualize. Mutually exclusive with region_name.
    region_name
        Name of the region to visualize (e.g., 'chr1:1000-2000'). Mutually exclusive with example_idx.
        Requires 'region_name' column in adata.obs.
    overlap_threshold
        Minimum distance (in base pairs) between seqlet labels to avoid overlap.
        Labels will be stacked vertically if they are too close together.
    show_unannotated
        Whether to show rectangles for seqlets without DBD annotations (default: False).
        When True, unannotated seqlets are shown in gray.
    cmap
        Colormap name for DNA-binding domain coloring (default: "tab20").
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include width, height, title, xlabel, ylabel, show, save_path.

    Returns
    -------
    Figure with contribution score visualization, or None if show=True.

    Examples
    --------
    >>> import tfmindi as tm
    >>> # Plot saliency for example 0
    >>> fig = tm.pl.plot_region_saliency(adata, example_idx=0)
    >>> # Plot saliency by region name
    >>> fig = tm.pl.plot_region_saliency(adata, region_name="chr1:1000-2000")
    >>> # Custom styling
    >>> tm.pl.plot_region_saliency(adata, example_idx=0, width=15, height=6)
    """
    if example_idx is None and region_name is None:
        raise ValueError("Either 'example_idx' or 'region_name' must be provided")
    if example_idx is not None and region_name is not None:
        raise ValueError("'example_idx' and 'region_name' are mutually exclusive - provide only one")
    if "example_oh" not in adata.obs.columns:
        raise ValueError("'example_oh' column not found in adata.obs")
    if "example_contrib" not in adata.obs.columns:
        raise ValueError("'example_contrib' column not found in adata.obs")
    if "cluster_dbd" not in adata.obs.columns:
        raise ValueError("'cluster_dbd' column not found in adata.obs")

    # Handle region_name to example_idx conversion
    if region_name is not None:
        if "region_name" not in adata.obs.columns:
            raise ValueError("'region_name' column not found in adata.obs. Cannot use region_name indexing.")

        # Find example_idx for the given region_name
        matching_rows = adata.obs[adata.obs["region_name"] == region_name]
        if len(matching_rows) == 0:
            raise ValueError(f"No region found with name '{region_name}'")

        # Get the example_idx (should be the same for all seqlets in the same region)
        example_idx = matching_rows["example_idx"].iloc[0]
        region_identifier = region_name
    else:
        region_identifier = f"example {example_idx}"

    hits = adata.obs.query("example_idx == @example_idx")[["start", "end", "cluster_dbd", "mean_contrib"]].copy()
    if len(hits) == 0:
        raise ValueError(f"No seqlets found for {region_identifier}")

    annotated_dbds = hits["cluster_dbd"].dropna().unique()
    colormap = plt.get_cmap(cmap)
    colors = colormap(np.linspace(0, 1, len(annotated_dbds)))
    dbd_color_map = dict(zip(annotated_dbds, colors, strict=False))

    contrib = adata.obs.loc[adata.obs["example_idx"] == example_idx, "example_contrib"].iloc[0]
    oh = adata.obs.loc[adata.obs["example_idx"] == example_idx, "example_oh"].iloc[0]

    region_length = contrib.shape[1]  # assuming contrib is shape (4, length)
    x_min = 0
    x_max = region_length

    fig, axs = plt.subplots(figsize=(15, 3), nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # Top panel: Saliency logo
    ax = axs[0]
    logo_data = pd.DataFrame((contrib * oh).T, columns=list("ACGT"))
    logomaker.Logo(logo_data, ax=ax, zorder=1)
    ax.set_rasterization_zorder(2)

    ymin, ymax = ax.get_ylim()

    # Add colored rectangles for seqlet regions
    for _i, (_, (start, end, dbd, _score)) in enumerate(hits.sort_values("start").iterrows()):
        if pd.notna(dbd) and dbd in dbd_color_map:
            rect = matplotlib.patches.Rectangle(
                xy=(start, ymin), width=end - start, height=ymax - ymin, facecolor=dbd_color_map[dbd], alpha=0.3
            )
            ax.add_patch(rect)
        elif show_unannotated and pd.isna(dbd):
            # Unannotated seqlets
            rect = matplotlib.patches.Rectangle(
                xy=(start, ymin), width=end - start, height=ymax - ymin, facecolor="gray", alpha=0.2
            )
            ax.add_patch(rect)

    # Bottom panel: DBD labels with better positioning
    ax_bottom = axs[1]

    sorted_hits = hits.sort_values("start")
    label_positions = []
    labeled_dbds = {}

    for _, (start, end, dbd, _score) in sorted_hits.iterrows():
        if pd.notna(dbd) and dbd in dbd_color_map:
            center_x = (start + end) / 2

            # Check if this DBD type already has a label in an overlapping region
            should_label = True

            if dbd in labeled_dbds:
                for existing_center in labeled_dbds[dbd]:
                    if abs(center_x - existing_center) < overlap_threshold:
                        should_label = False
                        break

            if should_label:
                # Find a y position that doesn't overlap with existing labels
                y_pos = 0.1
                min_distance = 50
                for existing_x, existing_y in label_positions:
                    if abs(center_x - existing_x) < min_distance:
                        y_pos = existing_y + 0.3
                label_positions.append((center_x, y_pos))
                if dbd not in labeled_dbds:
                    labeled_dbds[dbd] = []
                labeled_dbds[dbd].append(center_x)
                ax_bottom.text(
                    center_x,
                    y_pos,
                    dbd,
                    fontsize=8,
                    color=dbd_color_map[dbd],
                    fontweight="bold",
                    ha="center",  # Center horizontally
                    va="bottom",  # Align bottom
                )
    for ax in axs:
        ax.set_xlim(x_min, x_max)
        ax.set_axis_off()
    fig.tight_layout()

    render_kwargs = {
        "width": 15,
        "height": 3,
        "xlabel": "Position",
        "ylabel": "Contribution Score",
        **kwargs,
    }

    return render_plot(fig, **render_kwargs)
