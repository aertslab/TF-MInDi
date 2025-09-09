"""Saliency visualization functions."""

from __future__ import annotations

import logomaker
import matplotlib.patches
import matplotlib.pyplot as plt
import pandas as pd
from anndata import AnnData

from tfmindi.pl._utils import get_colors, render_plot


def region_contributions(
    adata: AnnData,
    example_idx: int | None = None,
    region_name: str | None = None,
    zoom_start: int | None = None,
    zoom_end: int | None = None,
    min_attribution: float | None = None,
    overlap_threshold=25,  # Base pairs - consider labels overlapping if within this distance
    show_unannotated: bool = False,
    dbd_names: str | list[str] | None = None,
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
        Must contain adata.uns['unique_examples'] with 'oh' and 'contrib' arrays,
        and adata.obs with 'example_oh_idx' and 'example_contrib_idx' columns.
    example_idx
        Index of the example/region to visualize. Mutually exclusive with region_name.
    region_name
        Name of the region to visualize (e.g., 'chr1:1000-2000'). Mutually exclusive with example_idx.
        Requires 'region_name' column in adata.obs.
    zoom_start
        Start position for zooming into a specific subregion (0-based indexing).
        Must be used together with zoom_end.
    zoom_end
        End position for zooming into a specific subregion (0-based indexing, exclusive).
        Must be used together with zoom_start.
    min_attribution
        Minimum absolute attribution score for seqlets to be highlighted with colored boxes.
        Seqlets with an absolute attribution below this threshold will not show colored rectangles.
        If None, all seqlets are highlighted regardless of attribution score.
    overlap_threshold
        Minimum distance (in base pairs) between seqlet labels to avoid overlap.
        Labels will be stacked vertically if they are too close together.
    show_unannotated
        Whether to show rectangles for seqlets without DBD annotations (default: False).
        When True, unannotated seqlets are shown in gray.
    dbd_names
        DNA-binding domain name(s) to display. Can be a single DBD name (string) or
        a list of DBD names. If None (default), all annotated DBDs are shown.
        Only seqlets with these specific DBD annotations will be highlighted and labeled.
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
    >>> fig = tm.pl.region_contributions(adata, example_idx=0)
    >>> # Plot saliency by region name
    >>> fig = tm.pl.region_contributions(adata, region_name="chr1:1000-2000")
    >>> # Zoom into a specific subregion (positions 50-150)
    >>> fig = tm.pl.region_contributions(adata, example_idx=0, zoom_start=50, zoom_end=150)
    >>> # Only show seqlets with high contribution scores
    >>> fig = tm.pl.region_contributions(adata, example_idx=0, min_attribution=0.5)
    >>> # Show only specific DBDs
    >>> fig = tm.pl.region_contributions(adata, example_idx=0, dbd_names="bZIP")
    >>> fig = tm.pl.region_contributions(adata, example_idx=0, dbd_names=["bZIP", "HLH"])
    >>> # Custom styling
    >>> tm.pl.region_contributions(adata, example_idx=0, width=15, height=6)
    """
    if example_idx is None and region_name is None:
        raise ValueError("Either 'example_idx' or 'region_name' must be provided")
    if example_idx is not None and region_name is not None:
        raise ValueError("'example_idx' and 'region_name' are mutually exclusive - provide only one")
    if (zoom_start is None) != (zoom_end is None):
        raise ValueError("'zoom_start' and 'zoom_end' must be provided together")
    if zoom_start is not None and zoom_end is not None:
        if zoom_start < 0 or zoom_end < 0:
            raise ValueError("'zoom_start' and 'zoom_end' must be non-negative")
        if zoom_start >= zoom_end:
            raise ValueError("'zoom_start' must be less than 'zoom_end'")
    if "unique_examples" not in adata.uns:
        raise ValueError("'unique_examples' not found in adata.uns. Use the new storage format.")
    if "oh" not in adata.uns["unique_examples"]:
        raise ValueError("'oh' array not found in unique_examples storage")
    if "contrib" not in adata.uns["unique_examples"]:
        raise ValueError("'contrib' array not found in unique_examples storage")
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

    hits = adata.obs.query("example_idx == @example_idx")[["start", "end", "cluster_dbd", "attribution"]].copy()
    if len(hits) == 0:
        raise ValueError(f"No seqlets found for {region_identifier}")

    # Handle DBD name filtering
    if dbd_names is not None:
        # Convert single string to list
        if isinstance(dbd_names, str):
            dbd_names = [dbd_names]

        # Validate that requested DBDs exist in the data
        available_dbds = hits["cluster_dbd"].dropna().unique()
        missing_dbds = set(dbd_names) - set(available_dbds)
        if missing_dbds:
            raise ValueError(f"DBD name(s) not found in data: {list(missing_dbds)}. Available: {list(available_dbds)}")

        # Filter to only show requested DBDs - keep all hits but filter annotated_dbds for coloring
        annotated_dbds = [dbd for dbd in available_dbds if dbd in dbd_names]
    else:
        annotated_dbds = hits["cluster_dbd"].dropna().unique()

    # Use stored colors for cluster_dbd column
    dbd_color_map = get_colors(adata, "cluster_dbd", cmap)

    # Filter color map to only include annotated DBDs
    dbd_color_map = {dbd: dbd_color_map[dbd] for dbd in annotated_dbds if dbd in dbd_color_map}

    # Find the seqlet index for this example
    matching_seqlets = adata.obs[adata.obs["example_idx"] == example_idx]
    if len(matching_seqlets) == 0:
        raise ValueError(f"No seqlets found for example_idx {example_idx}")
    # Use the first matching seqlet to get the example data
    seqlet_idx = adata.obs.index.get_loc(matching_seqlets.index[0])
    from tfmindi.pp.seqlets import get_example_contrib, get_example_oh

    contrib = get_example_contrib(adata, seqlet_idx)
    oh = get_example_oh(adata, seqlet_idx)

    region_length = contrib.shape[1]  # assuming contrib is shape (4, length)

    # Handle zooming
    if zoom_start is not None and zoom_end is not None:
        if zoom_end > region_length:
            raise ValueError(f"'zoom_end' ({zoom_end}) exceeds region length ({region_length})")
        if zoom_start >= region_length:
            raise ValueError(f"'zoom_start' ({zoom_start}) exceeds region length ({region_length})")

        contrib = contrib[:, zoom_start:zoom_end]
        oh = oh[:, zoom_start:zoom_end]

        hits = hits.copy()
        hits["start"] = hits["start"] - zoom_start
        hits["end"] = hits["end"] - zoom_start
        hits = hits[(hits["end"] > 0) & (hits["start"] < zoom_end - zoom_start)]
        hits.loc[hits["start"] < 0, "start"] = 0
        hits.loc[hits["end"] > zoom_end - zoom_start, "end"] = zoom_end - zoom_start

        x_min = 0
        x_max = zoom_end - zoom_start
    else:
        x_min = 0
        x_max = region_length

    fig, axs = plt.subplots(figsize=(15, 3), nrows=2, sharex=True, gridspec_kw={"height_ratios": [4, 1]})

    # Top panel: Saliency logo
    ax = axs[0]
    logo_data = pd.DataFrame((contrib * oh).T, columns=list("ACGT"))
    logomaker.Logo(logo_data, ax=ax, zorder=1)
    ax.set_rasterization_zorder(2)

    ymin, ymax = ax.get_ylim()

    # Add colored rectangles for seqlet regions
    for _i, (_, (start, end, dbd, score)) in enumerate(hits.sort_values("start").iterrows()):
        passes_threshold = min_attribution is None or abs(score) >= min_attribution

        if passes_threshold and pd.notna(dbd) and dbd in dbd_color_map:
            rect = matplotlib.patches.Rectangle(
                xy=(start, ymin), width=end - start, height=ymax - ymin, facecolor=dbd_color_map[dbd], alpha=0.3
            )
            ax.add_patch(rect)
        elif passes_threshold and show_unannotated and pd.isna(dbd):
            # Unannotated seqlets
            rect = matplotlib.patches.Rectangle(
                xy=(start, ymin), width=end - start, height=ymax - ymin, facecolor="gray", alpha=0.2
            )
            ax.add_patch(rect)

    # Bottom panel: DBD labels
    ax_bottom = axs[1]

    sorted_hits = hits.sort_values("start")
    label_positions = []
    labeled_dbds = {}
    use_above = False  # Simple alternation tracker

    for _, (start, end, dbd, score) in sorted_hits.iterrows():
        # Check if seqlet meets contribution threshold
        passes_threshold = min_attribution is None or abs(score) >= min_attribution

        if passes_threshold and pd.notna(dbd) and dbd in dbd_color_map:
            center_x = (start + end) / 2

            # Check if this DBD type already has a label in an overlapping region
            should_label = True

            if dbd in labeled_dbds:
                for existing_center in labeled_dbds[dbd]:
                    if abs(center_x - existing_center) < overlap_threshold:
                        should_label = False
                        break

            if should_label:
                y_pos = 0.2 if use_above else -0.2
                use_above = not use_above  # Flip for next label

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
