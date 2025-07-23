"""Logo plotting functions for DNA-binding domain visualization."""

from __future__ import annotations

import math

import logomaker
import matplotlib.pyplot as plt
import pandas as pd

from tfmindi.pl._utils import render_plot
from tfmindi.types import Pattern


def dbd_logos(
    patterns: dict[str, Pattern],
    ic_threshold: float = 0.2,
    min_nucleotides: int = 4,
    ncols: int | None = None,
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Create a subplot grid showing sequence logos for each DNA-binding domain.

    This function groups patterns by their DBD annotation and creates one subplot
    per DBD, showing a representative logo for that domain. Only patterns with
    DBD annotations are included in the plot.

    Parameters
    ----------
    patterns
        Dictionary mapping cluster IDs to Pattern objects with DBD annotations.
        Patterns without DBD annotations (dbd=None) are ignored.
    ic_threshold
        Information content threshold for logo trimming.
    min_nucleotides
        Minimum number of nucleotides required after trimming to show logo.
        Patterns with fewer nucleotides will be skipped.
    ncols
        Number of columns in the subplot grid. If None, automatically determined
        based on the number of DBDs (aim for roughly square grid).
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include title, show, save_path, dpi.

    Returns
    -------
    Figure with logo subplot grid, or None if show=True.

    Examples
    --------
    >>> import tfmindi as tm
    >>> # After clustering and creating patterns with DBD annotations
    >>> patterns = tm.tl.create_patterns(adata)
    >>> # Create logo plots for each DBD
    >>> fig = tm.pl.dbd_logos(patterns, title="DBD Logos")
    >>> # Custom layout and styling
    >>> tm.pl.dbd_logos(
    ...     patterns, ncols=3, ic_threshold=0.15, min_nucleotides=5, figsize=(12, 8), save_path="dbd_logos.png"
    ... )
    """
    # Group patterns by DBD annotation
    dbd_to_patterns = {}
    for pattern in patterns.values():
        if pattern.dbd is not None and pd.notna(pattern.dbd) and pattern.dbd != "nan":
            if pattern.dbd not in dbd_to_patterns:
                dbd_to_patterns[pattern.dbd] = []
            dbd_to_patterns[pattern.dbd].append(pattern)

    if not dbd_to_patterns:
        raise ValueError("No patterns with valid DBD annotations found")

    dbd_representatives = {}
    for dbd, dbd_patterns in dbd_to_patterns.items():
        # Select pattern with the highest average information content
        best_pattern = max(dbd_patterns, key=lambda p: p.ic().mean())
        dbd_representatives[dbd] = best_pattern

    n_dbds = len(dbd_representatives)

    # Calculate grid layout
    if ncols is None:
        ncols = math.ceil(math.sqrt(n_dbds))
    nrows = math.ceil(n_dbds / ncols)

    # Create subplot grid (let render_plot handle figsize)
    fig, axes = plt.subplots(nrows, ncols)

    if n_dbds == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    sorted_dbds = sorted(dbd_representatives.keys())
    for i, dbd in enumerate(sorted_dbds):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]

        pattern = dbd_representatives[dbd]

        # Apply IC trimming
        ic = pattern.ic()
        start_idx, end_idx = pattern.ic_trim(ic_threshold)

        # Check if pattern is too short after trimming
        if start_idx == end_idx or (end_idx - start_idx) < min_nucleotides:
            # Create empty plot with message
            ax.text(
                0.5,
                0.5,
                f"Pattern too short\n({end_idx - start_idx} nt)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set_title(dbd, fontsize=12, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Extract trimmed data
        trimmed_ppm = pattern.ppm[start_idx:end_idx]
        trimmed_ic = ic[start_idx:end_idx]

        logo_data = pd.DataFrame(trimmed_ppm * trimmed_ic[:, None], columns=["A", "C", "G", "T"])
        logomaker.Logo(logo_data, ax=ax, color_scheme="classic")

        ax.set_ylabel("Bits", fontsize=10)
        ax.set_xticks([])
        ax.set_title(dbd, fontsize=12, fontweight="bold")

    # Hide unused subplots
    for i in range(n_dbds, nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col] if nrows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout()

    # Apply render_plot styling
    render_kwargs = {
        "title": "DNA-Binding Domain Logos",
        **kwargs,
    }

    return render_plot(fig, **render_kwargs)


def dbd_cluster_logos(
    patterns: dict[str, Pattern],
    dbd_name: str,
    ic_threshold: float = 0.2,
    min_nucleotides: int = 4,
    ncols: int | None = None,
    sort_by: str = "ic",
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Create a subplot grid showing sequence logos for all clusters of a specific DBD.

    This function filters patterns by the specified DBD and creates one subplot
    per cluster/pattern, showing the diversity of motifs within that domain.

    Parameters
    ----------
    patterns
        Dictionary mapping cluster IDs to Pattern objects with DBD annotations.
    dbd_name
        Name of the DNA-binding domain to display (e.g., "bHLH", "Homeobox").
        Only patterns with this DBD annotation will be shown.
    ic_threshold
        Information content threshold for logo trimming.
    min_nucleotides
        Minimum number of nucleotides required after trimming to show logo.
        Patterns with fewer nucleotides will be skipped.
    ncols
        Number of columns in the subplot grid. If None, automatically determined
        based on the number of patterns (aim for roughly square grid).
    sort_by
        How to sort the patterns in the grid. Options:
        - "ic": Sort by average information content (descending)
        - "n_seqlets": Sort by number of seqlets (descending)
        - "cluster_id": Sort by cluster ID (ascending)
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include title, show, save_path, dpi.

    Returns
    -------
    Figure with logo subplot grid, or None if show=True.

    Examples
    --------
    >>> import tfmindi as tm
    >>> # After clustering and creating patterns with DBD annotations
    >>> patterns = tm.tl.create_patterns(adata)
    >>> # Show all bHLH patterns
    >>> fig = tm.pl.dbd_cluster_logos(patterns, "bHLH", title="bHLH Motif Variants")
    >>> # Custom layout and sorting
    >>> tm.pl.dbd_cluster_logos(patterns, "Homeobox", ncols=4, sort_by="n_seqlets", save_path="homeobox_variants.png")
    """
    # Filter patterns by the specified DBD
    dbd_patterns = []
    for pattern in patterns.values():
        if pattern.dbd == dbd_name:
            dbd_patterns.append(pattern)

    if not dbd_patterns:
        raise ValueError(f"No patterns found for DBD '{dbd_name}'")

    # Sort patterns according to specified criteria
    if sort_by == "ic":
        dbd_patterns.sort(key=lambda p: p.ic().mean(), reverse=True)
    elif sort_by == "n_seqlets":
        dbd_patterns.sort(key=lambda p: p.n_seqlets, reverse=True)
    elif sort_by == "cluster_id":
        dbd_patterns.sort(key=lambda p: p.cluster_id)
    else:
        raise ValueError(f"Invalid sort_by option: {sort_by}. Must be 'ic', 'n_seqlets', or 'cluster_id'")

    n_patterns = len(dbd_patterns)

    # Calculate grid layout
    if ncols is None:
        ncols = math.ceil(math.sqrt(n_patterns))
    nrows = math.ceil(n_patterns / ncols)

    # Create subplot grid (let render_plot handle figsize)
    fig, axes = plt.subplots(nrows, ncols)

    # Handle different subplot cases to ensure consistent indexing
    if n_patterns == 1:
        axes = [axes]
    else:
        # Flatten axes array to allow consistent [i] indexing
        axes = axes.flatten()

    for i, pattern in enumerate(dbd_patterns):
        ax = axes[i]

        # Apply IC trimming
        ic = pattern.ic()
        start_idx, end_idx = pattern.ic_trim(ic_threshold)

        if start_idx == end_idx or (end_idx - start_idx) < min_nucleotides:
            ax.text(
                0.5,
                0.5,
                f"Pattern too short\n({end_idx - start_idx} nt)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set_title(f"Cluster {pattern.cluster_id}\n({pattern.n_seqlets} seqlets)", fontsize=12, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        trimmed_ppm = pattern.ppm[start_idx:end_idx]
        trimmed_ic = ic[start_idx:end_idx]

        logo_data = pd.DataFrame(trimmed_ppm * trimmed_ic[:, None], columns=["A", "C", "G", "T"])

        logomaker.Logo(logo_data, ax=ax, color_scheme="classic")

        ax.set_ylabel("Bits", fontsize=10)
        ax.set_xticks([])

        title_parts = [f"Cluster {pattern.cluster_id}"]
        title_parts.append(f"({pattern.n_seqlets} seqlets)")
        if sort_by == "ic":
            title_parts.append(f"IC: {pattern.ic().mean():.2f}")

        ax.set_title("\n".join(title_parts), fontsize=10, fontweight="bold")

    # Hide unused subplots
    for i in range(n_patterns, nrows * ncols):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Apply render_plot styling
    render_kwargs = {
        "title": f"{dbd_name} Motif Variants",
        "height": 3 * nrows,
        **kwargs,
    }

    return render_plot(fig, **render_kwargs)
