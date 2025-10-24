"""Plotting functions for distance bias visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import zscore

from tfmindi.pl._utils import render_plot
from tfmindi.types import BiasDetectionResult


def distance_bias_profile(
    result: BiasDetectionResult,
    pattern_color: str = "red",
    peak_color: str = "orange",
    line_color: str = "black",
    **kwargs,
) -> plt.Figure | None:
    """Plot contribution score profile showing pattern location and detected peaks.

    Creates a line plot of the averaged z-scored contribution profile with vertical
    lines indicating the pattern location and detected distance bias peaks.

    Parameters
    ----------
    result
        BiasDetectionResult object from detect_distance_bias().
    pattern_color
        Color for pattern location markers (default: "red").
    peak_color
        Color for peak location markers (default: "orange").
    line_color
        Color for the profile line (default: "black").
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include width, height, title, xlabel, ylabel, show, save_path.

    Returns
    -------
    Figure with profile plot, or None if show=True.

    Examples
    --------
    >>> import tfmindi as tm
    >>> patterns = tm.tl.create_patterns(adata)
    >>> result = tm.tl.detect_distance_bias(adata, patterns["0"], window=20)
    >>> fig = tm.pl.distance_bias_profile(result, title="Pattern 0")
    """
    fig, ax = plt.subplots()

    # Plot the profile
    ax.plot(np.arange(result.profile.shape[0]), result.profile, color=line_color, linewidth=1.5)

    # Mark pattern location
    for pos in result.pattern_location:
        ax.axvline(pos, color=pattern_color, linestyle="--", linewidth=1.5, alpha=0.7)

    # Mark detected peaks
    if result.has_bias:
        for peak_start, peak_end in result.peak_windows:
            ax.axvline(peak_start, color=peak_color, linestyle="--", linewidth=1.5, alpha=0.7)
            ax.axvline(peak_end, color=peak_color, linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Position (bp)")
    ax.set_ylabel("Mean Z-score")

    render_kwargs = {
        "width": 10,
        "height": 4,
        "title": "Distance Bias Profile",
        **kwargs,
    }

    return render_plot(fig, **render_kwargs)


def distance_bias_heatmap(
    result: BiasDetectionResult,
    pattern_color: str = "red",
    peak_color: str = "orange",
    cmap: str = "gray_r",
    vmin: float | None = None,
    robust: bool = True,
    **kwargs,
) -> plt.Figure | None:
    """Plot heatmap of contribution scores sorted by maximum signal position.

    Creates a heatmap showing individual seqlet contribution scores (z-scored)
    sorted by the position of maximum signal, with overlaid markers for pattern
    and peak locations.

    Parameters
    ----------
    result
        BiasDetectionResult object from detect_distance_bias().
    pattern_color
        Color for pattern location markers (default: "red").
    peak_color
        Color for peak location markers (default: "orange").
    cmap
        Colormap for heatmap (default: "gray_r").
    vmin
        Minimum value for colormap. If None, determined automatically.
    robust
        Use robust quantile-based colormap limits (default: True).
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include width, height, title, xlabel, ylabel, show, save_path.

    Returns
    -------
    Figure with heatmap, or None if show=True.

    Examples
    --------
    >>> import tfmindi as tm
    >>> patterns = tm.tl.create_patterns(adata)
    >>> result = tm.tl.detect_distance_bias(adata, patterns["0"], window=20)
    >>> fig = tm.pl.distance_bias_heatmap(result, figsize=(6, 12))
    """
    # Sort contribution scores by position of maximum signal
    z_contribution_score = zscore(result.contribution_scores, axis=1)
    sorted_idx = np.argsort(np.argmax(z_contribution_score, 1))
    sorted_scores = z_contribution_score[sorted_idx]

    peak_window_scores = result.max_contrib_peak_windows

    width = 6 + 2 * len(peak_window_scores)
    height = 10
    # Create figure
    figsize = kwargs.pop("figsize", (width, height))
    fig, axs = plt.subplots(
        figsize=figsize,
        ncols=1 + len(peak_window_scores),
        width_ratios=[6, *[2 for _ in range(len(peak_window_scores))]],
    )

    if not isinstance(axs, np.ndarray):
        axs = np.ndarray([axs])

    # Plot heatmap
    heatmap_kwargs = {
        "cmap": cmap,
        "robust": robust,
        "ax": axs[0],
        "yticklabels": False,
        "cbar_kws": {"label": "Z-score"},
    }
    if vmin is not None:
        heatmap_kwargs["vmin"] = vmin

    sns.heatmap(sorted_scores, **heatmap_kwargs)

    # Mark pattern location
    for pos in result.pattern_location:
        axs[0].axvline(pos, color=pattern_color, linewidth=2, alpha=0.8)

    # Mark detected peaks
    if result.has_bias:
        for peak_start, peak_end in result.peak_windows:
            axs[0].axvline(peak_start, color=peak_color, linewidth=2, alpha=0.8)
            axs[0].axvline(peak_end, color=peak_color, linewidth=2, alpha=0.8)

    axs[0].set_xlabel("Position (bp)")
    axs[0].set_ylabel("Seqlets (sorted)")

    # max contrib per peak window plot
    for max_contrib, ax in zip(peak_window_scores, axs[1:], strict=True):
        ax.scatter(
            x=max_contrib,
            y=np.arange(len(max_contrib)),
            color="black",
            s=1,
        )
        ax.axvline(vmin, color="red", lw=1)

    render_kwargs = {
        "width": width,
        "height": height,
        "title": "Distance Bias Heatmap",
        **kwargs,
    }

    return render_plot(fig, **render_kwargs)
