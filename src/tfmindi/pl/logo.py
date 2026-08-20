"""Sequence-logo plotting for patterns."""

from __future__ import annotations

import math
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from tfmindi.pl._glyphs import draw_logo
from tfmindi.pl._utils import render_plot
from tfmindi.types import Pattern

# Per-panel canvas size, in inches. A logo panel carries a two-line bold title, so the
# figure has to grow with the grid -- a fixed 8x8 canvas squeezes the rows until the
# titles overlap the logos and tight_layout gives up.
_PANEL_WIDTH = 4.0
_PANEL_HEIGHT = 1.9


def _cluster_sort_key(pattern: Pattern) -> tuple[int, int, str]:
    """Order cluster IDs numerically when they are numbers, alphabetically otherwise.

    Parameters
    ----------
    pattern
        Pattern whose ``cluster_id`` is being ordered.

    Returns
    -------
    Sort key placing numeric IDs first, in numeric order.
    """
    cid = str(pattern.cluster_id)
    if cid.lstrip("-").isdigit():
        return (0, int(cid), "")
    return (1, 0, cid)


def _select_patterns(
    patterns: dict[str, Pattern],
    group_by: Literal["annotation"] | None,
    annotation: str | None,
    ic_threshold: float,
    min_nucleotides: int,
    sort_by: Literal["ic", "n_seqlets", "cluster_id"],
) -> tuple[list[Pattern], list[str]]:
    """Choose which patterns to draw and what to title them.

    Parameters
    ----------
    patterns
        Cluster ID -> Pattern.
    group_by
        ``"annotation"`` keeps one representative pattern per annotation, None keeps all.
    annotation
        Keep only patterns carrying this annotation.
    ic_threshold
        IC threshold used to judge which representative survives trimming.
    min_nucleotides
        Minimum trimmed length a representative should have.
    sort_by
        Ordering of the selected patterns.

    Returns
    -------
    ``(selected, titles)`` — the patterns to draw and their subplot titles.

    Raises
    ------
    ValueError
        If the selection is empty, or ``sort_by`` is not a known option.
    """
    selected = list(patterns.values())

    if annotation is not None:
        selected = [p for p in selected if p.dbd == annotation]
        if not selected:
            raise ValueError(f"No patterns found with annotation '{annotation}'")

    if group_by == "annotation":
        # One pattern per annotation: the highest-IC one that still has enough
        # nucleotides after trimming, falling back to the highest-IC one overall.
        by_annotation: dict[str, list[Pattern]] = {}
        for pattern in selected:
            key = pattern.dbd
            if key is None or pd.isna(key) or key == "nan":
                continue
            by_annotation.setdefault(key, []).append(pattern)
        if not by_annotation:
            raise ValueError("No patterns with valid annotations found")

        selected = []
        for key in sorted(by_annotation):
            candidates = sorted(by_annotation[key], key=lambda p: p.ic().mean(), reverse=True)
            representative = candidates[0]
            for pattern in candidates:
                start_idx, end_idx = pattern.ic_trim(ic_threshold)
                if start_idx != end_idx and (end_idx - start_idx) >= min_nucleotides:
                    representative = pattern
                    break
            selected.append(representative)
        return selected, [str(p.dbd) for p in selected]

    if sort_by == "ic":
        selected.sort(key=lambda p: p.ic().mean(), reverse=True)
    elif sort_by == "n_seqlets":
        selected.sort(key=lambda p: p.n_seqlets, reverse=True)
    elif sort_by == "cluster_id":
        selected.sort(key=_cluster_sort_key)
    else:
        raise ValueError(f"Invalid sort_by option: {sort_by}. Must be 'ic', 'n_seqlets', or 'cluster_id'")

    # The annotation is redundant when the selection is already filtered to one.
    def _title(pattern: Pattern) -> str:
        head = f"Cluster {pattern.cluster_id}"
        if annotation is None and pattern.dbd is not None and pd.notna(pattern.dbd) and pattern.dbd != "nan":
            head = f"{head} - {pattern.dbd}"
        return f"{head}\n({pattern.n_seqlets} seqlets)"

    return selected, [_title(p) for p in selected]


def pattern_logos(
    patterns: dict[str, Pattern],
    group_by: Literal["annotation"] | None = None,
    annotation: str | None = None,
    ic_threshold: float = 0.2,
    min_nucleotides: int = 4,
    ncols: int | None = None,
    sort_by: Literal["ic", "n_seqlets", "cluster_id"] = "ic",
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Draw a grid of sequence logos for patterns.

    Which patterns are drawn depends on ``group_by`` and ``annotation``:

    - neither: one logo per pattern.
    - ``group_by="annotation"``: one representative logo per annotation, useful as an
      overview of the motif families present.
    - ``annotation="bHLH"``: every pattern carrying that annotation, useful for judging
      whether the clustering split a family into real variants or over-clustered it.

    Patterns that trim to fewer than ``min_nucleotides`` get a placeholder panel instead
    of a logo, so the grid stays aligned with the cluster IDs.

    Parameters
    ----------
    patterns
        Dictionary mapping cluster IDs to Pattern objects, as returned by
        :func:`tfmindi.tl.create_patterns`. The annotation is ``Pattern.dbd``, filled
        from ``create_patterns``' ``annotation_col``.
    group_by
        Set to ``"annotation"`` to keep only one representative pattern per annotation
        (the highest-IC one that survives trimming). None (default) keeps every pattern.
    annotation
        Keep only patterns carrying this annotation. Combines with ``group_by``.
    ic_threshold
        Information content threshold for logo trimming.
    min_nucleotides
        Minimum number of nucleotides required after trimming to draw a logo.
    ncols
        Number of columns in the subplot grid. If None, a roughly square grid is used.
    sort_by
        Order of the logos: ``"ic"`` (mean information content, descending),
        ``"n_seqlets"`` (descending) or ``"cluster_id"`` (ascending). Ignored when
        ``group_by="annotation"``, which sorts by annotation name.
    **kwargs
        Additional arguments passed to render_plot() for styling and display options.
        Common options include title, show, save_path, dpi.

    Returns
    -------
    Figure with the logo grid, or None if show=True.

    Raises
    ------
    ValueError
        If no pattern matches the requested selection.

    Examples
    --------
    >>> import tfmindi as tm
    >>> patterns = tm.tl.create_patterns(adata, annotation_col=f"predicted_{res}_predicted_family")
    >>> # One logo per TF family
    >>> tm.pl.pattern_logos(patterns, group_by="annotation", ncols=3)
    >>> # Every variant the clustering found for one family
    >>> tm.pl.pattern_logos(patterns, annotation="MADS box", ncols=3, width=15, height=3)
    >>> # Every pattern, largest clusters first
    >>> tm.pl.pattern_logos(patterns, sort_by="n_seqlets", ncols=4)
    """
    selected, titles = _select_patterns(patterns, group_by, annotation, ic_threshold, min_nucleotides, sort_by)

    n_patterns = len(selected)
    if ncols is None:
        ncols = math.ceil(math.sqrt(n_patterns))
    nrows = math.ceil(n_patterns / ncols)

    # Let render_plot handle figsize
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    flat_axes = axes.flatten()

    for ax, pattern, title in zip(flat_axes, selected, titles, strict=False):
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
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
        else:
            trimmed_ppm = pattern.ppm[start_idx:end_idx]
            trimmed_ic = ic[start_idx:end_idx]
            draw_logo(ax, trimmed_ppm * trimmed_ic[:, None])
            ax.set_ylabel("Bits", fontsize=10)

        ax.set_xticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")

    # Hide unused subplots
    for ax in flat_axes[n_patterns:]:
        ax.set_visible(False)

    if annotation is not None:
        default_title = f"{annotation} motif variants"
    elif group_by == "annotation":
        default_title = "Motif per annotation"
    else:
        default_title = "Pattern logos"

    # Scale the canvas with the grid; an explicit width/height still wins. Reserve a
    # strip at the top for the suptitle, which tight_layout does not account for.
    height = kwargs.get("height", _PANEL_HEIGHT * nrows)
    defaults = {
        "title": default_title,
        "width": _PANEL_WIDTH * ncols,
        "height": height,
        "tight_rect": (0, 0, 1, 1 - min(0.4, 0.5 / height)),
    }
    return render_plot(fig, **{**defaults, **kwargs})
