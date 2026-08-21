"""Fast sequence-logo rendering for long tracks.

``logomaker`` builds a fresh ``TextPath`` for every character it draws, which is what makes
long logos expensive: a 2 kb contribution track takes ~8 s. The glyph outlines are the same
four shapes every time, so this module builds them once, places copies with an affine
transform, and adds them to the axes as a single ``PathCollection`` instead of one patch
per character. The same track then draws in ~0.03 s.

Geometry, colours and axis limits reproduce ``logomaker.Logo``'s defaults for an ACGT
alphabet, so the output is visually identical; only the object churn is gone. The
approach is adapted from fast-logomaker (https://github.com/evanseitz/fast-logomaker,
MIT), which is not depended on directly because its batch API needs every logo to share
one length and is slower than ``logomaker`` on the many-short-logos case.
"""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

# logomaker's "classic" scheme, which is the default it picks for an ACGT alphabet.
_COLORS = {"A": (0.0, 0.5, 0.0), "C": (0.0, 0.0, 1.0), "G": (1.0, 0.65, 0.0), "T": (1.0, 0.0, 0.0)}
_ALPHABET = "ACGT"

# logomaker Glyph defaults: a character fills 0.95 of its position, and is never stretched
# horizontally more than an "E" would be (this is what keeps a narrow "I" from ballooning).
_GLYPH_WIDTH = 0.95
_STRETCH_LIMIT_CHAR = "E"


@cache
def _glyph_shapes(font_name: str, font_weight: str) -> dict[str, tuple]:
    """Build the upright and flipped outline of each base, normalised to the unit box.

    Each outline is pre-translated so its lower-left corner sits at the origin and
    pre-scaled horizontally, leaving only the per-glyph vertical scale and translation to
    apply at draw time. Cached because the ``TextPath`` construction is the expensive part.

    Parameters
    ----------
    font_name
        Font family to render the bases in.
    font_weight
        Font weight to render the bases in.

    Returns
    -------
    Base -> ``(upright, flipped, x_shift)``, where the paths are unit-height outlines and
    ``x_shift`` centres the glyph within its position.
    """
    prop = FontProperties(family=font_name, weight=font_weight)
    limit_width = TextPath((0, 0), _STRETCH_LIMIT_CHAR, size=1, prop=prop).get_extents().width

    shapes = {}
    for char in _ALPHABET:
        raw = TextPath((0, 0), char, size=1, prop=prop)
        extents = raw.get_extents()
        hstretch = min(_GLYPH_WIDTH / extents.width, _GLYPH_WIDTH / limit_width)
        x_shift = (_GLYPH_WIDTH - hstretch * extents.width) / 2.0

        # Normalise to a unit-height box anchored at the origin. The flipped variant is
        # built from the flipped outline's own extents, as logomaker does, so that a
        # mirrored glyph still sits flush against its floor.
        normalised = []
        for path in (raw, Affine2D().scale(1, -1).transform_path(raw)):
            box = path.get_extents()
            normalised.append(
                Affine2D().translate(-box.xmin, -box.ymin).scale(hstretch, 1.0 / box.height).transform_path(path)
            )
        shapes[char] = (normalised[0], normalised[1], x_shift)
    return shapes


def draw_logo(
    ax: plt.Axes,
    matrix: np.ndarray,
    zorder: int = 0,
    font_name: str = "sans",
    font_weight: str = "bold",
) -> None:
    """
    Draw a DNA sequence logo on an existing axes.

    A drop-in replacement for ``logomaker.Logo(df, ax=ax, zorder=zorder)`` for ACGT data,
    including its "big on top" stacking, downward-flipped negative characters, baseline
    and axis limits.

    Parameters
    ----------
    ax
        Axes to draw on.
    matrix
        Array of shape ``(n_positions, 4)`` giving the height of A, C, G and T at each
        position. Negative values are drawn below the baseline.
    zorder
        Draw order of the characters, passed through to the collection.
    font_name
        Font family to render the bases in.
    font_weight
        Font weight to render the bases in.

    Returns
    -------
    None. The characters, baseline and axis limits are added to ``ax`` in place.

    Raises
    ------
    ValueError
        If ``matrix`` does not have four columns.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(_ALPHABET):
        raise ValueError(f"matrix must have shape (n_positions, 4); got {values.shape}")

    # logomaker's "big_on_top" stack: characters are laid down in ascending value, so the
    # negatives stack downward from the summed-negative floor and the positives upward.
    order = np.argsort(values, axis=1, kind="stable")
    ordered = np.take_along_axis(values, order, axis=1)
    heights = np.abs(ordered)
    ceilings = np.where(ordered < 0, ordered, 0.0).sum(axis=1)[:, None] + np.cumsum(heights, axis=1)
    floors = ceilings - heights

    shapes = _glyph_shapes(font_name, font_weight)
    positions, slots = np.nonzero(heights)
    paths = []
    colors = []
    for pos, slot in zip(positions, slots, strict=True):
        char = _ALPHABET[order[pos, slot]]
        upright, flipped, x_shift = shapes[char]
        shape = flipped if ordered[pos, slot] < 0 else upright
        transform = (
            Affine2D().scale(1.0, heights[pos, slot]).translate(pos - _GLYPH_WIDTH / 2.0 + x_shift, floors[pos, slot])
        )
        paths.append(transform.transform_path(shape))
        colors.append(_COLORS[char])

    collection = PathCollection(paths, facecolors=colors, edgecolors="none", linewidths=0, zorder=zorder)
    collection.set_transform(ax.transData)
    ax.add_collection(collection, autolim=False)

    ax.axhline(0, color="black", linewidth=0.5, zorder=-1)
    ax.set_xlim(-_GLYPH_WIDTH / 2.0, len(values) - 1 + _GLYPH_WIDTH / 2.0)
    ax.set_ylim(floors.min(), ceilings.max())
