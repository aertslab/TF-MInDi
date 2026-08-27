"""Distance dependent bias detector functionality."""

from typing import Any

import ncls  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from anndata import AnnData  # type: ignore
from scipy.signal import find_peaks, peak_widths  # type: ignore
from scipy.stats import zscore  # type: ignore
from tqdm import tqdm  # type: ignore

from tfmindi.pp.seqlets import _extract_seqlet_matrices
from tfmindi.types import BiasDetectionResult, Pattern


def _calc_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Calculate overlap between two intervals.

    A              |--*--*--|
    b           |--*--*--|

    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    0..1..2..3..4..5..6..7..8..9.10.11.12.13.14.15.16.17.18.19.20
    max(new_start, o_start) = 5
    min(new_end, o_start) = 7
    overlap = 7 - 5 = 2

    a                       |--*--*--|
    b           |--*--*--|

    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    0..1..2..3..4..5..6..7..8..9.10.11.12.13.14.15.16.17.18.19.20
    max(new_start, o_start) = 8
    min(new_end, o_start) = 7
    overlap = 7 - 8 = -1

    a              |--*--*--|
    b                             |--*--*--|

    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    0..1..2..3..4..5..6..7..8..9.10.11.12.13.14.15.16.17.18.19.20
    max(new_start, o_start) = 10
    min(new_end, o_start) = 8
    overlap = 8 - 10 = -2
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def detect_distance_bias(
    adata: AnnData,
    pattern: Pattern,
    window: int = 20,
    height: float = 0.25,
    min_distance_peak: int = 3,
    rel_height_peak_width: float = 0.5,
    **kwargs,
) -> BiasDetectionResult:
    """Detect distance bias by identifying TFBS instances at fixed distances from the given pattern.

    This function analyzes contribution scores around seqlets that were matched to that pattern to detect commonly occuring nearby peaks
    indicating cooperative transcription factor binding sites.

    Parameters
    ----------
    adata
        AnnData object with stored seqlet data.
        Must contain:
        - adata.uns["unique_examples"]["oh"]: Unique example one-hot sequences
        - adata.uns["unique_examples"]["contrib"]: Unique example contribution scores
    pattern
        A Pattern to detect distance bias for.
    window
        Number of basepairs to look up- and downstream of the pattern.
    height
        Required height of peaks in the contribution profile to be used in scipy.signal.find_peaks.
    min_distance_peak
        Minimal distance of a peak relative to the pattern instance.
    rel_height_peak_width
        Relative height to calculate the peak width at using scipy.signal.peak_widths.
    **kwargs
        Extra keyword arguments passed to scipy.signal.find_peaks.

    Returns
    -------
    BiasDetectionResult object containing detection results and methods for further analysis.

    Examples
    --------
    >>> import tfmindi as tm
    >>> patterns = tm.tl.create_patterns(adata)
    >>> result = tm.tl.detect_distance_bias(adata, patterns["0"], window=20)
    >>> if result.has_bias:
    >>>     print(f"Detected {len(result.peak_windows)} biased peaks")
    >>>     biased_seqlets = result.get_biased_seqlets(threshold=0.5)
    """
    n_seqlets = len(pattern.seqlets)
    length_pattern = pattern.ppm.shape[0]
    length_sequence = adata.uns["unique_examples"]["oh"].shape[2]

    contribution_scores = np.zeros((n_seqlets, length_pattern + 2 * window))

    for i, seqlet in enumerate(pattern.seqlets):
        # Calculate padding needed if the window extends beyond the sequence boundaries
        to_pad_up = abs(min(seqlet.start - window, 0))
        to_pad_down = max(seqlet.start + length_pattern + window, length_sequence) - length_sequence

        c_window = adata.uns["unique_examples"]["contrib"][
            seqlet.example_idx,
            :,
            max(seqlet.start - window, 0) : min(seqlet.start + length_pattern + window, length_sequence),
        ]
        o_window = adata.uns["unique_examples"]["oh"][
            seqlet.example_idx,
            :,
            max(seqlet.start - window, 0) : min(seqlet.start + length_pattern + window, length_sequence),
        ]
        co_window = (c_window * o_window).sum(0)  # We only care about the magnitude of the contribution.

        if seqlet.is_revcomp:
            co_window = co_window[::-1]

        # Finally do the actual padding
        co_window = np.pad(
            co_window,
            ((to_pad_up, to_pad_down),),
        )

        contribution_scores[i] = co_window

    # calculate profile and detect peaks
    profile = zscore(contribution_scores, axis=1).mean(0)  # type: ignore
    pattern_location = (window, window + length_pattern)

    peaks, _ = find_peaks(x=profile, height=height, **kwargs)

    # discard the peak in contribution formed by the pattern itself
    peaks = peaks[
        ~np.logical_and(
            peaks >= pattern_location[0] - min_distance_peak, peaks <= pattern_location[1] + min_distance_peak
        )
    ]

    if len(peaks) > 0:
        widths = peak_widths(profile, peaks, rel_height=rel_height_peak_width)
        peak_windows = np.array([widths[2], widths[3]], dtype=int).T
    else:
        peak_windows = np.array([]).reshape(0, 2)

    return BiasDetectionResult(
        profile=profile,
        pattern_location=pattern_location,
        peak_windows=peak_windows,
        contribution_scores=contribution_scores,
        pattern=pattern,
    )


def extend_biased_seqlets(
    adata: AnnData,
    bias_results: list[BiasDetectionResult],
    threshold: float | None = None,
    extra_flanks: int = 7,
    overlap_fraction_new: float = 0.7,
    overlap_fraction_old: float = 0.7,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Extend seqlets that show distance bias to capture nearby binding sites.

    This function takes bias detection results and extends seqlets that show evidence
    of nearby binding sites. It handles overlap detection and removal to avoid duplicate
    seqlets in the output. It only returns seqlets that were modified.

    Parameters
    ----------
    adata
        Seqlet AnnData object with seqlet information.
    bias_results
        List of BiasDetectionResult objects from tfmindi.tl.detect_distance_bias().
    threshold
        Z-score threshold to determine whether a seqlet has a neighboring binding site (default: None).
        When None no thresholding is performed.
    extra_flanks
        Extra basepairs to add beyond the detected peak locations (default: 7).
    overlap_fraction_new
        Fraction of overlap relative to new seqlet needed to call overlap (default: 0.7).
    overlap_fraction_old
        Fraction of overlap relative to old seqlet needed to call overlap (default: 0.7).

    Returns
    -------
    Tuple of (seqlets_dataframe, seqlet_matrices), for seqlets that were modified, where:
        - seqlets_dataframe: DataFrame with columns [example_idx, start, end]
        - seqlet_matrices: List of numpy arrays containing contribution scores for each seqlet

    Examples
    --------
    >>> import tfmindi as tm
    >>> patterns = tm.tl.create_patterns(adata)
    >>> results = [tm.tl.detect_distance_bias(adata, p) for p in patterns.values()]
    >>> results_with_bias = [r for r in results if r.has_bias]
    >>> new_seqlets_df, new_matrices = tm.tl.extend_biased_seqlets(
    >>>     adata, results_with_bias, threshold=0.5, extra_flanks=10
    >>> )
    """
    assert isinstance(adata.obs, pd.DataFrame), "adata.obs should return a pandas dataframe"
    # Mark overlapping seqlet in place
    adata.obs["to_remove"] = False
    # store index of this column so we can later use iloc
    c_idx_to_remove = adata.obs.columns.get_loc("to_remove")
    if not isinstance(c_idx_to_remove, int):
        raise ValueError("column `to_remove` occurs multiple times in adata.obs!")

    # same for example_oh_idx
    c_idx_example_oh_idx = adata.obs.columns.get_loc("example_oh_idx")
    if not isinstance(c_idx_example_oh_idx, int):
        raise ValueError("column `example_oh_idx` occurs multiple times in adata.obs!")

    oh_sequences = adata.uns["unique_examples"]["oh"]
    contrib_scores = adata.uns["unique_examples"]["contrib"]

    # List to keep extra seqlets (example_idx, start, end, to_remove)
    extra_seqlets: list[tuple[int, int, int, bool]] = []

    # cache ncls
    ex_idx_to_ncls: dict[Any, tuple[ncls.NCLS64, pd.DataFrame]] = {}

    for i, result in enumerate(bias_results):
        if not result.has_bias:
            continue

        if threshold is None:
            threshold = min([x.min() for x in result.max_contrib_peak_windows])  # type: ignore
        # Mark seqlets with bias to remove from the old seqlet dataframe
        # these seqlets will be extended and added to the extra seqlet list
        biased_indices = result.get_biased_seqlet_indices(threshold=threshold)  # type: ignore

        adata.obs.iloc[biased_indices, c_idx_to_remove] = True

        extension_up, extension_down = result.extension_distances

        for seqlet in tqdm(result.get_biased_seqlets(threshold=threshold), desc=f"{i + 1}/{len(bias_results)}"):  # type: ignore
            if not seqlet.is_revcomp:
                new_start = seqlet.start - extension_up - extra_flanks
                new_end = seqlet.end + extension_down + extra_flanks
            else:
                # In case of reverse complement, peak is detected in reverse direction
                # (i.e. subtract distance to upstream peak from start)
                new_start = seqlet.start - extension_down - extra_flanks
                new_end = seqlet.end + extension_up + extra_flanks

            # Cap new start and end between 0 and the length of the sequences
            new_start = max(new_start, 0)
            new_end = min(new_end, oh_sequences.shape[2])

            ex_idx = adata.obs.iloc[seqlet.seqlet_idx, c_idx_example_oh_idx]
            # Generate nested containment list (ncls) for quick overlap calculations
            if ex_idx in ex_idx_to_ncls:
                ex_ncls, tmp = ex_idx_to_ncls[ex_idx]
            else:
                tmp = adata.obs.query("example_oh_idx == @ex_idx").sort_values(["start", "end"])
                ex_ncls = ncls.NCLS(
                    starts=tmp["start"].astype(int).values, ends=tmp["end"].astype(int).values, ids=np.arange(len(tmp))
                )

                ex_idx_to_ncls[ex_idx] = (ex_ncls, tmp)

            for o_start, o_end, o_idx in ex_ncls.find_overlap(new_start, new_end):
                n_overlap = _calc_overlap((new_start, new_end), (o_start, o_end))
                # Fraction of overlap relative to the new interval
                frac_new = n_overlap / (new_end - new_start)
                # Fraction of overlap relative to an old interval that overlaps with the new one
                frac_old = n_overlap / (o_end - o_start)

                if frac_new >= overlap_fraction_new or frac_old >= overlap_fraction_old:
                    adata.obs.loc[tmp.index[o_idx], "to_remove"] = True

            extra_seqlets.append((ex_idx, new_start, new_end, False))  # type: ignore

    # WATCH OUT THE "example_idx" in this dataframe is NOT the same example_idx
    extra_seqlets_df = pd.DataFrame(extra_seqlets, columns=["example_idx", "start", "end", "to_remove"])

    extra_seqlet_matrices = _extract_seqlet_matrices(
        seqlets_df=extra_seqlets_df, contrib=contrib_scores, oh=oh_sequences
    )

    return extra_seqlets_df, extra_seqlet_matrices
