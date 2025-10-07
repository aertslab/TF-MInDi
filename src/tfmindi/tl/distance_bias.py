"""Distance dependent bias detector functionality."""

from dataclasses import dataclass

import ncls  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from anndata import AnnData  # type: ignore
from scipy.signal import find_peaks, peak_widths  # type: ignore
from scipy.stats import zscore  # type: ignore

from tfmindi.pp.seqlets import _extract_seqlet_matrices
from tfmindi.types import Pattern, Seqlet


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


@dataclass
class FixedDistanceBiasDetector:
    """A FixedDistanceBiasDetector object for detecting TFBS instances at a fixed distance from a given pattern.

    Parameters
    ----------
    contribution_scores
        Numpy array with aligned contribution scores of a Pattern.
    pattern_location
        Location of the pattern istelf in the contribution score array.
    pattern
        Pattern for which to detect distance bias
    """

    contribution_scores: np.ndarray
    pattern_location: tuple[int, int]
    pattern: Pattern
    _peak_windows: np.ndarray | None = None

    @property
    def has_bias(self) -> bool:
        """Does this pattern have fixed distance bias."""
        if self._peak_windows is None:
            raise RuntimeError("`_peak_windows` is not set! Please run detect_distance_bias first.")
        return self._peak_windows.shape[0] > 0

    @property
    def profile(self) -> np.ndarray:
        """Get contribution profile as average zscore per position."""
        return zscore(self.contribution_scores, axis=1).mean(0)

    @property
    def profile_plot_data(self) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
        """Get data to generate a profile plot.

        Returns
        -------
        A tuple containing
        - np.ndarray: the contribution score profile (z-score)
        - tuple[int, int]: pattern location
        - np.naddary of shape (n, 2): start and end locations of n identified peaks

        Example
        -------
        >>> import tfmindi as tm
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> patterns = tm.tl.create_patterns(adata)
        >>> bias_detector = tm.tl.distance_bias.detect_fixed_distance_bias(
            adata,
            patterns["0"]
        )
        >>> bias_detector.detect_distance_bias()
        >>> profile, pattern_location, peaks = bias_detector.profile_plot_data
        >>> fig, ax = plt.subplots()
        >>> ax.plot(
            np.arange(profile.shape[0]),
            profile,
            color="black"
        )
        >>> for start_end in pattern_location:
            # plot location of the pattern
            ax.axvline(start_end, color = "red")
        >>> for peak_left, peak_right in peaks:
            # plot location of identified peaks
            ax.axvline(peak_left, color = "orange")
            ax.axvline(peak_right, color = "orange")
        >>> fig.show()
        """
        if self._peak_windows is None:
            raise RuntimeError("`_peak_windows` is not set! Please run detect_distance_bias first.")

        return self.profile, self.pattern_location, self._peak_windows

    @property
    def heatmap_plot_data(self) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
        """Get data to generate a heatmap plot.

        Returns
        -------
        A tuple containing
        - np.ndarray: the contribution scores sorted by the position of the maximal signal
        - tuple[int, int]: pattern location
        - np.naddary of shape (n, 2): start and end locations of n identified peaks

        Example
        -------
        >>> import tfmindi as tm
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import seaborn as sns
        >>> patterns = tm.tl.create_patterns(adata)
        >>> bias_detector = tm.tl.distance_bias.detect_fixed_distance_bias(
            adata,
            patterns["0"]
        )
        >>> bias_detector.detect_distance_bias()
        >>> contribution, pattern_location, peaks = bias_detector.heatmap_plot_data
        >>> fig, ax = plt.subplots(figsize=(4, 10))
        >>> sns.heatmap(
            contribution,
            cmap="gray_r",
            robust=True,
            ax=ax,
            yticklabels=False
        )
        >>> for start_end in pattern_location:
            # plot location of the pattern
            ax.axvline(start_end, color = "red")
        >>> for peak_left, peak_right in peaks:
            # plot location of identified peaks
            ax.axvline(peak_left, color = "orange")
            ax.axvline(peak_right, color = "orange")
        >>> fig.show()
        """
        if self._peak_windows is None:
            raise RuntimeError("`_peak_windows` is not set! Please run detect_distance_bias first.")
        z_contribution_score = zscore(self.contribution_scores, axis=1)
        s_idx = np.argsort(np.argmax(z_contribution_score, 1))

        return z_contribution_score[s_idx], self.pattern_location, self._peak_windows

    @property
    def up_downstream_window(self) -> tuple[int, int]:
        """Get number of basepairs up and downstream of the pattern to exend in order to capture nearby motifs."""
        if self._peak_windows is None:
            raise RuntimeError("`_peak_windows` is not set! Please run detect_distance_bias first.")
        # If there is a contribution score peak upstream of the pattern, use that location, otherwise use 0.
        distance_upstream = max(self.pattern_location[0] - self._peak_windows.min(0)[0], 0)
        # If there is a contribution score peak downstream of the pattern, use that location, otherwise use 0.
        distance_downstream = max(self._peak_windows.max(0)[1] - self.pattern_location[1], 0)
        return distance_upstream, distance_downstream

    def detect_distance_bias(
        self, height: float = 0.25, min_distance_peak: int = 3, rel_height_peak_width: float = 0.5, **kwargs
    ):
        """Detect distance bias by finding peaks in the contribution score profile.

        Parameters
        ----------
        height
            Required height of the peak.
        kwargs
            Extra keyword arguments passed to `scipy.signa.find_peaks`.
        min_distance_peak
            Minimal distance of a peak relative to the pattern instance.
        rel_height_peak_width
            Relative height to calculate the peak width at.
        """
        peaks, _ = find_peaks(x=self.profile, height=height, **kwargs)

        # discard the peak in contribution formed by the pattern itself.
        peaks = peaks[~np.logical_and(peaks >= self.pattern_location[0] - 3, peaks <= self.pattern_location[1] + 3)]
        if len(peaks) > 0:
            widths = peak_widths(self.profile, peaks, rel_height=rel_height_peak_width)
            self._peak_windows = np.array([widths[2], widths[3]], dtype=int).T
        else:
            self._peak_windows = np.array([])

    def get_seqlets_idc_with_distance_bias(self, threshold: float) -> list[int]:
        """Get indeces of seqlets with distance bias.

        Parameters
        ----------
        threshold
            Threshold on the maximum contribution z-score within a peak to call seqlets with distance bias.

        Returns
        -------
        List with seqlets indices with bias.
        """
        seqlets_w_bias = self.get_seqlets_with_distance_bias(threshold)
        return [seqlet.seqlet_idx for seqlet in seqlets_w_bias]

    def get_seqlets_with_distance_bias(self, threshold: float) -> list[Seqlet]:
        """Get indeces of seqlets with distance bias.

        Parameters
        ----------
        threshold
            Threshold on the maximum contribution z-score within a peak to call seqlets with distance bias.

        Returns
        -------
        List with seqlets with bias.
        """
        if self._peak_windows is None:
            raise RuntimeError("`_peak_windows` is not set! Please run detect_distance_bias first.")
        z_contribution_score = zscore(self.contribution_scores, axis=1)
        has_distance_bias = np.logical_or.reduce(
            np.array([z_contribution_score[:, start:end].max(1) for start, end in self._peak_windows]) > threshold,
            axis=0,
        )
        seqlets = []
        for seqlet, has_bias in zip(self.pattern.seqlets, has_distance_bias, strict=False):
            if has_bias:
                seqlets.append(seqlet)
        return seqlets


def detect_fixed_distance_bias(adata: AnnData, pattern: Pattern, window: int) -> FixedDistanceBiasDetector:
    """
    Detect whether other TFBS instances occur at a fixed distance from the given pattern.

    Parameters
    ----------
    adata
        AnnData object with stored seqlet data.
        Must contain
        - adata.uns["unique_examples"]["oh"]: Unique example one-hot sequences
        - adata.uns["unique_examples"]["contrib"]: Unique example contribution scores
    pattern
        A `Pattern` to detect distance bias for.
    window
        Integer specifying the number of basepairs to look up- and downstream of the pattern.

    Returns
    -------
    A `FixedDistanceBiasDetector` object.
    """
    n_seqlets = len(pattern.seqlets)
    length_pattern = pattern.ppm.shape[0]
    length_sequence = adata.uns["unique_examples"]["oh"].shape[2]

    # init array to store contribution scores of each seqlet of the pattern
    # and flanking basepairs defined by window
    contribution_scores = np.zeros((n_seqlets, length_pattern + 2 * window))

    seqlet_idc: np.ndarray = np.zeros(n_seqlets, dtype=int)

    for i, seqlet in enumerate(pattern.seqlets):
        # if the window arround the seqlet runs of the sequence start
        # pad with this many zeros upstream
        to_pad_up = abs(min(seqlet.start - window, 0))

        # if the window arround the seqlet runs of the sequence end
        # pad with this many zeros downstream
        to_pad_down = max(seqlet.start + length_pattern + window, length_sequence) - length_sequence

        # get the contribution of the window
        c_window = adata.uns["unique_examples"]["contrib"][
            seqlet.example_idx,
            :,
            max(seqlet.start - window, 0) : min(seqlet.start + length_pattern + window, length_sequence),
        ]
        # get the onehot of the window
        o_window = adata.uns["unique_examples"]["oh"][
            seqlet.example_idx,
            :,
            max(seqlet.start - window, 0) : min(seqlet.start + length_pattern + window, length_sequence),
        ]
        co_window = (c_window * o_window).sum(0)  # We only care about the magnitude of the contribution.

        if seqlet.is_revcomp:
            co_window = co_window[::-1]

        # finally do the actual padding
        co_window = np.pad(
            co_window,
            ((to_pad_up, to_pad_down),),
        )

        contribution_scores[i] = co_window
        seqlet_idc[i] = seqlet.seqlet_idx
    return FixedDistanceBiasDetector(
        contribution_scores=contribution_scores, pattern_location=(window, window + length_pattern), pattern=pattern
    )


def create_seqlet_matrices_with_distance_bias(
    adata: AnnData,
    fixed_distance_bias_detectors: list[FixedDistanceBiasDetector],
    threshold: float = 1.0,
    extra_flanks_to_add: int = 7,
    f: float = 0.7,
    F: float = 0.7,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Create a new seqlet AnnData object where seqlets with fixed distance bias are extended (capturing more dimers etc.).

    Parameters
    ----------
    adata
        Seqlet `AnnData` object.
    fixed_distance_bias_detectors
        A list of `FixedDistanceBiasDetector` objects for the patterns of interest.
    threshold
        Threshold on z-score of contribution score to determine whether the given seqlet has a neighbouring binding site.
    extra_flanks_to_add
        Extra basepairs to add to the extended pattern location.
    f
        fraction of overlap relative to new seqlet needed to call overlap.
    F
        fraction of overlap relative to old seqlet needed to call overlap.

    Returns
    -------
    dataframe of new seqlets and list of new seqlet matrices
    """
    new_seqlets_df: pd.DataFrame = adata.obs[["example_oh_idx", "start", "end"]].copy()
    new_seqlets_df.index = new_seqlets_df.index.astype(int)
    # example_oh_idx will become the new example_idx
    # below we will use the sequences and contribution scores in adata.uns["unique_examples"]
    new_seqlets_df.rename({"example_oh_idx": "example_idx"}, axis=1, inplace=True)

    # to_remove is column 4 (index 3 below when using .iloc)
    new_seqlets_df["to_remove"] = False

    oh_sequences = adata.uns["unique_examples"]["oh"]
    contrib_scores = adata.uns["unique_examples"]["contrib"]

    # list to keep extra seqlets (example_idx, start, end, to_remove, is_revcomp)
    extra_seqlets: list[tuple[int, int, int, bool]] = []
    for bias_detector in fixed_distance_bias_detectors:
        if not bias_detector.has_bias:
            continue
        # mark seqlets with bias to remove from the old seqlet dataframe
        # these seqlets will be extended and added to the extra seqlet
        new_seqlets_df.iloc[bias_detector.get_seqlets_idc_with_distance_bias(threshold=threshold), 3] = True

        for seqlet in bias_detector.get_seqlets_with_distance_bias(threshold=threshold):
            if not seqlet.is_revcomp:
                new_start = seqlet.start - bias_detector.up_downstream_window[0] - extra_flanks_to_add
                new_end = seqlet.end + bias_detector.up_downstream_window[1] + extra_flanks_to_add
            else:
                # in case of reverse complement, peak is detected in reverse direction (i.e. substract distance to upstream peak from start)
                new_start = seqlet.start - bias_detector.up_downstream_window[1] - extra_flanks_to_add
                new_end = seqlet.end + bias_detector.up_downstream_window[0] + extra_flanks_to_add

            # Cap new start and end between 0 and the length of the sequences.
            new_start = max(new_start, 0)
            new_end = min(new_end, oh_sequences.shape[2])

            # mark old seqlets that overlap with this new seqlets with at least
            # a fraction of f and F to be removed.
            ex_idx = new_seqlets_df.iloc[seqlet.seqlet_idx, 0]
            tmp = new_seqlets_df.query("example_idx == @ex_idx").sort_values(["start", "end"])

            # generate nested containment list (ncls) for quick overlap calculations.
            ex_ncls = ncls.NCLS(
                starts=tmp["start"].astype(int).values, ends=tmp["end"].astype(int).values, ids=tmp.index.values
            )

            for o_start, o_end, o_idx in ex_ncls.find_overlap(new_start, new_end):
                n_overlap = _calc_overlap((new_start, new_end), (o_start, o_end))
                # Fraction of overlap relative to the new interval
                frac_f = n_overlap / (new_end - new_start)
                # Fraction of overlap relative to an old interval that overlaps with the new one.
                frac_F = n_overlap / (o_end - o_start)

                if frac_f >= f or frac_F >= F:
                    new_seqlets_df.loc[o_idx, "to_remove"] = True

            extra_seqlets.append((ex_idx, new_start, new_end, False))

    # Generate new seqlets dataframe.
    new_seqlets_df = pd.concat([new_seqlets_df, pd.DataFrame(extra_seqlets, columns=new_seqlets_df.columns)])
    # Remove overlapping seqlets (marked as `to_remove`) and delete `to_remove` column
    new_seqlets_df = (
        new_seqlets_df.loc[~new_seqlets_df["to_remove"].astype(bool), ["example_idx", "start", "end"]]
        .astype(int)
        .reset_index(drop=True)
    )

    seqlet_matrices = _extract_seqlet_matrices(seqlets_df=new_seqlets_df, contrib=contrib_scores, oh=oh_sequences)

    return new_seqlets_df, seqlet_matrices
