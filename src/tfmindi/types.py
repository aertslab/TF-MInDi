"""Custom data types for TF-MInDi package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Seqlet:
    """A seqlet object representing an aligned sequence instance.

    Attributes
    ----------
    seq_instance
        Aligned sequence instance (length x 4) one-hot encoded
    start
        Start position in the original sequence
    end
        End position in the original sequence
    region_one_hot
        Full one-hot encoded sequence this seqlet comes from (4 x seq_length)
    is_revcomp
        Whether this seqlet is reverse complemented
    contrib_scores
        Actual contribution scores masked by sequence content (length x 4).
        Non-zero only where nucleotides are present (seq_instance * raw_contributions)
    hypothetical_contrib_scores
        Raw contribution scores showing potential importance at each position (length x 4).
        Values for all nucleotides regardless of what's actually present
    """

    seq_instance: np.ndarray
    start: int
    end: int
    region_one_hot: np.ndarray
    is_revcomp: bool
    contrib_scores: np.ndarray | None = None
    hypothetical_contrib_scores: np.ndarray | None = None

    def __repr__(self):
        length = self.end - self.start
        strand = "(-)" if self.is_revcomp else "(+)"

        # Get consensus sequence
        consensus = self._get_consensus_sequence()

        # Show contrib info if available
        contrib_info = ""
        if self.contrib_scores is not None:
            mean_contrib = self.contrib_scores.sum() / length
            contrib_info = f", contrib={mean_contrib:.3f}"

        return f"Seqlet({self.start}-{self.end}{strand}, len={length}, seq='{consensus}'{contrib_info})"

    def _get_consensus_sequence(self) -> str:
        """Get consensus sequence string from one-hot encoding."""
        nucleotides = ["A", "C", "G", "T"]
        consensus = ""
        for pos in range(self.seq_instance.shape[0]):
            max_idx = self.seq_instance[pos].argmax()
            if self.seq_instance[pos, max_idx] > 0:
                consensus += nucleotides[max_idx]
            else:
                consensus += "N"
        return consensus


@dataclass
class Pattern:
    """A pattern object representing aligned seqlets from a cluster.

    Attributes
    ----------
    ppm
        Position probability matrix (length x 4) representing the consensus sequence
    contrib_scores
        Mean contribution scores (length x 4) for the pattern
    hypothetical_contrib_scores
        Mean hypothetical contribution scores (length x 4)
    seqlets
        List of aligned Seqlet objects in this pattern
    cluster_id
        The cluster ID this pattern represents
    n_seqlets
        Number of seqlets in this pattern
    dbd
        DNA-binding domain annotation for this pattern (optional)
    """

    ppm: np.ndarray
    contrib_scores: np.ndarray
    hypothetical_contrib_scores: np.ndarray
    seqlets: list[Seqlet]
    cluster_id: str
    n_seqlets: int
    dbd: str | None = None

    def ic(self, bg: np.ndarray = np.array([0.27, 0.23, 0.23, 0.27]), eps: float = 1e-3) -> np.ndarray:
        """Calculate information content for each position.

        Parameters
        ----------
        bg
            Background nucleotide frequencies [A, C, G, T]
        eps
            Small epsilon to avoid log(0)

        Returns
        -------
        Information content per position
        """
        return (self.ppm * np.log(self.ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)).sum(1)

    def ic_trim(self, min_v: float, **kwargs) -> tuple[int, int]:
        """Find trim indices based on information content threshold.

        Parameters
        ----------
        min_v
            Minimum information content threshold
        **kwargs
            Additional arguments passed to ic() method

        Returns
        -------
        Tuple of (start_index, end_index) for trimming
        """
        delta = np.where(np.diff((self.ic(**kwargs) > min_v) * 1))[0]
        if len(delta) == 0:
            return 0, 0
        start_index = min(delta)
        end_index = max(delta)
        return start_index, end_index + 1

    def __repr__(self):
        length = self.ppm.shape[0]

        consensus = self._get_consensus_sequence()

        mean_ic = self.ic().mean()

        if length > 20:
            display_consensus = consensus[:20] + "..."
        else:
            display_consensus = consensus

        dbd_str = f", dbd={self.dbd}" if self.dbd else ""
        return f"Pattern(cluster={self.cluster_id}, n_seqlets={self.n_seqlets}, len={length}, consensus='{display_consensus}', mean_ic={mean_ic:.2f}{dbd_str})"

    def _get_consensus_sequence(self) -> str:
        """Get consensus sequence string from PPM."""
        nucleotides = ["A", "C", "G", "T"]
        consensus = ""
        for pos in range(self.ppm.shape[0]):
            max_idx = self.ppm[pos].argmax()
            consensus += nucleotides[max_idx]
        return consensus
