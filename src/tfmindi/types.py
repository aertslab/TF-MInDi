"""Custom data types for TF-MInDi package."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

import numpy as np

_BASE_TO_BIN = {"A": 0, "C": 1, "G": 2, "T": 3}
_BIN_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


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
    example_idx: int
    seqlet_idx: int
    contrib_scores: np.ndarray | None = None
    hypothetical_contrib_scores: np.ndarray | None = None

    def __repr__(self):
        """Print the Seqlet object."""
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

    def get_kmers(self) -> list[Kmer]:
        """Get list of aligned kmers."""
        kmers: list[Kmer] = []
        for seqlet in self.seqlets:
            kmers.append(Kmer.from_str(seqlet._get_consensus_sequence()))
        return kmers

    def get_unique_kmers(self, return_counts: bool = False) -> list[Kmer] | tuple[list[Kmer], list[int]]:
        """Get list of unique aligned kmers, sorted by occurence (most common first).

        Parameters
        ----------
        retun_counts
            Return the count of each kmer

        Returns
        -------
        list of unique kmers ( sorted by occurence with most common first) and count if `return_counts` is `True`
        """
        kmers = self.get_kmers()
        k_s = {kmer.k for kmer in kmers}
        if len(k_s) > 1:
            raise ValueError("Kmers of different sizes detected. Can only compare Kmers of the same size")
        kmer_count: dict[Kmer, int] = {}
        for kmer in kmers:
            if kmer not in kmer_count:
                kmer_count[kmer] = 0
            kmer_count[kmer] += 1

        sorted_kmers = sorted(kmer_count.items(), key=lambda item: kmer_count[item[0]], reverse=True)

        unique_kmers: list[Kmer] = []
        unique_kmers_counts: list[int] = []

        for kmer, count in sorted_kmers:
            unique_kmers.append(kmer)
            unique_kmers_counts.append(count)

        if return_counts:
            return unique_kmers, unique_kmers_counts

        else:
            return unique_kmers

    def get_kmer_distances(self) -> list[tuple[int, int]]:
        """For each seqlet get the Hamming distance to the most common kmer.

        Returns
        -------
        A list of kmer indexes and Hamming distances.
        """
        most_common_kmer = self.get_unique_kmers()[0]
        assert isinstance(most_common_kmer, Kmer)

        kmer_idx_and_dist: list[tuple[int, int]] = []
        for kmer, seqlet in zip(self.get_kmers(), self.seqlets, strict=True):
            # for each kmer calculate the distance to the most common kmer,
            # for both forward and reverse complement,
            # and take the minimum
            kmer_idx_and_dist.append((seqlet.seqlet_idx, min(most_common_kmer - kmer, most_common_kmer - ~kmer)))
        return kmer_idx_and_dist

    def __repr__(self):
        """Print the Pattern object."""
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


@dataclass
class Kmer:
    """
    Class representing a kmer using 2bit DNA notation.

    A = 00
    C = 01
    G = 10
    T = 11

    This Kmer class is inpired on seqlang kmer object.
    """

    value: int
    k: int

    @classmethod
    def from_str(cls, s: str) -> Kmer:
        """Create kmer form string."""
        s = s.upper()
        if len(set(s) - {"A", "C", "G", "T"}) > 0:
            raise ValueError("input must be nucleotide sequence.")

        k = len(s)
        b_kmer = 0
        for i in range(k):
            b_kmer = (b_kmer << 2) | _BASE_TO_BIN[s[i]]
        return cls(value=b_kmer, k=k)

    def __str__(self) -> str:
        """Return str representation of kmer."""
        sequence = []
        for i in range(self.k - 1, -1, -1):
            base = (self.value >> (2 * i)) & 0b11
            sequence.append(_BIN_TO_BASE[base])
        return "".join(sequence)

    def __repr__(self) -> str:
        """Return string representation of kmer for debugging."""
        return f"Kmer({str(self)})"

    def __invert__(self) -> Kmer:
        """Generate reverse complement of kmer."""
        mask = (1 << (2 * self.k)) - 1
        complement = self.value ^ mask

        reverse_complement = 0
        for i in range(self.k):
            base = (complement >> (2 * i)) & 0b11
            reverse_complement |= base << (2 * (self.k - 1 - i))

        return Kmer(reverse_complement, self.k)

    def __sub__(self, other: Kmer) -> int:
        """Calculate Hamming distance using bit tricks from seqlang."""
        mask1 = 0
        mask2 = 0
        for _ in range(self.k):
            mask1 = (mask1 << 2) | 0b01
            mask2 = (mask2 << 2) | 0b10

        lsb_diff = (self.value & mask1) ^ (other.value & mask1)
        msb_diff = (self.value & mask2) ^ (other.value & mask2)
        return ((lsb_diff << 1) | msb_diff).bit_count()

    def __hash__(self) -> int:
        """Returl value as hash (this should always be unique for a given k)."""
        return self.value


class Kmers:
    """Kmers generating factory for kmers of size k."""

    def __init__(
        self,
        k: int,
    ):
        """Initialize new kmer generating factory with length k."""
        self.k = k
        # generate binary mask
        # dna sequence will be 2bit encoded so 2bits per k
        self.kmer_mask = (1 << (2 * k)) - 1

    def __call__(self, sequence: str) -> Generator[Kmer]:
        """Generate kmers for sequence."""
        if len(sequence) < self.k:
            raise ValueError(f"Sequence {len(sequence)} is shorter than k ({self.k})")

        # Generate first kmer
        b_current_kmer = 0
        for i in range(self.k):
            b_current_kmer = (b_current_kmer << 2) | _BASE_TO_BIN[sequence[i]]
        yield Kmer(b_current_kmer, self.k)

        # generate remaining kmers
        for i in range(self.k, len(sequence)):
            b_current_kmer = ((b_current_kmer << 2) | _BASE_TO_BIN[sequence[i]]) & self.kmer_mask
            yield Kmer(b_current_kmer, self.k)
