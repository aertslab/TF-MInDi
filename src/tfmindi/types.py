"""Custom data types for TF-MInDi package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Pattern:
    """
    Represents a sequence pattern with associated statistics.

    Used to store aligned PWM patterns generated from seqlet clusters.
    """

    pwm: np.ndarray
    """Position weight matrix (4 x length) representing the pattern."""

    name: str
    """Name or identifier of the pattern."""

    cluster_id: str
    """Cluster ID this pattern was derived from."""

    n_seqlets: int
    """Number of seqlets contributing to this pattern."""

    mean_contrib: float
    """Mean contribution score of seqlets in this pattern."""

    consensus: str
    """Consensus sequence string."""

    ic_profile: np.ndarray
    """Information content profile for each position."""

    statistics: dict[str, float] | None = None
    """Additional statistics about the pattern."""

    def __post_init__(self):
        """Validate pattern dimensions."""
        if self.pwm.shape[0] != 4:
            raise ValueError("PWM must have 4 rows (A, C, G, T)")
        if len(self.ic_profile) != self.pwm.shape[1]:
            raise ValueError("IC profile length must match PWM width")
        if len(self.consensus) != self.pwm.shape[1]:
            raise ValueError("Consensus sequence length must match PWM width")
