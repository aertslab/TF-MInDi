"""Tests for tfmindi.types module."""

import numpy as np

from tfmindi.types import Pattern, Seqlet


def create_test_pattern(ppm: np.ndarray) -> Pattern:
    """Helper to create a minimal Pattern for testing."""
    seqlet = Seqlet(
        seq_instance=np.array([[1, 0, 0, 0]]),
        start=0,
        end=1,
        region_one_hot=np.zeros((4, 100)),
        is_revcomp=False,
        example_idx=0,
        seqlet_idx=0,
    )

    return Pattern(
        ppm=ppm,
        contrib_scores=np.zeros_like(ppm),
        hypothetical_contrib_scores=np.zeros_like(ppm),
        seqlets=[seqlet],
        cluster_id="0",
        n_seqlets=1,
    )


def test_ic_trim_simple_motif():
    """Test trimming a motif with low-IC flanks on both ends."""
    # low IC - high IC - low IC
    ppm = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],  # pos 0: uniform, low IC
            [0.9, 0.03, 0.03, 0.04],  # pos 1: high IC
            [0.05, 0.9, 0.03, 0.02],  # pos 2: high IC
            [0.25, 0.25, 0.25, 0.25],  # pos 3: uniform, low IC
        ]
    )

    pattern = create_test_pattern(ppm)
    start, end = pattern.ic_trim(min_v=0.5)

    # Should trim to positions 1-2 only (indices [1:3])
    assert start == 1 and end == 3, f"Expected (1, 3), got ({start}, {end})"

    # Verify all trimmed positions are above threshold
    ic_values = pattern.ic()
    assert np.all(ic_values[start:end] > 0.5), "All trimmed positions should be above threshold"


def test_ic_trim_dimer_motif():
    """Test trimming a dimer-like motif with high-low-high IC pattern."""
    # low - HIGH - low spacer - HIGH - low
    ppm = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],  # pos 0: low IC flank
            [0.9, 0.03, 0.03, 0.04],  # pos 1: high IC (site 1)
            [0.05, 0.9, 0.03, 0.02],  # pos 2: high IC (site 1)
            [0.25, 0.25, 0.25, 0.25],  # pos 3: spacer (low IC)
            [0.03, 0.02, 0.9, 0.05],  # pos 4: high IC (site 2)
            [0.9, 0.05, 0.03, 0.02],  # pos 5: high IC (site 2)
            [0.25, 0.25, 0.25, 0.25],  # pos 6: low IC flank
        ]
    )

    pattern = create_test_pattern(ppm)
    start, end = pattern.ic_trim(min_v=0.5)

    # Should trim outer flanks but keep everything including the spacer
    # Expected: positions 1-5 (indices [1:6])
    assert start == 1 and end == 6, f"Expected (1, 6), got ({start}, {end})"


def test_ic_trim_all_positions_above_threshold():
    """Test when all positions are above threshold (no trimming needed)."""
    # All high IC
    ppm = np.array(
        [
            [0.9, 0.03, 0.03, 0.04],
            [0.05, 0.9, 0.03, 0.02],
            [0.03, 0.02, 0.9, 0.05],
        ]
    )

    pattern = create_test_pattern(ppm)
    start, end = pattern.ic_trim(min_v=0.5)

    # Should include all positions
    assert start == 0 and end == 3, f"Expected (0, 3), got ({start}, {end})"


def test_ic_trim_no_positions_above_threshold():
    """Test when no positions are above threshold."""
    # All low IC
    ppm = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.26, 0.24, 0.25, 0.25],
            [0.25, 0.25, 0.26, 0.24],
        ]
    )

    pattern = create_test_pattern(ppm)
    start, end = pattern.ic_trim(min_v=0.5)

    # Should return (0, 0) indicating no valid region
    assert start == 0 and end == 0, f"Expected (0, 0), got ({start}, {end})"
