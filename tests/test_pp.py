"""Tests for preprocessing functions."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import tfmindi as tm


class TestExtractSeqlets:
    """Test extract_seqlets function."""

    def test_extract_seqlets_real_data(self, sample_contrib_data, sample_oh_data):
        """Test extract_seqlets with real data."""
        seqlet_df, seqlet_matrices = tm.pp.extract_seqlets(sample_contrib_data, sample_oh_data)

        assert len(seqlet_df) == len(seqlet_matrices) == 295

        assert isinstance(seqlet_df, pd.DataFrame)
        assert isinstance(seqlet_matrices, list)

        assert np.all(seqlet_df["start"] < seqlet_df["end"])
        assert np.all(seqlet_df["start"] >= 0)

        # check that all values in seqlet matrices are between -1 and 1
        for matrix in seqlet_matrices:
            assert np.all(matrix >= -1) and np.all(matrix <= 1)


class TestCalculateMotifSimilarity:
    """Test calculate_motif_similarity function."""

    def test_calculate_motif_similarity_real_data(self, sample_contrib_data, sample_oh_data, sample_motifs):
        """Test calculate_motif_similarity with real seqlets and motifs."""
        # Extract seqlets from real data (use subset for speed)
        contrib_subset = sample_contrib_data[:10]  # First 10 examples
        oh_subset = sample_oh_data[:10]

        seqlets_df, seqlet_matrices = tm.pp.extract_seqlets(contrib_subset, oh_subset, threshold=0.1)

        # Use first few seqlets and motifs for testing
        test_seqlets = seqlet_matrices[:5] if len(seqlet_matrices) >= 5 else seqlet_matrices
        test_motifs = list(sample_motifs.values())[:3]  # First 3 motifs

        # Skip test if no seqlets found
        if len(test_seqlets) == 0:
            pytest.skip("No seqlets found in test data")

        # seq that len of seqlets PWM is same as in df
        for i, seqlet in enumerate(seqlet_matrices):
            assert seqlet.shape[1] == seqlets_df.iloc[i]["end"] - seqlets_df.iloc[i]["start"]

        # Calculate similarity
        result = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs)

        # Basic output checks
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(test_seqlets), len(test_motifs))
        assert not np.isnan(result).any()
        assert np.all(result >= 0.05)  # Check minimum clipping
        assert np.all(np.isfinite(result))

    def test_calculate_motif_similarity_small_real_data(self, sample_motifs):
        """Test calculate_motif_similarity with small real motif data."""
        # Create simple test seqlets (normalized contribution patterns)
        seqlet1 = np.array([[0.8, 0.0, 0.0, 0.2], [0.0, 0.0, 0.9, 0.1], [0.1, 0.8, 0.0, 0.1], [0.0, 0.1, 0.1, 0.8]])

        seqlet2 = np.array([[0.0, 0.9, 0.1, 0.0], [0.8, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 1.0], [0.2, 0.1, 0.7, 0.0]])

        test_seqlets = [seqlet1, seqlet2]
        test_motifs = list(sample_motifs.values())[:2]  # First 2 motifs

        result = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs)

        # Check output properties
        assert result.shape == (2, 2)
        assert not np.isnan(result).any()
        assert np.all(result >= 0.05)  # Minimum clipping
        assert np.all(result > 0)  # All positive after log transform and clipping

    def test_calculate_motif_similarity_empty_inputs(self):
        """Test behavior with empty input lists."""
        with patch("tfmindi.pp.seqlets.tomtom") as mock_tomtom:
            # Empty array that won't cause issues with .max()
            empty_array = np.array([]).reshape(0, 0)
            mock_tomtom.return_value = (empty_array, None, None, None, None)

            result = tm.pp.calculate_motif_similarity([], [])

            assert result.shape == (0, 0)

    def test_extract_seqlets_with_real_data(self, sample_contrib_data, sample_oh_data):
        """Test extract_seqlets with real data from the sample dataset."""
        # Use a subset of the real data
        contrib = sample_contrib_data[:5]  # First 5 examples
        oh = sample_oh_data[:5]

        # This should not raise any errors with real data
        seqlets_df, seqlet_matrices = tm.pp.extract_seqlets(contrib, oh, threshold=0.1)

        # Basic checks
        assert isinstance(seqlets_df, pd.DataFrame)
        assert isinstance(seqlet_matrices, list)
        assert len(seqlet_matrices) == len(seqlets_df)

        # Check that all seqlet matrices have correct number of channels
        for matrix in seqlet_matrices:
            assert matrix.shape[0] == 4
