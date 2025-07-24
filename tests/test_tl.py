"""Tests for analysis tools."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

import tfmindi as tm
from tfmindi.types import Seqlet


class TestClusterSeqlets:
    """Test cluster_seqlets function."""

    def test_cluster_seqlets_basic(self, sample_clustered_adata):
        """Test basic functionality of cluster_seqlets."""
        adata = sample_clustered_adata.copy()
        expected_obs_columns = ["leiden", "mean_contrib", "seqlet_dbd", "cluster_dbd"]
        for col in expected_obs_columns:
            assert col in adata.obs.columns, f"Missing column: {col}"

        expected_obsm_keys = ["X_pca", "X_tsne"]
        for key in expected_obsm_keys:
            assert key in adata.obsm.keys(), f"Missing obsm key: {key}"

        # Check data types and shapes
        assert adata.obs["leiden"].dtype == "category"
        assert adata.obs["mean_contrib"].dtype == float
        assert adata.obsm["X_pca"].shape[0] == adata.n_obs
        assert adata.obsm["X_tsne"].shape == (adata.n_obs, 2)

        n_clusters = adata.obs["leiden"].nunique()
        assert n_clusters > 1, "Should find multiple clusters"
        assert n_clusters < adata.n_obs, "Should have fewer clusters than seqlets"

        assert adata.obs["seqlet_dbd"].notna().sum() > 0, "Should have some DBD annotations"
        assert adata.obs["cluster_dbd"].notna().sum() > 0, "Should have some cluster DBD annotations"

    def test_cluster_seqlets_different_resolutions(self, sample_seqlet_adata):
        """Test that different resolutions produce different numbers of clusters."""
        # should give fewer clusters
        adata_low = sample_seqlet_adata.copy()
        tm.tl.cluster_seqlets(adata_low, resolution=0.5)
        n_clusters_low = adata_low.obs["leiden"].nunique()

        # should give more clusters
        adata_high = sample_seqlet_adata.copy()
        tm.tl.cluster_seqlets(adata_high, resolution=2.0)
        n_clusters_high = adata_high.obs["leiden"].nunique()

        assert n_clusters_high >= n_clusters_low

    def test_cluster_seqlets_output_structure(self, sample_clustered_adata):
        """Test the structure and content of cluster_seqlets output."""
        adata = sample_clustered_adata.copy()

        # Test mean_contrib calculation
        assert adata.obs["mean_contrib"].min() >= 0, "Mean contrib should be non-negative"

        # Test seqlet_dbd assignment (should match top motif for each seqlet)
        for i in range(min(5, adata.n_obs)):  # Check first 5 seqlets
            top_motif_idx = adata.X[i].argmax()
            top_motif_name = adata.var.index[top_motif_idx]
            expected_dbd = adata.var.loc[top_motif_name, "dbd"]
            actual_dbd = adata.obs.iloc[i]["seqlet_dbd"]
            assert actual_dbd == expected_dbd, f"DBD mismatch for seqlet {i}"

        # Test cluster_dbd consistency (all seqlets in same cluster should have same cluster_dbd)
        for cluster in adata.obs["leiden"].unique():
            cluster_mask = adata.obs["leiden"] == cluster
            cluster_dbds = adata.obs.loc[cluster_mask, "cluster_dbd"].unique()
            cluster_dbds_clean = cluster_dbds[pd.notna(cluster_dbds)]
            if len(cluster_dbds_clean) > 0:
                assert len(cluster_dbds_clean) == 1, f"Cluster {cluster} should have consistent DBD annotation"


class TestCreatePatterns:
    """Test create_patterns function."""

    def test_create_patterns_basic(self, sample_patterns):
        """Test basic functionality of create_patterns."""
        assert isinstance(sample_patterns, dict)
        assert len(sample_patterns) > 0, "Should create at least one pattern"

        for cluster_id, pattern in sample_patterns.items():
            assert isinstance(pattern, tm.tl.Pattern)
            assert isinstance(cluster_id, str)

            # Check pattern attributes
            assert pattern.ppm.ndim == 2, "PWM should be 2D"
            assert pattern.ppm.shape[1] == 4, "PWM should have 4 nucleotides"
            assert pattern.contrib_scores.shape == pattern.ppm.shape
            assert pattern.hypothetical_contrib_scores.shape == pattern.ppm.shape
            assert pattern.n_seqlets > 0, "Pattern should have seqlets"
            assert pattern.cluster_id == cluster_id

            # Check that seqlets have correct structure
            assert len(pattern.seqlets) == pattern.n_seqlets
            for seqlet in pattern.seqlets:
                assert isinstance(seqlet, Seqlet)
                assert seqlet.seq_instance.shape[1] == 4, "Seqlet should have 4 nucleotides"
                assert seqlet.start < seqlet.end, "Seqlet start should be before end"
                if seqlet.contrib_scores is not None:
                    assert seqlet.contrib_scores.shape == seqlet.seq_instance.shape

    def test_pattern_ic_calculation(self, sample_patterns):
        """Test information content calculation."""
        if len(sample_patterns) > 0:
            pattern = list(sample_patterns.values())[0]
            ic = pattern.ic()

            assert ic.shape[0] == pattern.ppm.shape[0], "IC should have one value per position"
            assert np.all(ic >= 0), "IC should be non-negative"
            assert np.all(ic <= 2), "IC should be at most 2 bits"

    def test_pattern_consensus_quality(self, sample_patterns):
        """Test that patterns represent reasonable consensus sequences."""
        for pattern in sample_patterns.values():
            # Check that PWM is properly normalized (sums to 1 at each position)
            position_sums = pattern.ppm.sum(axis=1)
            np.testing.assert_allclose(position_sums, 1.0, rtol=1e-6, err_msg="PWM positions should sum to 1")

            assert np.all(pattern.ppm >= 0), "PWM values should be non-negative"

            for seqlet in pattern.seqlets:
                assert np.all(seqlet.seq_instance >= 0), "Seqlet instances should be non-negative"
                assert np.all(seqlet.seq_instance <= 1), "Seqlet instances should be at most 1"

    def test_create_patterns_max_n_timing(self, sample_clustered_adata):
        """Test that subsampling with max_n speeds up pattern creation."""
        adata = sample_clustered_adata.copy()

        total_seqlets = adata.n_obs
        if total_seqlets < 100:
            pytest.skip("Test data too small for meaningful timing comparison")

        # Find the largest cluster to ensure we have enough seqlets for subsampling
        cluster_sizes = adata.obs["leiden"].value_counts()
        largest_cluster_size = cluster_sizes.max()
        max_n_small = max(10, largest_cluster_size // 3)
        if largest_cluster_size < 30:
            pytest.skip("Largest cluster too small for meaningful timing comparison")

        start_time = time.time()
        patterns_full = tm.tl.create_patterns(adata)
        time_full = time.time() - start_time

        start_time = time.time()
        patterns_subsampled = tm.tl.create_patterns(adata, max_n=max_n_small)
        time_subsampled = time.time() - start_time

        assert len(patterns_full) > 0, "Should create patterns without subsampling"
        assert len(patterns_subsampled) > 0, "Should create patterns with subsampling"

        # Verify we get the same number of patterns (same clusters)
        assert len(patterns_full) == len(patterns_subsampled), "Should have same number of patterns"

        # Verify that subsampling actually reduced the number of seqlets in large clusters
        for cluster_id in patterns_full.keys():
            full_n_seqlets = patterns_full[cluster_id].n_seqlets
            subsampled_n_seqlets = patterns_subsampled[cluster_id].n_seqlets

            if full_n_seqlets > max_n_small:
                assert subsampled_n_seqlets == max_n_small, (
                    f"Cluster {cluster_id} should be subsampled to {max_n_small}"
                )
            else:
                assert subsampled_n_seqlets == full_n_seqlets, f"Small cluster {cluster_id} should not be subsampled"

        # Subsampling should be faster
        print(f"Time without subsampling: {time_full:.3f}s")
        print(f"Time with subsampling (max_n={max_n_small}): {time_subsampled:.3f}s")
        print(f"Speedup: {time_full / time_subsampled:.2f}x")
        assert time_subsampled < time_full * 1.2, "Subsampling should be faster than full processing"

    def test_create_patterns_max_n_functionality(self, sample_clustered_adata):
        """Test that max_n parameter works correctly for subsampling."""
        adata = sample_clustered_adata.copy()

        # Test with different max_n values
        max_n_values = [5, 10, None]
        pattern_results = {}

        for max_n in max_n_values:
            patterns = tm.tl.create_patterns(adata, max_n=max_n)
            pattern_results[max_n] = patterns

            # Verify all patterns have correct number of seqlets
            for cluster_id, pattern in patterns.items():
                cluster_mask = adata.obs["leiden"] == cluster_id
                original_cluster_size = cluster_mask.sum()

                if max_n is None:
                    expected_n_seqlets = original_cluster_size
                else:
                    expected_n_seqlets = min(max_n, original_cluster_size)

                assert pattern.n_seqlets == expected_n_seqlets, (
                    f"Pattern {cluster_id} should have {expected_n_seqlets} seqlets"
                )

        # Verify patterns have same cluster IDs regardless of max_n
        for max_n in max_n_values:
            assert set(pattern_results[max_n].keys()) == set(pattern_results[None].keys()), (
                "Should have same cluster IDs"
            )
