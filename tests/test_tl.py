"""Tests for analysis tools."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tfmindi as tm


class TestClusterSeqlets:
    """Test cluster_seqlets function."""

    def test_cluster_seqlets_basic(self, sample_seqlet_adata):
        """Test basic functionality of cluster_seqlets."""
        adata = sample_seqlet_adata.copy()

        # Run clustering
        tm.tl.cluster_seqlets(adata, resolution=1.0)

        # Check that required columns were added to .obs
        expected_obs_columns = ["leiden", "mean_contrib", "seqlet_dbd", "cluster_dbd"]
        for col in expected_obs_columns:
            assert col in adata.obs.columns, f"Missing column: {col}"

        # Check that required matrices were added to .obsm
        expected_obsm_keys = ["X_pca", "X_tsne"]
        for key in expected_obsm_keys:
            assert key in adata.obsm.keys(), f"Missing obsm key: {key}"

        # Check data types and shapes
        assert adata.obs["leiden"].dtype == "category"
        assert adata.obs["mean_contrib"].dtype == float
        assert adata.obsm["X_pca"].shape[0] == adata.n_obs
        assert adata.obsm["X_tsne"].shape == (adata.n_obs, 2)

        # Check that we found some clusters
        n_clusters = adata.obs["leiden"].nunique()
        assert n_clusters > 1, "Should find multiple clusters"
        assert n_clusters < adata.n_obs, "Should have fewer clusters than seqlets"

        # Check that DBD annotations were assigned
        assert adata.obs["seqlet_dbd"].notna().sum() > 0, "Should have some DBD annotations"
        assert adata.obs["cluster_dbd"].notna().sum() > 0, "Should have some cluster DBD annotations"

    def test_cluster_seqlets_different_resolutions(self, sample_seqlet_adata):
        """Test that different resolutions produce different numbers of clusters."""
        # Test low resolution (should give fewer clusters)
        adata_low = sample_seqlet_adata.copy()
        tm.tl.cluster_seqlets(adata_low, resolution=0.5)
        n_clusters_low = adata_low.obs["leiden"].nunique()

        # Test high resolution (should give more clusters)
        adata_high = sample_seqlet_adata.copy()
        tm.tl.cluster_seqlets(adata_high, resolution=2.0)
        n_clusters_high = adata_high.obs["leiden"].nunique()

        # Higher resolution should generally give more clusters (or at least not fewer)
        assert n_clusters_high >= n_clusters_low

    def test_cluster_seqlets_without_seqlet_matrices(self, sample_seqlet_adata):
        """Test cluster_seqlets when seqlet matrices are not provided."""
        adata = sample_seqlet_adata.copy()

        # Remove seqlet matrices to test fallback behavior
        del adata.obs["seqlet_matrix"]

        # Should still work but with NaN mean_contrib
        tm.tl.cluster_seqlets(adata, resolution=1.0)

        assert "leiden" in adata.obs.columns
        assert "mean_contrib" in adata.obs.columns
        assert adata.obs["mean_contrib"].isna().all()

    def test_cluster_seqlets_without_dbd_annotations(self, sample_seqlet_adata):
        """Test cluster_seqlets when DBD annotations are not provided."""
        adata = sample_seqlet_adata.copy()

        # Remove DBD annotations to test fallback behavior
        del adata.var["dbd"]

        # Should still work but with NaN DBD annotations
        tm.tl.cluster_seqlets(adata, resolution=1.0)

        assert "leiden" in adata.obs.columns
        assert "seqlet_dbd" in adata.obs.columns
        assert "cluster_dbd" in adata.obs.columns
        assert adata.obs["seqlet_dbd"].isna().all()
        assert adata.obs["cluster_dbd"].isna().all()

    def test_cluster_seqlets_output_structure(self, sample_seqlet_adata):
        """Test the structure and content of cluster_seqlets output."""
        adata = sample_seqlet_adata.copy()

        tm.tl.cluster_seqlets(adata, resolution=1.0)

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

    def test_create_patterns_basic(self, sample_seqlet_adata):
        """Test basic functionality of create_patterns."""
        adata = sample_seqlet_adata.copy()

        # First run clustering to get clusters
        tm.tl.cluster_seqlets(adata, resolution=1.0)

        # Create patterns
        patterns = tm.tl.create_patterns(adata)

        # Check that we got patterns
        assert isinstance(patterns, dict)
        assert len(patterns) > 0, "Should create at least one pattern"

        # Check pattern structure
        for cluster_id, pattern in patterns.items():
            assert isinstance(pattern, tm.tl.Pattern)
            assert isinstance(cluster_id, str)

            # Check pattern attributes
            assert pattern.ppm.ndim == 2, "PWM should be 2D"
            assert pattern.ppm.shape[1] == 4, "PWM should have 4 nucleotides"
            assert pattern.contrib_scores.shape == pattern.ppm.shape
            assert pattern.hypothetical_contrib_scores.shape == pattern.ppm.shape
            assert pattern.n_seqlets > 0, "Pattern should have seqlets"
            assert pattern.cluster_id == cluster_id

            # Check that seqlet instances have correct structure
            assert pattern.seqlet_instances.shape[0] == pattern.n_seqlets
            assert pattern.seqlet_instances.shape[2] == 4
            assert pattern.seqlet_contribs.shape == pattern.seqlet_instances.shape

            # Check metadata
            assert len(pattern.seqlets_metadata) == pattern.n_seqlets
            assert "start" in pattern.seqlets_metadata.columns
            assert "end" in pattern.seqlets_metadata.columns

    def test_create_patterns_with_small_clusters(self, sample_seqlet_adata):
        """Test that create_patterns skips clusters with too few seqlets."""
        adata = sample_seqlet_adata.copy()

        # Force high resolution to create many small clusters
        tm.tl.cluster_seqlets(adata, resolution=10.0)

        patterns = tm.tl.create_patterns(adata)

        # Should create some patterns, but may skip very small clusters
        assert isinstance(patterns, dict)

        # Check that returned patterns have reasonable sizes
        for pattern in patterns.values():
            assert pattern.n_seqlets >= 2, "Returned patterns should have at least 2 seqlets"

    def test_create_patterns_missing_data(self, sample_seqlet_adata):
        """Test error handling when required data is missing."""
        adata = sample_seqlet_adata.copy()

        # Run clustering first
        tm.tl.cluster_seqlets(adata, resolution=1.0)

        # Remove required column
        del adata.obs["seqlet_matrix"]

        # Should raise ValueError
        with pytest.raises(ValueError, match="Missing required columns"):
            tm.tl.create_patterns(adata)

    def test_create_patterns_without_clustering(self, sample_seqlet_adata):
        """Test error handling when clustering hasn't been run."""
        adata = sample_seqlet_adata.copy()

        # Should raise ValueError since leiden column doesn't exist
        with pytest.raises(ValueError, match="Missing required columns"):
            tm.tl.create_patterns(adata)

    def test_pattern_ic_calculation(self, sample_seqlet_adata):
        """Test information content calculation."""
        adata = sample_seqlet_adata.copy()

        # Run clustering and create patterns
        tm.tl.cluster_seqlets(adata, resolution=1.0)
        patterns = tm.tl.create_patterns(adata)

        if len(patterns) > 0:
            pattern = list(patterns.values())[0]
            ic = pattern.ic()

            # Check IC properties
            assert ic.shape[0] == pattern.ppm.shape[0], "IC should have one value per position"
            assert np.all(ic >= 0), "IC should be non-negative"
            assert np.all(ic <= 2), "IC should be at most 2 bits"

    def test_pattern_consensus_quality(self, sample_seqlet_adata):
        """Test that patterns represent reasonable consensus sequences."""
        adata = sample_seqlet_adata.copy()

        # Run clustering and create patterns
        tm.tl.cluster_seqlets(adata, resolution=0.5)  # Lower resolution for better consensus
        patterns = tm.tl.create_patterns(adata)

        for pattern in patterns.values():
            # Check that PWM is properly normalized (sums to 1 at each position)
            position_sums = pattern.ppm.sum(axis=1)
            np.testing.assert_allclose(position_sums, 1.0, rtol=1e-6, err_msg="PWM positions should sum to 1")

            # Check that all values are non-negative
            assert np.all(pattern.ppm >= 0), "PWM values should be non-negative"

            # Check that seqlet instances have reasonable range
            assert np.all(pattern.seqlet_instances >= 0), "Seqlet instances should be non-negative"
            assert np.all(pattern.seqlet_instances <= 1), "Seqlet instances should be at most 1"
