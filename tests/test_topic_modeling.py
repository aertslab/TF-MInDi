"""Tests for topic modeling functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

import tfmindi as tm


class TestTopicModeling:
    """Test topic modeling functions."""

    def test_run_topic_modeling_basic(self, sample_clustered_adata):
        """Test basic functionality of run_topic_modeling."""
        adata = sample_clustered_adata.copy()

        # Run topic modeling with small parameters for testing
        model, region_topic, count_table = tm.tl.run_topic_modeling(
            adata,
            n_topics=5,
            n_iter=10,
            filter_unknown=False,
        )

        # Check model object
        assert hasattr(model, "n_topics")
        assert hasattr(model, "doc_topic_")
        assert hasattr(model, "topic_word_")
        assert model.n_topics == 5

        # Check region-topic matrix
        assert isinstance(region_topic, pd.DataFrame)
        assert region_topic.shape[1] == 5  # n_topics
        assert region_topic.shape[0] > 0  # Should have some regions

        # Check that probabilities sum to 1 (approximately)
        row_sums = region_topic.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)

        # Check column names
        expected_columns = [f"Topic_{i + 1}" for i in range(5)]
        assert list(region_topic.columns) == expected_columns

        # Check count table
        assert isinstance(count_table, pd.DataFrame)
        assert count_table.shape[0] > 0  # Should have some regions
        assert count_table.shape[1] > 0  # Should have some clusters

    def test_run_topic_modeling_filter_unknown(self, sample_clustered_adata):
        """Test filtering of unknown DBD annotations."""
        adata = sample_clustered_adata.copy()

        # Add some "nan" values to cluster_dbd
        adata.obs["cluster_dbd"] = adata.obs["cluster_dbd"].astype(str)
        adata.obs.loc[adata.obs.index[:5], "cluster_dbd"] = "nan"

        # Test with filtering enabled (default)
        _, region_topic1, _ = tm.tl.run_topic_modeling(adata, n_topics=3, n_iter=10, filter_unknown=True)

        # Test with filtering disabled
        _, region_topic2, _ = tm.tl.run_topic_modeling(adata, n_topics=3, n_iter=10, filter_unknown=False)

        # With filtering, should have fewer or equal regions
        assert region_topic1.shape[0] <= region_topic2.shape[0]

    def test_run_topic_modeling_parameters(self, sample_clustered_adata):
        """Test different parameter combinations."""
        adata = sample_clustered_adata.copy()

        # Test different n_topics
        model, region_topic, _ = tm.tl.run_topic_modeling(
            adata, n_topics=10, n_iter=5, alpha=25, eta=0.05, random_state=42
        )

        assert model.n_topics == 10
        assert region_topic.shape[1] == 10

    def test_get_topic_cluster_matrix(self, sample_clustered_adata):
        """Test topic-cluster matrix extraction."""
        adata = sample_clustered_adata.copy()

        # Run topic modeling
        model, _, count_table = tm.tl.run_topic_modeling(adata, n_topics=5, n_iter=10, filter_unknown=False)

        # Get topic-cluster matrix
        topic_cluster = tm.tl.get_topic_cluster_matrix(model, count_table)

        # Check structure
        assert isinstance(topic_cluster, pd.DataFrame)
        assert topic_cluster.shape[0] == len(count_table.columns)  # n_clusters
        assert topic_cluster.shape[1] == 5  # n_topics

        # Check that each column sums to 1 (topic probabilities over clusters)
        col_sums = topic_cluster.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, rtol=1e-5)

    def test_get_topic_dbd_matrix(self, sample_clustered_adata):
        """Test topic-DBD matrix creation."""
        adata = sample_clustered_adata.copy()

        # Run topic modeling
        model, _, count_table = tm.tl.run_topic_modeling(adata, n_topics=5, n_iter=10, filter_unknown=False)

        # Create cluster-to-DBD mapping
        cluster_to_dbd = {}
        for cluster in count_table.columns:
            cluster_mask = adata.obs["leiden"] == cluster
            if cluster_mask.any():
                dbd = adata.obs.loc[cluster_mask, "cluster_dbd"].iloc[0]
                cluster_to_dbd[str(cluster)] = str(dbd)

        # Get topic-DBD matrix
        topic_dbd = tm.tl.get_topic_dbd_matrix(model, count_table, cluster_to_dbd)

        # Check structure
        assert isinstance(topic_dbd, pd.DataFrame)
        assert topic_dbd.shape[1] == 5  # n_topics
        assert topic_dbd.shape[0] <= len(cluster_to_dbd)  # n_unique_dbds

        # Check that values are reasonable (probabilities)
        assert (topic_dbd >= 0).all().all()
        assert (topic_dbd <= 1).all().all()

    def test_topic_modeling_reproducibility(self, sample_clustered_adata):
        """Test that topic modeling results are reproducible with same random state."""
        adata = sample_clustered_adata.copy()

        # Run topic modeling twice with same random state
        _, region_topic1, _ = tm.tl.run_topic_modeling(adata, n_topics=5, n_iter=10, random_state=42)

        _, region_topic2, _ = tm.tl.run_topic_modeling(adata, n_topics=5, n_iter=10, random_state=42)

        # Results should be identical
        np.testing.assert_array_almost_equal(region_topic1.values, region_topic2.values, decimal=10)

    def test_topic_modeling_edge_cases(self, sample_clustered_adata):
        """Test edge cases and error conditions."""
        adata = sample_clustered_adata.copy()

        # Test with n_topics = 1
        model, region_topic, count_table = tm.tl.run_topic_modeling(adata, n_topics=1, n_iter=5)

        assert model.n_topics == 1
        assert region_topic.shape[1] == 1
        assert count_table.shape[0] > 0

        # All topic probabilities should be 1.0
        np.testing.assert_allclose(region_topic.values, 1.0, rtol=1e-5)

    def test_topic_modeling_with_small_data(self, sample_clustered_adata):
        """Test topic modeling with very small datasets."""
        adata = sample_clustered_adata.copy()

        # Keep only first few observations
        adata = adata[:10, :].copy()

        # Should still work with small data
        model, region_topic, count_table = tm.tl.run_topic_modeling(adata, n_topics=2, n_iter=5)

        assert model.n_topics == 2
        assert region_topic.shape[1] == 2
        assert region_topic.shape[0] > 0
        assert count_table.shape[0] > 0
