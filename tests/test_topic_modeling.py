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
        tm.tl.run_topic_modeling(
            adata,
            n_topics=5,
            n_iter=10,
            filter_unknown=False,
        )

        # Check that results are stored in AnnData
        assert "topic_modeling" in adata.uns
        topic_results = adata.uns["topic_modeling"]

        # Check model object
        model = topic_results["model"]
        assert hasattr(model, "n_topics")
        assert hasattr(model, "doc_topic_")
        assert hasattr(model, "topic_word_")
        assert model.n_topics == 5

        # Check region-topic matrix
        region_topic = topic_results["region_topic_matrix"]
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
        count_table = topic_results["count_matrix"]
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
        tm.tl.run_topic_modeling(adata, n_topics=3, n_iter=10, filter_unknown=True)
        region_topic1 = adata.uns["topic_modeling"]["region_topic_matrix"]

        # Test with filtering disabled
        tm.tl.run_topic_modeling(adata, n_topics=3, n_iter=10, filter_unknown=False)
        region_topic2 = adata.uns["topic_modeling"]["region_topic_matrix"]

        # With filtering, should have fewer or equal regions
        assert region_topic1.shape[0] <= region_topic2.shape[0]

    def test_run_topic_modeling_parameters(self, sample_clustered_adata):
        """Test different parameter combinations."""
        adata = sample_clustered_adata.copy()

        # Test different n_topics
        tm.tl.run_topic_modeling(adata, n_topics=10, n_iter=5, alpha=25, eta=0.05, random_state=42)

        topic_results = adata.uns["topic_modeling"]
        model = topic_results["model"]
        region_topic = topic_results["region_topic_matrix"]

        assert model.n_topics == 10
        assert region_topic.shape[1] == 10

    def test_topic_cluster_matrix_storage(self, sample_clustered_adata):
        """Test that topic-cluster matrix is correctly stored."""
        adata = sample_clustered_adata.copy()

        # Run topic modeling
        tm.tl.run_topic_modeling(adata, n_topics=5, n_iter=10, filter_unknown=False)

        # Get topic-cluster matrix from stored results
        topic_results = adata.uns["topic_modeling"]
        topic_cluster = topic_results["topic_cluster_matrix"]
        count_table = topic_results["count_matrix"]

        # Check structure
        assert isinstance(topic_cluster, pd.DataFrame)
        assert topic_cluster.shape[0] == len(count_table.columns)  # n_clusters
        assert topic_cluster.shape[1] == 5  # n_topics

        # Check that each column sums to 1 (topic probabilities over clusters)
        col_sums = topic_cluster.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, rtol=1e-5)

    def test_topic_dbd_matrix_creation(self, sample_clustered_adata):
        """Test that topic-DBD matrix can be created from stored results."""
        adata = sample_clustered_adata.copy()

        # Run topic modeling
        tm.tl.run_topic_modeling(adata, n_topics=5, n_iter=10, filter_unknown=False)

        # Get stored results
        topic_results = adata.uns["topic_modeling"]
        topic_cluster = topic_results["topic_cluster_matrix"]
        count_table = topic_results["count_matrix"]

        # Create cluster-to-DBD mapping (same as before)
        cluster_to_dbd = {}
        for cluster in count_table.columns:
            cluster_mask = adata.obs["leiden"] == cluster
            if cluster_mask.any():
                dbd = adata.obs.loc[cluster_mask, "cluster_dbd"].iloc[0]
                cluster_to_dbd[str(cluster)] = str(dbd)

        # Create topic-DBD matrix using pandas operations (same logic as removed function)
        topic_dbd = topic_cluster.groupby(cluster_to_dbd).mean()

        # Check structure
        assert isinstance(topic_dbd, pd.DataFrame)
        assert topic_dbd.shape[1] == 5  # n_topics
        assert topic_dbd.shape[0] <= len(set(cluster_to_dbd.values()))  # n_unique_dbds

        # Check that values are reasonable (probabilities)
        assert (topic_dbd >= 0).all().all()
        assert (topic_dbd <= 1).all().all()

    def test_topic_modeling_reproducibility(self, sample_clustered_adata):
        """Test that topic modeling results are reproducible with same random state."""
        adata = sample_clustered_adata.copy()

        # Run topic modeling twice with same random state
        tm.tl.run_topic_modeling(adata, n_topics=5, n_iter=10, random_state=42)
        region_topic1 = adata.uns["topic_modeling"]["region_topic_matrix"]

        tm.tl.run_topic_modeling(adata, n_topics=5, n_iter=10, random_state=42)
        region_topic2 = adata.uns["topic_modeling"]["region_topic_matrix"]

        # Results should be identical
        np.testing.assert_array_almost_equal(region_topic1.values, region_topic2.values, decimal=10)

    def test_topic_modeling_edge_cases(self, sample_clustered_adata):
        """Test edge cases and error conditions."""
        adata = sample_clustered_adata.copy()

        # Test with n_topics = 1
        tm.tl.run_topic_modeling(adata, n_topics=1, n_iter=5)
        topic_results = adata.uns["topic_modeling"]
        model = topic_results["model"]
        region_topic = topic_results["region_topic_matrix"]
        count_table = topic_results["count_matrix"]

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
        tm.tl.run_topic_modeling(adata, n_topics=2, n_iter=5)
        topic_results = adata.uns["topic_modeling"]
        model = topic_results["model"]
        region_topic = topic_results["region_topic_matrix"]
        count_table = topic_results["count_matrix"]

        assert model.n_topics == 2
        assert region_topic.shape[1] == 2
        assert region_topic.shape[0] > 0
        assert count_table.shape[0] > 0
