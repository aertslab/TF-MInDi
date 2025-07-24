"""Tests for preprocessing functions."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

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

        # seq that len of seqlets PPM is same as in df
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

    def test_calculate_motif_similarity_chunked_vs_non_chunked(
        self, sample_contrib_data, sample_oh_data, sample_motifs
    ):
        """Test that chunked and non-chunked processing produce identical results."""
        # Extract seqlets from real data
        contrib_subset = sample_contrib_data[:5]  # First 5 examples
        oh_subset = sample_oh_data[:5]

        seqlets_df, seqlet_matrices = tm.pp.extract_seqlets(contrib_subset, oh_subset, threshold=0.1)

        # Skip test if not enough seqlets found
        if len(seqlet_matrices) < 10:
            pytest.skip("Not enough seqlets found for chunking test")

        # Use subset of seqlets and motifs for testing
        test_seqlets = seqlet_matrices[:20] if len(seqlet_matrices) >= 20 else seqlet_matrices
        test_motifs = list(sample_motifs.values())[:5]  # First 5 motifs

        # Calculate similarity without chunking
        result_no_chunk = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs, chunk_size=None)

        # Calculate similarity with chunking (use small chunk size to force chunking)
        chunk_size = 7  # Smaller than test_seqlets length to force chunking
        result_chunked = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs, chunk_size=chunk_size)

        # Results should be identical
        assert result_no_chunk.shape == result_chunked.shape
        np.testing.assert_array_equal(
            result_no_chunk, result_chunked, err_msg="Chunked and non-chunked results should be identical"
        )

        # Also test with very small chunks
        chunk_size_small = 3
        result_small_chunks = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs, chunk_size=chunk_size_small)

        np.testing.assert_array_equal(
            result_no_chunk, result_small_chunks, err_msg="Small chunks should produce same results as non-chunked"
        )

    def test_calculate_motif_similarity_chunked_edge_cases(self, sample_motifs):
        """Test chunked processing with edge cases."""
        # Create test seqlets
        test_seqlets = [
            np.array([[0.8, 0.0, 0.0, 0.2], [0.0, 0.0, 0.9, 0.1], [0.1, 0.8, 0.0, 0.1], [0.0, 0.1, 0.1, 0.8]]),
            np.array([[0.0, 0.9, 0.1, 0.0], [0.8, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 1.0], [0.2, 0.1, 0.7, 0.0]]),
            np.array([[0.5, 0.2, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.2, 0.1, 0.6, 0.1], [0.2, 0.1, 0.1, 0.6]]),
        ]
        test_motifs = list(sample_motifs.values())[:2]

        # Test chunk size larger than data (should use non-chunked path)
        result_large_chunk = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs, chunk_size=10)
        result_no_chunk = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs, chunk_size=None)

        np.testing.assert_array_equal(
            result_large_chunk, result_no_chunk, err_msg="Large chunk size should produce same results as no chunking"
        )

        # Test chunk size equal to data size
        result_exact_chunk = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs, chunk_size=len(test_seqlets))
        np.testing.assert_array_equal(
            result_exact_chunk, result_no_chunk, err_msg="Chunk size equal to data size should produce same results"
        )

        # Test chunk size of 1 (most extreme chunking)
        result_single_chunk = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs, chunk_size=1)
        np.testing.assert_array_equal(
            result_single_chunk, result_no_chunk, err_msg="Single-item chunks should produce same results"
        )


class TestCreateSeqletAdata:
    """Test create_seqlet_adata function."""

    def test_create_seqlet_adata_basic(self):
        """Test basic functionality of create_seqlet_adata."""
        # Create simple test data
        n_seqlets, n_motifs = 5, 3
        similarity_matrix = np.random.rand(n_seqlets, n_motifs)

        seqlet_metadata = pd.DataFrame(
            {
                "example_idx": [0, 1, 2, 0, 1],
                "start": [10, 20, 30, 40, 50],
                "end": [25, 35, 45, 55, 65],
                "attribution": [0.8, -0.6, 0.9, -0.7, 0.5],
                "p-value": [1e-5, 1e-4, 1e-6, 1e-3, 1e-4],
            }
        )

        # Create seqlet matrices (4 x length for each seqlet)
        seqlet_matrices = [np.random.rand(4, 15) for _ in range(n_seqlets)]

        # Create oh sequences and contrib scores (examples x 4 x total_length)
        oh_sequences = np.random.randint(0, 2, size=(3, 4, 100)).astype(float)
        contrib_scores = np.random.randn(3, 4, 100)

        motif_names = [f"motif_{i}" for i in range(n_motifs)]

        adata = tm.pp.create_seqlet_adata(
            similarity_matrix,
            seqlet_metadata,
            seqlet_matrices=seqlet_matrices,
            oh_sequences=oh_sequences,
            contrib_scores=contrib_scores,
            motif_names=motif_names,
        )

        # Check basic structure
        assert isinstance(adata, AnnData)
        assert adata.shape == (n_seqlets, n_motifs)
        assert np.array_equal(adata.X, similarity_matrix)  # type: ignore

        # Check that metadata is preserved (excluding new array columns)
        metadata_cols = seqlet_metadata.columns
        assert all(col in adata.obs.columns for col in metadata_cols)
        pd.testing.assert_frame_equal(
            adata.obs[metadata_cols].reset_index(drop=True), seqlet_metadata.reset_index(drop=True)
        )

        # Check that seqlet matrices are stored in .obs
        assert "seqlet_matrix" in adata.obs.columns
        assert len(adata.obs["seqlet_matrix"]) == n_seqlets
        assert all(mat.shape[0] == 4 for mat in adata.obs["seqlet_matrix"])

        # Check that seqlet one-hot sequences are stored in .obs
        assert "seqlet_oh" in adata.obs.columns

        # Check that example-level data is stored in .obs (mapped to each seqlet)
        assert "example_oh" in adata.obs.columns
        assert "example_contrib" in adata.obs.columns
        assert len(adata.obs["example_oh"]) == n_seqlets
        assert len(adata.obs["example_contrib"]) == n_seqlets

        # Verify example mapping is correct
        for i, (_, row) in enumerate(seqlet_metadata.iterrows()):
            ex_idx = int(row["example_idx"])
            assert np.array_equal(adata.obs.iloc[i]["example_oh"], oh_sequences[ex_idx])
            assert np.array_equal(adata.obs.iloc[i]["example_contrib"], contrib_scores[ex_idx])

        # Check motif names in var
        assert list(adata.var.index) == motif_names

    def test_create_seqlet_adata_with_motif_collection(self):
        """Test create_seqlet_adata with motif_collection parameter."""
        n_seqlets, n_motifs = 3, 2
        similarity_matrix = np.random.rand(n_seqlets, n_motifs)

        seqlet_metadata = pd.DataFrame({"example_idx": [0, 1, 0], "start": [10, 20, 30], "end": [25, 35, 45]})

        # Create motif collection as dict
        motif_collection = {"TF1": np.random.rand(4, 8), "TF2": np.random.rand(4, 10)}

        adata = tm.pp.create_seqlet_adata(similarity_matrix, seqlet_metadata, motif_collection=motif_collection)

        # Check motif PPMs are stored in .var
        assert "motif_ppm" in adata.var.columns
        assert len(adata.var["motif_ppm"]) == n_motifs
        assert list(adata.var.index) == list(motif_collection.keys())

        # Check that motif PPMs are correctly stored
        for _, (motif_name, motif_ppm) in enumerate(motif_collection.items()):
            stored_ppm = adata.var.loc[motif_name, "motif_ppm"]
            assert np.array_equal(stored_ppm, motif_ppm)  # type: ignore

    def test_create_seqlet_adata_with_motif_annotations(self):
        """Test create_seqlet_adata with motif annotations and DBD data."""
        n_seqlets, n_motifs = 3, 2
        similarity_matrix = np.random.rand(n_seqlets, n_motifs)

        seqlet_metadata = pd.DataFrame({"example_idx": [0, 1, 0], "start": [10, 20, 30], "end": [25, 35, 45]})

        motif_names = ["TF1", "TF2"]

        # Create motif annotations DataFrame
        motif_annotations = pd.DataFrame(
            {
                "Direct_annot": ["GENE1", "GENE2"],
                "Motif_similarity_annot": ["SIMILAR1", None],
                "Orthology_annot": [None, "ORTHOLOG2"],
            },
            index=motif_names,
        )

        # Create motif to DBD mapping
        motif_to_dbd = {"TF1": "Homeodomain", "TF2": "STAT"}

        adata = tm.pp.create_seqlet_adata(
            similarity_matrix,
            seqlet_metadata,
            motif_names=motif_names,
            motif_annotations=motif_annotations,
            motif_to_dbd=motif_to_dbd,
        )

        # Check motif annotations are stored in .var
        assert "Direct_annot" in adata.var.columns
        assert "Motif_similarity_annot" in adata.var.columns
        assert "Orthology_annot" in adata.var.columns
        assert "dbd" in adata.var.columns

        # Check specific values
        assert adata.var.loc["TF1", "Direct_annot"] == "GENE1"
        assert adata.var.loc["TF2", "Direct_annot"] == "GENE2"
        assert adata.var.loc["TF1", "dbd"] == "Homeodomain"
        assert adata.var.loc["TF2", "dbd"] == "STAT"

        # Check None values are preserved
        assert pd.isna(adata.var.loc["TF1", "Orthology_annot"])
        assert pd.isna(adata.var.loc["TF2", "Motif_similarity_annot"])

    def test_create_seqlet_adata_real_data(self, sample_contrib_data, sample_oh_data, sample_motifs):
        """Test create_seqlet_adata with real extracted seqlets."""
        # Extract seqlets from a small subset
        contrib_subset = sample_contrib_data[:5]
        oh_subset = sample_oh_data[:5]

        seqlets_df, seqlet_matrices = tm.pp.extract_seqlets(contrib_subset, oh_subset, threshold=0.1)

        # Skip if no seqlets found
        if len(seqlets_df) == 0:
            pytest.skip("No seqlets found in test data")

        # Calculate similarity with subset of motifs
        test_motifs = dict(list(sample_motifs.items())[:3])
        motif_names = list(test_motifs.keys())
        similarity_matrix = tm.pp.calculate_motif_similarity(seqlet_matrices, test_motifs)

        # Create AnnData object with all data
        adata = tm.pp.create_seqlet_adata(
            similarity_matrix,
            seqlets_df,
            seqlet_matrices=seqlet_matrices,
            oh_sequences=oh_subset,
            contrib_scores=contrib_subset,
            motif_names=motif_names,
        )

        # Verify structure
        assert isinstance(adata, AnnData)
        assert adata.shape == (len(seqlets_df), len(test_motifs))
        assert np.array_equal(adata.X, similarity_matrix)  # type: ignore

        # Check metadata preservation
        expected_cols = ["example_idx", "start", "end", "attribution", "p-value"]
        assert all(col in adata.obs.columns for col in expected_cols)

        # Check that all data is stored properly in .obs columns
        assert "seqlet_matrix" in adata.obs.columns
        assert len(adata.obs["seqlet_matrix"]) == len(seqlets_df)
        assert "seqlet_oh" in adata.obs.columns
        assert "example_oh" in adata.obs.columns
        assert "example_contrib" in adata.obs.columns

        # Verify example-level data mapping
        for i, (_, row) in enumerate(seqlets_df.iterrows()):
            ex_idx = int(row["example_idx"])
            assert np.array_equal(adata.obs.iloc[i]["example_oh"], oh_subset[ex_idx])
            assert np.array_equal(adata.obs.iloc[i]["example_contrib"], contrib_subset[ex_idx])

        assert list(adata.var.index) == motif_names

    def test_create_seqlet_adata_empty_inputs(self):
        """Test behavior with empty inputs."""
        similarity_matrix = np.array([]).reshape(0, 0)
        seqlet_metadata = pd.DataFrame()
        seqlet_matrices = []
        oh_sequences = np.array([]).reshape(0, 4, 0)
        contrib_scores = np.array([]).reshape(0, 4, 0)

        adata = tm.pp.create_seqlet_adata(
            similarity_matrix,
            seqlet_metadata,
            seqlet_matrices=seqlet_matrices,
            oh_sequences=oh_sequences,
            contrib_scores=contrib_scores,
        )

        assert isinstance(adata, AnnData)
        assert adata.shape == (0, 0)
        assert "seqlet_matrix" in adata.obs.columns
        assert len(adata.obs["seqlet_matrix"]) == 0

    def test_create_seqlet_adata_dimension_mismatch(self):
        """Test error handling for dimension mismatches."""
        similarity_matrix = np.random.rand(5, 3)
        seqlet_metadata = pd.DataFrame({"example_idx": [0, 1, 2]})  # Only 3 rows instead of 5
        seqlet_matrices = [np.random.rand(4, 10) for _ in range(3)]  # Only 3 matrices instead of 5

        with pytest.raises(ValueError, match="Number of seqlets in similarity matrix"):
            tm.pp.create_seqlet_adata(similarity_matrix, seqlet_metadata, seqlet_matrices=seqlet_matrices)

    def test_create_seqlet_adata_minimal_required_params(self):
        """Test that function works with minimal required parameters."""
        n_seqlets, n_motifs = 3, 2
        similarity_matrix = np.random.rand(n_seqlets, n_motifs)
        seqlet_metadata = pd.DataFrame({"example_idx": [0, 1, 0], "start": [10, 20, 30], "end": [25, 35, 45]})

        # Should work with just similarity matrix and metadata
        adata = tm.pp.create_seqlet_adata(similarity_matrix, seqlet_metadata)

        assert isinstance(adata, AnnData)
        assert adata.shape == (n_seqlets, n_motifs)
        # Optional data should not be present
        assert "seqlet_matrix" not in adata.obs.columns
        assert "example_oh" not in adata.obs.columns
        assert "example_contrib" not in adata.obs.columns
