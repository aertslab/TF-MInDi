"""Tests for preprocessing functions."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from scipy.sparse import csr_array

import tfmindi as tm


def _make_sparse_similarity_matrix(dense_matrix):
    """Helper function to convert dense matrix to sparse with threshold."""
    # Apply same threshold as in the actual function
    dense_matrix[dense_matrix < 0.05] = 0
    return sparse.csr_array(dense_matrix)


class TestExtractSeqlets:
    """Test extract_seqlets function."""

    def test_extract_seqlets_real_data(self, sample_contrib_data, sample_oh_data):
        """Test extract_seqlets with real data."""
        seqlet_df, seqlet_matrices = tm.pp.extract_seqlets(sample_contrib_data, sample_oh_data)

        assert len(seqlet_df) == len(seqlet_matrices) == 227

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
        assert isinstance(result, sparse.csr_array)
        assert result.shape == (len(test_seqlets), len(test_motifs))
        result_dense = result.toarray()
        assert not np.isnan(result_dense).any()
        assert np.all(result_dense >= 0)  # All non-negative after log transform and clipping
        assert np.all(np.isfinite(result_dense))

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
        result_dense = result.toarray()
        assert not np.isnan(result_dense).any()
        assert np.all(result_dense >= 0)

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
            result_no_chunk.toarray(),
            result_chunked.toarray(),
            err_msg="Chunked and non-chunked results should be identical",
        )

        # Also test with very small chunks
        chunk_size_small = 3
        result_small_chunks = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs, chunk_size=chunk_size_small)

        np.testing.assert_array_equal(
            result_no_chunk.toarray(),
            result_small_chunks.toarray(),
            err_msg="Small chunks should produce same results as non-chunked",
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
            result_large_chunk.toarray(),
            result_no_chunk.toarray(),
            err_msg="Large chunk size should produce same results as no chunking",
        )

        # Test chunk size equal to data size
        result_exact_chunk = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs, chunk_size=len(test_seqlets))
        np.testing.assert_array_equal(
            result_exact_chunk.toarray(),
            result_no_chunk.toarray(),
            err_msg="Chunk size equal to data size should produce same results",
        )

        # Test chunk size of 1 (most extreme chunking)
        result_single_chunk = tm.pp.calculate_motif_similarity(test_seqlets, test_motifs, chunk_size=1)
        np.testing.assert_array_equal(
            result_single_chunk.toarray(),
            result_no_chunk.toarray(),
            err_msg="Single-item chunks should produce same results",
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

        # Convert dense matrix to sparse for the function
        sparse_similarity_matrix = _make_sparse_similarity_matrix(similarity_matrix)

        adata = tm.pp.create_seqlet_adata(
            sparse_similarity_matrix,
            seqlet_metadata,
            seqlet_matrices=seqlet_matrices,
            oh_sequences=oh_sequences,
            contrib_scores=contrib_scores,
            motif_names=motif_names,
        )

        # Check basic structure
        assert isinstance(adata, AnnData)
        assert adata.shape == (n_seqlets, n_motifs)
        # Check that X is sparse and has expected data
        assert isinstance(adata.X, sparse.csr_array)
        # Convert to dense for comparison (apply same threshold as helper function)
        expected_dense = similarity_matrix.astype(np.float32).copy()
        expected_dense[expected_dense < 0.05] = 0
        # Convert both to dense arrays for comparison
        actual_dense = adata.X.toarray()
        np.testing.assert_array_equal(actual_dense, expected_dense)

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

        # Check that example-level data is stored in .uns with unique examples
        assert "unique_examples" in adata.uns
        assert "oh" in adata.uns["unique_examples"]
        assert "contrib" in adata.uns["unique_examples"]
        assert "example_oh_idx" in adata.obs.columns
        assert "example_contrib_idx" in adata.obs.columns

        # Check that unique examples are stored efficiently
        unique_example_indices = seqlet_metadata["example_idx"].unique()
        assert adata.uns["unique_examples"]["oh"].shape[0] == len(unique_example_indices)
        assert adata.uns["unique_examples"]["contrib"].shape[0] == len(unique_example_indices)

        # Verify example mapping is correct using helper functions
        for i, (_, row) in enumerate(seqlet_metadata.iterrows()):
            ex_idx = int(row["example_idx"])
            retrieved_oh = tm.pp.seqlets.get_example_oh(adata, i)
            retrieved_contrib = tm.pp.seqlets.get_example_contrib(adata, i)
            expected_oh = oh_sequences[ex_idx].astype(np.float32)
            expected_contrib = contrib_scores[ex_idx].astype(np.float32)
            assert np.array_equal(retrieved_oh, expected_oh)
            assert np.array_equal(retrieved_contrib, expected_contrib)

        # Check motif names in var
        assert list(adata.var.index) == motif_names

    def test_create_seqlet_adata_with_motif_collection(self):
        """Test create_seqlet_adata with motif_collection parameter."""
        n_seqlets, n_motifs = 3, 2
        similarity_matrix = np.random.rand(n_seqlets, n_motifs)

        seqlet_metadata = pd.DataFrame({"example_idx": [0, 1, 0], "start": [10, 20, 30], "end": [25, 35, 45]})

        # Create motif collection as dict
        motif_collection = {"TF1": np.random.rand(4, 8), "TF2": np.random.rand(4, 10)}

        # Convert dense matrix to sparse for the function
        sparse_similarity_matrix = _make_sparse_similarity_matrix(similarity_matrix)
        adata = tm.pp.create_seqlet_adata(sparse_similarity_matrix, seqlet_metadata, motif_collection=motif_collection)

        # Check motif PPMs are stored in .var
        assert "motif_ppm" in adata.var.columns
        assert len(adata.var["motif_ppm"]) == n_motifs
        assert list(adata.var.index) == list(motif_collection.keys())

        # Check that motif PPMs are correctly stored
        for _, (motif_name, motif_ppm) in enumerate(motif_collection.items()):
            stored_ppm = adata.var.loc[motif_name, "motif_ppm"]
            assert np.array_equal(stored_ppm, motif_ppm.astype(np.float32))  # type: ignore

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
        # Check that X is sparse and has expected data
        assert isinstance(adata.X, sparse.csr_array)
        # Convert sparse similarity matrix to dense for comparison
        expected_dense = similarity_matrix.toarray().astype(np.float32)
        # Convert both to dense arrays for comparison
        actual_dense = adata.X.toarray()
        np.testing.assert_array_equal(actual_dense, expected_dense)

        # Check metadata preservation
        expected_cols = ["example_idx", "start", "end", "attribution", "p-value"]
        assert all(col in adata.obs.columns for col in expected_cols)

        # Check that variable-length data is stored properly in .obs columns
        assert "seqlet_matrix" in adata.obs.columns
        assert len(adata.obs["seqlet_matrix"]) == len(seqlets_df)
        assert "seqlet_oh" in adata.obs.columns

        # Check that example-level data is stored in .uns with unique examples
        assert "unique_examples" in adata.uns
        assert "oh" in adata.uns["unique_examples"]
        assert "contrib" in adata.uns["unique_examples"]
        assert "example_oh_idx" in adata.obs.columns
        assert "example_contrib_idx" in adata.obs.columns

        # Verify example-level data mapping using helper functions
        for i, (_, row) in enumerate(seqlets_df.iterrows()):
            ex_idx = int(row["example_idx"])
            retrieved_oh = tm.pp.seqlets.get_example_oh(adata, i)
            retrieved_contrib = tm.pp.seqlets.get_example_contrib(adata, i)
            expected_oh = oh_subset[ex_idx].astype(np.float32)
            expected_contrib = contrib_subset[ex_idx].astype(np.float32)
            assert np.array_equal(retrieved_oh, expected_oh)
            assert np.array_equal(retrieved_contrib, expected_contrib)

        motif_names_cleaned = [name[1] for name in motif_names if name is not None]  # only non-cluster name
        assert list(adata.var.index) == motif_names_cleaned

    def test_create_seqlet_adata_empty_inputs(self):
        """Test behavior with empty inputs."""
        similarity_matrix = csr_array(np.array([]).reshape(0, 0))
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
        # Empty inputs should not create empty columns
        assert "seqlet_matrix" not in adata.obs.columns

    def test_create_seqlet_adata_dimension_mismatch(self):
        """Test error handling for dimension mismatches."""
        similarity_matrix = csr_array(np.random.rand(5, 3))
        seqlet_metadata = pd.DataFrame({"example_idx": [0, 1, 2]})  # Only 3 rows instead of 5
        seqlet_matrices = [np.random.rand(4, 10) for _ in range(3)]  # Only 3 matrices instead of 5

        with pytest.raises(ValueError, match="Number of seqlets in similarity matrix"):
            tm.pp.create_seqlet_adata(similarity_matrix, seqlet_metadata, seqlet_matrices=seqlet_matrices)

    def test_create_seqlet_adata_dtype_precision_preservation(self):
        """Test that dtype conversion doesn't introduce significant numerical errors."""
        n_seqlets, n_motifs = 5, 3
        similarity_matrix = csr_array(
            np.array(
                [
                    [1.0, 0.5, 1e-7],  # Very small positive number
                    [0.0, -1e-7, 2.5],  # Very small negative number
                    [100.0, 0.001, 0.999],  # Range of typical values
                    [1e-6, 1e6, 0.1],  # Small and large numbers
                    [np.pi, np.e, 1.234567],  # Irrational numbers with precision
                ],
                dtype=np.float64,
            )
        )

        seqlet_metadata = pd.DataFrame(
            {"example_idx": [0, 1, 0, 1, 2], "start": [10, 20, 30, 40, 50], "end": [25, 35, 45, 55, 65]}
        )

        seqlet_matrices = [
            np.array([[1.0, 0.5], [1e-7, 2.5], [100.0, 0.001], [0.999, np.pi]], dtype=np.float64)
            for _ in range(n_seqlets)
        ]

        oh_sequences = np.array(
            [
                [[1.0, 0.5, 1e-7], [0.0, 1.0, 0.5], [0.5, 0.25, 1.0], [0.25, 0.125, 0.0]],
                [[0.9, 0.1, 1e-6], [0.8, 0.2, 0.1], [0.7, 0.3, 0.2], [0.6, 0.4, 0.3]],
                [[np.pi, np.e, 1.5], [2.5, 3.5, 4.5], [5.5, 6.5, 7.5], [8.5, 9.5, 10.5]],
            ],
            dtype=np.float64,
        )

        contrib_scores = np.array(
            [
                [[0.1, -0.1, 1e-8], [0.2, -0.2, 2e-8], [0.3, -0.3, 3e-8], [0.4, -0.4, 4e-8]],
                [[1.1, -1.1, 1e-7], [1.2, -1.2, 2e-7], [1.3, -1.3, 3e-7], [1.4, -1.4, 4e-7]],
                [[10.1, -10.1, 1e-6], [10.2, -10.2, 2e-6], [10.3, -10.3, 3e-6], [10.4, -10.4, 4e-6]],
            ],
            dtype=np.float64,
        )

        motif_collection = {
            (f"motif_{i}", f"motif_{i}"): np.random.rand(4, 8).astype(np.float64)
            * 100  # Larger values to test precision
            for i in range(n_motifs)
        }

        # Test with float32 dtype (default)
        adata = tm.pp.create_seqlet_adata(
            similarity_matrix,
            seqlet_metadata,
            seqlet_matrices=seqlet_matrices,
            oh_sequences=oh_sequences,
            contrib_scores=contrib_scores,
            motif_names=list(motif_collection.keys()),
            motif_collection=motif_collection,
            dtype=np.float32,
        )

        # Check that conversion preserves reasonable precision
        original_float32 = similarity_matrix.astype(np.float32)
        max_error = np.max(np.abs(adata.X - original_float32))  # type: ignore
        assert max_error == 0.0, f"Similarity matrix conversion introduced errors: {max_error}"

        for i, (original_matrix, stored_matrix) in enumerate(
            zip(seqlet_matrices, adata.obs["seqlet_matrix"], strict=False)
        ):
            original_f32 = original_matrix.astype(np.float32)
            max_abs_error = np.max(np.abs(stored_matrix - original_f32))
            assert max_abs_error == 0.0, f"Seqlet matrix {i} conversion introduced errors: {max_abs_error}"

        original_oh_f32 = oh_sequences.astype(np.float32)
        original_contrib_f32 = contrib_scores.astype(np.float32)

        # Check that we get the same results as direct conversion using helper functions
        for i in range(n_seqlets):
            ex_idx = seqlet_metadata.iloc[i]["example_idx"]
            retrieved_oh = tm.pp.seqlets.get_example_oh(adata, i)
            retrieved_contrib = tm.pp.seqlets.get_example_contrib(adata, i)
            np.testing.assert_array_equal(
                retrieved_oh, original_oh_f32[ex_idx], err_msg=f"Example OH data mismatch for seqlet {i}"
            )
            np.testing.assert_array_equal(
                retrieved_contrib,
                original_contrib_f32[ex_idx],
                err_msg=f"Example contrib data mismatch for seqlet {i}",
            )

        # For motif PPMs
        for (_, motif_name), original_ppm in motif_collection.items():
            stored_ppm = adata.var.loc[motif_name, "motif_ppm"]
            original_ppm_f32 = original_ppm.astype(np.float32)
            np.testing.assert_array_equal(
                stored_ppm,  # type: ignore
                original_ppm_f32,
                err_msg=f"Motif PPM conversion error for {motif_name}",
            )

        # Test that we can override dtype to float64 if needed
        adata_f64 = tm.pp.create_seqlet_adata(
            similarity_matrix, seqlet_metadata, seqlet_matrices=seqlet_matrices, dtype=np.float64
        )

        # With float64, should get exact match
        np.testing.assert_array_equal(
            adata_f64.X.todense(),  # type: ignore
            similarity_matrix.todense(),
            err_msg="Float64 conversion should preserve exact values",
        )

    def test_create_seqlet_adata_memory_optimization(self):
        """Test that float32 dtype actually reduces memory usage compared to float64."""
        n_seqlets, n_motifs = 20, 10

        # Create moderately sized test data to see memory difference
        from scipy.sparse import csr_array

        similarity_matrix = csr_array(np.random.rand(n_seqlets, n_motifs).astype(np.float64))
        seqlet_metadata = pd.DataFrame(
            {
                "example_idx": [i % 5 for i in range(n_seqlets)],
                "start": [i * 10 for i in range(n_seqlets)],
                "end": [(i * 10) + 15 for i in range(n_seqlets)],
            }
        )

        seqlet_matrices = [np.random.rand(4, 12).astype(np.float64) for _ in range(n_seqlets)]
        oh_sequences = np.random.rand(5, 4, 500).astype(np.float64)  # 5 examples
        contrib_scores = np.random.rand(5, 4, 500).astype(np.float64)
        motif_collection = {
            (f"motif_{i}", f"motif_{i}"): np.random.rand(4, 8).astype(np.float64) for i in range(n_motifs)
        }

        # Create AnnData with float32 (optimized)
        adata_f32 = tm.pp.create_seqlet_adata(
            similarity_matrix,
            seqlet_metadata,
            seqlet_matrices=seqlet_matrices,
            oh_sequences=oh_sequences,
            contrib_scores=contrib_scores,
            motif_names=list(motif_collection.keys()),
            motif_collection=motif_collection,
            dtype=np.float32,
        )

        # Create AnnData with float64 (unoptimized)
        adata_f64 = tm.pp.create_seqlet_adata(
            similarity_matrix,
            seqlet_metadata,
            seqlet_matrices=seqlet_matrices,
            oh_sequences=oh_sequences,
            contrib_scores=contrib_scores,
            motif_names=list(motif_collection.keys()),
            motif_collection=motif_collection,
            dtype=np.float64,
        )

        # Calculate memory usage for main numerical arrays
        def get_memory_usage(adata) -> int:
            memory = 0
            memory += adata.X.data.nbytes + adata.X.indptr.nbytes + adata.X.indices.nbytes
            # Updated to use new storage format
            if "unique_examples" in adata.uns:
                for arr in adata.uns["unique_examples"].values():
                    memory += arr.nbytes
            for matrices in adata.obs["seqlet_matrix"]:
                memory += matrices.nbytes
            for matrices in adata.obs["seqlet_oh"]:
                memory += matrices.nbytes
            for ppm in adata.var["motif_ppm"]:
                memory += ppm.nbytes
            return memory

        memory_f32 = get_memory_usage(adata_f32)
        memory_f64 = get_memory_usage(adata_f64)

        # Float32 should use approximately half the memory of float64
        memory_ratio: float = memory_f32 / memory_f64

        print(f"Memory usage - float32: {memory_f32:,} bytes, float64: {memory_f64:,} bytes")
        print(f"Memory ratio (f32/f64): {memory_ratio:.3f}")

        # should be close to 0.5
        assert memory_ratio < 0.6, f"Float32 should use significantly less memory. Ratio: {memory_ratio:.3f}"
        assert memory_ratio > 0.4, f"Memory reduction too extreme, check implementation. Ratio: {memory_ratio:.3f}"

        # Verify dtypes are correct
        assert isinstance(adata_f32.X, csr_array) and adata_f32.X.dtype == np.float32
        assert isinstance(adata_f64.X, csr_array) and adata_f64.X.dtype == np.float64
        example_oh_f32 = adata_f32.uns["unique_examples"]["oh"]
        assert isinstance(example_oh_f32, np.ndarray) and example_oh_f32.dtype == np.float32
        example_oh_f64 = adata_f64.uns["unique_examples"]["oh"]
        assert isinstance(example_oh_f64, np.ndarray) and example_oh_f64.dtype == np.float64

    def test_create_seqlet_adata_minimal_required_params(self):
        """Test that function works with minimal required parameters."""
        n_seqlets, n_motifs = 3, 2
        similarity_matrix = np.random.rand(n_seqlets, n_motifs)
        seqlet_metadata = pd.DataFrame({"example_idx": [0, 1, 0], "start": [10, 20, 30], "end": [25, 35, 45]})

        # Should work with just similarity matrix and metadata
        # Convert dense matrix to sparse for the function
        sparse_similarity_matrix = _make_sparse_similarity_matrix(similarity_matrix)
        adata = tm.pp.create_seqlet_adata(sparse_similarity_matrix, seqlet_metadata)

        assert isinstance(adata, AnnData)
        assert adata.shape == (n_seqlets, n_motifs)
        # Optional data should not be present
        assert "seqlet_matrix" not in adata.obs.columns
        assert "unique_examples" not in adata.uns
