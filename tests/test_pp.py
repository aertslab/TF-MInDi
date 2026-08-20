"""Tests for preprocessing functions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from scipy.sparse import csr_array

import tfmindi as tm
from tfmindi.pp.seqlets import _prepare_motifs_for_finemo, finemo_fit_contrib


def _make_sparse_similarity_matrix(dense_matrix):
    """Helper function to convert dense matrix to sparse with threshold."""
    # Apply same threshold as in the actual function
    dense_matrix[dense_matrix < 0.05] = 0
    return sparse.csr_array(dense_matrix)


def _consensus_pfm(consensus, sharp=0.97, background=0.01):
    """Build a PFM of shape (4, len(consensus)) that is sharply informative at every position."""
    width = len(consensus)
    pfm = np.full((4, width), background)
    for j, base in enumerate(consensus):
        pfm[base, j] = sharp
    return pfm / pfm.sum(axis=0, keepdims=True)


def _embed_motif(consensus, n=3, length=100, insert_pos=40, seed=0):
    """Build synthetic (contrib, oh) with `consensus` embedded at `insert_pos` in every example."""
    rng = np.random.default_rng(seed)
    motif_len = len(consensus)
    oh = np.zeros((n, 4, length))
    contrib = np.zeros((n, 4, length))
    for i in range(n):
        idx = rng.integers(0, 4, length)
        oh[i, idx, np.arange(length)] = 1
        for j, base in enumerate(consensus):
            oh[i, :, insert_pos + j] = 0
            oh[i, base, insert_pos + j] = 1
        contrib[i] = rng.normal(scale=0.01, size=(4, length))
        for j, base in enumerate(consensus):
            contrib[i, base, insert_pos + j] = 2.0
    return contrib, oh, motif_len


class TestExtractSeqlets:
    """Test extract_seqlets function."""

    def test_extract_seqlets_real_data(self, sample_contrib_data, sample_oh_data):
        """Test extract_seqlets with the default (recursive_q99_abs_smooth) method."""
        seqlet_df, seqlet_matrices = tm.pp.extract_seqlets(sample_contrib_data, sample_oh_data)

        # Default method is recursive_q99_abs_smooth.
        assert len(seqlet_df) == len(seqlet_matrices) == 262

        assert isinstance(seqlet_df, pd.DataFrame)
        assert isinstance(seqlet_matrices, list)
        assert list(seqlet_df.columns) == ["example_idx", "start", "end", "attribution", "score"]

        assert np.all(seqlet_df["start"] < seqlet_df["end"])
        assert np.all(seqlet_df["start"] >= 0)

        # check that all values in seqlet matrices are between -1 and 1
        for matrix in seqlet_matrices:
            assert np.all(matrix >= -1) and np.all(matrix <= 1)

    def test_extract_seqlets_recursive_raw_reproduces_default(self, sample_contrib_data, sample_oh_data):
        """method='recursive_raw' reproduces the pre-change recursive-on-raw-track behaviour."""
        seqlet_df, seqlet_matrices = tm.pp.extract_seqlets(sample_contrib_data, sample_oh_data, method="recursive_raw")
        assert len(seqlet_df) == len(seqlet_matrices) == 227

    @pytest.mark.parametrize(
        "method",
        ["recursive_q99_abs_smooth", "recursive_raw", "hysteresis", "local_contrast", "wavelet_otsu"],
    )
    def test_extract_seqlets_all_methods(self, sample_contrib_data, sample_oh_data, method):
        """Every selectable caller returns a valid seqlet DataFrame + matrices."""
        seqlet_df, seqlet_matrices = tm.pp.extract_seqlets(sample_contrib_data, sample_oh_data, method=method)

        assert isinstance(seqlet_df, pd.DataFrame)
        assert list(seqlet_df.columns) == ["example_idx", "start", "end", "attribution", "score"]
        assert len(seqlet_matrices) == len(seqlet_df)
        assert len(seqlet_df) > 0

        assert np.all(seqlet_df["start"] < seqlet_df["end"])
        assert np.all(seqlet_df["start"] >= 0)
        for matrix in seqlet_matrices:
            assert matrix.shape[0] == 4
            assert np.all(matrix >= -1) and np.all(matrix <= 1)

    def test_extract_seqlets_invalid_method(self, sample_contrib_data, sample_oh_data):
        """An unknown method name raises ValueError."""
        with pytest.raises(ValueError, match="method must be one of"):
            tm.pp.extract_seqlets(sample_contrib_data, sample_oh_data, method="not_a_method")

    def test_extract_seqlets_method_kwargs(self, sample_contrib_data, sample_oh_data):
        """method_kwargs are forwarded to the selected caller."""
        base, _ = tm.pp.extract_seqlets(sample_contrib_data, sample_oh_data, method="hysteresis")
        tuned, _ = tm.pp.extract_seqlets(sample_contrib_data, sample_oh_data, method="hysteresis", seed_z=4.0)
        # A stricter seed threshold should not increase the number of calls.
        assert len(tuned) <= len(base)

    def test_extract_seqlets_unknown_kwarg_raises(self, sample_contrib_data, sample_oh_data):
        """Passing a kwarg the chosen caller does not accept raises TypeError."""
        with pytest.raises(TypeError):
            tm.pp.extract_seqlets(sample_contrib_data, sample_oh_data, method="hysteresis", threshold=0.1)

    def test_extract_seqlets_finemo_requires_motifs(self, sample_contrib_data, sample_oh_data):
        """method='finemo_fit_contrib' without `motifs` raises a clear ValueError."""
        with pytest.raises(ValueError, match="requires `motifs`"):
            tm.pp.extract_seqlets(sample_contrib_data, sample_oh_data, method="finemo_fit_contrib")


class TestPrepareMotifsForFinemo:
    """Test the _prepare_motifs_for_finemo helper."""

    def test_variant_count_and_shapes(self):
        """Default include_rc/include_neg expand each motif into 4 variants, padded to the widest motif."""
        motifs = {
            "m1": _consensus_pfm([0, 1, 2, 3, 0, 1]),  # width 6
            "m2": _consensus_pfm([2, 3, 0, 1, 2, 3, 0, 1, 2]),  # width 9
        }
        motif_data, icms, trim_masks = _prepare_motifs_for_finemo(motifs)

        assert len(motif_data) == icms.shape[0] == trim_masks.shape[0] == 2 * 2 * 2
        assert icms.shape == (8, 4, 9)
        assert trim_masks.shape == (8, 9)
        assert not np.isnan(icms).any()

        assert [md.motif_name for md in motif_data] == ["m1"] * 4 + ["m2"] * 4
        assert [md.strand for md in motif_data] == ["+", "+", "-", "-"] * 2
        assert [md.sign for md in motif_data] == [1, -1, 1, -1] * 2

    def test_no_rc_no_neg_single_variant_per_motif(self):
        """include_rc=False, include_neg=False keeps exactly one (forward, positive) variant per motif."""
        motifs = {"m1": _consensus_pfm([0, 1, 2, 3]), "m2": _consensus_pfm([1, 2, 3, 0])}
        motif_data, icms, trim_masks = _prepare_motifs_for_finemo(motifs, include_rc=False, include_neg=False)

        assert len(motif_data) == icms.shape[0] == trim_masks.shape[0] == 2
        assert all(md.strand == "+" and md.sign == 1 for md in motif_data)

    def test_sign_negation_flips_values_only(self):
        """The sign=-1 variant is the exact negation of sign=1, with identical trim coordinates."""
        motifs = {"m1": _consensus_pfm([0, 1, 2, 3])}
        motif_data, icms, trim_masks = _prepare_motifs_for_finemo(motifs, include_rc=False, include_neg=True)

        assert motif_data[0].sign == 1
        assert motif_data[1].sign == -1
        np.testing.assert_allclose(icms[1].astype(np.float64), -icms[0].astype(np.float64), atol=1e-2)
        assert motif_data[0].motif_start == motif_data[1].motif_start
        assert motif_data[0].motif_end == motif_data[1].motif_end
        np.testing.assert_array_equal(trim_masks[0], trim_masks[1])

    def test_reverse_complement_matches_flip(self):
        """The '-' strand variant is the reverse-complement (flip both axes) of the '+' strand."""
        motifs = {"m1": _consensus_pfm([0, 1, 2, 3, 0, 1])}
        motif_data, icms, trim_masks = _prepare_motifs_for_finemo(motifs, include_rc=True, include_neg=False)

        assert motif_data[0].strand == "+"
        assert motif_data[1].strand == "-"
        fwd = icms[0].astype(np.float64)
        rev = icms[1].astype(np.float64)
        np.testing.assert_allclose(rev, fwd[::-1, ::-1], atol=1e-2)

        width = icms.shape[2]
        assert motif_data[1].motif_start == width - motif_data[0].motif_end
        assert motif_data[1].motif_end == width - motif_data[0].motif_start

    def test_ic_trim_threshold_excludes_low_ic_flanks(self):
        """Uniform (background-like) flanking positions are trimmed from [motif_start, motif_end)."""
        core = [0, 1, 2, 3, 0]
        flank = 3
        width = flank + len(core) + flank
        pfm = np.full((4, width), 0.25)
        for j, base in enumerate(core):
            pfm[:, flank + j] = 0.01
            pfm[base, flank + j] = 0.97
        pfm = pfm / pfm.sum(axis=0, keepdims=True)

        motif_data, _, trim_masks = _prepare_motifs_for_finemo(
            {"m1": pfm}, include_rc=False, include_neg=False, ic_trim_threshold=0.2
        )

        md = motif_data[0]
        assert md.motif_start == flank
        assert md.motif_end == flank + len(core)
        assert trim_masks[0, :flank].sum() == 0
        assert trim_masks[0, flank : flank + len(core)].sum() == len(core)
        assert trim_masks[0, flank + len(core) :].sum() == 0

    def test_padding_symmetric_and_trim_mask_excludes_padding(self):
        """Motifs narrower than max_width are zero-padded symmetrically; padding is excluded from the trim mask."""
        short = _consensus_pfm([0, 1, 2, 3])  # width 4
        long_ = _consensus_pfm([0, 1, 2, 3, 0, 1, 2])  # width 7
        motif_data, icms, trim_masks = _prepare_motifs_for_finemo(
            {"short": short, "long": long_}, include_rc=False, include_neg=False
        )

        max_width = 7
        assert icms.shape[2] == max_width
        short_md = motif_data[0]
        # pad_left = (7 - 4) // 2 = 1, pad_right = 2
        assert short_md.motif_start == 1
        assert short_md.motif_end == 5
        assert trim_masks[0, 0] == 0
        np.testing.assert_array_equal(trim_masks[0, 1:5], 1)
        assert trim_masks[0, 5] == 0
        assert trim_masks[0, 6] == 0

    def test_invalid_background_length_raises(self):
        """A background vector that isn't length-4 raises ValueError."""
        motifs = {"m1": _consensus_pfm([0, 1, 2, 3])}
        with pytest.raises(ValueError, match="Background need a length of 4"):
            _prepare_motifs_for_finemo(motifs, background=(0.25, 0.25, 0.25))

    def test_icm_forward_unit_l2_norm(self):
        """The forward, unsigned icm is normalized to unit L2 norm (matching TF-MoDISco hcwm scaling)."""
        motifs = {"m1": _consensus_pfm([0, 1, 2, 3, 0, 1])}
        _, icms, _ = _prepare_motifs_for_finemo(motifs, include_rc=False, include_neg=False)
        norm = np.sqrt((icms[0].astype(np.float64) ** 2).sum())
        assert norm == pytest.approx(1.0, abs=0.05)


class TestFinemoFitContrib:
    """Test the finemo_fit_contrib seqlet caller."""

    @pytest.fixture(autouse=True)
    def _require_finemo(self):
        pytest.importorskip("finemo")

    def test_tuple_motif_keys_report_the_motif_name(self):
        """``(file_name, motif_name)`` keys, as MotifCollectionData yields, report the name alone."""
        consensus = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        contrib, oh, _ = _embed_motif(consensus, n=3, insert_pos=40)
        motifs = {("collection.meme", "M03434_2.00"): _consensus_pfm(consensus)}

        df = finemo_fit_contrib()(contrib, oh, motifs, compile_optimizer=False)

        assert (df["finemo_hit_motif_names"] == "M03434_2.00").all()

    def test_finds_embedded_motif(self):
        """A motif planted at a known position is recovered with the right coordinates and name."""
        consensus = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        insert_pos = 40
        contrib, oh, motif_len = _embed_motif(consensus, n=3, insert_pos=insert_pos)
        motifs = {"m1": _consensus_pfm(consensus)}

        caller = finemo_fit_contrib()
        df = caller(contrib, oh, motifs, compile_optimizer=False)

        assert list(df.columns) == [
            "example_idx",
            "start",
            "end",
            "attribution",
            "score",
            "finemo_hit_coefficients",
            "finemo_hit_motif_names",
        ]
        assert len(df) == 3
        assert (df["example_idx"].to_numpy() == np.arange(3)).all()
        assert (df["start"] == insert_pos).all()
        assert (df["end"] == insert_pos + motif_len).all()
        assert (df["finemo_hit_motif_names"] == "m1").all()
        assert (df["attribution"] > 0).all()

    def test_no_hits_on_pure_noise(self):
        """Random noise with no motif signal yields an empty (but correctly-shaped) DataFrame."""
        rng = np.random.default_rng(1)
        n, length = 3, 60
        oh = np.zeros((n, 4, length))
        for i in range(n):
            idx = rng.integers(0, 4, length)
            oh[i, idx, np.arange(length)] = 1
        contrib = rng.normal(scale=0.01, size=(n, 4, length)) * oh
        motifs = {"m1": _consensus_pfm([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])}

        caller = finemo_fit_contrib()
        df = caller(contrib, oh, motifs, compile_optimizer=False)

        assert list(df.columns) == [
            "example_idx",
            "start",
            "end",
            "attribution",
            "score",
            "finemo_hit_coefficients",
            "finemo_hit_motif_names",
        ]
        assert len(df) == 0

    def test_via_extract_seqlets(self):
        """extract_seqlets(method='finemo_fit_contrib', motifs=...) dispatches (contrib, oh, motifs) correctly."""
        consensus = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        insert_pos = 40
        contrib, oh, motif_len = _embed_motif(consensus, n=3, insert_pos=insert_pos)
        motifs = {"m1": _consensus_pfm(consensus)}

        seqlets_df, seqlet_matrices = tm.pp.extract_seqlets(
            contrib, oh, method="finemo_fit_contrib", motifs=motifs, compile_optimizer=False
        )

        assert len(seqlets_df) == len(seqlet_matrices) == 3
        for matrix in seqlet_matrices:
            assert matrix.shape == (4, motif_len)
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
                "score": [1e-5, 1e-4, 1e-6, 1e-3, 1e-4],
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

        # Per-seqlet arrays are derived from unique_examples, not stored in .obs
        assert "seqlet_matrix" not in adata.obs.columns
        assert "seqlet_oh" not in adata.obs.columns
        assert all(tm.pp.seqlets.get_seqlet_matrix(adata, i).shape[0] == 4 for i in range(n_seqlets))
        assert all(tm.pp.seqlets.get_seqlet_oh(adata, i).shape[0] == 4 for i in range(n_seqlets))

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
            expected_oh = (oh_sequences[ex_idx] > 0).astype(np.uint8)
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
        expected_cols = ["example_idx", "start", "end", "attribution", "score"]
        assert all(col in adata.obs.columns for col in expected_cols)

        # Per-seqlet arrays are derived from unique_examples, not stored in .obs
        assert "seqlet_matrix" not in adata.obs.columns
        assert "seqlet_oh" not in adata.obs.columns

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
            expected_oh = (oh_subset[ex_idx] > 0).astype(np.uint8)
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
            tm.pp.create_seqlet_adata(similarity_matrix, seqlet_metadata)

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

        original_oh_u8 = (oh_sequences > 0).astype(np.uint8)
        original_contrib_f32 = contrib_scores.astype(np.float32)

        # Check that we get the same results as direct conversion using helper functions
        for i in range(n_seqlets):
            ex_idx = seqlet_metadata.iloc[i]["example_idx"]
            retrieved_oh = tm.pp.seqlets.get_example_oh(adata, i)
            retrieved_contrib = tm.pp.seqlets.get_example_contrib(adata, i)
            np.testing.assert_array_equal(
                retrieved_oh, original_oh_u8[ex_idx], err_msg=f"Example OH data mismatch for seqlet {i}"
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
        adata_f64 = tm.pp.create_seqlet_adata(similarity_matrix, seqlet_metadata, dtype=np.float64)

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
        oh_sequences = np.random.randint(0, 2, size=(5, 4, 500)).astype(np.float64)  # 5 examples
        contrib_scores = np.random.rand(5, 4, 500).astype(np.float64)
        motif_collection = {
            (f"motif_{i}", f"motif_{i}"): np.random.rand(4, 8).astype(np.float64) for i in range(n_motifs)
        }

        # Create AnnData with float32 (optimized)
        adata_f32 = tm.pp.create_seqlet_adata(
            similarity_matrix,
            seqlet_metadata,
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
        # One-hot ignores `dtype` and is always uint8; `dtype` governs contributions and PPMs.
        for adata in (adata_f32, adata_f64):
            assert adata.uns["unique_examples"]["oh"].dtype == np.uint8
        assert adata_f32.uns["unique_examples"]["contrib"].dtype == np.float32
        assert adata_f64.uns["unique_examples"]["contrib"].dtype == np.float64

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


class _StubCollection:
    """Minimal stand-in for MotifCollectionData: only the two accessors used for projection."""

    def __init__(self, motif_names, pcs):
        """Store the reference motif order and the PC loadings aligned to it."""
        self._names = list(motif_names)
        self._pcs = np.asarray(pcs, dtype=np.float64)

    def get_motif_names(self, n_motifs_per_cluster):
        """Return the reference motif names, in reference order."""
        return self._names

    def get_pca_data(self, n_motifs_per_cluster):
        """Return an object exposing `.pcs`, as the real accessor does."""
        return SimpleNamespace(pcs=self._pcs)


class TestStreamedReferenceProjection:
    """Test projecting into a reference space while the full profile is still in memory."""

    @staticmethod
    def _inputs():
        """Build tiny seqlets and a name-keyed motif dict."""
        rng = np.random.default_rng(0)
        seqlets = [rng.random((4, 8)) for _ in range(6)]
        motifs = {f"m{i}": rng.random((4, 6)) for i in range(5)}
        return seqlets, motifs

    def test_matches_projecting_the_unpruned_matrix(self):
        """The streamed projection must equal projecting the full matrix afterwards."""
        seqlets, motifs = self._inputs()
        # Reference order deliberately differs from the motif dict order, which is the case
        # for the real collection and the thing the column permutation has to get right.
        ref_names = ["m3", "m0", "m4", "m1", "m2"]
        pcs = np.random.default_rng(1).random((len(ref_names), 3))
        collection = _StubCollection(ref_names, pcs)

        full = tm.pp.calculate_motif_similarity(seqlets, motifs, chunk_size=2)
        _, streamed = tm.pp.calculate_motif_similarity(
            seqlets, motifs, chunk_size=2, n_nearest=2, reference=collection, n_motifs_per_reference_cluster=20
        )

        # Reproduce what tl.project does to the full matrix, restricted to the reference columns.
        order = [list(motifs).index(name) for name in ref_names]
        dense = full.toarray()[:, order]
        expected = dense @ pcs - np.outer(dense.mean(axis=1), pcs.sum(axis=0))
        np.testing.assert_allclose(streamed, expected, rtol=1e-5, atol=1e-4)

    def test_prunes_the_stored_matrix_but_not_the_projection(self):
        """n_nearest must bound what is stored without reaching the projection."""
        seqlets, motifs = self._inputs()
        ref_names = list(motifs)
        collection = _StubCollection(ref_names, np.random.default_rng(1).random((len(ref_names), 3)))

        sim, streamed = tm.pp.calculate_motif_similarity(
            seqlets, motifs, chunk_size=3, n_nearest=2, reference=collection, n_motifs_per_reference_cluster=20
        )
        assert np.all(np.diff(sim.tocsr().indptr) <= 2)
        assert streamed.shape == (len(seqlets), 3)

    def test_reference_alone_leaves_the_matrix_unchanged(self):
        """Without n_nearest the stored matrix must match the plain call exactly."""
        seqlets, motifs = self._inputs()
        collection = _StubCollection(list(motifs), np.random.default_rng(1).random((len(motifs), 3)))

        plain = tm.pp.calculate_motif_similarity(seqlets, motifs, chunk_size=2).tocsr()
        with_ref, _ = tm.pp.calculate_motif_similarity(
            seqlets, motifs, chunk_size=2, reference=collection, n_motifs_per_reference_cluster=20
        )
        with_ref = with_ref.tocsr()
        np.testing.assert_array_equal(plain.indices, with_ref.indices)
        np.testing.assert_array_equal(plain.data, with_ref.data)

    def test_requires_a_reference_budget(self):
        """The budget selects the PCA embedding, so it cannot be inferred."""
        seqlets, motifs = self._inputs()
        with pytest.raises(ValueError, match="n_motifs_per_reference_cluster is required"):
            tm.pp.calculate_motif_similarity(seqlets, motifs, reference=_StubCollection([], np.zeros((0, 2))))

    def test_requires_named_motifs(self):
        """Reference columns are matched by name, which a bare list cannot supply."""
        seqlets, motifs = self._inputs()
        with pytest.raises(ValueError, match="must be a dict"):
            tm.pp.calculate_motif_similarity(
                seqlets,
                list(motifs.values()),
                reference=_StubCollection([], np.zeros((0, 2))),
                n_motifs_per_reference_cluster=20,
            )

    def test_rejects_missing_reference_motifs(self):
        """A collection asking for motifs that were never scored cannot be aligned."""
        seqlets, motifs = self._inputs()
        collection = _StubCollection(["m0", "absent"], np.zeros((2, 3)))
        with pytest.raises(ValueError, match="reference motifs are absent"):
            tm.pp.calculate_motif_similarity(seqlets, motifs, reference=collection, n_motifs_per_reference_cluster=20)
