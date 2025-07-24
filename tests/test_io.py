"""Tests for input/output functions."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import tfmindi as tm


@pytest.fixture
def sample_adata_with_arrays():
    """Create sample AnnData with numpy arrays in .obs for testing."""
    # Create basic AnnData
    X = np.random.rand(10, 5)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(10)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(5)])
    adata = AnnData(X=X, obs=obs, var=var)

    # Add numpy arrays to obs (similar to seqlet data)
    seqlet_matrices = [np.random.rand(4, np.random.randint(10, 20)) for _ in range(10)]
    seqlet_oh = [np.random.randint(0, 2, (4, np.random.randint(10, 20))) for _ in range(10)]
    example_oh = [np.random.randint(0, 2, (4, 1000)) for _ in range(10)]

    adata.obs["seqlet_matrix"] = seqlet_matrices
    adata.obs["seqlet_oh"] = seqlet_oh
    adata.obs["example_oh"] = example_oh

    # Add regular columns
    adata.obs["cluster"] = [f"cluster_{i % 3}" for i in range(10)]
    adata.obs["score"] = np.random.rand(10)

    # Add numpy arrays to var (similar to motif data)
    motif_pwms = [np.random.rand(4, np.random.randint(8, 15)) for _ in range(5)]
    adata.var["motif_pwm"] = motif_pwms

    return adata


def test_save_h5ad_basic(sample_adata_with_arrays):
    """Test basic save_h5ad functionality."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "test.h5ad"

        # Should not raise an error
        tm.save_h5ad(sample_adata_with_arrays, filepath)

        # File should exist
        assert filepath.exists()


def test_save_load_h5ad_roundtrip(sample_adata_with_arrays):
    """Test that save and load preserves numpy arrays correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "test.h5ad"

        # Save
        tm.save_h5ad(sample_adata_with_arrays, filepath)

        # Load
        loaded_adata = tm.load_h5ad(filepath)

        # Check basic structure
        assert loaded_adata.shape == sample_adata_with_arrays.shape
        assert loaded_adata.obs.shape == sample_adata_with_arrays.obs.shape
        assert loaded_adata.var.shape == sample_adata_with_arrays.var.shape

        # Check that numpy array columns are preserved in obs
        assert "seqlet_matrix" in loaded_adata.obs.columns
        assert "seqlet_oh" in loaded_adata.obs.columns
        assert "example_oh" in loaded_adata.obs.columns

        # Check that numpy array columns are preserved in var
        assert "motif_pwm" in loaded_adata.var.columns

        # Check that arrays are actually numpy arrays
        assert isinstance(loaded_adata.obs["seqlet_matrix"].iloc[0], np.ndarray)
        assert isinstance(loaded_adata.obs["seqlet_oh"].iloc[0], np.ndarray)
        assert isinstance(loaded_adata.obs["example_oh"].iloc[0], np.ndarray)
        assert isinstance(loaded_adata.var["motif_pwm"].iloc[0], np.ndarray)

        # Check array contents are preserved in obs
        for i in range(len(sample_adata_with_arrays)):
            np.testing.assert_array_equal(
                loaded_adata.obs["seqlet_matrix"].iloc[i], sample_adata_with_arrays.obs["seqlet_matrix"].iloc[i]
            )
            np.testing.assert_array_equal(
                loaded_adata.obs["seqlet_oh"].iloc[i], sample_adata_with_arrays.obs["seqlet_oh"].iloc[i]
            )
            np.testing.assert_array_equal(
                loaded_adata.obs["example_oh"].iloc[i], sample_adata_with_arrays.obs["example_oh"].iloc[i]
            )

        # Check array contents are preserved in var
        for i in range(len(sample_adata_with_arrays.var)):
            np.testing.assert_array_equal(
                loaded_adata.var["motif_pwm"].iloc[i], sample_adata_with_arrays.var["motif_pwm"].iloc[i]
            )

        # Check that regular columns are preserved
        assert "cluster" in loaded_adata.obs.columns
        assert "score" in loaded_adata.obs.columns
        # Check values are preserved (dtype might change for strings)
        assert loaded_adata.obs["cluster"].astype(str).tolist() == sample_adata_with_arrays.obs["cluster"].tolist()
        np.testing.assert_array_equal(loaded_adata.obs["score"], sample_adata_with_arrays.obs["score"])


def test_save_h5ad_no_arrays():
    """Test save_h5ad works with AnnData that has no numpy arrays in obs."""
    # Create AnnData without numpy arrays
    X = np.random.rand(5, 3)
    obs = pd.DataFrame({"cluster": ["A", "B", "A", "B", "A"]}, index=[f"cell_{i}" for i in range(5)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(3)])
    adata = AnnData(X=X, obs=obs, var=var)

    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "test.h5ad"

        # Should not raise an error
        tm.save_h5ad(adata, filepath)

        # Load and check
        loaded_adata = tm.load_h5ad(filepath)
        assert loaded_adata.shape == adata.shape
        # Note: string columns may be converted to categorical during save/load
        assert list(loaded_adata.obs.columns) == list(adata.obs.columns)
        # Check that the string values are the same even if dtype changed
        assert loaded_adata.obs["cluster"].astype(str).tolist() == adata.obs["cluster"].tolist()


def test_load_h5ad_standard_file():
    """Test that load_h5ad works with standard h5ad files (no numpy arrays)."""
    # Create and save with standard AnnData
    X = np.random.rand(5, 3)
    obs = pd.DataFrame({"cluster": ["A", "B", "A", "B", "A"]}, index=[f"cell_{i}" for i in range(5)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(3)])
    adata = AnnData(X=X, obs=obs, var=var)

    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "test.h5ad"

        # Save with standard AnnData
        adata.write_h5ad(filepath)

        # Load with our custom function
        loaded_adata = tm.load_h5ad(filepath)

        # Should work fine
        assert loaded_adata.shape == adata.shape
        # Check that values are preserved (dtype might change due to AnnData processing)
        assert loaded_adata.obs["cluster"].astype(str).tolist() == adata.obs["cluster"].tolist()


def test_save_h5ad_with_compression(sample_adata_with_arrays):
    """Test save_h5ad with compression options."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "test.h5ad"

        # Should not raise an error with compression
        tm.save_h5ad(sample_adata_with_arrays, filepath, compression="gzip")

        # Load and check arrays are preserved
        loaded_adata = tm.load_h5ad(filepath)
        assert isinstance(loaded_adata.obs["seqlet_matrix"].iloc[0], np.ndarray)
        assert isinstance(loaded_adata.var["motif_pwm"].iloc[0], np.ndarray)


def test_save_h5ad_preserves_original(sample_adata_with_arrays):
    """Test that save_h5ad doesn't modify the original AnnData object."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "test.h5ad"

        # Store original arrays for comparison
        original_obs_array = sample_adata_with_arrays.obs["seqlet_matrix"].iloc[0].copy()
        original_var_array = sample_adata_with_arrays.var["motif_pwm"].iloc[0].copy()

        # Save
        tm.save_h5ad(sample_adata_with_arrays, filepath)

        # Check that original is unchanged
        assert isinstance(sample_adata_with_arrays.obs["seqlet_matrix"].iloc[0], np.ndarray)
        assert isinstance(sample_adata_with_arrays.var["motif_pwm"].iloc[0], np.ndarray)
        np.testing.assert_array_equal(sample_adata_with_arrays.obs["seqlet_matrix"].iloc[0], original_obs_array)
        np.testing.assert_array_equal(sample_adata_with_arrays.var["motif_pwm"].iloc[0], original_var_array)
