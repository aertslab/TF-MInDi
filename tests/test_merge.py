"""Tests for merging functionality."""

import anndata
import numpy as np
import pandas as pd
import pytest


def test_merge_non_match_idx(sample_seqlet_adata: anndata.AnnData):
    import tfmindi as tm

    adata_a = sample_seqlet_adata
    adata_b = sample_seqlet_adata.copy()

    adata_c = tm.concat({"a": adata_a, "b": adata_b}, idx_match=False, index_unique="-")

    assert adata_c.shape[0] == adata_a.shape[0] + adata_b.shape[0]

    assert "unique_examples" in adata_c.uns.keys()
    assert "oh" in adata_c.uns["unique_examples"].keys()
    assert "contrib" in adata_c.uns["unique_examples"].keys()

    pd.testing.assert_frame_equal(adata_c.var, adata_a.var)

    assert (
        adata_c.uns["unique_examples"]["oh"].shape[0]
        == adata_a.uns["unique_examples"]["oh"].shape[0] + adata_b.uns["unique_examples"]["oh"].shape[0]
    )

    a_index = [f"{orig_idx}-a" for orig_idx in adata_a.obs_names]
    b_index = [f"{orig_idx}-b" for orig_idx in adata_b.obs_names]

    np.testing.assert_array_equal(
        adata_c.uns["unique_examples"]["oh"][adata_c.obs.loc[a_index, "example_oh_idx"]],
        adata_a.uns["unique_examples"]["oh"][adata_a.obs["example_oh_idx"]],
    )

    np.testing.assert_array_equal(
        adata_c.uns["unique_examples"]["oh"][adata_c.obs.loc[b_index, "example_oh_idx"]],
        adata_b.uns["unique_examples"]["oh"][adata_b.obs["example_oh_idx"]],
    )

    np.testing.assert_array_equal(
        adata_c.uns["unique_examples"]["contrib"][adata_c.obs.loc[a_index, "example_contrib_idx"]],
        adata_a.uns["unique_examples"]["contrib"][adata_a.obs["example_contrib_idx"]],
    )

    np.testing.assert_array_equal(
        adata_c.uns["unique_examples"]["contrib"][adata_c.obs.loc[b_index, "example_contrib_idx"]],
        adata_b.uns["unique_examples"]["contrib"][adata_b.obs["example_contrib_idx"]],
    )


def test_merge_match_idx(sample_seqlet_adata: anndata.AnnData):
    import tfmindi as tm

    # fake data with some overlap
    adata_a = sample_seqlet_adata[0:150].copy()
    adata_b = sample_seqlet_adata[100:250].copy()

    adata_c = tm.concat({"a": adata_a, "b": adata_b}, idx_match=True, index_unique="-")

    assert adata_c.shape[0] == adata_a.shape[0] + adata_b.shape[0]

    assert "unique_examples" in adata_c.uns.keys()
    assert "oh" in adata_c.uns["unique_examples"].keys()
    assert "contrib" in adata_c.uns["unique_examples"].keys()

    pd.testing.assert_frame_equal(adata_c.var, adata_a.var)

    assert adata_c.uns["unique_examples"]["oh"].shape[0] == adata_a.uns["unique_examples"]["oh"].shape[0]

    a_index = [f"{orig_idx}-a" for orig_idx in adata_a.obs_names]
    b_index = [f"{orig_idx}-b" for orig_idx in adata_b.obs_names]

    np.testing.assert_array_equal(
        adata_c.uns["unique_examples"]["oh"][adata_c.obs.loc[a_index, "example_oh_idx"]],
        adata_a.uns["unique_examples"]["oh"][adata_a.obs["example_oh_idx"]],
    )

    np.testing.assert_array_equal(
        adata_c.uns["unique_examples"]["oh"][adata_c.obs.loc[b_index, "example_oh_idx"]],
        adata_b.uns["unique_examples"]["oh"][adata_b.obs["example_oh_idx"]],
    )

    np.testing.assert_array_equal(
        adata_c.uns["unique_examples"]["contrib"][adata_c.obs.loc[a_index, "example_contrib_idx"]],
        adata_a.uns["unique_examples"]["contrib"][adata_a.obs["example_contrib_idx"]],
    )

    np.testing.assert_array_equal(
        adata_c.uns["unique_examples"]["contrib"][adata_c.obs.loc[b_index, "example_contrib_idx"]],
        adata_b.uns["unique_examples"]["contrib"][adata_b.obs["example_contrib_idx"]],
    )


def test_idx_match_with_mismatched_data(sample_seqlet_adata):
    """Test that ValueError is raised when idx_match=True but data differs"""
    import tfmindi as tm

    adata_a = sample_seqlet_adata.copy()
    adata_b = sample_seqlet_adata.copy()
    # Modify unique_examples to be different
    adata_b.uns["unique_examples"]["oh"] = np.zeros_like(adata_b.uns["unique_examples"]["oh"])

    with pytest.raises(ValueError, match="should be the same across adatas"):
        tm.concat([adata_a, adata_b], idx_match=True)


def test_invalid_index_unique_type():
    """Test that non-string index_unique raises ValueError"""
    import tfmindi as tm

    with pytest.raises(ValueError, match="index_unique should be a string"):
        tm.concat([...], index_unique=None)


def test_original_adatas_unchanged(sample_seqlet_adata):
    """Ensure original adatas are restored after concat with idx_match=False"""
    import tfmindi as tm

    adata_a = sample_seqlet_adata.copy()
    adata_b = sample_seqlet_adata.copy()
    # Store original values
    tm.concat([adata_a, adata_b], idx_match=False)

    pd.testing.assert_frame_equal(adata_a.obs, sample_seqlet_adata.obs)
    pd.testing.assert_frame_equal(adata_b.obs, sample_seqlet_adata.obs)
