"""Tests for plotting functions."""

import numpy as np
import pandas as pd
from anndata import AnnData

from tfmindi.pl._utils import get_point_colors


class TestColorUtils:
    """Test color utility functions."""

    def test_get_point_colors_with_nan_categorical(self):
        """Test that get_point_colors handles NaN values in categorical data without KeyError."""
        # Create test data with NaN values in categorical column
        adata = AnnData(X=np.random.rand(10, 5))

        # Create categorical data with some NaN values
        categories = ["Type1", "Type2", "Type3"]
        values = ["Type1", "Type2", np.nan, "Type1", np.nan, "Type3", "Type2", np.nan, "Type1", "Type2"]
        adata.obs["test_category"] = pd.Categorical(values, categories=categories)

        # This should not raise a KeyError
        point_colors, color_map = get_point_colors(adata, "test_category")

        # Verify that all point colors are valid (no NaN in the color list)
        assert len(point_colors) == 10
        assert all(color is not None for color in point_colors)

        # Verify that "Unknown" is in the color map
        assert "Unknown" in color_map
        assert color_map["Unknown"] == "#D3D3D3"  # lightgray as defined in ensure_colors

        # Verify colors are stored in scanpy format
        assert "test_category_colors" in adata.uns
        assert "Unknown" in adata.uns["test_category_colors"]

    def test_get_point_colors_with_nan_object_dtype(self):
        """Test that get_point_colors handles NaN values in object dtype columns."""
        # Create test data with NaN values in object column
        adata = AnnData(X=np.random.rand(8, 5))

        # Create object data with some NaN values
        values = ["DBD1", "DBD2", np.nan, "DBD1", np.nan, "DBD3", "DBD2", np.nan]
        adata.obs["test_dbd"] = values

        # This should not raise a KeyError
        point_colors, color_map = get_point_colors(adata, "test_dbd")

        # Verify that all point colors are valid
        assert len(point_colors) == 8
        assert all(color is not None for color in point_colors)

        # Verify that "Unknown" is in the color map
        assert "Unknown" in color_map

        # Verify colors are stored in scanpy format
        assert "test_dbd_colors" in adata.uns

    def test_get_point_colors_with_nan_stored_colors(self):
        """Test NaN handling when using stored colors."""
        # Create test data
        adata = AnnData(X=np.random.rand(6, 5))

        # Create categorical data with NaN values
        categories = ["A", "B", "C"]
        values = ["A", "B", np.nan, "A", np.nan, "C"]
        adata.obs["test_col"] = pd.Categorical(values, categories=categories)

        # Test with stored colors (use_stored_colors=True)
        point_colors, color_map = get_point_colors(adata, "test_col", use_stored_colors=True)

        # Should work without KeyError
        assert len(point_colors) == 6
        assert "Unknown" in color_map

        # Verify colors are stored in scanpy format
        assert "test_col_colors" in adata.uns

    def test_get_point_colors_with_nan_no_stored_colors(self):
        """Test NaN handling when not using stored colors."""
        # Create test data
        adata = AnnData(X=np.random.rand(6, 5))

        # Create categorical data with NaN values
        categories = ["X", "Y", "Z"]
        values = ["X", "Y", np.nan, "X", np.nan, "Z"]
        adata.obs["test_col"] = pd.Categorical(values, categories=categories)

        # Test without stored colors (use_stored_colors=False)
        point_colors, color_map = get_point_colors(adata, "test_col", use_stored_colors=False)

        # Should work without KeyError
        assert len(point_colors) == 6
        assert "Unknown" in color_map
        assert color_map["Unknown"] == "lightgray"  # As defined in the non-stored path
