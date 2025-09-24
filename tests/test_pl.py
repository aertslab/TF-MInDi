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

    def test_colormap_fallback_behavior(self):
        """Test that correct colormaps are used based on number of categories."""
        from tfmindi.pl._utils import ensure_colors

        # Test with small number of categories (should use tab10)
        adata_small = AnnData(X=np.random.rand(10, 5))
        categories_small = [f"Cat{i}" for i in range(8)]  # 8 categories
        adata_small.obs["test_col"] = pd.Categorical(
            np.random.choice(categories_small, 10), categories=categories_small
        )

        colors_small = ensure_colors(adata_small, "test_col", cmap="tab10")
        assert len(colors_small) == 8
        # Should be able to use tab10 directly

        # Test with medium number of categories (should upgrade to tab20)
        adata_medium = AnnData(X=np.random.rand(25, 5))
        categories_medium = [f"Cat{i}" for i in range(15)]  # 15 categories
        adata_medium.obs["test_col"] = pd.Categorical(
            np.random.choice(categories_medium, 25), categories=categories_medium
        )

        colors_medium = ensure_colors(adata_medium, "test_col", cmap="tab10")
        assert len(colors_medium) == 15
        # Should upgrade to tab20

        # Test with many categories (should use random colors)
        adata_large = AnnData(X=np.random.rand(50, 5))
        categories_large = [f"Cat{i}" for i in range(25)]  # 25 categories
        adata_large.obs["test_col"] = pd.Categorical(
            np.random.choice(categories_large, 50), categories=categories_large
        )

        colors_large = ensure_colors(adata_large, "test_col", cmap="tab10")
        assert len(colors_large) == 25
        # Should use random colors - all should be valid hex colors
        for color in colors_large.values():
            if color != "#D3D3D3":  # Skip "Unknown" color
                assert color.startswith("#")
                assert len(color) == 7
                # Should be a valid hex color
                int(color[1:], 16)

        # Test with tab20 starting point
        adata_tab20 = AnnData(X=np.random.rand(30, 5))
        categories_tab20 = [f"Cat{i}" for i in range(25)]  # 25 categories
        adata_tab20.obs["test_col"] = pd.Categorical(
            np.random.choice(categories_tab20, 30), categories=categories_tab20
        )

        colors_tab20 = ensure_colors(adata_tab20, "test_col", cmap="tab20")
        assert len(colors_tab20) == 25
        # Should use random colors since >20 categories
        for color in colors_tab20.values():
            if color != "#D3D3D3":
                assert color.startswith("#")
                assert len(color) == 7
                int(color[1:], 16)

    def test_dbd_colormap_regeneration_on_reclustering(self):
        """Test that DBD colormap gets updated when new DBDs appear after reclustering."""
        from tfmindi.pl._utils import ensure_colors

        # Create initial test data with specific DBDs
        adata = AnnData(X=np.random.rand(30, 10))
        adata.obs["cluster_dbd"] = pd.Categorical(["ETS", "bZIP", "Homeodomain"] * 10)

        initial_colors = ensure_colors(adata, "cluster_dbd", cmap="tab10")
        initial_dbd_set = set(initial_colors.keys())

        print(f"Initial DBDs: {initial_dbd_set}")
        assert initial_dbd_set == {"ETS", "bZIP", "Homeodomain"}

        # Simulate reclustering that introduces new DBD types
        new_dbds = ["ETS", "bZIP", "Homeodomain", "STAT", "Nuclear receptor"] * 6
        adata.obs["cluster_dbd"] = pd.Categorical(new_dbds)

        # This should detect new DBDs and update the colormap
        updated_colors = ensure_colors(adata, "cluster_dbd", cmap="tab10")
        updated_dbd_set = set(updated_colors.keys())

        print(f"Updated DBDs: {updated_dbd_set}")
        print(f"Current data DBDs: {set(adata.obs['cluster_dbd'].unique())}")

        # All DBDs present in the data should have colors
        data_dbds = set(adata.obs["cluster_dbd"].unique())
        missing_colors = data_dbds - updated_dbd_set

        if missing_colors:
            print(f"Missing colors for: {missing_colors}")

        assert missing_colors == set(), f"Missing colors for DBDs: {missing_colors}"

        assert "STAT" in updated_colors, "STAT should have a color assigned"
        assert "Nuclear receptor" in updated_colors, "Nuclear receptor should have a color assigned"
        assert "ETS" in updated_colors, "ETS should still have a color assigned"
