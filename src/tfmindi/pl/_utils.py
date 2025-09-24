"""Utility functions for plotting."""

from __future__ import annotations

import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData


def _generate_random_colors(n_colors: int, seed: int = 42) -> list[str]:
    """
    Generate n distinct random hex colors.

    Parameters
    ----------
    n_colors
        Number of colors to generate
    seed
        Random seed for reproducibility

    Returns
    -------
    list[str]
        List of hex color strings
    """
    random.seed(seed)
    return [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(n_colors)]


def render_plot(
    fig: plt.Figure,
    width: int = 8,
    height: int = 8,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    supxlabel: str | None = None,
    supylabel: str | None = None,
    tight_rect: tuple | None = None,
    title_fontsize: int = 16,
    x_label_fontsize: int = 14,
    y_label_fontsize: int = 14,
    x_tick_fontsize: int = 12,
    y_tick_fontsize: int = 12,
    x_label_rotation: int = 0,
    y_label_rotation: int = 0,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
    **kwargs,
) -> None | plt.Figure:
    """
    Render a plot with customization options.

    Note
    ----
    This function should never be called directly. Rather, the other plotting functions call this function.

    Parameters
    ----------
    fig
        The figure object to render.
    width
        Width of the plot (inches).
    height
        Height of the plot (inches).
    title
        Title of the plot.
    xlabel
        Label for the X-axis.
    ylabel
        Label for the Y-axis.
    supxlabel
        Suplabel for the X-axis.
    supylabel
        Suplabel for the Y-axis.
    tight_rect
        Normalized coordinates in which subplots will fit.
    title_fontsize
        Font size for the title.
    x_label_fontsize
        Font size for the X-axis labels.
    y_label_fontsize
        Font size for the Y-axis labels.
    x_tick_fontsize
        Font size for the X-axis ticks.
    y_tick_fontsize
        Font size for the Y-axis ticks
    x_label_rotation
        Rotation of the X-axis labels in degrees.
    y_label_rotation
        Rotation of the Y-axis labels in degrees.
    show
        Whether to display the plot. Set this to False if you want to return the figure object to customize it further.
    save_path
        Optional path to save the figure. If None, the figure is displayed but not saved.
    dpi
        Resolution for saving figures.
    **kwargs
        Additional arguments passed to plt.savefig().

    Returns
    -------
    None or matplotlib.Figure
        Returns figure object if show=False and save_path=None, otherwise None.
    """
    fig.set_size_inches(width, height)

    if title:
        fig.suptitle(title, fontsize=title_fontsize)
    if supxlabel:
        fig.supxlabel(supxlabel)
    if supylabel:
        fig.supylabel(supylabel)

    for ax in fig.axes:
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=x_label_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=y_label_fontsize)

        for label in ax.get_xticklabels():
            label.set_rotation(x_label_rotation)
            label.set_fontsize(x_tick_fontsize)
        for label in ax.get_yticklabels():
            label.set_fontsize(y_tick_fontsize)
            label.set_rotation(y_label_rotation)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "This figure includes Axes that are not compatible with tight_layout")
        if tight_rect:
            fig.tight_layout(rect=tight_rect)
        else:
            fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, **kwargs)

    if show:
        plt.show()

    if not show and not save_path:
        return fig

    return None


def ensure_colors(
    adata: AnnData,
    column: str,
    cmap: str = "tab20",
    force_regenerate: bool = False,
) -> dict[str, str]:
    """
    Ensure colors exist for a categorical column, generating them if needed.

    Colors are stored in adata.uns[f'{column}_colors'] following scanpy conventions.

    Parameters
    ----------
    adata
        AnnData object containing the data
    column
        Column name in adata.obs to generate/retrieve colors for
    cmap
        Matplotlib colormap name to use for color generation
    force_regenerate
        If True, regenerate colors even if they already exist

    Returns
    -------
    dict[str, str]
        Dictionary mapping category values to hex color codes

    Raises
    ------
    ValueError
        If column doesn't exist in adata.obs
    """
    if column not in adata.obs.columns:
        raise ValueError(f"Column '{column}' not found in adata.obs")

    # Check if colors already exist and don't need regeneration
    color_key = f"{column}_colors"
    if color_key in adata.uns and not force_regenerate:
        existing_colors = adata.uns[color_key]

        # Check if current data has new categories not in existing colormap
        values = adata.obs[column]
        if values.dtype == "category":
            current_categories = set(values.cat.categories)
            # Add "Unknown" if there are NaN values
            if values.isnull().any():
                current_categories.add("Unknown")
        else:
            current_categories = set(values.dropna().unique())
            if values.isnull().any():
                current_categories.add("Unknown")

        existing_categories = set(existing_colors.keys())

        # If all current categories have colors, return existing colormap
        if current_categories.issubset(existing_categories):
            return existing_colors
        # Otherwise, we need to regenerate (continue to below)

    # Get unique values, handling NaN
    values = adata.obs[column]
    if values.dtype == "category":
        # For categorical data, include all categories even if not present
        unique_values = list(values.cat.categories)
        # Add "Unknown" if there are NaN values
        if values.isnull().any() and "Unknown" not in unique_values:
            unique_values.append("Unknown")
    else:
        # For non-categorical, get unique values and handle NaN
        unique_values = values.dropna().unique().tolist()
        if values.isnull().any():
            unique_values.append("Unknown")

    # Generate colors
    colormap = plt.get_cmap(cmap)
    if len(unique_values) <= colormap.N:
        # Use discrete colors from colormap
        colors = [colormap(i) for i in range(len(unique_values))]
    else:
        # Too many categories for the discrete colormap - use fallback chain
        if cmap == "tab10":
            # Upgrade to tab20 if more than 10 categories
            colormap = plt.get_cmap("tab20")
            if len(unique_values) <= 20:
                colors = [colormap(i) for i in range(len(unique_values))]
            else:
                # More than 20 categories - use random colors
                hex_colors = _generate_random_colors(len(unique_values))
                colors = [plt.matplotlib.colors.to_rgba(color) for color in hex_colors]
        elif cmap == "tab20" and len(unique_values) > 20:
            # Use random colors for more than 20 categories
            hex_colors = _generate_random_colors(len(unique_values))
            colors = [plt.matplotlib.colors.to_rgba(color) for color in hex_colors]
        else:
            # For other colormaps, use them as continuous
            colormap = plt.get_cmap(cmap)
            colors = colormap(np.linspace(0, 1, len(unique_values)))

    # Convert to hex colors
    hex_colors = [plt.matplotlib.colors.to_hex(color) for color in colors]

    # Create color mapping
    color_map = dict(zip(unique_values, hex_colors, strict=False))
    # Special handling for "Unknown" - always use light gray
    if "Unknown" in color_map:
        color_map["Unknown"] = "#D3D3D3"  # lightgray

    # Store in AnnData using scanpy convention
    adata.uns[color_key] = color_map

    return color_map


def get_colors(
    adata: AnnData,
    column: str,
    cmap: str = "tab20",
) -> dict[str, str]:
    """
    Get colors for a categorical column, generating them if they don't exist.

    Parameters
    ----------
    adata
        AnnData object containing the data
    column
        Column name in adata.obs to get colors for
    cmap
        Matplotlib colormap name to use if colors need to be generated

    Returns
    -------
    dict[str, str]
        Dictionary mapping category values to hex color codes
    """
    return ensure_colors(adata, column, cmap, force_regenerate=False)


def set_colors(
    adata: AnnData,
    column: str,
    color_dict: dict[str, str],
) -> None:
    """
    Manually set colors for a categorical column.

    Parameters
    ----------
    adata
        AnnData object to store colors in
    column
        Column name in adata.obs to set colors for
    color_dict
        Dictionary mapping category values to color codes (hex, named, or RGB)
    """
    # Convert all colors to hex format
    hex_colors = {}
    for value, color in color_dict.items():
        try:
            hex_colors[value] = plt.matplotlib.colors.to_hex(color)
        except ValueError:
            # If conversion fails, keep original (might be a valid color name)
            hex_colors[value] = color

    # Store using scanpy convention
    adata.uns[f"{column}_colors"] = hex_colors


def reset_colors(
    adata: AnnData,
    column: str | None = None,
) -> None:
    """
    Reset stored colors, either for a specific column or all columns.

    Parameters
    ----------
    adata
        AnnData object containing stored colors
    column
        Column name to reset colors for. If None, reset all colors.
    """
    if column is None:
        # Reset all colors (find all keys ending with '_colors')
        color_keys = [key for key in adata.uns.keys() if key.endswith("_colors")]
        for key in color_keys:
            del adata.uns[key]
    else:
        # Reset specific column
        color_key = f"{column}_colors"
        if color_key in adata.uns:
            del adata.uns[color_key]


def get_point_colors(
    adata: AnnData,
    column: str,
    cmap: str = "tab20",
    use_stored_colors: bool = True,
) -> tuple[list[str], dict[str, str] | None]:
    """
    Get point colors for plotting based on a categorical column.

    Parameters
    ----------
    adata
        AnnData object containing the data
    column
        Column name in adata.obs to color by
    cmap
        Matplotlib colormap name (used if not using stored colors)
    use_stored_colors
        Whether to use/generate stored colors or create fresh ones

    Returns
    -------
    tuple[list[str], dict[str, str] | None]
        - List of colors for each point
        - Color mapping dictionary (None for continuous data)
    """
    if column not in adata.obs.columns:
        raise ValueError(f"Column '{column}' not found in adata.obs")

    color_values = adata.obs[column]

    # Handle categorical data
    if color_values.dtype == "category" or color_values.dtype == object:
        if use_stored_colors:
            color_map = get_colors(adata, column, cmap)
        else:
            # Generate fresh colors without storing
            if color_values.dtype == "category":
                if "Unknown" not in color_values.cat.categories:
                    color_values = color_values.cat.add_categories(["Unknown"])
                color_values = color_values.fillna("Unknown")
            else:
                color_values = color_values.fillna("Unknown")

            unique_values = color_values.unique()
            colormap = plt.get_cmap(cmap)
            colors = colormap(np.linspace(0, 1, len(unique_values)))
            color_map = dict(zip(unique_values, colors, strict=False))
            if "Unknown" in color_map:
                color_map["Unknown"] = "lightgray"

        # Create a copy for color mapping and convert NaN to "Unknown"
        color_values_for_mapping = color_values.copy()
        if color_values.dtype == "category":
            if "Unknown" not in color_values_for_mapping.cat.categories:
                color_values_for_mapping = color_values_for_mapping.cat.add_categories(["Unknown"])
            color_values_for_mapping = color_values_for_mapping.fillna("Unknown")
        else:
            color_values_for_mapping = color_values_for_mapping.fillna("Unknown")

        point_colors = [color_map[val] for val in color_values_for_mapping]
        return point_colors, color_map
    else:
        # Continuous data - return values directly
        return color_values, None
