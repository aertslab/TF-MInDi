"""Custom H5AD save/load functions with numpy array handling."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from anndata import AnnData, read_h5ad


def save_h5ad(
    adata: AnnData,
    filename: str | Path,
    compression: str | None = None,
    compression_opts: int | None = None,
    as_dense: str | None = None,
    **kwargs,
) -> None:
    """
    Save AnnData object to H5AD format with proper handling of numpy arrays in .obs and .var.

    This function wraps AnnData.write_h5ad() with additional preprocessing to handle
    numpy arrays stored in .obs and .var columns, which would otherwise cause HDF5 serialization
    errors. The numpy arrays are temporarily converted to string representations for
    serialization, with metadata stored to restore them during loading.

    Parameters
    ----------
    adata
        AnnData object to save
    filename
        Path to the output H5AD file
    compression
        Compression algorithm to use (e.g., 'gzip', 'lzf')
    compression_opts
        Compression options
    as_dense
        Write sparse data as dense arrays
    **kwargs
        Additional arguments passed to AnnData.write_h5ad()

    Examples
    --------
    >>> import tfmindi as tm
    >>> tm.save_h5ad(adata, "my_data.h5ad")
    >>> tm.save_h5ad(adata, "my_data.h5ad", compression="gzip")
    """
    # Create a copy to avoid modifying the original
    adata_copy = adata.copy()

    # Track which columns contain numpy arrays
    numpy_array_obs_columns = []
    numpy_array_var_columns = []

    # Convert numpy array columns in obs to string representation
    for col in adata_copy.obs.columns:
        if adata_copy.obs[col].dtype == "object":
            # Check if the column contains numpy arrays
            first_non_null = adata_copy.obs[col].dropna().iloc[0] if not adata_copy.obs[col].dropna().empty else None
            if first_non_null is not None and isinstance(first_non_null, np.ndarray):
                numpy_array_obs_columns.append(col)
                # Convert numpy arrays to pickle strings for serialization
                adata_copy.obs[col] = (
                    adata_copy.obs[col]
                    .apply(lambda x: pickle.dumps(x).hex() if isinstance(x, np.ndarray) else x)
                    .astype(str)
                    .astype("category")
                )

    # Convert numpy array columns in var to string representation
    for col in adata_copy.var.columns:
        if adata_copy.var[col].dtype == "object":
            # Check if the column contains numpy arrays
            first_non_null = adata_copy.var[col].dropna().iloc[0] if not adata_copy.var[col].dropna().empty else None
            if first_non_null is not None and isinstance(first_non_null, np.ndarray):
                numpy_array_var_columns.append(col)
                # Convert numpy arrays to pickle strings for serialization
                adata_copy.var[col] = (
                    adata_copy.var[col]
                    .apply(lambda x: pickle.dumps(x).hex() if isinstance(x, np.ndarray) else x)
                    .astype(str)
                    .astype("category")
                )

    # Store metadata about numpy array columns
    if numpy_array_obs_columns:
        adata_copy.uns["_tfmindi_numpy_array_obs_columns"] = numpy_array_obs_columns
    if numpy_array_var_columns:
        adata_copy.uns["_tfmindi_numpy_array_var_columns"] = numpy_array_var_columns

    # Save using standard AnnData method
    write_kwargs = {"filename": filename, "compression": compression, "compression_opts": compression_opts, **kwargs}

    # Only pass as_dense if it's not None
    if as_dense is not None:
        write_kwargs["as_dense"] = as_dense

    adata_copy.write_h5ad(**write_kwargs)


def load_h5ad(filename: str | Path, backed: str | None = None, **kwargs) -> AnnData:
    """
    Load AnnData object from H5AD format with restoration of numpy arrays in .obs and .var.

    This function wraps AnnData.read_h5ad() with additional postprocessing to restore
    numpy arrays that were stored in .obs and .var columns using save_h5ad().

    Parameters
    ----------
    filename
        Path to the H5AD file to load
    backed
        Load in backed mode to save memory. Use 'r' for read-only access.
    **kwargs
        Additional arguments passed to AnnData.read_h5ad()

    Returns
    -------
    AnnData object with numpy arrays restored in .obs columns

    Examples
    --------
    >>> import tfmindi as tm
    >>> adata = tm.load_h5ad("my_data.h5ad")
    >>> print(type(adata.obs["seqlet_matrix"].iloc[0]))
    <class 'numpy.ndarray'>

    >>> # Memory-efficient loading for large files
    >>> adata = tm.load_h5ad("my_data.h5ad", backed="r")
    """
    # Load using standard AnnData method with memory optimizations
    load_kwargs = {"backed": backed, **kwargs}
    adata = read_h5ad(filename, **load_kwargs)

    # Check if there are numpy array columns to restore in obs
    if "_tfmindi_numpy_array_obs_columns" in adata.uns:
        numpy_array_obs_columns = adata.uns["_tfmindi_numpy_array_obs_columns"]

        # Restore numpy arrays from pickle strings in obs
        for col in numpy_array_obs_columns:
            if col in adata.obs.columns:
                _restore_numpy_arrays_inplace(adata.obs, col)

        # Clean up metadata
        del adata.uns["_tfmindi_numpy_array_obs_columns"]

    # Check if there are numpy array columns to restore in var
    if "_tfmindi_numpy_array_var_columns" in adata.uns:
        numpy_array_var_columns = adata.uns["_tfmindi_numpy_array_var_columns"]

        # Restore numpy arrays from pickle strings in var
        for col in numpy_array_var_columns:
            if col in adata.var.columns:
                _restore_numpy_arrays_inplace(adata.var, col)

        # Clean up metadata
        del adata.uns["_tfmindi_numpy_array_var_columns"]

    return adata


def _restore_numpy_arrays_inplace(df, col):
    """Memory-efficient in-place restoration of numpy arrays from pickle strings."""
    import pandas as pd

    # Get the series - convert categorical to string without creating copy
    series = df[col]
    if hasattr(series, "cat"):
        # For categorical data, work with categories to minimize memory
        categories = series.cat.categories.astype(str)
        restored_categories = [pickle.loads(bytes.fromhex(cat)) for cat in categories]

        cat_mapping = dict(zip(categories, restored_categories, strict=False))
        df[col] = series.cat.categories[series.cat.codes].map(cat_mapping)
    else:
        # For non-categorical data, process in chunks to limit memory usage
        chunk_size = 1000
        restored_values = []

        for i in range(0, len(series), chunk_size):
            chunk = series.iloc[i : i + chunk_size]
            chunk_restored = [pickle.loads(bytes.fromhex(x)) if isinstance(x, str) else x for x in chunk.astype(str)]
            restored_values.extend(chunk_restored)

        df[col] = pd.Series(restored_values, index=series.index)
