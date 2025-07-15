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


def load_h5ad(filename: str | Path, **kwargs) -> AnnData:
    """
    Load AnnData object from H5AD format with restoration of numpy arrays in .obs and .var.

    This function wraps AnnData.read_h5ad() with additional postprocessing to restore
    numpy arrays that were stored in .obs and .var columns using save_h5ad().

    Parameters
    ----------
    filename
        Path to the H5AD file to load
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
    """
    # Load using standard AnnData method
    adata = read_h5ad(filename, **kwargs)

    # Check if there are numpy array columns to restore in obs
    if "_tfmindi_numpy_array_obs_columns" in adata.uns:
        numpy_array_obs_columns = adata.uns["_tfmindi_numpy_array_obs_columns"]

        # Restore numpy arrays from pickle strings in obs
        for col in numpy_array_obs_columns:
            if col in adata.obs.columns:
                # Convert categorical back to object first, then restore arrays
                adata.obs[col] = (
                    adata.obs[col]
                    .astype(str)
                    .apply(lambda x: pickle.loads(bytes.fromhex(x)) if isinstance(x, str) else x)
                )

        # Clean up metadata
        del adata.uns["_tfmindi_numpy_array_obs_columns"]

    # Check if there are numpy array columns to restore in var
    if "_tfmindi_numpy_array_var_columns" in adata.uns:
        numpy_array_var_columns = adata.uns["_tfmindi_numpy_array_var_columns"]

        # Restore numpy arrays from pickle strings in var
        for col in numpy_array_var_columns:
            if col in adata.var.columns:
                # Convert categorical back to object first, then restore arrays
                adata.var[col] = (
                    adata.var[col]
                    .astype(str)
                    .apply(lambda x: pickle.loads(bytes.fromhex(x)) if isinstance(x, str) else x)
                )

        # Clean up metadata
        del adata.uns["_tfmindi_numpy_array_var_columns"]

    # Handle legacy metadata key for backwards compatibility
    if "_tfmindi_numpy_array_columns" in adata.uns:
        numpy_array_columns = adata.uns["_tfmindi_numpy_array_columns"]

        # Restore numpy arrays from pickle strings (assuming they were in obs)
        for col in numpy_array_columns:
            if col in adata.obs.columns:
                # Convert categorical back to object first, then restore arrays
                adata.obs[col] = (
                    adata.obs[col]
                    .astype(str)
                    .apply(lambda x: pickle.loads(bytes.fromhex(x)) if isinstance(x, str) else x)
                )

        # Clean up metadata
        del adata.uns["_tfmindi_numpy_array_columns"]

    return adata
