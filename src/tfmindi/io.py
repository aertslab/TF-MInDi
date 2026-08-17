"""Custom H5AD save/load functions with numpy array handling."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import h5py  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from anndata import AnnData, read_h5ad  # type: ignore

from tfmindi.types import _PATTERN_SPEC, _SEQLET_SPEC, Pattern, Seqlet

# File-level group holding one one-hot array per region, keyed by example_idx.
_REGION_STORE = "_regions"


def _sanitize_hdf5_keys(data):
    """Recursively sanitize dictionary keys for HDF5 storage by replacing problematic characters."""
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Replace forward slashes with a safe placeholder
            sanitized_key = str(key).replace("/", "__SLASH__")
            sanitized[sanitized_key] = _sanitize_hdf5_keys(value)
        return sanitized
    elif isinstance(data, list | tuple):
        return [_sanitize_hdf5_keys(item) for item in data]
    else:
        return data


def _unsanitize_hdf5_keys(data):
    """Recursively restore original dictionary keys by converting placeholders back."""
    if isinstance(data, dict):
        unsanitized = {}
        for key, value in data.items():
            # Restore forward slashes from placeholder
            original_key = str(key).replace("__SLASH__", "/")
            unsanitized[original_key] = _unsanitize_hdf5_keys(value)
        return unsanitized
    elif isinstance(data, list | tuple):
        return [_unsanitize_hdf5_keys(item) for item in data]
    else:
        return data


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
    # Stringify the numpy-array columns of .obs and .var, remembering the originals so the
    # caller's object is restored untouched in the finally block below.
    originals: dict[str, dict[str, pd.Series]] = {}
    for axis, df in (("obs", adata.obs), ("var", adata.var)):
        columns = _numpy_array_columns(df)
        originals[axis] = {col: df[col].copy() for col in columns}
        for col in columns:
            _convert_numpy_arrays_to_strings(df, col)
        if columns:
            adata.uns[f"_tfmindi_numpy_array_{axis}_columns"] = columns

    # Handle HDF5 key sanitization for .uns dictionary
    original_uns = adata.uns.copy()
    adata.uns.clear()
    adata.uns.update(_sanitize_hdf5_keys(original_uns))

    try:
        # Save using standard AnnData method
        write_kwargs = {
            "filename": filename,
            "compression": compression,
            "compression_opts": compression_opts,
            **kwargs,
        }

        # Only pass as_dense if it's not None
        if as_dense is not None:
            write_kwargs["as_dense"] = as_dense

        adata.write_h5ad(**write_kwargs)

    finally:
        # Restore original data structures
        for col, original_data in originals["obs"].items():
            adata.obs[col] = original_data
        for col, original_data in originals["var"].items():
            adata.var[col] = original_data

        # Restore original .uns dictionary with unsanitized keys
        adata.uns.clear()
        adata.uns.update(original_uns)

        for axis in ("obs", "var"):
            adata.uns.pop(f"_tfmindi_numpy_array_{axis}_columns", None)


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

    # Unsanitize HDF5 keys in .uns dictionary
    adata.uns.update(_unsanitize_hdf5_keys(dict(adata.uns)))

    # Restore numpy arrays from the pickle strings written by save_h5ad
    for axis, df in (("obs", adata.obs), ("var", adata.var)):
        key = f"_tfmindi_numpy_array_{axis}_columns"
        for col in adata.uns.pop(key, []):
            if col in df.columns:
                _restore_numpy_arrays_inplace(df, col)

    return adata


def _numpy_array_columns(df: pd.DataFrame) -> list[str]:
    """List the object columns of ``df`` whose values are numpy arrays.

    Such columns cannot be written by plain ``write_h5ad`` and need the pickle-to-string
    detour below.

    Parameters
    ----------
    df
        ``.obs`` or ``.var`` of the AnnData being saved.

    Returns
    -------
    Names of the columns holding numpy arrays.
    """
    columns = []
    for col in df.columns:
        if df[col].dtype == "object":
            non_null = df[col].dropna()
            if not non_null.empty and isinstance(non_null.iloc[0], np.ndarray):
                columns.append(col)
    return columns


def _restore_numpy_arrays_inplace(df, col):
    """Restore a column of numpy arrays from the pickle strings written by save_h5ad."""
    series = df[col]
    if hasattr(series, "cat"):
        # For categorical data, work with categories to minimize memory
        categories = series.cat.categories.astype(str)
        restored_categories = np.empty(len(categories) + 1, dtype=object)
        restored_categories[: len(categories)] = [pickle.loads(bytes.fromhex(cat)) for cat in categories]
        # Trailing slot holds None so that code -1 (missing value) maps to None rather
        # than silently aliasing the last category.
        restored_categories[len(categories)] = None

        df[col] = pd.Series(restored_categories[series.cat.codes.to_numpy()], index=series.index)
    else:
        restored = [pickle.loads(bytes.fromhex(x)) if isinstance(x, str) else x for x in series.astype(str)]
        df[col] = pd.Series(restored, index=series.index)


def _convert_numpy_arrays_to_strings(df, col):
    """Convert a column of numpy arrays to pickle strings so h5ad can store it."""
    series = df[col]
    converted = [pickle.dumps(x).hex() if isinstance(x, np.ndarray) else x for x in series]
    # Categorical, so that load can unpickle each distinct value once instead of per row.
    df[col] = pd.Series(converted, index=series.index).astype(str).astype("category")


def _save_seqlet(seqlet: Seqlet, grp: h5py.Group, regions: h5py.Group) -> None:
    """Save seqlet to h5 group, putting its region one-hot in the shared region store."""
    grp.attrs["version"] = _SEQLET_SPEC
    for k, v in seqlet.__dict__.items():
        if v is None or k == "region_one_hot":
            continue
        grp[k] = v

    # Every seqlet of a region carries the same region one-hot, so writing it per seqlet
    # made pattern files scale with n_seqlets x region_length instead of n_regions x region_length.
    key = str(seqlet.example_idx)
    if key not in regions:
        regions.create_dataset(key, data=seqlet.region_one_hot, compression="gzip")


def _save_pattern(pattern: Pattern, grp: h5py.Group, regions: h5py.Group) -> None:
    """Save pattern to h5 group."""
    grp.attrs["version"] = _PATTERN_SPEC
    for k, v in pattern.__dict__.items():
        if k == "seqlets":
            continue
        if v is None:
            continue
        grp[k] = v
    seqlets_grp = grp.create_group("seqlets")
    for i, seqlet in enumerate(pattern.seqlets):
        seqlet_grp = seqlets_grp.create_group(f"seqlet_{i}")
        _save_seqlet(seqlet, seqlet_grp, regions)


def _read_seqlet(grp: h5py.Group, regions: h5py.Group | None, cache: dict[str, np.ndarray]) -> Seqlet:
    """Load seqlet from h5 group, resolving its region one-hot through the region store.

    Parameters
    ----------
    grp
        Group holding the seqlet attributes.
    regions
        File-level region store, or None for files written before seqlet spec 2.0.
    cache
        Region arrays already read from ``regions``, so seqlets of the same region share
        one array in memory as they do when patterns are first created.
    """
    kwargs = {}
    if grp.attrs["version"] != _SEQLET_SPEC:
        warnings.warn(
            f"The version of the seqlet on disk ({grp.attrs['version']}) does not match with the seqlet version in TF-MInDi ({_SEQLET_SPEC})! Will try to read anyway.",
            stacklevel=1,
        )
    for k in grp.keys():
        value = grp[k][()]  # type: ignore
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        kwargs[k] = value

    # Before spec 2.0 the region one-hot lived in the seqlet group itself.
    if "region_one_hot" not in kwargs and regions is not None:
        key = str(kwargs["example_idx"])
        if key not in cache:
            cache[key] = regions[key][()]  # type: ignore
        kwargs["region_one_hot"] = cache[key]
    return Seqlet(**kwargs)


def _load_pattern(grp: h5py.Group, regions: h5py.Group | None, cache: dict[str, np.ndarray]) -> Pattern:
    """Load pattern from h5 group."""
    kwargs = {}
    if grp.attrs["version"] != _PATTERN_SPEC:
        warnings.warn(
            f"The version of the pattern on disk ({grp.attrs['version']}) does not match with the pattern version in TF-MInDi ({_PATTERN_SPEC})! Will try to read anyway.",
            stacklevel=1,
        )
    for k in grp.keys():
        if k == "seqlets":
            continue
        value = grp[k][()]  # type: ignore
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        kwargs[k] = value
    seqlets: list[Seqlet] = []
    # Sorted to make sure that the order of the seqlets is the same as when they were saved.
    for seqlet_key in sorted(grp["seqlets"].keys(), key=lambda x: int(x.split("_")[1])):  # type: ignore
        seqlets.append(_read_seqlet(grp["seqlets"][seqlet_key], regions, cache))  # type: ignore
    kwargs["seqlets"] = seqlets
    return Pattern(**kwargs)


def save_patterns(patterns: dict[str, Pattern], filename: str | Path) -> None:
    """Save dict of Patterns to disk.

    Paramaters
    ----------
    patterns
        Dict of patterns.
    filename
        output filename.
    """
    with h5py.File(filename, "w") as h5_handle:
        regions = h5_handle.create_group(_REGION_STORE)
        for key, pattern in patterns.items():
            pattern_grp = h5_handle.create_group(f"pattern_{key}")
            _save_pattern(pattern, pattern_grp, regions)


def load_patterns(filename: str | Path) -> dict[str, Pattern]:
    """Load patterns from disk.

    Parameters
    ----------
    filename
        input filename.
    """
    patterns: dict[str, Pattern] = {}
    with h5py.File(filename, "r") as h5_handle:
        regions = h5_handle.get(_REGION_STORE)
        cache: dict[str, np.ndarray] = {}
        for pattern_name in h5_handle.keys():
            if pattern_name == _REGION_STORE:
                continue
            patterns[pattern_name.replace("pattern_", "")] = _load_pattern(h5_handle[pattern_name], regions, cache)
    return patterns
