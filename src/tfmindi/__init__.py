"""Package initialization for tfmindi."""

import warnings
from importlib.metadata import version

# Suppress numba hashing warning from tangermeme
warnings.filterwarnings("ignore", message=".*FNV hashing is not implemented in Numba.*", category=UserWarning)

# Pre-import rapids_singlecell to establish proper CUDA environment
# This must happen BEFORE any other CUDA libraries are imported
try:
    import rapids_singlecell  # type: ignore
except ImportError:
    pass

from tfmindi import pl, pp, tl  # noqa: E402
from tfmindi.backends import get_backend, is_gpu_available, set_backend  # noqa: E402
from tfmindi.datasets import (  # noqa: E402
    fetch_motif_annotations,
    fetch_motif_collection,
    load_motif_annotations,
    load_motif_collection,
    load_motif_to_dbd,
)
from tfmindi.io import load_h5ad, save_h5ad  # noqa: E402
from tfmindi.types import Pattern, Seqlet  # noqa: E402

__all__ = [
    "pl",
    "pp",
    "tl",
    "Pattern",
    "Seqlet",
    "get_backend",
    "set_backend",
    "is_gpu_available",
    "fetch_motif_collection",
    "fetch_motif_annotations",
    "load_motif_collection",
    "load_motif_annotations",
    "load_motif_to_dbd",
    "save_h5ad",
    "load_h5ad",
]

__version__ = version("tfmindi")
