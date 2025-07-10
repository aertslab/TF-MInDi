from importlib.metadata import version

from tfmindi import pl, pp, tl
from tfmindi.datasets import (
    fetch_motif_annotations,
    fetch_motif_collection,
    load_motif_annotations,
    load_motif_collection,
    load_motif_to_dbd,
)

__all__ = [
    "pl",
    "pp",
    "tl",
    "fetch_motif_collection",
    "fetch_motif_annotations",
    "load_motif_collection",
    "load_motif_annotations",
    "load_motif_to_dbd",
]

__version__ = version("tfmindi")
