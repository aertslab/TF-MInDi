"""Fixtures for testing TF-MInDi."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest


@pytest.fixture
def adata():
    adata = ad.AnnData(X=np.array([[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]]).astype(np.float32))
    adata.layers["scaled"] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(np.float32)

    return adata


@pytest.fixture
def motif_collection_folder():
    return "tests/data/singletons"


@pytest.fixture
def motif_annotations_file():
    return "tests/data/motif_annotations.tbl"


@pytest.fixture
def sample_contrib_data():
    """Load sample contribution scores for testing."""
    test_data_dir = Path(__file__).parent / "data"
    contrib_file = test_data_dir / "sample_contrib.npz"
    return np.load(contrib_file)["contrib"]


@pytest.fixture
def sample_oh_data():
    """Load sample one-hot encoded sequences for testing."""
    test_data_dir = Path(__file__).parent / "data"
    oh_file = test_data_dir / "sample_oh.npz"
    return np.load(oh_file)["oh"]


@pytest.fixture
def sample_cell_labels():
    """Load sample cell type labels for testing."""
    test_data_dir = Path(__file__).parent / "data"
    labels_file = test_data_dir / "sample_labels.txt"
    with open(labels_file) as f:
        return [line.strip() for line in f]


@pytest.fixture
def sample_motifs():
    """Load sample motif collection from singletons folder for testing."""
    import tfmindi as tm

    motif_collection_folder = Path(__file__).parent / "data" / "singletons"
    return tm.datasets.load_motif_collection(str(motif_collection_folder))
