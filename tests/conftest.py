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


@pytest.fixture(scope="session")
def sample_contrib_data():
    """Load sample contribution scores for testing."""
    test_data_dir = Path(__file__).parent / "data"
    contrib_file = test_data_dir / "sample_contrib.npz"
    return np.load(contrib_file)["contrib"]


@pytest.fixture(scope="session")
def sample_oh_data():
    """Load sample one-hot encoded sequences for testing."""
    test_data_dir = Path(__file__).parent / "data"
    oh_file = test_data_dir / "sample_oh.npz"
    return np.load(oh_file)["oh"]


@pytest.fixture(scope="session")
def sample_cell_labels():
    """Load sample cell type labels for testing."""
    test_data_dir = Path(__file__).parent / "data"
    labels_file = test_data_dir / "sample_labels.txt"
    with open(labels_file) as f:
        return [line.strip() for line in f]


@pytest.fixture(scope="session")
def sample_motifs():
    """Load sample motif collection from singletons folder for testing."""
    import tfmindi as tm

    motif_collection_folder = Path(__file__).parent / "data" / "singletons"
    return tm.load_motif_collection(str(motif_collection_folder))


@pytest.fixture(scope="session")
def sample_seqlet_adata(sample_contrib_data, sample_oh_data, sample_motifs):
    """Create a comprehensive test AnnData object with seqlets and motif data."""
    import pandas as pd

    import tfmindi as tm

    # Use all the contribution data
    contrib = sample_contrib_data
    oh = sample_oh_data

    # Extract seqlets
    seqlets_df, seqlet_matrices = tm.pp.extract_seqlets(contrib, oh, threshold=0.1)

    # Skip if no seqlets found
    if len(seqlets_df) == 0:
        pytest.skip("No seqlets found in test data")

    # Use subset of motifs for testing
    test_motifs = dict(list(sample_motifs.items())[:10])
    motif_names = list(test_motifs.keys())

    # Calculate similarity matrix
    similarity_matrix = tm.pp.calculate_motif_similarity(seqlet_matrices, test_motifs)

    # Create motif annotations DataFrame
    motif_annotations = pd.DataFrame(
        {
            "Direct_annot": [f"TF_{i}" for i in range(len(motif_names))],
            "Motif_similarity_annot": [f"SIM_{i}" if i % 2 == 0 else None for i in range(len(motif_names))],
            "Orthology_annot": [f"ORTH_{i}" if i % 3 == 0 else None for i in range(len(motif_names))],
        },
        index=motif_names,
    )

    # Create motif to DBD mapping
    dbd_types = ["Homeodomain", "STAT", "bZIP", "Forkhead", "ETS", "C2H2 ZF", "bHLH", "Nuclear receptor"]
    motif_to_dbd = {name: dbd_types[i % len(dbd_types)] for i, name in enumerate(motif_names)}

    # Create comprehensive AnnData object
    adata = tm.pp.create_seqlet_adata(
        similarity_matrix,
        seqlets_df,
        seqlet_matrices=seqlet_matrices,
        oh_sequences=oh,
        contrib_scores=contrib,
        motif_collection=test_motifs,
        motif_annotations=motif_annotations,
        motif_to_dbd=motif_to_dbd,
    )

    return adata


@pytest.fixture(scope="session")
def sample_clustered_adata(sample_seqlet_adata):
    """Create a test AnnData object with clustering already performed at resolution=1.0."""
    import tfmindi as tm

    adata = sample_seqlet_adata.copy()
    tm.tl.cluster_seqlets(adata, resolution=1.0)
    return adata


@pytest.fixture(scope="session")
def sample_patterns(sample_clustered_adata):
    """Create patterns from the clustered AnnData object."""
    import tfmindi as tm

    patterns = tm.tl.create_patterns(sample_clustered_adata)
    return patterns
