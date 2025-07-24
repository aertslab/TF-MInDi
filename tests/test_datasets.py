"""Tests for core TF-MInDi functions."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

import tfmindi as tm


class TestLoadMotifCollection:
    """Test load_motif_collection function."""

    def test_load_motif_collection_success(self, motif_collection_folder):
        """Test load_motif_collection with valid motif files."""
        result = tm.load_motif_collection(motif_collection_folder)

        assert isinstance(result, dict)
        assert len(result) == 26
        assert "jaspar__MA0936.1" in result
        assert "jaspar__MA1785.1" in result
        assert result["jaspar__MA0936.1"].shape == (4, 8)
        assert result["jaspar__MA1785.1"].shape == (4, 11)

    def test_load_motif_collection_no_files(self):
        """Test load_motif_collection with directory containing no .cb files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = tm.load_motif_collection(temp_dir)
            assert result == {}

    def test_load_motif_collection_nonexistent_dir(self):
        """Test load_motif_collection with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            tm.load_motif_collection("/nonexistent/path")

    def test_load_motif_collection_with_motif_names(self, motif_collection_folder):
        """Test load_motif_collection with specific motif names."""
        # Test with subset of motifs
        target_motifs = ["jaspar__MA0936.1", "jaspar__MA1785.1"]
        result = tm.load_motif_collection(motif_collection_folder, motif_names=target_motifs)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert set(result.keys()) == set(target_motifs)
        assert result["jaspar__MA0936.1"].shape == (4, 8)
        assert result["jaspar__MA1785.1"].shape == (4, 11)

        # Test with motif names that don't exist
        non_existent_motifs = ["fake_motif1", "fake_motif2"]
        result_empty = tm.load_motif_collection(motif_collection_folder, motif_names=non_existent_motifs)
        assert result_empty == {}

        # Test with mixed existing and non-existing motifs
        mixed_motifs = ["jaspar__MA0936.1", "fake_motif1"]
        result_mixed = tm.load_motif_collection(motif_collection_folder, motif_names=mixed_motifs)
        assert len(result_mixed) == 1
        assert "jaspar__MA0936.1" in result_mixed
        assert "fake_motif1" not in result_mixed


class TestFetchMotifAnnotations:
    """Test fetch_motif_annotations function."""

    def test_fetch_motif_annotations_invalid_species(self):
        """Test fetch_motif_annotations with invalid species."""
        with pytest.raises(AssertionError, match="Species invalid_species with version v10nr_clust is not recognised"):
            tm.fetch_motif_annotations(species="invalid_species")


class TestLoadMotifAnnotations:
    """Test load_motif_annotations function."""

    def test_load_motif_annotations_success(self, motif_annotations_file):
        """Test load_motif_annotations with valid TSV file."""
        result = tm.load_motif_annotations(motif_annotations_file)

        assert isinstance(result, pd.DataFrame)
        assert "Direct_annot" in result.columns
        assert "Motif_similarity_annot" in result.columns
        assert "Orthology_annot" in result.columns
        assert "Motif_similarity_and_Orthology_annot" in result.columns
        # Check that direct annotations are present
        assert len(result[result["Direct_annot"].notna()]) >= 1

    def test_load_motif_annotations_with_filtering(self):
        """Test load_motif_annotations with filtering parameters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("#motif_id\tgene_name\tmotif_similarity_qvalue\torthologous_identity\tdescription\n")
            f.write("motif1\tTF1\t0.0005\t0.9\tgene is directly annotated\n")
            f.write("motif2\tTF2\t0.002\t0.8\tgene is directly annotated\n")  # Should be filtered out
            f.write("motif3\tTF3\t0.0003\t0.4\tgene is similar to motif\n")  # Should be filtered out
            temp_file = f.name

        try:
            result = tm.load_motif_annotations(
                temp_file, motif_similarity_fdr=0.001, orthologous_identity_threshold=0.5
            )

            assert isinstance(result, pd.DataFrame)
            # Only motif1 should pass both filters
            assert len(result[result["Direct_annot"].notna()]) == 1
        finally:
            os.unlink(temp_file)

    def test_load_motif_annotations_nonexistent_file(self):
        """Test load_motif_annotations with non-existent file."""
        with pytest.raises(FileNotFoundError):
            tm.load_motif_annotations("/nonexistent/file.tsv")
