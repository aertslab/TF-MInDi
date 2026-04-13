"""Regression tests for the seqlet DBD assignment bugfix.

See thoughts/shared/plans/2026-04-09-bugfix-seqlet-assignment.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import tfmindi as tm


class TestLoadMotifToDbdHierarchy:
    """Contract for the rewritten load_motif_to_dbd."""

    @pytest.fixture
    def fake_tf_to_dbd(self, monkeypatch):
        """Patch the human TF CSV download with a synthetic in-memory table."""
        table = pd.DataFrame(
            {
                "HGNC symbol": [
                    "GATA1",
                    "GATA2",
                    "GATA3",
                    "GATA4",
                    "GATA5",
                    "GATA6",
                    "NFIC",
                    "NFIA",
                    "E2F1",
                    "FOS",
                    "JUN",
                    "SOX2",
                    "POU5F1",
                ],
                "DBD": [
                    "GATA",
                    "GATA",
                    "GATA",
                    "GATA",
                    "GATA",
                    "GATA",
                    "CTF/NF-I",
                    "CTF/NF-I",
                    "E2F",
                    "bZIP",
                    "bZIP",
                    "HMG/Sox",
                    "Homeodomain",
                ],
            }
        )
        table.index = range(1, len(table) + 1)

        def fake_read_csv(url, index_col=0):
            assert "humantfs" in url
            return table

        monkeypatch.setattr("tfmindi.datasets.pd.read_csv", fake_read_csv)
        return table

    def test_direct_single_family_wins(self, fake_tf_to_dbd):
        annots = pd.DataFrame(
            {
                "Direct_annot": ["GATA1"],
                "Motif_similarity_annot": [None],
                "Orthology_annot": [None],
                "Motif_similarity_and_Orthology_annot": [None],
            },
            index=["motif_clean_gata"],
        )
        annots.index.name = "MotifID"
        out = tm.load_motif_to_dbd(annots)
        assert out["motif_clean_gata"] == "GATA"

    def test_direct_multi_family_is_composite(self, fake_tf_to_dbd):
        """The tfdimers__MD00378 failure case."""
        annots = pd.DataFrame(
            {
                "Direct_annot": ["NFIC, NFIA, E2F1"],
                "Motif_similarity_annot": [None],
                "Orthology_annot": [None],
                "Motif_similarity_and_Orthology_annot": [None],
            },
            index=["tfdimers__MD00378"],
        )
        annots.index.name = "MotifID"
        out = tm.load_motif_to_dbd(annots)
        assert out["tfdimers__MD00378"] == "Composite"

    def test_similarity_family_collapse(self, fake_tf_to_dbd):
        """GATA1..GATA6 listed only under similarity should still collapse to GATA."""
        annots = pd.DataFrame(
            {
                "Direct_annot": [None],
                "Motif_similarity_annot": ["GATA1, GATA2, GATA3, GATA4, GATA5, GATA6"],
                "Orthology_annot": [None],
                "Motif_similarity_and_Orthology_annot": [None],
            },
            index=["motif_gata_family"],
        )
        annots.index.name = "MotifID"
        out = tm.load_motif_to_dbd(annots)
        assert out["motif_gata_family"] == "GATA"

    def test_orthology_beats_similarity(self, fake_tf_to_dbd):
        """Orthology tier is consulted before similarity tier."""
        annots = pd.DataFrame(
            {
                "Direct_annot": [None],
                "Motif_similarity_annot": ["FOS, JUN"],  # bZIP
                "Orthology_annot": ["GATA2"],  # GATA
                "Motif_similarity_and_Orthology_annot": [None],
            },
            index=["motif_ortho_gata"],
        )
        annots.index.name = "MotifID"
        out = tm.load_motif_to_dbd(annots)
        assert out["motif_ortho_gata"] == "GATA"

    def test_lower_tier_ambiguous_falls_through(self, fake_tf_to_dbd):
        """Similarity listing TFs from many families and no other evidence → NaN."""
        annots = pd.DataFrame(
            {
                "Direct_annot": [None],
                "Motif_similarity_annot": ["GATA1, FOS, SOX2, POU5F1"],
                "Orthology_annot": [None],
                "Motif_similarity_and_Orthology_annot": [None],
            },
            index=["motif_noisy"],
        )
        annots.index.name = "MotifID"
        out = tm.load_motif_to_dbd(annots)
        assert "motif_noisy" not in out or pd.isna(out.get("motif_noisy"))

    def test_no_evidence_means_absent(self, fake_tf_to_dbd):
        annots = pd.DataFrame(
            {
                "Direct_annot": [None],
                "Motif_similarity_annot": [None],
                "Orthology_annot": [None],
                "Motif_similarity_and_Orthology_annot": [None],
            },
            index=["motif_empty"],
        )
        annots.index.name = "MotifID"
        out = tm.load_motif_to_dbd(annots)
        assert "motif_empty" not in out or pd.isna(out.get("motif_empty"))

    def test_missing_tf_from_human_table_is_ignored(self, fake_tf_to_dbd):
        """TFs not in the human TF CSV must not raise KeyError."""
        annots = pd.DataFrame(
            {
                "Direct_annot": ["FOO_NOT_HUMAN, GATA1"],
                "Motif_similarity_annot": [None],
                "Orthology_annot": [None],
                "Motif_similarity_and_Orthology_annot": [None],
            },
            index=["motif_partial"],
        )
        annots.index.name = "MotifID"
        out = tm.load_motif_to_dbd(annots)
        assert out["motif_partial"] == "GATA"


class TestSeqletDbdTopKVote:
    """Unit-level contract for the rewritten per-seqlet block in cluster_seqlets."""

    def _build_minimal_adata(self, similarity_rows, var_dbds, n_filler: int = 30):
        """Build a minimal AnnData that cluster_seqlets' full pipeline can consume.

        The provided ``similarity_rows`` become the leading rows of ``X``; the
        remaining ``n_filler`` rows are filled with reproducible random noise so
        that scanpy PCA/neighbors/tSNE/Leiden do not degenerate on a trivially
        rank-deficient matrix. Only the leading rows should be asserted on.
        """
        import anndata as ad

        rng = np.random.default_rng(0)
        test_rows = np.asarray(similarity_rows, dtype=np.float32)
        n_test, n_vars = test_rows.shape
        filler = rng.uniform(0.1, 1.0, size=(n_filler, n_vars)).astype(np.float32)
        full = np.vstack([test_rows, filler])
        X = sp.csr_array(full)
        n_obs = X.shape[0]
        obs = pd.DataFrame(
            {
                "seqlet_matrix": [np.zeros((4, 6), dtype=np.float32)] * n_obs,
            },
            index=[str(i) for i in range(n_obs)],
        )
        var = pd.DataFrame({"dbd": var_dbds}, index=[f"motif_{i}" for i in range(X.shape[1])])
        return ad.AnnData(X=X, obs=obs, var=var)

    def test_composite_top1_is_dropped_from_vote(self):
        """Seqlet whose rank-1 is Composite but ranks 2-5 are bZIP → labelled bZIP."""
        row = [5.0, 4.0, 3.9, 3.8, 3.7, 0.0]  # 6 motifs
        var_dbds = ["Composite", "bZIP", "bZIP", "bZIP", "bZIP", "GATA"]
        adata = self._build_minimal_adata([row], var_dbds)
        tm.tl.cluster_seqlets(adata, resolution=1.0, top_k_motifs=5, dbd_vote_min_share=0.4)
        assert adata.obs["seqlet_dbd"].iloc[0] == "bZIP"

    def test_nan_motifs_are_dropped_from_vote(self):
        row = [10.0, 4.0, 3.9, 3.8, 3.7, 0.0]
        var_dbds = [np.nan, "bZIP", "bZIP", "bZIP", "bZIP", "GATA"]
        adata = self._build_minimal_adata([row], var_dbds)
        tm.tl.cluster_seqlets(adata, resolution=1.0, top_k_motifs=5, dbd_vote_min_share=0.4)
        assert adata.obs["seqlet_dbd"].iloc[0] == "bZIP"

    def test_empty_row_returns_nan(self):
        row = [0.0] * 6
        var_dbds = ["GATA", "bZIP", "Ets", "bHLH", "Forkhead", "NR"]
        adata = self._build_minimal_adata([row], var_dbds)
        tm.tl.cluster_seqlets(adata, resolution=1.0, top_k_motifs=5, dbd_vote_min_share=0.4)
        assert pd.isna(adata.obs["seqlet_dbd"].iloc[0])

    def test_rejection_threshold_triggers_on_ties(self):
        row = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        var_dbds = ["GATA", "bZIP", "Ets", "bHLH", "Forkhead", "NR"]
        adata = self._build_minimal_adata([row], var_dbds)
        tm.tl.cluster_seqlets(adata, resolution=1.0, top_k_motifs=5, dbd_vote_min_share=0.4)
        # Each family has 0.2 share → winner_share < 0.4 → NaN
        assert pd.isna(adata.obs["seqlet_dbd"].iloc[0])

    def test_clear_winner_above_threshold(self):
        row = [5.0, 4.5, 1.0, 1.0, 1.0, 0.0]
        var_dbds = ["bZIP", "bZIP", "GATA", "Ets", "Forkhead", "NR"]
        adata = self._build_minimal_adata([row], var_dbds)
        tm.tl.cluster_seqlets(adata, resolution=1.0, top_k_motifs=5, dbd_vote_min_share=0.4)
        assert adata.obs["seqlet_dbd"].iloc[0] == "bZIP"

    def test_composite_excluded_from_cluster_background(self):
        """'Composite' must not appear in cluster_dbd."""
        row = [5.0, 4.0, 0.0, 0.0, 0.0, 0.0]
        var_dbds = ["Composite", "bZIP", "GATA", "Ets", "Forkhead", "NR"]
        adata = self._build_minimal_adata([row] * 10, var_dbds)
        tm.tl.cluster_seqlets(adata, resolution=1.0, top_k_motifs=5, dbd_vote_min_share=0.4)
        assert "Composite" not in set(adata.obs["cluster_dbd"].dropna().unique())
