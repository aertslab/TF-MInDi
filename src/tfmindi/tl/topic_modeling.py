"""Topic modeling for discovering co-occurring motif patterns.

This module extends the original TF-MINDI topic modeling implementation with:
* New fixed loglikelihood calculation (corrects a previous bug and is now based on a C++ implementation).
* Robust sparse construction of the region × cluster count matrix (no dense `pd.crosstab`).
* Optional "short-text" style preprocessing:
  - document-frequency (DF) filtering (remove ultra-rare and near-ubiquitous clusters)
  - optional binarization (presence/absence per region)
* Multiple topic model backends:
  - "lda_gibbs"   : existing `lda` package (collapsed Gibbs sampler)
  - "lda_sklearn" : sklearn `LatentDirichletAllocation` (VB), handles sparse inputs
  - "nmf"         : sklearn `NMF` baseline
  - "btm"         : Biterm Topic Model for short-text-like data
* Multi-metric topic model evaluation (similar to pycisTopic).
* Optional multi-start fitting to reduce sensitivity to random initialization.

The default behavior remains compatible with the original TF-MINDI API:
* `tm.tl.run_topic_modeling(adata, ...)` still runs Gibbs LDA by default.
* Results are stored in `adata.uns['topic_modeling']` under the same keys.

Notes on "short-text" relevance
-------------------------------
The region×cluster matrix is typically very sparse and low-count per region
(many regions have only a handful of clusters), which exhibits the same failure
modes as short-text topic modeling (high variance, stop-pattern domination).
"""

from __future__ import annotations

import lda
import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from typing import Literal
from dataclasses import dataclass


TopicMethod = Literal["lda_gibbs", "lda_sklearn", "nmf", "btm"]
Weighting = Literal["count", "tfidf"]
BTMImpl = Literal["internal_em", "bitermplus"]
BTMTokenMode = Literal["binary", "count"]


@dataclass(frozen=True)
class _CountMatrix:
    X: sp.csr_matrix
    region_names: list
    pattern_names: list[str]


# -----------------------------------------------------------------------------
# Likelihood / coherence utilities
# -----------------------------------------------------------------------------


def loglikelihood(nzw: np.ndarray, ndz: np.ndarray, alpha: float, eta: float) -> float:
    """Calculate log-likelihood of LDA model parameters given count matrices.

    Adapted from https://shusei-e.github.io/natural%20language%20processing/LDA-CGS-loglikelihood/.

    Parameters
    ----------
    nzw
        (K, V) Topic-word count matrix
    ndz
        (D, K) Document-topic count matrix
    alpha
        scalar Dirichlet prior for document-topic distributions
    eta
        scalar Dirichlet prior parameter for topic-word distributions
    """
    K, V = nzw.shape
    D, K2 = ndz.shape
    assert K == K2

    # Calculate log p(w|z) - topic-word distributions
    topic_ll = 0.0
    for k in range(K):
        nw = nzw[k, :].sum()
        topic_ll += math.lgamma(V * eta) - math.lgamma(V * eta + nw)
        for v in range(V):
            topic_ll += math.lgamma(nzw[k, v] + eta) - math.lgamma(eta)

    # Calculate log p(z) - document-topic distributions
    doc_ll = 0.0
    for d in range(D):
        nd = ndz[d, :].sum()
        doc_ll += math.lgamma(K * alpha) - math.lgamma(K * alpha + nd)
        for k in range(K):
            doc_ll += math.lgamma(ndz[d, k] + alpha) - math.lgamma(alpha)

    return doc_ll + topic_ll


def topic_coherence_umass_per_topic(
    X_bin: sp.csr_matrix,
    topic_word: np.ndarray,
    top_n: int = 20,
    eps: float = 1.0,
) -> np.ndarray:
    """
    UMass-style topic coherence using co-occurrence within regions (per topic).

    For each topic k:

        coherence_k = sum_{m=2..M} sum_{l=1..m-1} log((D(w_m, w_l)+eps) / D(w_l))

    Where D(w) is the number of regions containing pattern w (binary),
    and D(w_i, w_j) is the number of regions containing both patterns.

    Parameters
    ----------
    X_bin
        (D, V) binary CSR matrix (presence/absence).
    topic_word
        (K, V) topic-word weights/probabilities (higher = more associated with topic).
    top_n
        Number of top patterns per topic (M in the formula).
    eps
        Smoothing constant.

    Returns
    -------
    np.ndarray
        Array of shape (K,) with coherence per topic.
        Empty topics (no positive weight in top_n) are set to -inf.
    """
    if not sp.isspmatrix_csr(X_bin):
        X_bin = X_bin.tocsr()

    _, V = X_bin.shape
    K = topic_word.shape[0]

    if V == 0 or K == 0:
        return np.full((K,), float("-inf"), dtype=np.float64)

    df = np.asarray(X_bin.sum(axis=0)).ravel()  # D(w)
    scores = np.full((K,), float("-inf"), dtype=np.float64)

    # Pairwise co-occurrence for top_n (cheap when top_n ~ 20)
    for k in range(K):
        top_idx = np.argsort(topic_word[k])[::-1][: int(top_n)]
        if top_idx.size == 0:
            continue
        if np.all(topic_word[k, top_idx] <= 0):
            continue

        score_k = 0.0
        for m in range(1, len(top_idx)):
            wm = int(top_idx[m])
            col_m = X_bin[:, wm]
            for l in range(m):
                wl = int(top_idx[l])
                co = float(col_m.multiply(X_bin[:, wl]).sum())  # D(wm, wl)
                denom = float(df[wl])
                if denom > 0:
                    score_k += math.log((co + eps) / denom)

        scores[k] = score_k

    return scores


def topic_coherence_umass(
    X_bin: sp.csr_matrix,
    topic_word: np.ndarray,
    top_n: int = 20,
    eps: float = 1.0,
) -> float:
    """
    UMass-style topic coherence averaged across non-empty topics.

    This is a convenience wrapper around :func:`topic_coherence_umass_per_topic`.
    """
    scores = topic_coherence_umass_per_topic(X_bin=X_bin, topic_word=topic_word, top_n=top_n, eps=eps)
    finite = np.isfinite(scores)
    if finite.sum() == 0:
        return float("-inf")
    return float(scores[finite].mean())


def _build_region_cluster_counts(
    seqlets: pd.DataFrame,
    region_col: str,
    cluster_col: str,
) -> _CountMatrix:
    """Build sparse CSR region×cluster count matrix from long-form seqlet table."""
    # Group and count
    grp = seqlets.groupby([region_col, cluster_col], observed=True).size()
    regions = grp.index.get_level_values(0)
    patterns = grp.index.get_level_values(1)

    # Keep regions as-is; cast patterns to str for consistent downstream behavior
    r_codes, r_uniques = pd.factorize(regions, sort=True)
    p_codes, p_uniques = pd.factorize(patterns.astype(str), sort=True)

    data = grp.values.astype(np.int32)
    X = sp.coo_matrix((data, (r_codes, p_codes)), shape=(len(r_uniques), len(p_uniques))).tocsr()

    return _CountMatrix(
        X=X,
        region_names=r_uniques.tolist(),
        pattern_names=[str(x) for x in p_uniques.tolist()],
    )


def _maybe_binarize(X: sp.csr_matrix, binarize: bool) -> sp.csr_matrix:
    if not binarize:
        return X
    Xb = X.copy()
    Xb.data = np.ones_like(Xb.data, dtype=np.int32)
    return Xb


def _filter_patterns_by_df(
    X: sp.csr_matrix,
    pattern_names: list[str],
    min_regions: int = 1,
    max_regions_frac: float = 1.0,
) -> tuple[sp.csr_matrix, list[str], np.ndarray]:
    """Filter patterns by document frequency (#regions where pattern appears)."""
    if min_regions <= 1 and max_regions_frac >= 1.0:
        keep = np.ones(X.shape[1], dtype=bool)
        return X, pattern_names, keep

    X_bin = X.copy()
    X_bin.data = np.ones_like(X_bin.data)
    df = np.asarray(X_bin.sum(axis=0)).ravel()
    D = X.shape[0]

    max_regions = int(math.floor(max_regions_frac * D))
    keep = (df >= min_regions) & (df <= max_regions)

    if keep.sum() == 0:
        raise ValueError(
            "After DF filtering, zero clusters remain. "
            "Relax min_regions_per_pattern / max_regions_frac."
        )

    X_f = X[:, keep].tocsr()
    names_f = [n for n, k in zip(pattern_names, keep) if bool(k)]
    return X_f, names_f, keep


def _apply_weighting(X: sp.csr_matrix, weighting: Weighting) -> sp.csr_matrix:
    if weighting == "count":
        return X.astype(np.float64)

    if weighting == "tfidf":
        from sklearn.feature_extraction.text import TfidfTransformer

        tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=True)
        return tfidf.fit_transform(X.astype(np.float64))

    raise ValueError(f"Unknown weighting: {weighting}")


def _df_to_csr(df: pd.DataFrame) -> sp.csr_matrix:
    """Convert a pandas DataFrame (dense or sparse) to CSR matrix."""
    if hasattr(df, "sparse") and getattr(df, "dtypes", None) is not None:
        # If any column is SparseDtype, use the sparse accessor.
        if any(isinstance(dt, pd.SparseDtype) for dt in df.dtypes):
            return df.sparse.to_coo().tocsr()
    return sp.csr_matrix(df.values)


# -----------------------------------------------------------------------------
# Biterm Topic Model (BTM) – internal EM implementation
# -----------------------------------------------------------------------------


@dataclass
class BTMModel:
    """Lightweight BTM container with an LDA-like attribute interface."""

    n_topics: int
    topic_word_: np.ndarray  # (K, V) p(word|topic)
    doc_topic_: np.ndarray  # (D, K) p(topic|doc)
    theta_: np.ndarray  # (K,) global topic proportions
    loglikelihood_: float
    n_iter: int
    converged: bool


def _btm_em_fit(
    X: sp.csr_matrix,
    n_topics: int,
    alpha: float,
    beta: float,
    n_iter: int,
    random_state: int,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
    """Fit a Biterm Topic Model using EM on aggregated biterm counts.

    This implementation uses global co-occurrence counts derived from a
    region×cluster matrix.

    Important: `alpha` is interpreted as the total concentration.
    Internally we convert it to a symmetric per-topic prior alpha/K.

    Parameters
    ----------
    X
        (D, V) CSR matrix. If binary, co-occurrence counts reflect #regions
        containing both patterns. If count-valued, co-occurrence counts reflect
        Σ_d c_di * c_dj (token-pair weighting, excluding diagonal).
    n_topics
        Number of topics (K).
    alpha
        Total Dirichlet mass over topics (will be scaled to alpha/K).
    beta
        Symmetric Dirichlet prior for topic-word distributions (per word).
    n_iter
        Maximum EM iterations.
    random_state
        Random seed for reproducibility.
    tol
        Convergence tolerance on log-likelihood improvement.

    Returns
    -------
    theta : (K,)
        Global topic proportions.
    phi : (K, V)
        Topic-word distributions p(word|topic).
    ll : float
        Final biterm log-likelihood.
    iters : int
        Iterations run.
    converged : bool
        Whether EM converged by the tolerance criterion.
    """
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()

    rng = np.random.default_rng(int(random_state))

    D, V = X.shape
    if V < 2:
        raise ValueError("BTM requires at least 2 clusters in the vocabulary.")

    # Global co-occurrence counts: C[i,j] = Σ_d X[d,i] * X[d,j]
    # For binary X this is document-level co-occurrence; for count X this is
    # token-pair weighting (off-diagonal).
    C = (X.T @ X).astype(np.float64).tocoo()

    # Extract upper-triangular biterms and counts (exclude diagonal)
    mask = (C.row < C.col) & (C.data > 0)
    iu = C.row[mask].astype(np.int32)
    ju = C.col[mask].astype(np.int32)
    counts = C.data[mask].astype(np.float64)

    if counts.size == 0:
        raise ValueError("No biterms found (need regions with >=2 distinct clusters).")

    # Symmetric priors
    alpha_total = max(float(alpha), 1e-9)
    alpha_k = alpha_total / float(n_topics)  # per-topic concentration
    beta_v = max(float(beta), 1e-12)

    # Initialize parameters
    theta = rng.dirichlet(np.ones(int(n_topics), dtype=np.float64))  # (K,)
    phi = rng.dirichlet(np.ones(int(V), dtype=np.float64) * beta_v, size=int(n_topics))  # (K,V)

    ll_prev = -np.inf
    converged = False
    eps = 1e-32

    for it in range(1, int(n_iter) + 1):
        # E-step: responsibilities r[b,k]
        # p(b|k) = phi[k, i] * phi[k, j]
        p_bk = phi[:, iu] * phi[:, ju]  # (K, B)
        p_bk = (theta[:, None] * p_bk) + eps
        denom = p_bk.sum(axis=0, keepdims=True)  # (1,B)
        r = p_bk / denom  # (K,B)

        # M-step
        Nk = (r * counts[None, :]).sum(axis=1)  # (K,)
        theta = Nk + alpha_k
        theta = theta / theta.sum()

        # Update phi
        phi_counts = np.zeros_like(phi)
        weighted = r * counts[None, :]  # (K,B)
        # each biterm contributes one count to each of its two words
        for k in range(int(n_topics)):
            np.add.at(phi_counts[k], iu, weighted[k])
            np.add.at(phi_counts[k], ju, weighted[k])

        phi = phi_counts + beta_v
        phi = phi / phi.sum(axis=1, keepdims=True)

        # Log-likelihood over biterms
        p_b = (theta[:, None] * (phi[:, iu] * phi[:, ju])).sum(axis=0) + eps
        ll = float((counts * np.log(p_b)).sum())

        improvement = ll - ll_prev
        # declare convergence on very small positive improvement
        if (improvement >= 0.0) and (improvement < float(tol)):
            converged = True
            ll_prev = ll
            break
        ll_prev = ll

    return theta, phi, ll_prev, it, converged


def _btm_infer_doc_topics(
    X: sp.csr_matrix,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Infer per-region topic mixtures from fitted BTM parameters.

    Uses a fast biterm-based approximation:

        p(z|d) ∝ θ_z * Σ_{i<j in d} c_i c_j φ_{z,i} φ_{z,j}

    With a fallback for regions with <2 active words.

    Notes
    -----
    * This works for both binary and count-valued X.
    * Diagonal (i==j) biterms are not explicitly modeled (consistent with the fit).
    """
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()

    D, _ = X.shape
    K = int(theta.shape[0])

    theta = np.asarray(theta, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64)
    phi = np.clip(phi, 1e-12, 1.0)
    phi = phi / (phi.sum(axis=1, keepdims=True) + 1e-12)

    doc_topic = np.zeros((D, K), dtype=np.float64)
    for d in range(D):
        start, end = X.indptr[d], X.indptr[d + 1]
        idx = X.indices[start:end]
        cnt = X.data[start:end].astype(np.float64)

        if idx.size == 0:
            doc_topic[d] = theta
            continue

        # Weighted sum of φ over active words (counts-aware)
        s = (phi[:, idx] * cnt[None, :]).sum(axis=1)  # (K,)

        if idx.size >= 2:
            # Σ_i (c_i * φ_i)^2 (counts-aware) -> excludes diagonal (i==j) biterms
            sq = ((phi[:, idx] * cnt[None, :]) ** 2).sum(axis=1)  # (K,)
            biterm_sum = 0.5 * (s**2 - sq)
            scores = theta * np.maximum(biterm_sum, 0.0)
        else:
            # fallback for singleton docs
            scores = theta * np.maximum(s, 0.0)

        if scores.sum() <= 0:
            doc_topic[d] = theta
        else:
            doc_topic[d] = scores / scores.sum()

    return doc_topic


def _btm_generate_biterms(
    X: sp.csr_matrix,
    token_mode: BTMTokenMode,
    max_biterms_per_region: int,
    random_state: int,
    allow_self_pairs: bool = False,
) -> list[list[list[int]]]:
    """Generate per-region biterms in the format expected by `bitermplus.BTM.fit`.

    Parameters
    ----------
    X
        (D, V) CSR matrix (binary or counts).
    token_mode
        - "binary": treat clusters as presence/absence per region (unique indices)
        - "count" : treat clusters as multiplicity (counts) and sample biterms with
          probabilities proportional to counts
    max_biterms_per_region
        Cap on number of biterms generated per region (when sampling is required).
    random_state
        Random seed for reproducibility.
    allow_self_pairs
        Only used for token_mode="count". If True, allow self-biterms (i,i). If False,
        self pairs are rejected (default).

    Returns
    -------
    biterms : list[list[list[int]]]
        Nested list where biterms[d] is the list of biterms for document/region d and each
        biterm is [i, j] with i <= j.
    """
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()

    rng = np.random.default_rng(int(random_state))
    biterms: list[list[list[int]]] = []

    D = X.shape[0]
    mmax = int(max_biterms_per_region)

    for d in range(D):
        start, end = X.indptr[d], X.indptr[d + 1]
        idx = X.indices[start:end].astype(np.int32)

        # Keep alignment with document ids: append empty list for docs with <2 tokens
        if idx.size < 2:
            biterms.append([])
            continue

        if token_mode == "binary":
            # ensure uniqueness in case the matrix has duplicates
            idx = np.unique(idx)
            if idx.size < 2:
                biterms.append([])
                continue

            m = int(idx.size)
            possible_pairs = (m * (m - 1)) // 2
            doc_biterms: list[list[int]] = []

            if possible_pairs <= mmax:
                # enumerate all distinct pairs
                for a in range(m - 1):
                    ia = int(idx[a])
                    for b in range(a + 1, m):
                        ib = int(idx[b])
                        doc_biterms.append([ia, ib] if ia <= ib else [ib, ia])

            else:
                # uniform sampling over distinct words
                for _ in range(mmax):
                    i1, i2 = rng.choice(idx, size=2, replace=False)
                    i1 = int(i1)
                    i2 = int(i2)
                    doc_biterms.append([i1, i2] if i1 <= i2 else [i2, i1])

            biterms.append(doc_biterms)
            continue

        if token_mode == "count":
            cnt = X.data[start:end].astype(np.float64)
            s = float(cnt.sum())
            if s <= 0.0:
                biterms.append([])
                continue

            p = cnt / s
            doc_biterms: list[list[int]] = []

            # Sample biterms proportional to counts (multiplicity-aware).
            # We sample two words with replacement. If allow_self_pairs is False, reject i==j.
            for _ in range(mmax):

                i1 = int(rng.choice(idx, p=p))
                i2 = int(rng.choice(idx, p=p))

                if (not allow_self_pairs) and (i1 == i2):
                    ok = False
                    for __ in range(10):
                        i2 = int(rng.choice(idx, p=p))
                        if i2 != i1:
                            ok = True
                            break
                    if not ok:
                        continue

                doc_biterms.append([i1, i2] if i1 <= i2 else [i2, i1])

            biterms.append(doc_biterms)
            continue

        raise ValueError(f"Unknown token_mode: {token_mode}")

    if sum(len(b) for b in biterms) == 0:
        raise ValueError("No biterms generated (need regions with >=2 distinct clusters).")

    return biterms


def _btm_docs_vec_from_csr(
    X: sp.csr_matrix,
    token_mode: BTMTokenMode,
    max_tokens_per_region: int | None,
    random_state: int,
) -> list[np.ndarray]:
    """Build docs_vec (list of token id arrays) for bitermplus-style APIs.

    Parameters
    ----------
    X
        (D, V) CSR matrix.
    token_mode
        "binary": return unique indices per region.
        "count": return repeated indices according to counts; if max_tokens_per_region
        is set, downsample with replacement using count-proportional probabilities.
    """
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()

    rng = np.random.default_rng(int(random_state))
    docs_vec: list[np.ndarray] = []
    D = X.shape[0]

    cap = None if max_tokens_per_region is None else int(max_tokens_per_region)

    for d in range(D):
        start, end = X.indptr[d], X.indptr[d + 1]
        idx = X.indices[start:end].astype(np.int32)

        if idx.size == 0:
            docs_vec.append(np.array([], dtype=np.int32))
            continue

        if token_mode == "binary":
            docs_vec.append(idx.copy())
            continue

        cnt = X.data[start:end].astype(np.int64)
        total = int(cnt.sum())
        if total <= 0:
            docs_vec.append(np.array([], dtype=np.int32))
            continue

        if cap is None or total <= cap:
            docs_vec.append(np.repeat(idx, cnt).astype(np.int32))
        else:
            p = cnt.astype(np.float64)
            p = p / p.sum()
            sampled = rng.choice(idx, size=cap, replace=True, p=p)
            docs_vec.append(sampled.astype(np.int32))

    return docs_vec


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------


def run_topic_modeling(
    adata: AnnData,
    n_topics: int = 25,
    alpha: float = 50,
    eta: float = 0.1,
    n_iter: int = 150,
    random_state: int = 123,
    filter_unknown: bool = True,
    method: TopicMethod = "lda_gibbs",
    binarize: bool = False,
    min_regions_per_pattern: int = 1,
    max_regions_frac: float = 1.0,
    weighting: Weighting = "count",
    n_starts: int = 1,
    store_key: str = "topic_modeling",
    btm_impl: BTMImpl = "internal_em",
    btm_token_mode: BTMTokenMode = "binary",
    btm_max_biterms_per_region: int = 5000,
    btm_max_tokens_per_region: int | None = 500,
    btm_window_size: int | None = None,
    btm_allow_self_pairs: bool = False,
    region_ids: set[str] | None = None,
    verbose: bool = True,
) -> None:
    """Discover co-occurring motif patterns using topic modeling on region-level data.

    Parameters
    ----------
    adata
        AnnData object with cluster assignments and genomic coordinates.
        Must contain:
        - adata.obs["leiden"]: cluster assignments
        - adata.obs["example_idx"]: region identifier
        - adata.obs["start"], adata.obs["end"]: seqlet coordinates
        - adata.obs["cluster_dbd"]: optional DBD annotations
    n_topics
        Number of topics to discover
    alpha
        Dirichlet prior for document-topic distribution
    eta
        Dirichlet prior for topic-word distribution
    n_iter
        Number of LDA iterations
    random_state
        Random seed for reproducibility
    filter_unknown
        Whether to filter out seqlets with unknown DBD annotations
    method
        Topic modeling method:
        - "lda_gibbs"   : collapsed Gibbs LDA using `lda` package (default)
        - "lda_sklearn" : variational LDA using sklearn
        - "nmf"         : Non-negative matrix factorization (strong sparse baseline)
        - "btm"         : Biterm Topic Model (short-text oriented)
    binarize
        If True, treat cluster presence as binary per region.
    min_regions_per_pattern / max_regions_frac
        Document-frequency (DF) filter to remove ultra-rare and near-ubiquitous clusters.
    weighting
        Input weighting ("count", "tfidf") for methods accepting float sparse inputs ("lda_sklearn", "nmf").
        For NMF, TF-IDF is often strong.
    n_starts
        Number of random restarts; best model selected by internal objective.
        Default is 1 for backwards compatibility.
    store_key
        Key under `adata.uns` where results are stored (default: "topic_modeling").
        Useful for side-by-side comparison of multiple methods/configs.
    btm_impl
        BTM implementation to use when method="btm":
        - "internal_em" : built-in EM (no additional dependencies)
        - "bitermplus"  : use `bitermplus` if installed
    btm_token_mode
        Token interpretation for BTM:
        - "binary": treat clusters as presence/absence per region (recommended, robust)
        - "count" : treat clusters as counts (multiplicity / sampling weights)
    btm_max_biterms_per_region
        For btm_impl="bitermplus": maximum number of biterms to sample per region when
        generating biterms explicitly (prevents memory blow-up).
    btm_max_tokens_per_region
        For btm_impl="bitermplus" and btm_token_mode="count": cap the number of tokens per
        region when building doc vectors (downsample with replacement).
    btm_window_size
        For bitermplus BTMClassifier fallback: sliding window size for biterm generation.
        If None, set to cover the whole document (order-free).
    btm_allow_self_pairs
        For btm_impl="bitermplus" and btm_token_mode="count": if True, allow self-biterms (i,i)
        when sampling biterms in count mode. Default False (more robust).

    Returns
    -------
    None
        Results are stored in `adata.uns[store_key]`.
    """
    # Check required columns
    required_cols = ["leiden", "example_idx", "start", "end"]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in adata.obs: {missing_cols}")

    # Create deduplicated seqlets table (do not mutate adata.obs)
    obs = adata.obs.copy()
    obs["region_id"] = obs["example_idx"].astype(str)
    dedup_cols = ["region_id", "start", "end", "leiden"]
    if "cluster_dbd" in obs.columns:
        dedup_cols.append("cluster_dbd")
    seqlets_dedup = obs[dedup_cols].drop_duplicates()

    # Filter out unknown DBD annotations if requested
    if filter_unknown and "cluster_dbd" in seqlets_dedup.columns:
        initial_count = len(seqlets_dedup)
        seqlets_dedup = seqlets_dedup.loc[seqlets_dedup["cluster_dbd"].notna()]
        seqlets_dedup = seqlets_dedup.loc[seqlets_dedup["cluster_dbd"].astype(str) != "nan"]
        print(f"Filtered {initial_count - len(seqlets_dedup)} seqlets with unknown DBD annotations")

    # Optional: restrict to a subset of regions (used for train/test evaluation workflows)
    if region_ids is not None:
        region_ids = {str(r) for r in region_ids}
        before = len(seqlets_dedup)
        seqlets_dedup = seqlets_dedup.loc[seqlets_dedup["region_id"].astype(str).isin(region_ids)]
        if verbose:
            print(
                f"Filtered to {seqlets_dedup['region_id'].nunique()} regions "
                f"({before} -> {len(seqlets_dedup)} seqlets)"
            )
        if seqlets_dedup["region_id"].nunique() == 0:
            raise ValueError("No regions remain after applying region_ids filter.")

    print(
        f"Using {len(seqlets_dedup)} deduplicated seqlets across {seqlets_dedup['region_id'].nunique()} regions"
    )

    # Build sparse region×cluster counts
    cm = _build_region_cluster_counts(seqlets_dedup, region_col="region_id", cluster_col="leiden")
    X_counts = cm.X
    region_names = cm.region_names
    cluster_names = cm.pattern_names

    # DF filter (optional)
    X_counts, cluster_names, keep_mask = _filter_patterns_by_df(
        X_counts,
        pattern_names=cluster_names,
        min_regions=min_regions_per_pattern,
        max_regions_frac=max_regions_frac,
    )
    if keep_mask.sum() != keep_mask.size:
        print(f"DF filter kept {keep_mask.sum()}/{keep_mask.size} clusters")

    # Optional binarization
    X_model = _maybe_binarize(X_counts, binarize=binarize)

    # Build a pandas count table for storage/compatibility
    count_table = pd.DataFrame.sparse.from_spmatrix(X_model, index=region_names, columns=cluster_names)
    count_table.index.name = "region_id"
    count_table.columns.name = "cluster"

    print(f"Count matrix shape: {count_table.shape} (regions × clusters)")

    # Prepare training matrix
    if method in {"lda_sklearn", "nmf"}:
        X_train = _apply_weighting(X_model, weighting=weighting)
    else:
        X_train = X_model

    best: dict | None = None
    best_score: float | None = None

    doc_topic_prior_used = None
    topic_word_prior_used = None

    for s in range(max(1, int(n_starts))):
        seed = int(random_state) + s

        if method == "lda_gibbs":
            # `lda` expects dense integer counts
            Xt = X_train
            if sp.issparse(Xt):
                Xt = Xt.toarray().astype(np.int32)
            else:
                Xt = np.asarray(Xt).astype(np.int32)

            model = lda.LDA(
                n_topics=int(n_topics),
                n_iter=int(n_iter),
                random_state=seed,
                alpha=float(alpha) / float(n_topics),
                eta=float(eta),
            )
            model.fit(Xt)

            ll = float(loglikelihood(model.nzw_, model.ndz_, float(alpha) / float(n_topics), float(eta)))
            score = ll

            region_topic = np.asarray(model.doc_topic_, dtype=np.float64)
            topic_word = np.asarray(model.topic_word_, dtype=np.float64)  # (K,V)

        elif method == "lda_sklearn":
            from sklearn.decomposition import LatentDirichletAllocation

            # scikit-learn constrains priors to [0, 1]
            # clip the per-topic prior if needed (common default alpha=50 would otherwise fail)
            doc_topic_prior = float(alpha) / float(n_topics)
            topic_word_prior = float(eta)
            doc_topic_prior_used = min(max(doc_topic_prior, 1e-6), 1.0)
            topic_word_prior_used = min(max(topic_word_prior, 1e-6), 1.0)
            if verbose and (
                abs(doc_topic_prior_used - doc_topic_prior) > 1e-12
                or abs(topic_word_prior_used - topic_word_prior) > 1e-12
            ):
                print(
                    f"[lda_sklearn] Clipped priors to satisfy sklearn constraints: "
                    f"doc_topic_prior={doc_topic_prior_used:.4g} (from {doc_topic_prior:.4g}), "
                    f"topic_word_prior={topic_word_prior_used:.4g} (from {topic_word_prior:.4g})"
                )

            model = LatentDirichletAllocation(
                n_components=int(n_topics),
                doc_topic_prior=doc_topic_prior_used,
                topic_word_prior=topic_word_prior_used,
                max_iter=int(n_iter),
                learning_method="batch",
                random_state=seed,
                evaluate_every=-1,
            )
            model.fit(X_train)

            score = float(model.score(X_train))
            region_topic = np.asarray(model.transform(X_train), dtype=np.float64)
            tw = np.asarray(model.components_, dtype=np.float64)
            topic_word = tw / (tw.sum(axis=1, keepdims=True) + 1e-12)

        elif method == "nmf":
            from sklearn.decomposition import NMF

            model = NMF(
                n_components=int(n_topics),
                init="nndsvda",
                random_state=seed,
                max_iter=int(max(200, int(n_iter))),
            )
            W = np.asarray(model.fit_transform(X_train), dtype=np.float64)
            H = np.asarray(model.components_, dtype=np.float64)
            score = -float(model.reconstruction_err_)
            region_topic = W
            topic_word = H / (H.sum(axis=1, keepdims=True) + 1e-12)

        elif method == "btm":
            # BTM is designed for short-text-like data. For TF-MINDI's region×cluster matrix,
            # regions behave like "documents" (bags of motif-cluster tokens). Because there is
            # no inherent token order, we get all unordered pairs of tokens within each region.

            # token_mode:
            #   - "binary": presence/absence per region (recommended default; robust)
            #   - "count" : use counts as multiplicity / sampling weights (optional)
            if btm_token_mode not in ("binary", "count"):
                raise ValueError(f"Unknown btm_token_mode: {btm_token_mode}")

            if btm_token_mode == "binary":
                X_tok = _maybe_binarize(X_counts, binarize=True)
            else:
                # Use raw counts; keep sparse
                X_tok = X_counts.copy()
                # Ensure integer dtype for downstream expectations
                X_tok.data = X_tok.data.astype(np.int32, copy=False)

            if btm_impl == "internal_em":
                theta, phi, ll, iters, conv = _btm_em_fit(
                    X_tok,
                    n_topics=int(n_topics),
                    alpha=float(alpha), 
                    beta=float(eta),
                    n_iter=int(n_iter),
                    random_state=seed,
                )
                region_topic = _btm_infer_doc_topics(X_tok, theta=theta, phi=phi)
                topic_word = phi
                model = BTMModel(
                    n_topics=int(n_topics),
                    topic_word_=topic_word,
                    doc_topic_=region_topic,
                    theta_=theta,
                    loglikelihood_=float(ll),
                    n_iter=int(iters),
                    converged=bool(conv),
                )
                score = float(ll)

            elif btm_impl == "bitermplus":
                try:
                    import bitermplus as btm  # type: ignore
                except Exception as e:  # noqa: BLE001
                    raise ImportError(
                        "BTM with btm_impl='bitermplus' requires `bitermplus`. "
                        "Install it with `pip install bitermplus`."
                    ) from e

                # Prefer the low-level BTM API (directly takes doc-term matrix + biterms),
                # because it avoids text tokenization and makes the 'all unordered pairs' logic explicit.
                if hasattr(btm, "BTM"):
                    # Generate global biterms list
                    biterms = _btm_generate_biterms(
                        X_tok,
                        token_mode=btm_token_mode,
                        max_biterms_per_region=int(btm_max_biterms_per_region),
                        random_state=seed,
                        allow_self_pairs=bool(btm_allow_self_pairs),
                    )

                    vocabulary = np.asarray(cluster_names, dtype=object)

                    # bitermplus uses per-topic alpha
                    alpha_k = float(alpha) / float(n_topics)

                    model_bp = btm.BTM(
                        X_tok,
                        vocabulary,
                        T=int(n_topics),
                        alpha=float(alpha_k),
                        beta=float(eta),
                        seed=int(seed),
                    )
                    model_bp.fit(biterms, iterations=int(n_iter))

                    # topic-word (K,V)
                    tw = np.asarray(getattr(model_bp, "matrix_topics_words_", None), dtype=np.float64)
                    if tw.ndim != 2:
                        raise RuntimeError("bitermplus BTM did not expose matrix_topics_words_.")
                    topic_word = tw / (tw.sum(axis=1, keepdims=True) + 1e-12)

                    # doc-topic (D,K): use bitermplus transform if available
                    region_topic = None
                    if hasattr(model_bp, "transform"):
                        docs_vec = _btm_docs_vec_from_csr(
                            X_tok,
                            token_mode=btm_token_mode,
                            max_tokens_per_region=btm_max_tokens_per_region,
                            random_state=seed,
                        )
                        _ = model_bp.transform(docs_vec)
                        dt = getattr(model_bp, "matrix_docs_topics_", None)
                        if dt is not None:
                            region_topic = np.asarray(dt, dtype=np.float64)

                    if region_topic is None:
                        # fallback: infer from parameters (fast, order-free)
                        theta_tmp = np.full((int(n_topics),), 1.0 / float(n_topics), dtype=np.float64)
                        region_topic = _btm_infer_doc_topics(X_tok, theta=theta_tmp, phi=topic_word)

                    # Estimate global theta from doc-topic mixtures
                    theta_est = np.asarray(region_topic.mean(axis=0), dtype=np.float64)
                    theta_est = theta_est / (theta_est.sum() + 1e-12)

                    # Use a simple biterm log-likelihood (over the generated training biterms) for model selection
                    # p(b) = Σ_k θ_k φ_k,i φ_k,j
                    eps = 1e-32
                    bt = np.asarray([b for doc in biterms for b in doc], dtype=np.int32)
                    i_idx = bt[:, 0]
                    j_idx = bt[:, 1]
                    p_b = (theta_est[:, None] * (topic_word[:, i_idx] * topic_word[:, j_idx])).sum(axis=0) + eps
                    ll = float(np.log(p_b).sum())

                    model = BTMModel(
                        n_topics=int(n_topics),
                        topic_word_=topic_word,
                        doc_topic_=region_topic,
                        theta_=theta_est,
                        loglikelihood_=float(ll),
                        n_iter=int(n_iter),
                        converged=bool(getattr(model_bp, "converged_", False)),
                     )
                    score = float(ll)

                elif hasattr(btm, "BTMClassifier"):
                    # Fallback: BTMClassifier operates on text docs and uses a sliding window for biterm generation.
                    # To emulate "all unordered pairs" for unordered TF-MInDi tokens, we set window_size to cover
                    # the whole (possibly downsampled) document.
                    X_tok_csr = X_tok.tocsr()

                    rng = np.random.default_rng(int(seed))
                    docs: list[str] = []
                    max_len = 0

                    for d in range(X_tok_csr.shape[0]):
                        start, end = X_tok_csr.indptr[d], X_tok_csr.indptr[d + 1]
                        idx = X_tok_csr.indices[start:end]
                        if idx.size == 0:
                            docs.append("")
                            continue

                        if btm_token_mode == "binary":
                            tokens = [cluster_names[i] for i in idx]
                        else:
                            cnt = X_tok_csr.data[start:end].astype(np.int64)
                            total = int(cnt.sum())
                            cap = None if btm_max_tokens_per_region is None else int(btm_max_tokens_per_region)
                            if cap is None or total <= cap:
                                rep_idx = np.repeat(idx, cnt)
                            else:
                                p = cnt.astype(np.float64)
                                p = p / p.sum()
                                rep_idx = rng.choice(idx, size=cap, replace=True, p=p)
                            tokens = [cluster_names[i] for i in rep_idx]

                        docs.append(" ".join(tokens))
                        if len(tokens) > max_len:
                            max_len = len(tokens)

                    # window_size: ensure it spans the document to get all pairs
                    win = int(btm_window_size) if btm_window_size is not None else max(2, max_len)

                    alpha_k = float(alpha) / float(n_topics)

                    try:
                        model_bp = btm.BTMClassifier(
                            n_topics=int(n_topics),
                            alpha=float(alpha_k),
                            beta=float(eta),
                            max_iter=int(n_iter),
                            random_state=seed,
                            window_size=win,
                            vectorizer_params={
                                "vocabulary": cluster_names,
                                "lowercase": False,
                                "token_pattern": r"(?u)\b\w+\b",
                            },
                        )
                    except TypeError:
                        # Older bitermplus versions may not accept window_size
                        model_bp = btm.BTMClassifier(
                            n_topics=int(n_topics),
                            alpha=float(alpha_k),
                            beta=float(eta),
                            max_iter=int(n_iter),
                            random_state=seed,
                            vectorizer_params={
                                "vocabulary": cluster_names,
                                "lowercase": False,
                                "token_pattern": r"(?u)\b\w+\b",
                            },
                        )

                    region_topic = np.asarray(model_bp.fit_transform(docs), dtype=np.float64)

                    wt = getattr(model_bp, "df_words_topics_", None)
                    if wt is None:
                        raise RuntimeError("bitermplus model did not expose df_words_topics_.")
                    wt = wt.reindex(cluster_names)  # align to our vocabulary
                    topic_word = wt.values.T.astype(np.float64)
                    topic_word = topic_word / (topic_word.sum(axis=1, keepdims=True) + 1e-12)

                    theta_est = np.asarray(region_topic.mean(axis=0), dtype=np.float64)
                    theta_est = theta_est / (theta_est.sum() + 1e-12)

                    ll = float(getattr(model_bp, "loglikelihood_", np.nan))
                    model = BTMModel(
                        n_topics=int(n_topics),
                        topic_word_=topic_word,
                        doc_topic_=region_topic,
                        theta_=theta_est,
                        loglikelihood_=ll,
                        n_iter=int(n_iter),
                        converged=bool(getattr(model_bp, "converged_", False)),
                    )
                    score = float(ll) if np.isfinite(ll) else float(theta_est.sum())

                else:
                    raise ImportError(
                        "Installed bitermplus does not expose BTM/BTMClassifier. "
                        "Please update bitermplus or use btm_impl='internal_em'."
                    )
            else:
                raise ValueError(f"Unknown btm_impl: {btm_impl}")

        else:
            raise ValueError(f"Unknown method: {method}")

        if best_score is None or float(score) > float(best_score):
            best_score = float(score)
            best = {
                "model": model,
                "region_topic": region_topic,
                "topic_word": topic_word,
                "seed": seed,
                "internal_score": float(score),
            }

    assert best is not None

    # Normalize region-topic (safe for NMF)
    rt = np.asarray(best["region_topic"], dtype=np.float64)
    rt_sum = rt.sum(axis=1, keepdims=True)
    rt_sum[rt_sum == 0] = 1.0
    rt = rt / rt_sum

    topic_cols = [f"Topic_{i + 1}" for i in range(int(n_topics))]
    region_topic_df = pd.DataFrame(rt, index=region_names, columns=topic_cols)

    # Normalize topic-word to probability simplex
    tw = np.asarray(best["topic_word"], dtype=np.float64)
    tw_sum = tw.sum(axis=1, keepdims=True)
    tw_sum[tw_sum == 0] = 1.0
    tw = tw / tw_sum

    topic_cluster_df = pd.DataFrame(
        tw.T,
        index=[str(x) for x in cluster_names],
        columns=topic_cols,
    )

    # Store results
    adata.uns[store_key] = {
        "model": best["model"],
        "params": {
            "method": method,
            "n_topics": int(n_topics),
            "alpha": float(alpha),
            "eta": float(eta),
            "n_iter": int(n_iter),
            "random_state": int(random_state),
            "filter_unknown": bool(filter_unknown),
            "binarize": bool(binarize),
            "min_regions_per_pattern": int(min_regions_per_pattern),
            "max_regions_frac": float(max_regions_frac),
            "weighting": weighting,
            "n_starts": int(n_starts),
            "best_seed": int(best["seed"]),
            "internal_score": float(best["internal_score"]),
            "doc_topic_prior_used": (doc_topic_prior_used if method == "lda_sklearn" else None),
            "topic_word_prior_used": (topic_word_prior_used if method == "lda_sklearn" else None),
            "btm_impl": btm_impl if method == "btm" else None,
            "btm_token_mode": btm_token_mode if method == "btm" else None,
            "btm_max_biterms_per_region": int(btm_max_biterms_per_region) if method == "btm" else None,
            "btm_max_tokens_per_region": (None if btm_max_tokens_per_region is None else int(btm_max_tokens_per_region)) if method == "btm" else None,
            "btm_window_size": (None if btm_window_size is None else int(btm_window_size)) if method == "btm" else None,
        },
        "count_matrix": count_table,
        "topic_cluster_matrix": topic_cluster_df,
        "region_names": list(region_names),
        "region_topic_matrix": region_topic_df,
    }

    if verbose:
        print(f"Stored topic modeling results in adata.uns['{store_key}']")


def evaluate_topic_models(
    adata: AnnData,
    n_topics_range: list[int] | None = None,
    alpha: float = 50,
    eta: float = 0.1,
    n_iter: int = 150,
    random_state: int = 123,
    method: TopicMethod = "lda_gibbs",
    metric: str = "loglikelihood",
    metrics: list[str] | None = None,
    metric_weights: dict[str, float] | None = None,
    select_model: int | None = None,
    label_key: str | None = None,
    top_n_coherence: int = 20,
    knn_k: int = 30,
    min_topics_coh: int = 5,
    plot: bool = False,
    plot_metrics: bool = False,
    figsize: tuple[float, float] = (6.4, 4.8),
    save_path: str | None = None,
    store_key: str = "topic_modeling",
    return_df: bool = False,
    **kwargs,
) -> dict[int, float] | pd.DataFrame:
    """Evaluate topic models over a range of topic numbers and select an optimal model.

    This function extends TF-MINDI's original `evaluate_topic_models` in a
    backwards-compatible way:

    * **Single-metric mode (legacy)**:
        - set ``metrics=None`` (default)
        - provide ``metric=...`` (default: ``"loglikelihood"``)
        - returns ``dict[n_topics -> score]`` where **higher is better** (for
          minimization metrics, the returned values are sign-inverted).

    * **Multi-metric mode (pycisTopic-style)**:
        - provide ``metrics=[...]``
        - returns a tidy ``pd.DataFrame`` with raw and rescaled metrics, plus a
          ``combined_score`` used for selection
        - optionally plots rescaled metrics over ``n_topics`` and marks the
          selected model.

    Parameters
    ----------
    method
        Topic model backend to evaluate (must match `run_topic_modeling`).
    metric
        Single-metric score to optimize (legacy mode). Higher is better.
        Common values:
          - "loglikelihood"      (lda_gibbs, lda_sklearn, btm)
          - "perplexity"         (lda_sklearn; returned as -perplexity)
          - "reconstruction_err" (nmf; returned as -reconstruction_err)
          - "coherence_umass"    (all methods)
    metrics
        List of metric names for multi-metric selection. Recommended defaults:
          ["coherence_umass", "topic_diversity", "redundancy_cosine"]
        Optional label-driven tie-breakers (if label_key is provided):
          ["label_knn_purity", "label_ami_argmax"]
    metric_weights
        Optional weights (name -> weight) used when computing `combined_score`.
        Unspecified metrics default to 1.0.
    select_model
        If provided, overrides automatic selection and forces a specific n_topics.
    label_key
        Optional label (e.g. "cell_type") in `adata.obs` used for extrinsic
        validation metrics (kNN purity, AMI).
    min_topics_coh
        If coherence is used in `metrics`, coherence values for n_topics < min_topics_coh
        are ignored in the combined score (matches pycisTopic behavior).
    plot
        Plot rescaled metrics over number of topics (multi-metric mode).
    plot_metrics
        Also generate per-metric plots (multi-metric mode).
    save_path
        If provided, saves the combined plot. If it ends with ".pdf" and plot_metrics=True,
        all plots are saved into the same PDF.
    store_key
        Where to store the best model. Intermediate models overwrite this key.
    return_df
        If True in single-metric mode, also returns a DataFrame with the single metric.

    Returns
    -------
    dict[int, float] or pd.DataFrame
        Legacy single-metric mode: dict mapping n_topics to score.
        Multi-metric mode: DataFrame indexed by n_topics with metric columns and
        `combined_score`. The best model is re-fit and stored in `adata.uns[store_key]`.
    """
    if n_topics_range is None:
        n_topics_range = [10, 15, 20, 25, 30, 35, 40, 50]

    # Metric directions are defined on the *raw* scale.
    # For multi-metric selection, we invert "minimize" metrics so that larger is better
    # before rescaling to [0,1].
    metric_direction: dict[str, str] = {
        "loglikelihood": "max",
        "perplexity": "min",
        "reconstruction_err": "min",
        "coherence_umass": "max",
        # pycisTopic-style coherence alias (Mimno et al., 2011)
        "Mimno_2011": "max",
        "redundancy_cosine": "min",
        # pycisTopic-style density-based metric (Cao Juan et al., 2009)
        "Cao_Juan_2009": "min",
        "topic_diversity": "max",
        "median_effective_topics": "min",
        # pycisTopic-style divergence-based metric (Arun et al., 2010)
        "Arun_2010": "min",
        "label_ami_argmax": "max",
        "label_knn_purity": "max",
        # convenience: expose internal objective (already "higher is better" for our runs)
        "internal_score": "max",
    }

    def _safe_rescale(v: pd.Series) -> pd.Series:
        """Rescale a numeric series to [0,1] robustly (NaNs preserved)."""
        vv = v.astype(float)
        mask = vv.notna()
        if mask.sum() <= 1:
            out = vv.copy()
            out[mask] = 0.0
            return out
        lo = float(vv[mask].min())
        hi = float(vv[mask].max())
        if hi - lo < 1e-12:
            out = vv.copy()
            out[mask] = 0.0
            return out
        out = (vv - lo) / (hi - lo)
        return out

    # ----------------------------
    # Legacy: single metric
    # ----------------------------
    if metrics is None:
        print(f"Evaluating {len(n_topics_range)} models: method={method}, metric={metric}")

        scores: dict[int, float] = {}
        best_n: int | None = None
        best_score: float | None = None

        for n_topics in n_topics_range:
            run_topic_modeling(
                adata,
                n_topics=int(n_topics),
                alpha=float(alpha),
                eta=float(eta),
                n_iter=int(n_iter),
                random_state=int(random_state),
                method=method,
                store_key=store_key,
                **kwargs,
            )
            res = adata.uns[store_key]

            # Compute a "higher is better" score for the requested metric.
            if metric in {
                "coherence_umass",
                "Mimno_2011",
                "redundancy_cosine",
                "Cao_Juan_2009",
                "Arun_2010",
                "topic_diversity",
                "median_effective_topics",
            } or (
                metric.startswith("label_") and label_key is not None
            ):
                q = compute_topic_model_quality(
                    adata,
                    topic_key=store_key,
                    label_key=label_key,
                    top_n_coherence=int(top_n_coherence),
                    knn_k=int(knn_k),
                )
                if metric not in q:
                    raise ValueError(f"Metric '{metric}' not available (label_key required?)")
                raw = float(q[metric])
                if metric_direction.get(metric, "max") == "min":
                    sc = -raw
                else:
                    sc = raw

            elif metric == "loglikelihood":
                m = res["model"]
                if method == "lda_gibbs":
                    sc = float(
                        loglikelihood(
                            m.nzw_,
                            m.ndz_,
                            float(alpha) / float(n_topics),
                            float(eta),
                        )
                    )
                elif method == "lda_sklearn":
                    X = _apply_weighting(_df_to_csr(res["count_matrix"]), weighting=res["params"].get("weighting", "count"))
                    sc = float(m.score(X))
                elif method == "btm":
                    sc = float(getattr(m, "loglikelihood_", res["params"].get("internal_score", np.nan)))
                else:
                    raise ValueError("metric='loglikelihood' requires method='lda_gibbs', 'lda_sklearn', or 'btm'")

            elif metric == "perplexity":
                if method != "lda_sklearn":
                    raise ValueError("metric='perplexity' requires method='lda_sklearn'")
                m = res["model"]
                X = _apply_weighting(_df_to_csr(res["count_matrix"]), weighting=res["params"].get("weighting", "count"))
                sc = -float(m.perplexity(X))  # higher is better

            elif metric == "reconstruction_err":
                if method != "nmf":
                    raise ValueError("metric='reconstruction_err' requires method='nmf'")
                m = res["model"]
                sc = -float(getattr(m, "reconstruction_err_", np.nan))  # higher is better

            elif metric == "internal_score":
                sc = float(res["params"].get("internal_score", np.nan))

            else:
                raise ValueError(f"Unknown metric: {metric}")

            scores[int(n_topics)] = float(sc)
            if best_score is None or float(sc) > float(best_score):
                best_score = float(sc)
                best_n = int(n_topics)

            print(f"  score({n_topics}) = {sc:.4f}")

        assert best_n is not None

        # Refit best model to ensure it is stored in `store_key`
        run_topic_modeling(
            adata,
            n_topics=int(best_n),
            alpha=float(alpha),
            eta=float(eta),
            n_iter=int(n_iter),
            random_state=int(random_state),
            method=method,
            store_key=store_key,
            **kwargs,
        )

        print(f"Best n_topics by {metric}: {best_n} (score={best_score:.4f})")

        if not return_df:
            return scores

        # Optional: convert to DataFrame for convenience
        df = pd.DataFrame({"n_topics": list(scores.keys()), metric: list(scores.values())}).set_index("n_topics")
        df.attrs["best_n_topics"] = best_n
        return df

    # ----------------------------
    # Multi-metric selection
    # ----------------------------
    metrics = [str(m) for m in metrics]
    unknown = [m for m in metrics if m not in metric_direction]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Supported: {sorted(metric_direction.keys())}")

    print(f"Evaluating {len(n_topics_range)} models: method={method}, metrics={metrics}")

    rows: list[dict[str, float]] = []
    for n_topics in n_topics_range:
        run_topic_modeling(
            adata,
            n_topics=int(n_topics),
            alpha=float(alpha),
            eta=float(eta),
            n_iter=int(n_iter),
            random_state=int(random_state),
            method=method,
            store_key=store_key,
            **kwargs,
        )
        res = adata.uns[store_key]
        m = res["model"]

        row: dict[str, float] = {"n_topics": float(n_topics)}

        # Always compute interpretability/diagnostic metrics.
        q = compute_topic_model_quality(
            adata,
            topic_key=store_key,
            label_key=label_key,
            top_n_coherence=int(top_n_coherence),
            knn_k=int(knn_k),
        )
        row.update({k: float(v) for k, v in q.items()})

        # Fit/likelihood metrics (method-dependent)
        row["internal_score"] = float(res["params"].get("internal_score", np.nan))

        if method == "lda_gibbs":
            row["loglikelihood"] = float(
                loglikelihood(
                    m.nzw_,
                    m.ndz_,
                    float(alpha) / float(n_topics),
                    float(eta),
                )
            )

        elif method == "lda_sklearn":
            X = _apply_weighting(_df_to_csr(res["count_matrix"]), weighting=res["params"].get("weighting", "count"))
            row["loglikelihood"] = float(m.score(X))
            row["perplexity"] = float(m.perplexity(X))

        elif method == "nmf":
            row["reconstruction_err"] = float(getattr(m, "reconstruction_err_", np.nan))

        elif method == "btm":
            row["loglikelihood"] = float(getattr(m, "loglikelihood_", res["params"].get("internal_score", np.nan)))

        rows.append(row)

    df = pd.DataFrame(rows).set_index("n_topics").sort_index()

    # Build rescaled + combined score
    weights = metric_weights or {}
    combined = pd.Series(0.0, index=df.index, dtype=float)

    for met in metrics:
        raw = df[met].astype(float) if met in df.columns else pd.Series(np.nan, index=df.index)

        # Ignore coherence for small K if requested (pycisTopic behavior)
        if met in {"coherence_umass", "Mimno_2011"} and int(min_topics_coh) > 0:
            raw = raw.where(df.index.astype(int) >= int(min_topics_coh))

        # Convert to "higher is better" for rescaling
        if metric_direction.get(met, "max") == "min":
            raw = -raw

        scaled = _safe_rescale(raw)
        w = float(weights.get(met, 1.0))
        df[f"{met}__scaled"] = scaled
        combined = combined + (w * scaled.fillna(0.0))

    df["combined_score"] = combined

    # Select best model
    if select_model is not None:
        best_n = int(select_model)
        if best_n not in df.index.astype(int).tolist():
            raise ValueError(f"select_model={best_n} not in n_topics_range")
    else:
        # If combined is all NaN, fall back to max internal_score
        if df["combined_score"].notna().sum() == 0:
            best_n = int(df["internal_score"].astype(float).idxmax())
        else:
            best_n = int(df["combined_score"].astype(float).idxmax())

    df["is_best"] = df.index.astype(int) == best_n
    df.attrs["best_n_topics"] = best_n
    df.attrs["metrics_used"] = metrics

    # Refit best model and keep it stored in `store_key`
    run_topic_modeling(
        adata,
        n_topics=int(best_n),
        alpha=float(alpha),
        eta=float(eta),
        n_iter=int(n_iter),
        random_state=int(random_state),
        method=method,
        store_key=store_key,
        **kwargs,
    )

    # Store evaluation table for later inspection
    adata.uns[f"{store_key}_evaluation"] = df

    print(f"Best n_topics by combined metrics: {best_n}")

    # Plot (combined plot of rescaled metrics)
    if plot or save_path is not None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except Exception as e:  # noqa: BLE001
            raise ImportError("Plotting requires matplotlib.") from e


        def _make_combined_fig() -> "plt.Figure":
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

            # plot rescaled metrics (minimization metrics shown as inverted)
            # do not plot the combined score curve (selection is still based on it)

            pretty = {
                "loglikelihood": "Loglikelihood",
            }

            for met in metrics:
                y = df.get(f"{met}__scaled", None)
                if y is None:
                    continue
                label = pretty.get(met, met)
                if metric_direction.get(met, "max") == "min":
                    label = f"Inv_{label}"
                ax.plot(df.index.astype(int), y.values, linestyle="--", marker="o", label=label)

            ax.axvline(best_n, linestyle="--", color="grey")
            ax.set_xlabel(f"Number of topics\nOptimal number of topics: {best_n}")
            ax.set_ylabel("Rescaled metric")
            ax.legend(bbox_to_anchor=(1.04, 1.0), loc="upper left")
            fig.tight_layout()
            return fig

        figs = []
        combined_fig = _make_combined_fig()
        figs.append(combined_fig)

        if plot_metrics:
            for met in metrics:
                if met not in df.columns:
                    continue
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)
                ax.plot(df.index.astype(int), df[met].values, linestyle="--", marker="o")
                ax.axvline(best_n, linestyle="--", color="grey")
                direction = "Maximize" if metric_direction.get(met, "max") == "max" else "Minimize"
                title = met
                if met == "loglikelihood":
                    title = "Loglikelihood"
                ax.set_title(f"{title} - {direction}")
                ax.set_xlabel("Number of topics")
                ax.set_ylabel(met)
                fig.tight_layout()
                figs.append(fig)

        if save_path is not None and str(save_path).lower().endswith(".pdf"):
            with PdfPages(save_path) as pdf:
                for fig in figs:
                    pdf.savefig(fig, bbox_inches="tight")
        elif save_path is not None:
            # Save only the combined figure to the provided path
            combined_fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if plot:
            plt.show()
        else:
            # Close figures
            for fig in figs:
                plt.close(fig)

    return df

# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------


def top_patterns_per_topic(
    adata: AnnData,
    topic_key: str = "topic_modeling",
    top_n: int = 20,
) -> pd.DataFrame:
    """Return top clusters per topic as a DataFrame."""
    if topic_key not in adata.uns:
        raise ValueError(f"No topic modeling results found at adata.uns['{topic_key}']")

    tc = adata.uns[topic_key]["topic_cluster_matrix"]  # clusters × topics
    rows = []
    for topic in tc.columns:
        s = tc[topic].sort_values(ascending=False).head(top_n)
        for rank, (cluster, weight) in enumerate(s.items(), start=1):
            rows.append({"topic": topic, "rank": rank, "cluster": str(cluster), "weight": float(weight)})
    return pd.DataFrame(rows)


def _arun_2010_metric(
    X_counts: sp.csr_matrix,
    theta: np.ndarray,
    phi: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Arun et al. (2010) divergence-based metric for selecting K (lower is better).

    This implementation follows the common definition:

        KL( s / ||s||_1  ||  q / ||q||_1 )

    where:
      - s are the singular values of the topic-word matrix (phi, shape K×V)
      - q_k = sum_d n_d * theta_{d,k} is the topic usage weighted by document lengths

    Notes
    -----
    - This metric is only meaningful when V >= K (enough features to support K topics).
    - We add eps smoothing for numerical stability.

    Returns
    -------
    float
        KL divergence (lower is better). Returns NaN if it cannot be computed reliably.
    """
    if not sp.isspmatrix_csr(X_counts):
        X_counts = X_counts.tocsr()

    D, V = X_counts.shape
    K = int(theta.shape[1]) if theta.ndim == 2 else 0
    if D == 0 or V == 0 or K == 0:
        return float("nan")

    if phi.shape[0] != K:
        return float("nan")

    if V < K:
        # Under-determined (more topics than features); metric becomes unreliable
        return float("nan")

    # Document lengths
    n_d = np.asarray(X_counts.sum(axis=1)).ravel().astype(np.float64)
    if float(n_d.sum()) <= 0:
        return float("nan")

    # Topic usage weighted by document length
    q = (theta.T @ n_d).astype(np.float64)
    q = q + eps
    q = q / q.sum()

    # Singular values of topic-word matrix
    try:
        s = np.linalg.svd(phi, compute_uv=False)
    except Exception:
        return float("nan")

    s = np.asarray(s, dtype=np.float64)
    if s.size != q.size:
        return float("nan")

    p = s + eps
    p = p / p.sum()

    kl = float(np.sum(p * np.log(p / q)))
    return kl


def _compute_topic_model_quality_from_result(
    adata: AnnData,
    res: dict,
    label_key: str | None = None,
    top_n_coherence: int = 20,
    top_topics_coh: int = 5,
    knn_k: int = 30,
    random_state: int = 0,
) -> dict[str, float]:
    """
    Compute lightweight quality metrics given a topic model result dict.

    This is used by both:
      - compute_topic_model_quality (when results are stored in adata.uns)
      - model selection utilities (when results are cached for multiple K)

    Returns
    -------
    dict[str, float]
        Metric name -> value.
    """
    count_df = res.get("count_matrix")
    topic_cluster = res.get("topic_cluster_matrix")
    region_topic = res.get("region_topic_matrix")

    if count_df is None or topic_cluster is None or region_topic is None:
        raise ValueError("Result dict must contain 'count_matrix', 'topic_cluster_matrix', and 'region_topic_matrix'")

    # Convert count table to sparse counts
    X_counts = _df_to_csr(count_df)

    # Binary view for coherence (UMass)
    X_bin = X_counts.copy()
    X_bin.data = np.ones_like(X_bin.data)

    # topic_word as (K,V)
    tw = topic_cluster.T.values.astype(np.float64)
    theta = region_topic.values.astype(np.float64)

    # Coherence per topic + aggregates
    per_topic_coh = topic_coherence_umass_per_topic(
        X_bin=X_bin,
        topic_word=tw,
        top_n=int(top_n_coherence),
        eps=1.0,
    )
    finite = np.isfinite(per_topic_coh)
    if finite.sum() == 0:
        coh_mean = float("-inf")
        mimno = float("-inf")
    else:
        coh_mean = float(per_topic_coh[finite].mean())
        # pycisTopic "Mimno_2011": average of top coherence values to reduce K bias
        top_topics = int(max(1, min(int(top_topics_coh), finite.sum())))
        mimno = float(np.sort(per_topic_coh[finite])[-top_topics:].mean())

    # Topic redundancy (Cao Juan 2009 style; lower is better)
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(tw)
    K = sims.shape[0]
    if K <= 1:
        redundancy = 0.0
    else:
        redundancy = float((sims.sum() - np.trace(sims)) / (K * (K - 1)))

    # Topic diversity (unique top words across topics)
    top_idx = np.argsort(tw, axis=1)[:, ::-1][:, : int(top_n_coherence)]
    unique = len(set(top_idx.ravel().tolist()))
    diversity = float(unique / (K * int(top_n_coherence))) if K > 0 else 0.0

    # Effective topics per region
    p = np.clip(theta, 1e-12, 1.0)
    ent = -np.sum(p * np.log(p), axis=1)
    eff = np.exp(ent)
    median_eff = float(np.median(eff)) if eff.size > 0 else float("nan")

    # Arun 2010 divergence (lower is better)
    arun = _arun_2010_metric(X_counts=X_counts, theta=theta, phi=tw)

    out: dict[str, float] = {
        # TF-MINDI-native names
        "coherence_umass": coh_mean,
        "redundancy_cosine": redundancy,
        "topic_diversity": diversity,
        "median_effective_topics": median_eff,
        # pycisTopic-compatible names
        # (Mimno et al., 2011 coherence; Cao Juan et al., 2009 density; Arun et al., 2010 divergence)
        "Mimno_2011": mimno,
        "Cao_Juan_2009": redundancy,
        "Arun_2010": arun,
    }

    # Optional: alignment to known labels (e.g. cell_type)
    if label_key is not None and label_key in adata.obs.columns:
        # Map region_id -> label
        tmp = adata.obs[["example_idx", label_key]].copy()
        tmp = tmp.dropna()
        tmp["example_idx"] = tmp["example_idx"].astype(str)

        def _mode(s: pd.Series) -> str:
            m = s.mode(dropna=True)
            if len(m) > 0:
                return str(m.iloc[0])
            return str(s.iloc[0])

        region_label = tmp.groupby("example_idx", observed=True)[label_key].apply(_mode)

        # Align to region_topic index
        idx = pd.Index(region_topic.index.astype(str))
        y = region_label.reindex(idx).fillna("Unknown").astype(str).values
        Xr = theta.astype(np.float64)

        # Discrete agreement between labels and the dominant topic per region
        try:
            from sklearn.metrics import adjusted_mutual_info_score

            pred = region_topic.idxmax(axis=1).astype(str).values
            mask = y != "Unknown"
            if mask.sum() > 1:
                out["label_ami_argmax"] = float(adjusted_mutual_info_score(y[mask], pred[mask]))
            else:
                out["label_ami_argmax"] = float("nan")
        except Exception:
            out["label_ami_argmax"] = float("nan")

        # kNN purity in topic space
        try:
            from sklearn.neighbors import NearestNeighbors

            k = int(min(knn_k, max(2, Xr.shape[0] - 1)))
            nn = NearestNeighbors(n_neighbors=k, metric="cosine")
            nn.fit(Xr)
            neigh = nn.kneighbors(return_distance=False)
            # exclude self (first neighbor)
            neigh = neigh[:, 1:]
            pur = []
            for i in range(neigh.shape[0]):
                if y[i] == "Unknown":
                    continue
                pur.append(float(np.mean(y[neigh[i]] == y[i])))
            out["label_knn_purity"] = float(np.mean(pur)) if pur else float("nan")
        except Exception:
            out["label_knn_purity"] = float("nan")

    return out


def compute_topic_model_quality(
    adata: AnnData,
    topic_key: str = "topic_modeling",
    label_key: str | None = None,
    top_n_coherence: int = 20,
    top_topics_coh: int = 5,
    knn_k: int = 30,
    random_state: int = 0,
) -> dict[str, float]:
    """Compute lightweight quality metrics for a stored topic model.

    This is primarily intended for **comparative evaluation** of topic models
    for biological interpretability, not for strict statistical model selection.

    Parameters
    ----------
    topic_key
        Key under `adata.uns` containing topic modeling results.
    label_key
        Optional `adata.obs` column to evaluate label-topic alignment (e.g. cell_type).
    top_n_coherence
        Number of top patterns per topic used for coherence/diversity.
    top_topics_coh
        Number of top-coherent topics to average for Mimno_2011 (pycisTopic-style).
        Default 5 matches pycisTopic.
    knn_k
        k for kNN purity in region-topic space.

    Returns
    -------
    dict[str, float]
        Metrics for the stored topic model.
    """
    if topic_key not in adata.uns:
        raise KeyError(f"Topic modeling results not found in adata.uns['{topic_key}']")

    res = adata.uns[topic_key]
    return _compute_topic_model_quality_from_result(
        adata,
        res,
        label_key=label_key,
        top_n_coherence=top_n_coherence,
        top_topics_coh=top_topics_coh,
        knn_k=knn_k,
    )


def benchmark_topic_models(
    adata: AnnData,
    configs: list[dict],
    label_key: str | None = None,
    select_by: str = "coherence_umass",
    store_best: bool = True,
    best_store_key: str = "topic_modeling",
    candidates_key: str | None = None,
) -> pd.DataFrame:
    """Run and compare multiple topic modeling configurations.

    Each config is a dict of keyword arguments passed to `run_topic_modeling`.
    Recommended that each config contains a `name` key.
    """
    rows = []

    for i, cfg in enumerate(configs):
        cfg = dict(cfg)
        name = str(cfg.pop("name", f"model_{i+1}"))
        tmp_key = cfg.pop("store_key", f"_topic_modeling_{name}")

        run_topic_modeling(adata, store_key=tmp_key, **cfg)
        q = compute_topic_model_quality(adata, topic_key=tmp_key, label_key=label_key)
        params = adata.uns[tmp_key]["params"].copy()

        row = {"name": name, "store_key": tmp_key, **q, **params}
        rows.append(row)

    df = pd.DataFrame(rows)

    if candidates_key is not None:
        # Store the benchmark summary for later inspection
        adata.uns[candidates_key] = df

    if select_by in df.columns and store_best:
        best_idx = df[select_by].astype(float).idxmax()
        best_key = df.loc[best_idx, "store_key"]
        # Copy selected model into the standard location
        adata.uns[best_store_key] = adata.uns[best_key]
        print(f"Selected best model by {select_by}: {df.loc[best_idx, 'name']} -> adata.uns['{best_store_key}']")

    return df


# ----------------------------
# model selection helpers
# ----------------------------

def fit_topic_models(
    adata: AnnData,
    n_topics_range: list[int] | None = None,
    method: TopicMethod = "lda_gibbs",
    alpha: float = 50.0,
    eta: float = 0.1,
    n_iter: int = 150,
    random_state: int = 123,
    store_key: str = "topic_modeling",
    models_key: str = "topic_modeling_models",
    overwrite: bool = True,
    keep_model_objects: bool = True,
    verbose: bool = True,
    heldout_fraction: float = 0.0,
    heldout_random_state: int = 0,
    fit_scope: Literal["full", "train"] | None = None,
    **kwargs,
) -> dict[int, dict]:
    """
    Fit a collection of topic models (varying K) and cache them in `adata.uns`.

    Split-aware best practice
    -------------------------
    If ``heldout_fraction > 0`` (and ``fit_scope`` is not explicitly set), models
    are fitted on the Train split of regions and held-out metrics are computed
    later on the Test split without refitting. After selecting K, you can
    call :func:`select_topic_model(..., refit_final=True)` to refit once on
    all regions for downstream tasks.

    Notes
    -----
    - If ``keep_model_objects=True`` (default), full estimator objects are stored in
      ``adata.uns[models_key]``. This is convenient for interactive work, but may
      prevent saving the AnnData to disk (depending on the writer and the estimator).
    - Set ``keep_model_objects=False`` to store only serializable outputs
      (topic matrices + params), at the cost of not being able to re-use the fitted
      estimator object directly.

    Returns
    -------
    dict[int, dict]
        Mapping {n_topics -> result_dict}, where each result_dict has the same structure
        as produced by :func:`run_topic_modeling`.
    """
    if n_topics_range is None:
        n_topics_range = [10, 15, 20, 25, 30, 35, 40, 50]
    n_topics_range = sorted({int(x) for x in n_topics_range})
    if len(n_topics_range) == 0:
        raise ValueError("n_topics_range must contain at least one value")

    # Get fit_scope
    if fit_scope is None:
        fit_scope = "train" if float(heldout_fraction) > 0 else "full"
    if fit_scope not in {"full", "train"}:
        raise ValueError("fit_scope must be one of {'full','train'}")

    train_region_ids: set[str] | None = None
    test_region_ids: set[str] | None = None

    meta: dict = {
        "fit_scope": str(fit_scope),
        "heldout_fraction": float(heldout_fraction),
        "heldout_random_state": int(heldout_random_state),
        "train_region_ids": None,
        "test_region_ids": None,
    }

    if str(fit_scope) == "train":
        if float(heldout_fraction) <= 0:
            raise ValueError("fit_scope='train' requires heldout_fraction > 0")

        # Split on regions, not seqlets
        all_regions = pd.unique(adata.obs["example_idx"].astype(str))
        n_docs = len(all_regions)
        if n_docs < 2:
            raise ValueError("Need at least 2 regions to compute held-out metrics.")

        n_test = int(max(1, math.floor(float(heldout_fraction) * n_docs)))
        rng = np.random.default_rng(int(heldout_random_state))
        test = set(rng.choice(all_regions, size=n_test, replace=False).tolist())
        train = set(all_regions.tolist()) - test

        train_region_ids = train
        test_region_ids = test
        meta["train_region_ids"] = sorted(train)
        meta["test_region_ids"] = sorted(test)

        if verbose:
            print(f"[fit_topic_models] Hold-out split: train={len(train)} regions, test={len(test)} regions")

    tmp_key = f"{store_key}__tmp_fit_topic_models"
    models: dict[int, dict] = {}

    for n_topics in n_topics_range:
        if verbose:
            print(f"Fitting {method} with n_topics={n_topics}")

        run_topic_modeling(
            adata,
            n_topics=int(n_topics),
            alpha=float(alpha),
            eta=float(eta),
            n_iter=int(n_iter),
            random_state=int(random_state),
            method=method,
            store_key=tmp_key,
            verbose=bool(verbose),
            region_ids=train_region_ids if str(fit_scope) == "train" else None,
            **kwargs,
        )
        res = adata.uns[tmp_key]

        if not keep_model_objects and "model" in res:
            # drop the estimator object to make serialization safer
            res = {k: v for k, v in res.items() if k != "model"}

        models[int(n_topics)] = res

    # Clean up temp key
    if tmp_key in adata.uns:
        del adata.uns[tmp_key]

    # Reduce memory by sharing the same count_matrix object across cached models
    shared_count = None
    for k in sorted(models.keys()):
        cm = models[k].get("count_matrix")
        if shared_count is None and cm is not None:
            shared_count = cm
        elif shared_count is not None and "count_matrix" in models[k]:
            models[k]["count_matrix"] = shared_count

    # Store with string keys to be safe for AnnData writers
    payload = {str(k): v for k, v in models.items()}
    payload["__meta__"] = meta

    if (not overwrite) and (models_key in adata.uns) and isinstance(adata.uns[models_key], dict):
        adata.uns[models_key].update(payload)
    else:
        adata.uns[models_key] = payload

    return models


def compute_topic_model_metrics_table(
    adata: AnnData,
    models_key: str = "topic_modeling_models",
    out_key: str | None = None,
    label_key: str | None = None,
    top_n_coherence: int = 20,
    top_topics_coh: int = 5,
    knn_k: int = 30,
    compute_fit_metrics: bool = True,
    heldout_fraction: float = 0.0,
    heldout_random_state: int = 0,
) -> pd.DataFrame:
    """
    Compute model selection metrics for a cached set of fitted topic models.

    Held-out metrics
    ----------------
    If ``heldout_fraction > 0`` and the models were fitted on a Train split via
    :func:`fit_topic_models(..., fit_scope='train')`, this function will compute
    held-out metrics on the cached Test split without refitting the models.

    If split metadata is not available (e.g. cached models were fit on all data),
    we fall back to the standard behavior: refit a model per K on a random train split
    of the cached count matrix. However, this is slower and mixes metric scopes.

    Parameters
    ----------
    models_key
        Key under ``adata.uns`` where models were stored by :func:`fit_topic_models`.
    out_key
        Where to store the resulting DataFrame. Defaults to ``f"{models_key}_metrics"``.
    label_key
        Optional `adata.obs` column used for label-alignment metrics.
    top_n_coherence
        Number of top patterns per topic for coherence/diversity.
    top_topics_coh
        Number of top-coherent topics used for Mimno_2011 (pycisTopic-style).
    compute_fit_metrics
        If True, compute fit metrics (loglikelihood/perplexity/etc.) when the underlying
        estimator object is available in the cached results.
    heldout_fraction
        If > 0, compute held-out loglikelihood/perplexity for LDA methods.
    heldout_random_state
        Random seed for the held-out split (used only in the fallback mode).

    Returns
    -------
    pandas.DataFrame
        Rows correspond to n_topics; columns correspond to metrics.
    """
    if out_key is None:
        out_key = f"{models_key}_metrics"

    if models_key not in adata.uns:
        raise KeyError(f"Cached models not found in adata.uns['{models_key}']. Run fit_topic_models(...) first.")

    raw_models = adata.uns[models_key]
    if not isinstance(raw_models, dict):
        raise TypeError(f"adata.uns['{models_key}'] must be a dict (got {type(raw_models)!r})")

    # Optional metadata stored by `fit_topic_models` (e.g. train/test split information)
    meta = raw_models.get("__meta__", {}) if isinstance(raw_models, dict) else {}
    if not isinstance(meta, dict):
        meta = {}

    # Sort by integer K (keys might be strings)
    items: list[tuple[int, dict]] = []
    for k_str, res in raw_models.items():
        try:
            k_int = int(k_str)
        except Exception:
            continue
        if isinstance(res, dict):
            items.append((k_int, res))
    items = sorted(items, key=lambda x: x[0])
    if len(items) == 0:
        raise ValueError(f"No valid models found in adata.uns['{models_key}']")

    # Shared matrix for held-out scoring / TF-IDF fitting
    shared_count_df = None
    for _, res in items:
        if isinstance(res, dict) and res.get("count_matrix") is not None:
            shared_count_df = res["count_matrix"]
            break

    rows: list[dict[str, float]] = []
    for k_int, res in items:
        row: dict[str, float] = {"n_topics": float(k_int)}

        # Interpretability + diagnostic metrics
        q = _compute_topic_model_quality_from_result(
            adata,
            res,
            label_key=label_key,
            top_n_coherence=int(top_n_coherence),
            top_topics_coh=int(top_topics_coh),
            knn_k=int(knn_k),
        )
        row.update({str(k): float(v) for k, v in q.items()})

        # Fit metrics (method-dependent)
        params = res.get("params", {}) if isinstance(res.get("params", {}), dict) else {}
        method = params.get("method", None)

        if compute_fit_metrics:
            model_obj = res.get("model", None)
            row["internal_score"] = float(params.get("internal_score", np.nan))

            try:
                if method == "lda_gibbs" and model_obj is not None:
                    row["loglikelihood"] = float(
                        loglikelihood(
                            model_obj.nzw_,
                            model_obj.ndz_,
                            float(params.get("alpha", 50.0)) / float(k_int),
                            float(params.get("eta", 0.1)),
                        )
                    )

                elif method == "lda_sklearn" and model_obj is not None:
                    X = _apply_weighting(_df_to_csr(res["count_matrix"]), weighting=str(params.get("weighting", "count")))
                    row["loglikelihood"] = float(model_obj.score(X))
                    row["perplexity"] = float(model_obj.perplexity(X))

                elif method == "nmf" and model_obj is not None:
                    row["reconstruction_err"] = float(getattr(model_obj, "reconstruction_err_", np.nan))

                elif method == "btm" and model_obj is not None:
                    row["loglikelihood"] = float(getattr(model_obj, "loglikelihood_", params.get("internal_score", np.nan)))

            except Exception:
                # don't fail metric table construction if a backend can't be scored
                pass

        rows.append(row)

    df = pd.DataFrame(rows).set_index("n_topics").sort_index()

    # ------------------------------------------------------------------
    # Held-out scoring (LDA/BTM only)
    # ------------------------------------------------------------------
    if float(heldout_fraction) > 0:
        df["heldout_loglikelihood"] = np.nan
        df["heldout_perplexity"] = np.nan

        can_use_cached_split = (
            meta.get("fit_scope") == "train"
            and isinstance(meta.get("test_region_ids"), list)
            and len(meta.get("test_region_ids")) > 0
        )

        if can_use_cached_split:
            try:
                if shared_count_df is None:
                    raise RuntimeError("No shared count_matrix available for held-out evaluation.")

                # Training matrix and vocabulary (from cached results)
                X_train = _df_to_csr(shared_count_df)
                cluster_names = list(shared_count_df.columns.astype(str))

                # Preprocessing flags from the first cached model
                params0 = items[0][1].get("params", {}) if isinstance(items[0][1].get("params", {}), dict) else {}
                binarize = bool(params0.get("binarize", False))
                filter_unknown = bool(params0.get("filter_unknown", True))

                test_regions = {str(x) for x in meta["test_region_ids"]}

                # Build deduplicated seqlets table (same logic as run_topic_modeling)
                obs = adata.obs.copy()
                obs["region_id"] = obs["example_idx"].astype(str)
                dedup_cols = ["region_id", "start", "end", "leiden"]
                if "cluster_dbd" in obs.columns:
                    dedup_cols.append("cluster_dbd")
                seqlets_dedup = obs[dedup_cols].drop_duplicates()

                if filter_unknown and "cluster_dbd" in seqlets_dedup.columns:
                    seqlets_dedup = seqlets_dedup.loc[seqlets_dedup["cluster_dbd"].notna()]
                    seqlets_dedup = seqlets_dedup.loc[seqlets_dedup["cluster_dbd"].astype(str) != "nan"]

                seqlets_dedup["leiden"] = seqlets_dedup["leiden"].astype(str)

                # Restrict to held-out regions
                seqlets_test = seqlets_dedup.loc[seqlets_dedup["region_id"].isin(test_regions)].copy()
                if len(seqlets_test) == 0:
                    raise RuntimeError("No seqlets in held-out regions after filtering/deduplication.")

                # Build test region×cluster matrix aligned to training clusters
                grp = seqlets_test.groupby(["region_id", "leiden"], observed=True).size()
                regions = grp.index.get_level_values(0).astype(str)
                patterns = grp.index.get_level_values(1).astype(str)

                r_codes, r_uniques = pd.factorize(regions, sort=True)

                col_map = {c: i for i, c in enumerate(cluster_names)}
                p_idx = pd.Index(patterns).map(col_map)
                keep = p_idx.notna()

                X_test = sp.coo_matrix(
                    (grp.values[keep].astype(np.int32), (r_codes[keep], p_idx[keep].astype(int))),
                    shape=(len(r_uniques), len(cluster_names)),
                ).tocsr()

                if binarize:
                    X_test = X_test.copy()
                    X_test.data = np.ones_like(X_test.data, dtype=np.int32)

                # Helper: fold-in theta for fixed phi (simple EM)
                def _infer_theta_fixed_phi(
                    X: sp.csr_matrix, phi: np.ndarray, alpha_vec: np.ndarray, n_iters: int = 50
                ) -> np.ndarray:
                    if not sp.isspmatrix_csr(X):
                        X = X.tocsr()
                    D = X.shape[0]
                    K = phi.shape[0]
                    theta = np.full((D, K), 1.0 / K, dtype=np.float64)
                    phi = np.clip(phi, 1e-12, 1.0)
                    phi = phi / phi.sum(axis=1, keepdims=True)

                    for d in range(D):
                        start, end = X.indptr[d], X.indptr[d + 1]
                        idx = X.indices[start:end]
                        cnt = X.data[start:end].astype(np.float64)
                        if idx.size == 0:
                            continue
                        th = theta[d]
                        for _ in range(int(n_iters)):
                            denom = (th[:, None] * phi[:, idx]).sum(axis=0) + 1e-12
                            r = (th[:, None] * phi[:, idx]) / denom[None, :]
                            th_new = alpha_vec + (r * cnt[None, :]).sum(axis=1)
                            th_new = th_new / (th_new.sum() + 1e-12)
                            if np.linalg.norm(th_new - th, ord=1) < 1e-6:
                                th = th_new
                                break
                            th = th_new
                        theta[d] = th
                    return theta

                def _doc_mixture_loglikelihood(X: sp.csr_matrix, theta: np.ndarray, phi: np.ndarray) -> float:
                    if not sp.isspmatrix_csr(X):
                        X = X.tocsr()
                    phi = np.clip(phi, 1e-12, 1.0)
                    phi = phi / phi.sum(axis=1, keepdims=True)
                    ll = 0.0
                    for d in range(X.shape[0]):
                        start, end = X.indptr[d], X.indptr[d + 1]
                        idx = X.indices[start:end]
                        cnt = X.data[start:end].astype(np.float64)
                        if idx.size == 0:
                            continue
                        pw = (theta[d][:, None] * phi[:, idx]).sum(axis=0) + 1e-12
                        ll += float((cnt * np.log(pw)).sum())
                    return ll

                # Precompute held-out biterm counts (for BTM heldout metrics)
                if method == "btm":
                    X_test_bin_for_btm = X_test.copy()
                    if X_test_bin_for_btm.nnz > 0:
                        X_test_bin_for_btm.data = np.ones_like(X_test_bin_for_btm.data, dtype=np.int32)

                    C_bin = (X_test_bin_for_btm.T @ X_test_bin_for_btm).tocoo()
                    m_bin = C_bin.row < C_bin.col
                    btm_bin_i = C_bin.row[m_bin].astype(int)
                    btm_bin_j = C_bin.col[m_bin].astype(int)
                    btm_bin_cnt = C_bin.data[m_bin].astype(np.float64)

                    C_cnt = (X_test.T @ X_test).tocoo()
                    m_cnt = C_cnt.row < C_cnt.col
                    btm_cnt_i = C_cnt.row[m_cnt].astype(int)
                    btm_cnt_j = C_cnt.col[m_cnt].astype(int)
                    btm_cnt_cnt = C_cnt.data[m_cnt].astype(np.float64)

                    # Self-pair counts for count mode (unordered): sum_d choose(x_di, 2)
                    col_sum_cnt = np.asarray(X_test.sum(axis=0)).ravel().astype(np.float64)
                    col_sq_cnt = np.asarray(X_test.power(2).sum(axis=0)).ravel().astype(np.float64)
                    btm_cnt_self = 0.5 * (col_sq_cnt - col_sum_cnt)
                    btm_cnt_self_idx = np.where(btm_cnt_self > 0)[0].astype(int)
                    btm_cnt_self_cnt = btm_cnt_self[btm_cnt_self_idx].astype(np.float64)

                    def _btm_biterm_loglikelihood_from_pairs(
                        theta: np.ndarray,
                        phi: np.ndarray,
                        i_idx: np.ndarray,
                        j_idx: np.ndarray,
                        pair_cnt: np.ndarray,
                        self_idx: np.ndarray | None = None,
                        self_cnt: np.ndarray | None = None,
                    ) -> tuple[float, float]:
                        """
                        Compute BTM-style log-likelihood from precomputed biterm pairs.

                        Returns (ll, n_biterms), where n_biterms is the total (weighted) number of biterms.
                        """
                        theta = np.asarray(theta, dtype=np.float64).ravel()
                        if theta.size != phi.shape[0]:
                            theta = np.ones((phi.shape[0],), dtype=np.float64)
                        theta = theta / max(theta.sum(), 1e-12)

                        phi = np.asarray(phi, dtype=np.float64)
                        phi = np.clip(phi, 1e-12, 1.0)
                        phi = phi / phi.sum(axis=1, keepdims=True)

                        ll = 0.0
                        n_biterms = 0.0

                        if pair_cnt is not None and pair_cnt.size > 0:
                            p = (theta[:, None] * phi[:, i_idx] * phi[:, j_idx]).sum(axis=0) + 1e-12
                            ll += float((pair_cnt * np.log(p)).sum())
                            n_biterms += float(pair_cnt.sum())

                        if self_idx is not None and self_cnt is not None and self_cnt.size > 0:
                            p2 = (theta[:, None] * phi[:, self_idx] * phi[:, self_idx]).sum(axis=0) + 1e-12
                            ll += float((self_cnt * np.log(p2)).sum())
                            n_biterms += float(self_cnt.sum())

                        return ll, n_biterms

                for k_int, res in items:
                    params = res.get("params", {}) if isinstance(res.get("params", {}), dict) else {}
                    method_k = params.get("method", None)

                    if method_k == "lda_gibbs":
                        # Use cached topic-word distribution (K,V)
                        phi = res["topic_cluster_matrix"].T.values.astype(np.float64)
                        alpha_total = float(params.get("alpha", 50.0))
                        alpha_vec = np.full((int(k_int),), alpha_total / float(k_int), dtype=np.float64)

                        theta_test = _infer_theta_fixed_phi(X_test, phi, alpha_vec, n_iters=50)
                        ll = _doc_mixture_loglikelihood(X_test, theta_test, phi)
                        n_tokens = float(X_test.sum())
                        df.loc[float(k_int), "heldout_loglikelihood"] = float(ll)
                        df.loc[float(k_int), "heldout_perplexity"] = float(math.exp(-ll / max(n_tokens, 1.0)))

                    elif method_k == "lda_sklearn":
                        model_obj = res.get("model", None)
                        if model_obj is None:
                            continue

                        weighting = str(params.get("weighting", "count"))
                        if weighting == "tfidf":
                            from sklearn.feature_extraction.text import TfidfTransformer

                            tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=True)
                            X_train_w = tfidf.fit_transform(X_train.astype(np.float64))
                            X_test_w = tfidf.transform(X_test.astype(np.float64))
                        else:
                            X_test_w = X_test.astype(np.float64)

                        df.loc[float(k_int), "heldout_loglikelihood"] = float(model_obj.score(X_test_w))
                        df.loc[float(k_int), "heldout_perplexity"] = float(model_obj.perplexity(X_test_w))

                    elif method_k == "btm":
                        # Use cached topic-word distribution (K,V)
                        phi = res["topic_cluster_matrix"].T.values.astype(np.float64)

                        model_obj = res.get("model", None)
                        theta = getattr(model_obj, "theta_", None) if model_obj is not None else None
                        if theta is None:
                            # fall back to mean region-topic usage if available
                            rt = res.get("region_topic_matrix", None)
                            if isinstance(rt, pd.DataFrame):
                                theta = rt.mean(axis=0).values.astype(np.float64)
                            else:
                                theta = np.ones((int(k_int),), dtype=np.float64)

                        token_mode = str(params.get("btm_token_mode", "binary")).lower()
                        allow_self_pairs = bool(params.get("btm_allow_self_pairs", False))

                        if token_mode == "count":
                            ll_btm, n_b = _btm_biterm_loglikelihood_from_pairs(
                                theta=np.asarray(theta),
                                phi=phi,
                                i_idx=btm_cnt_i,
                                j_idx=btm_cnt_j,
                                pair_cnt=btm_cnt_cnt,
                                self_idx=btm_cnt_self_idx if allow_self_pairs else None,
                                self_cnt=btm_cnt_self_cnt if allow_self_pairs else None,
                            )
                        else:
                            # binary / default
                            ll_btm, n_b = _btm_biterm_loglikelihood_from_pairs(
                                theta=np.asarray(theta),
                                phi=phi,
                                i_idx=btm_bin_i,
                                j_idx=btm_bin_j,
                                pair_cnt=btm_bin_cnt,
                                self_idx=None,
                                self_cnt=None,
                            )

                        df.loc[float(k_int), "heldout_loglikelihood"] = float(ll_btm)
                        df.loc[float(k_int), "heldout_perplexity"] = float(math.exp(-ll_btm / max(n_b, 1.0)))

                    else:
                        continue
                df.attrs["heldout_scope"] = "cached_split"
                df.attrs["n_test_regions"] = int(len(test_regions))

            except Exception as e:
                df.attrs["heldout_scope"] = "cached_split_failed"
                df.attrs["heldout_error"] = repr(e)

        else:
            # ------------------------------------------------------------------
            # Fallback: refit per K on a random split of the cached matrix.
            # ------------------------------------------------------------------
            if shared_count_df is not None:
                try:
                    X_full = _df_to_csr(shared_count_df)
                    rng = np.random.default_rng(int(heldout_random_state))
                    n_docs = X_full.shape[0]
                    n_test = int(max(1, math.floor(float(heldout_fraction) * n_docs)))
                    test_idx = rng.choice(n_docs, size=n_test, replace=False)
                    train_mask = np.ones(n_docs, dtype=bool)
                    train_mask[test_idx] = False
                    X_train = X_full[train_mask]
                    X_test = X_full[~train_mask]

                    # Helper: fold-in theta for fixed phi (simple EM)
                    def _infer_theta_fixed_phi(
                        X: sp.csr_matrix, phi: np.ndarray, alpha_vec: np.ndarray, n_iters: int = 50
                    ) -> np.ndarray:
                        if not sp.isspmatrix_csr(X):
                            X = X.tocsr()
                        D = X.shape[0]
                        K = phi.shape[0]
                        theta = np.full((D, K), 1.0 / K, dtype=np.float64)
                        phi = np.clip(phi, 1e-12, 1.0)
                        phi = phi / phi.sum(axis=1, keepdims=True)

                        for d in range(D):
                            start, end = X.indptr[d], X.indptr[d + 1]
                            idx = X.indices[start:end]
                            cnt = X.data[start:end].astype(np.float64)
                            if idx.size == 0:
                                continue
                            th = theta[d]
                            for _ in range(int(n_iters)):
                                denom = (th[:, None] * phi[:, idx]).sum(axis=0) + 1e-12
                                r = (th[:, None] * phi[:, idx]) / denom[None, :]
                                th_new = alpha_vec + (r * cnt[None, :]).sum(axis=1)
                                th_new = th_new / (th_new.sum() + 1e-12)
                                if np.linalg.norm(th_new - th, ord=1) < 1e-6:
                                    th = th_new
                                    break
                                th = th_new
                            theta[d] = th
                        return theta

                    def _doc_mixture_loglikelihood(X: sp.csr_matrix, theta: np.ndarray, phi: np.ndarray) -> float:
                        if not sp.isspmatrix_csr(X):
                            X = X.tocsr()
                        phi = np.clip(phi, 1e-12, 1.0)
                        phi = phi / phi.sum(axis=1, keepdims=True)
                        ll = 0.0
                        for d in range(X.shape[0]):
                            start, end = X.indptr[d], X.indptr[d + 1]
                            idx = X.indices[start:end]
                            cnt = X.data[start:end].astype(np.float64)
                            if idx.size == 0:
                                continue
                            pw = (theta[d][:, None] * phi[:, idx]).sum(axis=0) + 1e-12
                            ll += float((cnt * np.log(pw)).sum())
                        return ll

                    for k_int, res in items:
                        params = res.get("params", {}) if isinstance(res.get("params", {}), dict) else {}
                        method_k = params.get("method", None)

                        if method_k == "lda_sklearn":
                            from sklearn.decomposition import LatentDirichletAllocation

                            model = LatentDirichletAllocation(
                                n_components=int(k_int),
                                doc_topic_prior=min(max(float(params.get("alpha", 50.0)) / float(k_int), 1e-6), 1.0),
                                topic_word_prior=min(max(float(params.get("eta", 0.1)), 1e-6), 1.0),
                                max_iter=max(10, int(params.get("n_iter", 150)) // 10),
                                learning_method="batch",
                                random_state=int(params.get("random_state", heldout_random_state)),
                                evaluate_every=-1,
                            )
                            weighting = str(params.get("weighting", "count"))
                            if weighting == "tfidf":
                                from sklearn.feature_extraction.text import TfidfTransformer

                                tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=True)
                                Xw_train = tfidf.fit_transform(X_train.astype(np.float64))
                                Xw_test = tfidf.transform(X_test.astype(np.float64))
                            else:
                                Xw_train = X_train.astype(np.float64)
                                Xw_test = X_test.astype(np.float64)
                            model.fit(Xw_train)
                            df.loc[float(k_int), "heldout_loglikelihood"] = float(model.score(Xw_test))
                            df.loc[float(k_int), "heldout_perplexity"] = float(model.perplexity(Xw_test))

                        elif method_k == "lda_gibbs" and lda is not None:
                            Xt = X_train.toarray().astype(np.int32)
                            model = lda.LDA(
                                n_topics=int(k_int),
                                n_iter=int(params.get("n_iter", 150)),
                                random_state=int(params.get("random_state", heldout_random_state)),
                                alpha=float(params.get("alpha", 50.0)) / float(k_int),
                                eta=float(params.get("eta", 0.1)),
                            )
                            model.fit(Xt)

                            phi = model.topic_word_.astype(np.float64)
                            alpha_vec = np.full((int(k_int),), float(params.get("alpha", 50.0)) / float(k_int), dtype=np.float64)
                            theta_test = _infer_theta_fixed_phi(X_test, phi, alpha_vec, n_iters=50)

                            ll = _doc_mixture_loglikelihood(X_test, theta_test, phi)
                            n_tokens = float(X_test.sum())
                            df.loc[float(k_int), "heldout_loglikelihood"] = float(ll)
                            df.loc[float(k_int), "heldout_perplexity"] = float(math.exp(-ll / max(n_tokens, 1.0)))

                        else:
                            continue

                    df.attrs["heldout_scope"] = "refit_fallback"

                except Exception as e:
                    df.attrs["heldout_scope"] = "refit_fallback_failed"
                    df.attrs["heldout_error"] = repr(e)

    # Store for reuse
    adata.uns[out_key] = df
    df.attrs["models_key"] = models_key
    df.attrs["label_key"] = label_key
    df.attrs["top_n_coherence"] = int(top_n_coherence)
    df.attrs["top_topics_coh"] = int(top_topics_coh)
    df.attrs["heldout_fraction"] = float(heldout_fraction)
    df.attrs["fit_scope"] = str(meta.get("fit_scope", "unknown"))

    return df


def score_topic_models(
    metrics_df: pd.DataFrame,
    metrics: list[str],
    metric_weights: dict[str, float] | None = None,
    min_topics_coh: int = 5,
    select_model: int | None = None,
) -> pd.DataFrame:
    """
    Rescale, combine, and optionally select an optimal number of topics.

    Parameters
    ----------
    metrics_df
        Raw metric table produced by :func:`compute_topic_model_metrics_table`.
    metrics
        List of metric names (columns in metrics_df) to include in the combined score.
    metric_weights
        Optional weight per metric (default 1.0).
    min_topics_coh
        Ignore coherence metrics for K < min_topics_coh (pycisTopic behavior).
    select_model
        If provided, force selection to this K instead of automatic argmax.

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with additional columns:
          - <metric>__scaled in [0,1]
          - combined_score
          - is_best
        The selected K is also stored in df.attrs["best_n_topics"].
    """
    if not isinstance(metrics_df, pd.DataFrame):
        raise TypeError("metrics_df must be a pandas.DataFrame")

    df = metrics_df.copy()
    df.index = df.index.astype(float)

    metric_direction: dict[str, str] = {
        "loglikelihood": "max",
        "perplexity": "min",
        "reconstruction_err": "min",
        "coherence_umass": "max",
        "Mimno_2011": "max",
        "redundancy_cosine": "min",
        "Cao_Juan_2009": "min",
        "Arun_2010": "min",
        "topic_diversity": "max",
        "median_effective_topics": "min",
        "label_ami_argmax": "max",
        "label_knn_purity": "max",
        "internal_score": "max",
        "heldout_loglikelihood": "max",
        "heldout_perplexity": "min",
    }

    unknown = [m for m in metrics if m not in df.columns]
    if unknown:
        raise ValueError(f"Metrics not present in metrics_df: {unknown}")

    weights = metric_weights or {}
    combined = pd.Series(0.0, index=df.index, dtype=float)

    def _safe_rescale(v: pd.Series) -> pd.Series:
        vv = v.astype(float)
        m = vv.min(skipna=True)
        M = vv.max(skipna=True)
        if not np.isfinite(m) or not np.isfinite(M) or float(M - m) == 0.0:
            return pd.Series(np.nan, index=vv.index)
        return (vv - m) / (M - m)

    for met in metrics:
        raw = df[met].astype(float)

        # pycisTopic behavior: ignore coherence for very small K
        if met in {"coherence_umass", "Mimno_2011"} and int(min_topics_coh) > 0:
            raw = raw.where(df.index.astype(int) >= int(min_topics_coh))

        # Convert to "higher is better" for rescaling
        if metric_direction.get(met, "max") == "min":
            raw = -raw

        scaled = _safe_rescale(raw)
        df[f"{met}__scaled"] = scaled
        w = float(weights.get(met, 1.0))
        combined = combined + (w * scaled.fillna(0.0))

    df["combined_score"] = combined

    # Select best K
    if select_model is not None:
        best_n = int(select_model)
        if best_n not in df.index.astype(int).tolist():
            raise ValueError(f"select_model={best_n} not present in metrics_df")
    else:
        if df["combined_score"].notna().sum() == 0:
            # fall back to internal_score if everything is NaN
            if "internal_score" in df.columns:
                best_n = int(df["internal_score"].astype(float).idxmax())
            else:
                best_n = int(df.index.astype(int).max())
        else:
            best_n = int(df["combined_score"].astype(float).idxmax())

    df["is_best"] = df.index.astype(int) == best_n
    df.attrs["best_n_topics"] = best_n
    df.attrs["metrics_used"] = list(metrics)
    df.attrs["metric_weights"] = dict(metric_weights or {})

    return df


def plot_topic_model_selection(
    scored_df: pd.DataFrame,
    metrics: list[str],
    selected_n: int | None = None,
    figsize: tuple[float, float] = (7.5, 5.0),
    save_path: str | None = None,
    show: bool = True,
):
    """
    Plot rescaled model selection metrics across K (pycisTopic-style).

    Parameters
    ----------
    scored_df
        Output of :func:`score_topic_models` (must contain <metric>__scaled columns).
    metrics
        Metrics to plot (must match those passed to score_topic_models).
    selected_n
        If provided, draw a vertical line at this K. If None, will use
        ``scored_df.attrs['best_n_topics']`` if available.
    """
    import matplotlib.pyplot as plt

    metric_direction: dict[str, str] = {
        "loglikelihood": "max",
        "perplexity": "min",
        "reconstruction_err": "min",
        "coherence_umass": "max",
        "Mimno_2011": "max",
        "redundancy_cosine": "min",
        "Cao_Juan_2009": "min",
        "Arun_2010": "min",
        "topic_diversity": "max",
        "median_effective_topics": "min",
        "label_ami_argmax": "max",
        "label_knn_purity": "max",
        "internal_score": "max",
        "heldout_loglikelihood": "max",
        "heldout_perplexity": "min",
    }

    pretty = {
        "loglikelihood": "Loglikelihood",
    }

    if selected_n is None:
        selected_n = int(scored_df.attrs.get("best_n_topics", int(scored_df.index.astype(int).max())))

    fig = plt.figure(figsize=figsize)

    x = scored_df.index.astype(int).values
    for met in metrics:
        col = f"{met}__scaled"
        if col not in scored_df.columns:
            raise ValueError(f"Missing scaled column '{col}'. Run score_topic_models(...) first.")
        label = pretty.get(met, met)
        if metric_direction.get(met, "max") == "min":
            label = f"Inv_{label}"
        plt.plot(x, scored_df[col].values, linestyle="--", marker="o", label=label)

    plt.axvline(int(selected_n), linestyle="--", color="grey")
    plt.xlabel(f"Number of topics\nOptimal number of topics: {int(selected_n)}")
    plt.ylabel("Rescaled metric")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def select_topic_model(
    adata: AnnData,
    n_topics: int,
    models_key: str = "topic_modeling_models",
    store_key: str = "topic_modeling",
    refit_final: bool = False,
    verbose: bool = True,
) -> None:
    """
    Select a cached model (by K) and store it under `adata.uns[store_key]`.

    This enables downstream TF-MINDI plotting/analysis functions to operate on the
    selected model without re-fitting.

    Parameters
    ----------
    n_topics
        Number of topics to select (K).
    models_key
        Key under `adata.uns` where cached models were stored by :func:`fit_topic_models`.
    store_key
        Key under `adata.uns` where the selected model should be stored for downstream
        TF-MINDI plotting/analysis (default: ``"topic_modeling"``).
    refit_final
        If True, refit on all regions using the cached configuration before
        storing under `store_key`. This is recommended when cached models were fitted
        on a train/test split for model selection.
    verbose
        Print progress information.

    Notes
    -----
    If the cached results were saved with ``keep_model_objects=False``, the stored
    result will not contain the underlying estimator object. Re-fitting with
    ``refit_final=True`` will restore the estimator object in `adata.uns[store_key]["model"]`.
    """
    if models_key not in adata.uns:
        raise KeyError(f"Cached models not found in adata.uns['{models_key}']")

    raw_models = adata.uns[models_key]
    if not isinstance(raw_models, dict):
        raise TypeError(f"adata.uns['{models_key}'] must be a dict (got {type(raw_models)!r})")

    key = str(int(n_topics))
    if key not in raw_models:
        raise KeyError(f"n_topics={n_topics} not found in cached models under '{models_key}'")

    res = raw_models[key]

    if not refit_final:
        adata.uns[store_key] = res
        return

    # Refit once on all data with the same configuration for downstream interpretation/plots.
    params = res.get("params", {}) if isinstance(res.get("params", {}), dict) else {}
    method = params.get("method", "lda_gibbs")

    run_topic_modeling(
        adata,
        n_topics=int(n_topics),
        alpha=float(params.get("alpha", 50.0)),
        eta=float(params.get("eta", 0.1)),
        n_iter=int(params.get("n_iter", 150)),
        random_state=int(params.get("random_state", 123)),
        filter_unknown=bool(params.get("filter_unknown", True)),
        method=method,
        binarize=bool(params.get("binarize", False)),
        min_regions_per_pattern=int(params.get("min_regions_per_pattern", 1)),
        max_regions_frac=float(params.get("max_regions_frac", 1.0)),
        weighting=str(params.get("weighting", "count")),
        n_starts=int(params.get("n_starts", 1)),
        btm_impl=str(params.get("btm_impl", "internal_em")) if method == "btm" else "internal_em",
        store_key=store_key,
        verbose=bool(verbose),
        region_ids=None,
    )