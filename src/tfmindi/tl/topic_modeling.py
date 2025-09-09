"""Topic modeling for discovering co-occurring motif patterns."""

from __future__ import annotations

import math

import lda
import pandas as pd
from anndata import AnnData


def loglikelihood(nzw, ndz, alpha, eta):
    """Calculate log-likelihood of LDA model parameters (from pycisTopic)."""
    D = ndz.shape[0]
    n_topics = ndz.shape[1]
    vocab_size = nzw.shape[1]

    const_prior = (n_topics * math.lgamma(alpha) - math.lgamma(alpha * n_topics)) * D
    const_ll = (vocab_size * math.lgamma(eta) - math.lgamma(eta * vocab_size)) * n_topics

    # calculate log p(w|z)
    topic_ll = 0
    for k in range(n_topics):
        sum = eta * vocab_size
        for w in range(vocab_size):
            if nzw[k, w] > 0:
                topic_ll = math.lgamma(nzw[k, w] + eta)
                sum += nzw[k, w]
        topic_ll -= math.lgamma(sum)

    # calculate log p(z)
    doc_ll = 0
    for d in range(D):
        sum = alpha * n_topics
        for k in range(n_topics):
            if ndz[d, k] > 0:
                doc_ll = math.lgamma(ndz[d, k] + alpha)
                sum += ndz[d, k]
        doc_ll -= math.lgamma(sum)

    ll = doc_ll - const_prior + topic_ll - const_ll
    return ll


def run_topic_modeling(
    adata: AnnData,
    n_topics: int = 40,
    alpha: float = 50,
    eta: float = 0.1,
    n_iter: int = 150,
    random_state: int = 123,
    filter_unknown: bool = True,
) -> None:
    """
    Discover co-occurring motif patterns using topic modeling on region-level data.

    This function performs the following steps:
    1. Group seqlets by genomic regions using stored coordinates
    2. Create region-cluster count matrix from leiden assignments
    3. Fit LDA model to discover topics (co-occurring cluster patterns)
    4. Store fitted model and results in adata.uns and adata.obsm

    Parameters
    ----------
    adata
        AnnData object with cluster assignments and genomic coordinates.
        Must contain:
        - adata.obs["leiden"]: Cluster assignments
        - adata.obs["example_idx"]: Example indices for region grouping
        - adata.obs["start"]: Seqlet start positions
        - adata.obs["end"]: Seqlet end positions
        - adata.obs["cluster_dbd"]: DBD annotations per cluster (optional)
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

    Returns
    -------
    None
        Results are stored in adata:
        - adata.uns['topic_modeling']: Model, parameters, and all topic-related matrices

    Examples
    --------
    >>> import tfmindi as tm
    >>> # adata with clustering results
    >>> tm.tl.run_topic_modeling(adata, n_topics=40)
    >>> print(f"Discovered {adata.uns['topic_modeling']['params']['n_topics']} topics")
    >>> print(f"Region-topic matrix shape: {adata.obsm['X_topics'].shape}")
    >>> # Now can plot directly from adata
    >>> tm.pl.dbd_topic_heatmap(adata)
    >>> tm.pl.region_topic_tsne(adata)
    """
    # Check required columns
    required_cols = ["leiden", "example_idx", "start", "end"]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in adata.obs: {missing_cols}")

    # Create deduplicated seqlets table
    adata.obs["region_id"] = adata.obs["example_idx"]
    dedup_cols = ["region_id", "start", "end", "leiden"]
    if "cluster_dbd" in adata.obs.columns:
        dedup_cols.append("cluster_dbd")
    seqlets_dedup = adata.obs[dedup_cols].drop_duplicates()

    # Filter out unknown DBD annotations if requested
    if filter_unknown and "cluster_dbd" in seqlets_dedup.columns:
        initial_count = len(seqlets_dedup)
        seqlets_dedup = seqlets_dedup.loc[seqlets_dedup["cluster_dbd"] != "nan"]
        seqlets_dedup = seqlets_dedup.loc[seqlets_dedup["cluster_dbd"].notna()]
        print(f"Filtered {initial_count - len(seqlets_dedup)} seqlets with unknown DBD annotations")

    print(f"Using {len(seqlets_dedup)} deduplicated seqlets across {seqlets_dedup['region_id'].nunique()} regions")

    # Create region-cluster count matrix
    count_table = pd.crosstab(seqlets_dedup["region_id"].values, seqlets_dedup["leiden"].values)
    count_table.index.name = "region_id"
    count_table.columns.name = "cluster"

    print(f"Count matrix shape: {count_table.shape} (regions × clusters)")

    # Fit LDA model
    print(f"Fitting LDA model with {n_topics} topics...")

    model = lda.LDA(
        n_topics=n_topics,
        n_iter=n_iter,
        random_state=random_state,
        alpha=alpha / n_topics,  # Normalize alpha by n_topics
        eta=eta,
    )

    model.fit(count_table.values)

    # Create region-topic matrix
    region_topic = pd.DataFrame(
        model.doc_topic_, index=count_table.index.values, columns=[f"Topic_{x + 1}" for x in range(model.n_topics)]
    )

    # Create topic-cluster matrix
    topic_cluster = pd.DataFrame(
        model.topic_word_.T,
        index=count_table.columns.values.astype(str),
        columns=[f"Topic_{x + 1}" for x in range(model.n_topics)],
    )

    # Store results in AnnData
    adata.uns["topic_modeling"] = {
        "model": model,
        "params": {
            "n_topics": n_topics,
            "alpha": alpha,
            "eta": eta,
            "n_iter": n_iter,
            "random_state": random_state,
            "filter_unknown": filter_unknown,
        },
        "count_matrix": count_table,
        "topic_cluster_matrix": topic_cluster,
        "region_names": list(count_table.index.values),
    }

    # Store region-topic probabilities in uns (topics are region-level, not seqlet-level)
    adata.uns["topic_modeling"]["region_topic_matrix"] = region_topic

    print("Stored topic modeling results in adata.uns['topic_modeling']")


def evaluate_topic_models(
    adata: AnnData,
    n_topics_range: list[int] | None = None,
    alpha: float = 50,
    eta: float = 0.1,
    n_iter: int = 150,
    random_state: int = 123,
    **kwargs,
) -> dict[int, float]:
    """
    Evaluate multiple topic models to find optimal number of topics.

    Parameters
    ----------
    adata
        AnnData object with cluster assignments and genomic coordinates
    n_topics_range
        List of topic numbers to evaluate (default: [10, 15, 20, 25, 30, 35, 40, 50])
    alpha
        Dirichlet prior for document-topic distribution (default: 50)
    eta
        Dirichlet prior for topic-word distribution (default: 0.1)
    n_iter
        Number of LDA iterations (default: 150)
    random_state
        Random seed for reproducibility (default: 123)
    **kwargs
        Additional arguments passed to run_topic_modeling

    Returns
    -------
    Mapping of n_topics to log-likelihood scores

    Note: The best-performing model is automatically stored in adata

    Examples
    --------
    >>> import tfmindi as tm
    >>> # Evaluate different numbers of topics
    >>> scores = tm.tl.evaluate_topic_models(adata, n_topics_range=[10, 20, 30, 40])
    >>> best_n_topics = max(scores, key=scores.get)
    >>> print(f"Best number of topics: {best_n_topics}")
    >>> # Best model is already stored in adata for plotting
    """
    if n_topics_range is None:
        n_topics_range = [10, 15, 20, 25, 30, 35, 40, 50]

    print(f"Evaluating {len(n_topics_range)} different topic models...")

    model_to_ll = {}
    best_model_info = None
    best_ll = -float("inf")

    for n_topics in n_topics_range:
        print(f"Training model with {n_topics} topics...")
        run_topic_modeling(
            adata, n_topics=n_topics, alpha=alpha, eta=eta, n_iter=n_iter, random_state=random_state, **kwargs
        )

        # Get the model that was just stored
        model = adata.uns["topic_modeling"]["model"]
        ll = loglikelihood(model.nzw_, model.ndz_, alpha / n_topics, eta)
        model_to_ll[n_topics] = ll
        print(f"Model with {n_topics} topics: log-likelihood = {ll:.2f}")

        # Track the best model
        if ll > best_ll:
            best_ll = ll
            best_model_info = n_topics

    # Ensure the best model is stored in adata (rerun if needed)
    if best_model_info != n_topics:  # If best model isn't the last one trained
        print(f"Storing best model with {best_model_info} topics...")
        run_topic_modeling(
            adata, n_topics=best_model_info, alpha=alpha, eta=eta, n_iter=n_iter, random_state=random_state, **kwargs
        )

    return model_to_ll
