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
) -> tuple[lda.LDA, pd.DataFrame, pd.DataFrame]:
    """
    Discover co-occurring motif patterns using topic modeling on region-level data.

    This function performs the following steps:
    1. Group seqlets by genomic regions using stored coordinates
    2. Create region-cluster count matrix from leiden assignments
    3. Fit LDA model to discover topics (co-occurring cluster patterns)
    4. Return fitted model and region-topic assignments

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
        Number of topics to discover (default: 40)
    alpha
        Dirichlet prior for document-topic distribution (default: 50)
    eta
        Dirichlet prior for topic-word distribution (default: 0.1)
    n_iter
        Number of LDA iterations (default: 150)
    random_state
        Random seed for reproducibility (default: 123)
    filter_unknown
        Whether to filter out seqlets with unknown DBD annotations (default: True)
    **kwargs
        Additional arguments (reserved for future use)

    Returns
    -------
    tuple[lda.LDA, pd.DataFrame, pd.DataFrame]
        Fitted LDA model, region-topic matrix (regions × topics), and count table (regions × clusters)

    Examples
    --------
    >>> import tfmindi as tm
    >>> # adata with clustering results
    >>> lda_model, region_topics, count_table = tm.tl.run_topic_modeling(adata, n_topics=40)
    >>> print(f"Discovered {lda_model.n_topics} topics")
    >>> print(f"Region-topic matrix shape: {region_topics.shape}")
    >>> print(f"Count table shape: {count_table.shape}")
    >>> # Now can easily get topic-cluster matrix
    >>> topic_cluster = tm.tl.get_topic_cluster_matrix(lda_model, count_table)
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
    return model, region_topic, count_table


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

    Examples
    --------
    >>> import tfmindi as tm
    >>> # Evaluate different numbers of topics
    >>> scores = tm.tl.evaluate_topic_models(adata, n_topics_range=[10, 20, 30, 40])
    >>> best_n_topics = max(scores, key=scores.get)
    >>> print(f"Best number of topics: {best_n_topics}")
    """
    if n_topics_range is None:
        n_topics_range = [10, 15, 20, 25, 30, 35, 40, 50]

    print(f"Evaluating {len(n_topics_range)} different topic models...")

    models = {}
    for n_topics in n_topics_range:
        print(f"Training model with {n_topics} topics...")
        model, _, _ = run_topic_modeling(
            adata, n_topics=n_topics, alpha=alpha, eta=eta, n_iter=n_iter, random_state=random_state, **kwargs
        )
        models[n_topics] = model

    # Calculate log-likelihood for each model
    model_to_ll = {}
    for n_topics, model in models.items():
        ll = loglikelihood(model.nzw_, model.ndz_, alpha / n_topics, eta)
        model_to_ll[n_topics] = ll
        print(f"Model with {n_topics} topics: log-likelihood = {ll:.2f}")

    return model_to_ll


def get_topic_cluster_matrix(model: lda.LDA, count_table: pd.DataFrame) -> pd.DataFrame:
    """
    Extract topic-cluster matrix from fitted LDA model.

    Parameters
    ----------
    model
        Fitted LDA model
    count_table
        Original count table used for fitting

    Returns
    -------
    Topic-cluster matrix (clusters × topics)
    """
    return pd.DataFrame(
        model.topic_word_.T,
        index=count_table.columns.values.astype(str),
        columns=[f"Topic_{x + 1}" for x in range(model.n_topics)],
    )


def get_topic_dbd_matrix(model: lda.LDA, count_table: pd.DataFrame, cluster_to_dbd: dict[str, str]) -> pd.DataFrame:
    """
    Create topic-DBD matrix by grouping clusters by DNA-binding domain.

    Parameters
    ----------
    model
        Fitted LDA model
    count_table
        Original count table used for fitting
    cluster_to_dbd
        Mapping from cluster IDs to DBD annotations

    Returns
    -------
    pd.DataFrame
        Topic-DBD matrix (DBDs × topics)
    """
    # Get topic-cluster matrix
    topic_cluster = get_topic_cluster_matrix(model, count_table)

    # Group by DBD
    topic_dbd = topic_cluster.groupby(cluster_to_dbd).mean()  # type: ignore

    return topic_dbd
