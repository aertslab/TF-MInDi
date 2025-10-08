"""Topic modeling for discovering co-occurring motif patterns."""

from __future__ import annotations

import lda  # type: ignore
import pandas as pd  # type: ignore
from anndata import AnnData  # type: ignore
from tmtoolkit.topicmod import tm_lda  # type: ignore


def run_topic_modeling(
    adata: AnnData,
    n_topics: int | list[int],
    alpha: float | list[float] = 0.04,
    eta: float | list[float] = 0.1,
    n_iter: int = 2_000,
    random_state: int = 123,
    filter_unknown: bool = True,
) -> tuple[list, pd.DataFrame]:
    """
    Discover co-occurring motif patterns using topic modeling on region-level data.

    This function performs the following steps:
    1. Group seqlets by genomic regions using stored coordinates
    2. Create region-cluster count matrix from leiden assignments
    3. Fit LDA model to discover topics (co-occurring cluster patterns)

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

    For the following parameters either a list of values or a single value is accepted:
    - n_topics
    - alpha
    - eta
    Multiple models will be fitted to cover all parameters specified by the list of values.

    Returns
    -------
    list of topic modeling results and count table
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

    print(f"Count matrix shape: {count_table.shape} (regions x clusters)")

    # Fit LDA model
    constant_params: dict[str, int | float] = {"n_iter": n_iter, "random_state": random_state}
    variable_params: dict[str, list[int] | list[float]] = {}

    if isinstance(n_topics, list):
        variable_params["n_topics"] = n_topics
    else:
        constant_params["n_topics"] = n_topics

    if isinstance(alpha, list):
        variable_params["alpha"] = alpha
    else:
        constant_params["alpha"] = alpha

    if isinstance(eta, list):
        variable_params["eta"] = eta
    else:
        constant_params["eta"] = eta

    print("Fitting model")
    print("\tConstant parameters:")
    for _param, _val in constant_params.items():
        print(f"\t\t{_param}={_val}")
    print("\tVariable parameters:")
    for _param, _val in variable_params.items():  # type: ignore
        print(f"\t\t{_param}={_val}")

    # fit multiple models in parallel
    topic_modeling_results = tm_lda.evaluate_topic_models(
        data=count_table.values,
        varying_parameters=variable_params,
        constant_parameters=constant_params,
        metric=tm_lda.AVAILABLE_METRICS,
        return_models=True,
    )

    return topic_modeling_results, count_table


def add_topic_modeling_result(adata: AnnData, model: lda.LDA, count_table: pd.DataFrame) -> None:
    """Add topic model to AnnData object.

    Parameters
    ----------
    adata
        AnnData object.

    model
        LDA model.

    count_table
        seqlet by region count table.

    Returns
    -------
    Stores topic modeling result in adata.uns["topic_modeling"]
    """
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
            "n_topics": model.n_topics,
            "alpha": model.alpha,
            "eta": model.eta,
            "n_iter": model.n_iter,
            "random_state": model.random_state,
        },
        "count_matrix": count_table,
        "topic_cluster_matrix": topic_cluster,
        "region_names": list(count_table.index.values),
    }

    # Store region-topic probabilities in uns (topics are region-level, not seqlet-level)
    adata.uns["topic_modeling"]["region_topic_matrix"] = region_topic

    print("Stored topic modeling results in adata.uns['topic_modeling']")
