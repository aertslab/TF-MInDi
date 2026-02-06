"""Topic probability plotting functions for regions and clusters."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData

from tfmindi.pl._utils import render_plot


def dbd_topic_heatmap(
    adata: AnnData,
    cluster_column: str = "leiden",
    dbd_column: str = "cluster_dbd",
    vmax: float = 0.01,
    cmap: str = "RdPu",
    show_labels: bool = True,
    mode: Literal["prob", "enrichment"]="prob",
    baseline: Literal["data", "model", "uniform"]="data",
    eps: float = 1e-9,
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Plot heatmap of topic signal grouped by DNA-binding domain (DBD) family.

    This shows how different DBD families are associated with specific topics.

    By default (mode="prob"), plots the average topic probability 
    that belong to a given DBD family.

    For cross-method comparisons (LDA vs BTM vs NMF), mode="enrichment" is often
    more interpretable: it plots log2 enrichment of the topic mass assigned to
    a DBD family relative to a baseline frequency of that family in the data.

        enrichment(f, k) = log2( (P(f|topic=k) + eps) / (P(f) + eps) )

    where P(f|topic=k) is the sum of topic-word probability mass across
    clusters in family f (so each topic sums to 1 across families), and P(f) is a
    baseline frequency of that family (by default derived from the empirical
    cluster counts).

    Parameters
    ----------
    adata
        AnnData object containing cluster and DBD annotations in .obs and stored topic modeling results
    cluster_column
        Column name in adata.obs containing cluster assignments
    dbd_column
        Column name in adata.obs containing DBD annotations per cluster
    vmax
        Maximum value for colormap
    cmap
        Colormap name
    show_labels
        Whether to show axis labels
    mode
        - "prob": average topic probability per cluster within each DBD family.
        - "enrichment": log2 enrichment vs baseline.
    baseline
        Only used when mode="enrichment":
        - "data": use empirical cluster frequencies from ``adata.uns["topic_modeling"]["count_matrix"]``.
        - "model": use model-implied cluster frequencies (theta·phi).
        - "uniform": assume all DBD families equally frequent.
    eps
        Smoothing constant to avoid log(0) in enrichment mode.

    Notes
    -----
    * In enrichment mode, the heatmap is centered around 0 (no enrichment) with
      vmin = -vmax and vmax = +vmax.
    **kwargs
        Additional arguments passed to render_plot()

    Returns
    -------
    matplotlib Figure or None if show=False

    Examples
    --------
    >>> import tfmindi as tmi
    >>> # After clustering and topic modeling
    >>> tm.tl.run_topic_modeling(adata, n_topics=40)
    >>> fig = tmi.pl.dbd_topic_heatmap(adata)
    """
    # Check if topic modeling results exist
    if "topic_modeling" not in adata.uns:
        raise ValueError("No topic modeling results found. Run tm.tl.run_topic_modeling() first.")

    # Get topic-cluster matrix from stored results (clusters × topics)
    cluster_topic_matrix: pd.DataFrame = adata.uns["topic_modeling"]["topic_cluster_matrix"]
    cluster_topic_matrix = cluster_topic_matrix.copy()
    cluster_topic_matrix.index = cluster_topic_matrix.index.astype(str)

    # Create cluster → DBD mapping from AnnData object
    if cluster_column not in adata.obs.columns:
        raise ValueError(f"Column '{cluster_column}' not found in adata.obs")
    if dbd_column not in adata.obs.columns:
        raise ValueError(f"Column '{dbd_column}' not found in adata.obs")

    cluster_dbd_df = adata.obs[[cluster_column, dbd_column]].dropna().copy()
    cluster_dbd_df[cluster_column] = cluster_dbd_df[cluster_column].astype(str)
    cluster_dbd_df[dbd_column] = cluster_dbd_df[dbd_column].astype(str)
    cluster_to_dbd = (
        cluster_dbd_df.groupby(cluster_column, observed=True)[dbd_column]
        .first()
        .to_dict()
    )

    # Assign a DBD label to every cluster in the topic-word matrix
    dbd_labels = cluster_topic_matrix.index.map(lambda c: cluster_to_dbd.get(str(c), "Unknown"))

    # ------------------------------------------------------------------
    # Compute DBD × topic matrix
    # ------------------------------------------------------------------
    dbd_topic_mean = cluster_topic_matrix.groupby(dbd_labels).mean()  # legacy view
    dbd_topic_mass = cluster_topic_matrix.groupby(dbd_labels).sum()   # probability mass view

    if mode == "prob":
        plot_mat = dbd_topic_mean
        vmin = 0.0
        vmax_plot = float(vmax)
        cbar_label = "DBD topic prob."

    elif mode == "enrichment":
        # Compute baseline family frequencies P(f)
        dbd_freq = None

        if baseline == "data":
            count_mat = adata.uns["topic_modeling"].get("count_matrix", None)
            if isinstance(count_mat, pd.DataFrame):
                # empirical cluster frequencies
                cluster_freq = count_mat.sum(axis=0).astype(float)
                cluster_freq.index = cluster_freq.index.astype(str)
                lbl = cluster_freq.index.map(lambda c: cluster_to_dbd.get(str(c), "Unknown"))
                dbd_freq = cluster_freq.groupby(lbl).sum()
                s = float(dbd_freq.sum())
                if s > 0:
                    dbd_freq = dbd_freq / s
            else:
                # fall back if dense count_matrix not available
                baseline = "model"

        if dbd_freq is None and baseline == "model":
            model_obj = adata.uns["topic_modeling"].get("model", None)
            theta = getattr(model_obj, "theta_", None)
            if theta is None:
                # fall back to mean topic usage across regions
                rt = adata.uns["topic_modeling"]["region_topic_matrix"]
                theta = np.asarray(rt.mean(axis=0).values, dtype=float)
            theta = np.asarray(theta, dtype=float).ravel()
            if theta.size != cluster_topic_matrix.shape[1]:
                # topic dimension mismatch; fall back to uniform
                theta = np.ones((cluster_topic_matrix.shape[1],), dtype=float)
            theta = theta / max(theta.sum(), 1e-12)

            # phi: (K,V) from topic-word matrix (clusters×topics)
            phi = cluster_topic_matrix.values.T.astype(float)  # (topics, clusters)
            p_cluster = (theta[:, None] * phi).sum(axis=0)
            cluster_freq = pd.Series(p_cluster, index=cluster_topic_matrix.index)
            lbl = cluster_freq.index.map(lambda c: cluster_to_dbd.get(str(c), "Unknown"))
            dbd_freq = cluster_freq.groupby(lbl).sum()
            s = float(dbd_freq.sum())
            if s > 0:
                dbd_freq = dbd_freq / s

        if dbd_freq is None and baseline == "uniform":
            dbd_freq = pd.Series(1.0, index=dbd_topic_mass.index, dtype=float)
            dbd_freq = dbd_freq / float(dbd_freq.sum())

        if dbd_freq is None:
            # ultimate fallback
            dbd_freq = pd.Series(1.0, index=dbd_topic_mass.index, dtype=float)
            dbd_freq = dbd_freq / float(dbd_freq.sum())

        # Align baseline to plotted families
        base = dbd_freq.reindex(dbd_topic_mass.index).fillna(0.0).values[:, None]
        plot_mat = np.log2((dbd_topic_mass.values + float(eps)) / (base + float(eps)))
        plot_mat = pd.DataFrame(plot_mat, index=dbd_topic_mass.index, columns=dbd_topic_mass.columns)

        vmin = -abs(float(vmax))
        vmax_plot = abs(float(vmax))
        cbar_label = "log2 enrichment"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    # Sorting (keep similar style to original)
    # ------------------------------------------------------------------
    # Sort topics by number of families above a small activity threshold
    sorted_topics = list((dbd_topic_mean > 0.005).sum(axis=0).sort_values(ascending=False).index)

    # Sort DBD families by the topic in which they are maximal
    dbd_to_top_topic = dbd_topic_mean[sorted_topics].idxmax(axis=1)
    dbd_order = dbd_to_top_topic.map(lambda t: sorted_topics.index(t) if t in sorted_topics else 1e9)
    sorted_dbds = list(dbd_order.sort_values().index)

    # Create topic labels as numbers (extract number from "Topic_X" format)
    topic_labels = []
    for topic in sorted_topics:
        if isinstance(topic, str) and topic.startswith("Topic_"):
            topic_labels.append(topic.replace("Topic_", ""))
        else:
            topic_labels.append(str(topic))

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        plot_mat.loc[sorted_dbds, sorted_topics],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax_plot,
        xticklabels=topic_labels if show_labels else False,
        yticklabels=sorted_dbds if show_labels else False,
        linewidths=0.5,
        linecolor="black",
        cbar=False,
        square=False,
        ax=ax,
    )

    ax.set_xlabel("Topic")
    ax.set_ylabel("")
    ax.set_position([0.1, 0.1, 0.65, 0.75])  # [x, y, width, height]

    # Manually create a vertical colorbar to the right of heatmap, aligned with bottom
    heatmap_pos = ax.get_position()
    cbar_ax = fig.add_axes([heatmap_pos.x1 + 0.3, heatmap_pos.y0, 0.03, 0.2])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax_plot))

    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")

    cbar.set_label(cbar_label, rotation=90, labelpad=-75)

    # Add black border around colorbar
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)

    return render_plot(fig, **kwargs)


def region_topic_tsne(
    adata: AnnData,
    topics_to_show: list[str] | None = None,
    vmin: float = 0.0,
    vmax: float = 0.6,
    point_size: float = 2.0,
    cmap: str = "viridis",
    ncols: int = 3,
    perplexity: float = 30.0,
    random_state: int = 42,
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Plot t-SNE visualization of regions colored by topic probabilities.

    This function computes t-SNE coordinates from the region-topic matrix and shows
    how different topics are distributed across the region t-SNE space.

    Parameters
    ----------
    adata
        AnnData object with stored topic modeling results
    topics_to_show
        List of topic names to plot. If None, plots all topics
    vmin
        Minimum value for colormap
    vmax
        Maximum value for colormap
    point_size
        Size of scatter points
    cmap
        Colormap name
    ncols
        Number of columns in subplot grid
    perplexity
        t-SNE perplexity parameter
    random_state
        Random seed for t-SNE reproducibility
    **kwargs
        Additional arguments passed to render_plot()

    Returns
    -------
    matplotlib Figure or None if show=False

    Examples
    --------
    >>> import tfmindi as tmi
    >>> # After topic modeling
    >>> tm.tl.run_topic_modeling(adata, n_topics=5)
    >>> fig = tmi.pl.region_topic_tsne(adata, topics_to_show=["Topic_1", "Topic_2", "Topic_3"])
    """
    from sklearn.manifold import TSNE

    # Check if topic modeling results exist
    if "topic_modeling" not in adata.uns:
        raise ValueError("No topic modeling results found. Run tm.tl.run_topic_modeling() first.")

    # Get region-topic matrix from stored results
    region_topic_matrix = adata.uns["topic_modeling"]["region_topic_matrix"]

    if topics_to_show is None:
        topics_to_show = list(region_topic_matrix.columns)

    # Compute t-SNE coordinates from region-topic matrix
    print("Computing t-SNE coordinates from region-topic matrix...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    tsne_coords = tsne.fit_transform(region_topic_matrix.values)

    n_topics = len(topics_to_show)
    nrows = (n_topics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    else:
        axes = axes.flatten()

    x_coords = tsne_coords[:, 0]
    y_coords = tsne_coords[:, 1]

    for i, topic in enumerate(topics_to_show):
        ax = axes[i]

        # Get topic values and normalize
        topic_values = region_topic_matrix[topic].values
        if topic_values.max() > topic_values.min():
            topic_values_norm = (topic_values - topic_values.min()) / (topic_values.max() - topic_values.min())
        else:
            topic_values_norm = topic_values

        # Sort points by intensity for better visualization
        sort_idx = np.argsort(topic_values_norm)

        scatter = ax.scatter(
            x_coords[sort_idx],
            y_coords[sort_idx],
            c=topic_values_norm[sort_idx],
            vmin=vmin,
            vmax=vmax,
            s=point_size,
            cmap=cmap,
        )

        # Extract topic number for title
        topic_num = topic.replace("Topic_", "") if topic.startswith("Topic_") else topic
        ax.set_title(f"Topic {topic_num}")
        ax.set_axis_off()

        # Add colorbar for each subplot
        plt.colorbar(scatter, ax=ax, shrink=0.8)

    # Hide unused subplots
    for i in range(n_topics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    return render_plot(fig, **kwargs)


def region_topic_tsne_by_label(
    adata: AnnData,
    label_key: str = "cell_type",
    region_key: str = "example_idx",
    topic_key: str = "topic_modeling",
    perplexity: float = 30.0,
    random_state: int = 42,
    point_size: float = 5.0,
    alpha: float = 0.8,
    cmap: str = "tab20",
    show_legend: bool = True,
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Plot t-SNE visualization of regions colored by a region-level label.

    This function computes t-SNE coordinates from the region-topic matrix and shows
    how different cell types/clusters are distributed across the region t-SNE space.

    - each point is a region (example_idx)
    - the embedding uses the region-topic matrix
    - points are colored by `label_key` aggregated per region
    """
    from sklearn.manifold import TSNE

    if topic_key not in adata.uns:
        raise ValueError(
            f"No topic modeling results found at adata.uns['{topic_key}']. "
            "Run tm.tl.run_topic_modeling() first."
        )
    if label_key not in adata.obs.columns:
        raise ValueError(f"Column '{label_key}' not found in adata.obs")
    if region_key not in adata.obs.columns:
        raise ValueError(f"Column '{region_key}' not found in adata.obs")

    region_topic = adata.uns[topic_key]["region_topic_matrix"].copy()

    # Build region->label mapping (majority vote)
    tmp = adata.obs[[region_key, label_key]].copy().dropna()
    tmp[region_key] = tmp[region_key].astype(str)

    def _mode(s: pd.Series) -> str:
        m = s.mode(dropna=True)
        if len(m) > 0:
            return str(m.iloc[0])
        return str(s.iloc[0])

    region_to_label = tmp.groupby(region_key, observed=True)[label_key].apply(_mode)
    labels = region_to_label.reindex(region_topic.index.astype(str)).fillna("Unknown").astype(str)

    # Compute t-SNE on region-topic matrix
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    coords = tsne.fit_transform(region_topic.values)

    x = coords[:, 0]
    y = coords[:, 1]

    # Color mapping
    uniq = labels.unique().tolist()
    if "Unknown" in uniq:
        # move Unknown to the end for a cleaner legend
        uniq = [u for u in uniq if u != "Unknown"] + ["Unknown"]

    # seaborn palettes include e.g. "tab20", "Set2"
    try:
        palette = sns.color_palette(cmap, n_colors=len(uniq)).as_hex()
        color_map = dict(zip(uniq, palette, strict=False))
    except Exception:
        # fallback to matplotlib
        colormap = plt.get_cmap(cmap)
        color_map = {u: plt.matplotlib.colors.to_hex(colormap(i / max(1, len(uniq) - 1))) for i, u in enumerate(uniq)}

    # enforce Unknown gray
    if "Unknown" in color_map:
        color_map["Unknown"] = "#D3D3D3"

    colors = [color_map[str(v)] for v in labels.values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c=colors, s=point_size, alpha=alpha)
    ax.set_axis_off()
    ax.set_title(f"Regions embedded by topics (colored by {label_key})")

    if show_legend:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", color="w", label=str(u), markerfacecolor=color_map[u], markersize=8)
            for u in uniq
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", title=label_key)

    return render_plot(fig, **kwargs)


def topic_label_heatmap(
    adata: AnnData,
    label_key: str = "cell_type",
    region_key: str = "example_idx",
    topic_key: str = "topic_modeling",
    normalize: Literal["none", "row", "col"] = "row",
    cmap: str = "RdPu",
    vmax: float | None = None,
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Plot heatmap of average topic probabilities per label (e.g., cell type).

    This shows how the different cell types are associated with specific topics.

    - rows are labels (e.g., cell types, clusters, etc)
    - columns are topics
    - values are the average topic probability across regions with that label

    Parameters
    ----------
    normalize
        - "row": each label sums to 1 (recommended; highlights which topics define a label)
        - "col": each topic sums to 1 (highlights which labels a topic is enriched in)
        - "none": no normalization
    """
    if topic_key not in adata.uns:
        raise ValueError(
            f"No topic modeling results found at adata.uns['{topic_key}']. "
            "Run tm.tl.run_topic_modeling() first."
        )
    if label_key not in adata.obs.columns:
        raise ValueError(f"Column '{label_key}' not found in adata.obs")
    if region_key not in adata.obs.columns:
        raise ValueError(f"Column '{region_key}' not found in adata.obs")

    region_topic = adata.uns[topic_key]["region_topic_matrix"].copy()

    tmp = adata.obs[[region_key, label_key]].copy().dropna()
    tmp[region_key] = tmp[region_key].astype(str)

    def _mode(s: pd.Series) -> str:
        m = s.mode(dropna=True)
        if len(m) > 0:
            return str(m.iloc[0])
        return str(s.iloc[0])

    region_to_label = tmp.groupby(region_key, observed=True)[label_key].apply(_mode)
    labels = region_to_label.reindex(region_topic.index.astype(str)).fillna("Unknown").astype(str)

    mat = region_topic.copy()
    mat["__label__"] = labels.values
    by = mat.groupby("__label__", observed=True).mean(numeric_only=True)

    if normalize == "row":
        by = by.div(by.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    elif normalize == "col":
        by = by.div(by.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be one of {'none','row','col'}")

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * by.shape[0])))
    sns.heatmap(by, cmap=cmap, vmin=0.0, vmax=vmax, ax=ax)
    ax.set_xlabel("Topic")
    ax.set_ylabel(label_key)
    ax.set_title(f"Mean topic usage per {label_key}")

    return render_plot(fig, **kwargs)
