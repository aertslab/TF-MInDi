"""Topic probability plotting functions for regions and clusters."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData

from tfmindi.pl._utils import render_plot


def dbd_topic_heatmap(
    adata: AnnData,
    cluster_topic_matrix: pd.DataFrame,
    cluster_column: str = "leiden",
    dbd_column: str = "cluster_dbd",
    vmax: float = 0.01,
    cmap: str = "RdPu",
    show_labels: bool = True,
    **kwargs,
) -> plt.Figure | None:  # type: ignore[return]
    """
    Plot heatmap of average topic probabilities grouped by DNA-binding domain (DBD).

    This shows how different DBD families are associated with specific topics.

    Parameters
    ----------
    adata
        AnnData object containing cluster and DBD annotations in .obs
    cluster_topic_matrix
        DataFrame with clusters as rows and topics as columns (cluster-topic probabilities)
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
    **kwargs
        Additional arguments passed to render_plot()

    Returns
    -------
    matplotlib Figure or None if show=False

    Examples
    --------
    >>> import tfmindi as tmi
    >>> # After clustering and topic modeling
    >>> fig = tmi.pl.dbd_topic_heatmap(adata=seqlet_adata, cluster_topic_matrix=leiden_topic)
    """
    # Create cluster to DBD mapping from AnnData object
    cluster_dbd_df = adata.obs[[cluster_column, dbd_column]].dropna()
    cluster_to_dbd = cluster_dbd_df.groupby(cluster_column, observed=True)[dbd_column].first().to_dict()
    dbd_topic = cluster_topic_matrix.groupby(cluster_to_dbd).mean()

    # Sort topics by total activity (most active topics first)
    sorted_topics = list((dbd_topic > 0.005).sum().sort_values(ascending=False).index)

    topic_to_dbd = pd.DataFrame(dbd_topic.T.idxmax()).reset_index()
    topic_to_dbd["order"] = [sorted_topics.index(x) for x in topic_to_dbd[0]]
    sorted_dbds = list(topic_to_dbd.sort_values("order")["index"])

    # Create topic labels as numbers (extract number from "Topic_X" format)
    topic_labels = []
    for topic in sorted_topics:
        if isinstance(topic, str) and topic.startswith("Topic_"):
            topic_labels.append(topic.replace("Topic_", ""))
        else:
            topic_labels.append(str(topic))

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        dbd_topic.loc[sorted_dbds, sorted_topics],
        cmap=cmap,
        vmin=0,
        vmax=vmax,
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
    cbar_ax = fig.add_axes(
        [
            heatmap_pos.x1 + 0.3,
            heatmap_pos.y0,
            0.03,
            0.2,
        ]
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="vertical")

    cbar.set_label("DBD topic prob.", rotation=90, labelpad=-75)

    # Add black border around colorbar
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)

    return render_plot(fig, **kwargs)


def region_topic_tsne(
    region_topic_matrix: pd.DataFrame,
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
    region_topic_matrix
        DataFrame with regions as rows and topics as columns (from run_topic_modeling)
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
    >>> model, region_topics, count_table = tmi.tl.run_topic_modeling(adata, n_topics=5)
    >>> fig = tmi.pl.region_topic_tsne(
    ...     region_topic_matrix=region_topics, topics_to_show=["Topic_1", "Topic_2", "Topic_3"]
    ... )
    """
    from sklearn.manifold import TSNE

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
