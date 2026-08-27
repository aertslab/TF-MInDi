"""Plotting functions for TF-MInDi."""

from tfmindi.pl._utils import (
    ensure_colors,
    get_colors,
    get_point_colors,
    render_plot,
    reset_colors,
    set_colors,
)
from tfmindi.pl.contributions import region_contributions
from tfmindi.pl.dbd_heatmap import dbd_heatmap
from tfmindi.pl.distance_bias import distance_bias_heatmap, distance_bias_profile
from tfmindi.pl.logo import dbd_cluster_logos, dbd_logos, pattern_logo
from tfmindi.pl.region_topics import dbd_topic_heatmap, region_topic_tsne
from tfmindi.pl.tsne import tsne, tsne_logos

__all__ = [
    "tsne",
    "tsne_logos",
    "dbd_heatmap",
    "dbd_topic_heatmap",
    "region_contributions",
    "dbd_logos",
    "dbd_cluster_logos",
    "pattern_logo",
    "dbd_topic_heatmap",
    "region_topic_tsne",
    "distance_bias_profile",
    "distance_bias_heatmap",
    "render_plot",
    "ensure_colors",
    "get_colors",
    "set_colors",
    "reset_colors",
    "get_point_colors",
]
