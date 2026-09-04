"""Plotting functions for TF-MInDi."""

from tfmindi.pl._utils import (
    ensure_colors,
    get_colors,
    get_point_colors,
    render_plot,
)
from tfmindi.pl.code import code_table_dotplot
from tfmindi.pl.contributions import region_contributions
from tfmindi.pl.dbd_heatmap import dbd_heatmap
from tfmindi.pl.logo import pattern_logos
from tfmindi.pl.region_topics import dbd_topic_heatmap, plot_top_motifs, region_topic_tsne
from tfmindi.pl.tsne import region_tsne, tsne, tsne_logos

__all__ = [
    "tsne",
    "tsne_logos",
    "dbd_heatmap",
    "dbd_topic_heatmap",
    "region_contributions",
    "pattern_logos",
    "region_topic_tsne",
    "render_plot",
    "ensure_colors",
    "get_colors",
    "get_point_colors",
    "region_tsne",
    "plot_top_motifs",
    "code_table_dotplot",
]
