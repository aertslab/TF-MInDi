# TF-MInDi: Transcription Factor Motifs and Instances Discovery

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/aertslab/tfmindi/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/tfmindi

**TF-MInDi** is a Python package for analyzing transcription factor binding patterns from deep learning model attribution scores. It identifies and clusters sequence motifs from contribution scores, maps them to DNA-binding domains, and provides comprehensive visualization tools for regulatory genomics analysis.

## Key Features

- **Seqlet Extraction**: Identifies important sequence regions from contribution scores using recursive seqlet calling
- **Motif Similarity Analysis**: Compares extracted seqlets to known motif databases using TomTom
- **Clustering & Dimensionality Reduction**: Groups similar seqlets using Leiden clustering and t-SNE visualization
- **DNA-Binding Domain Annotation**: Maps seqlet clusters to transcription factor families
- **Pattern Generation**: Creates consensus motifs from clustered seqlets with alignment
- **Comprehensive Visualization**: Region-level contribution plots, t-SNE embeddings, motif logos, and heatmaps
- **Scalable Processing**: Memory-efficient chunked processing for large datasets

## Quick Start

```python
import tfmindi as tm

# Extract seqlets from contribution scores
seqlets_df, seqlet_matrices = tm.pp.extract_seqlets(
    contrib=contrib_scores,  # (n_examples, 4, length)
    oh=one_hot_sequences,    # (n_examples, 4, length)
    threshold=0.05
)

# Calculate motif similarity
motif_collection = tm.load_motif_collection(
    tm.fetch_motif_collection()
)
similarity_matrix = tm.pp.calculate_motif_similarity(
    seqlet_matrices,
    motif_collection,
    chunk_size=10000
)

# Create AnnData object for analysis
adata = tm.pp.create_seqlet_adata(
    similarity_matrix,
    seqlets_df,
    seqlet_matrices=seqlet_matrices,
    oh_sequences=one_hot_sequences,
    contrib_scores=contrib_scores,
    motif_collection=motif_collection
)

# Cluster seqlets and annotate with DNA-binding domains
tm.tl.cluster_seqlets(adata, resolution=3.0)

# Generate consensus logos for each cluster
patterns = tm.tl.create_patterns(adata)

# Visualize results
tm.pl.tsne(adata, color_by="cluster_dbd")
tm.pl.region_contributions(adata, example_idx=0)
tm.pl.dbd_heatmap(adata)
```

## Installation

You need to have Python 3.10 or newer installed on your system.

```bash
pip install tfmindi
```

## Core Workflow

TF-MInDi follows a scanpy-inspired workflow:

1. **Preprocessing (`tm.pp`)**: Extract seqlets, calculate motif similarities, and create an Anndata object
2. **Tools (`tm.tl`)**: Cluster seqlets and create consensus patterns
3. **Plotting (`tm.pl`)**: Visualize results

### Data Requirements

- **Contribution scores**: Attribution values from deep learning models (e.g., DeepSHAP, Integrated Gradients)
- **One-hot sequences**: Corresponding genomic sequences in one-hot encoding
- **Motif database**: Known transcription factor motifs

## Getting Started

Please refer to the [documentation][] for detailed tutorials and examples,
in particular, the [API documentation][].

## Release Notes

See the [changelog][].

## Contact

If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/aertslab/tfmindi/issues
[tests]: https://github.com/aertslab/tfmindi/actions/workflows/test.yaml
[documentation]: https://tfmindi.readthedocs.io
[changelog]: https://tfmindi.readthedocs.io/en/latest/changelog.html
[api documentation]: https://tfmindi.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/tfmindi
