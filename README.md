<div align="center">
    <img src="https://raw.githubusercontent.com/aertslab/TF-MINDI/main/docs/_static/TF-MINDI_LOGO_nobg_notext.png"
    height=50%
    >
</div>

# TF-MINDI: Transcription Factor Motif Instance Neighborhood Decomposition and Interpretation

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/aertslab/TF-MInDi/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/tf-mindi

**TF-MINDI** is a Python package for analyzing transcription factor binding patterns from deep learning model attribution scores. It identifies and clusters sequence motifs from contribution scores, maps them to DNA-binding domains, and provides comprehensive visualization tools for regulatory genomics analysis.

<div align="center">
   <img src="https://raw.githubusercontent.com/aertslab/TF-MINDI/main/docs/_static/tf_mindi_overview.png"
   height=700>
</div>

## Getting Started

Please refer to the [documentation](https://tf-mindi.readthedocs.io/en/latest/index.html) for detailed tutorials and examples,
in particular, the [API documentation](https://tf-mindi.readthedocs.io/en/latest/api.html) and [Tutorials](https://tf-mindi.readthedocs.io/en/latest/tutorials.html)

## Key Features

- **Seqlet Extraction**: Identifies important sequence regions from contribution scores using recursive seqlet calling from `tangermeme`
- **Motif Similarity Analysis**: Compares extracted seqlets to known motif databases using TomTom
- **Clustering & Dimensionality Reduction**: Groups similar seqlets using Leiden clustering and t-SNE visualization
- **DNA-Binding Domain Annotation**: Maps seqlet clusters to transcription factor families
- **Pattern Generation**: Creates consensus motifs from clustered seqlets with alignment
- **Comprehensive Visualization**: Region-level contribution plots, t-SNE embeddings, motif logos, and heatmaps

## Installation

tfmindi is compatible with python version 3.10-3.12.

### CPU Version (Default)
```bash
pip install tfmindi
```

### GPU-Accelerated Version (Recommended for large datasets)
```bash
# Requires CUDA-compatible GPU (CUDA 12.X)
pip install tfmindi[gpu]
```

The GPU version provides significant speedups for:
- PCA computation
- Neighborhood graph construction
- t-SNE embedding
- Leiden clustering

We're still working on making the tfmindi package as GPU-compatible as possible.
If `tfmindi` can't find the GPU, try importing `rapids_singlecell` directly in python and see what errors you get.
You might have to explicitly set your LD_LIBRARY_PATH for cuml as described [here](https://github.com/rapidsai/cuml/issues/404).

## Quick Start

TF-MINDI follows a scanpy-inspired workflow:

1. **Preprocessing (`tm.pp`)**: Extract seqlets, calculate motif similarities, and create an Anndata object
2. **Tools (`tm.tl`)**: Cluster seqlets and create consensus patterns
3. **Plotting (`tm.pl`)**: Visualize results


```python
import tfmindi as tm

# Optional: Check GPU availability and set backend
print(f"GPU available: {tm.is_gpu_available()}")
print(f"Current backend: {tm.get_backend()}")
# tm.set_backend('gpu')  # Force GPU backend
# tm.set_backend('cpu')  # Swap back to CPU backend

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

## Release Notes

See the [changelog](https://tf-mindi.readthedocs.io/en/latest/changelog.html).

## Contact

If you found a bug, please use the [issue tracker](https://github.com/aertslab/TF-MInDi/issues).

## Citation

> [De Winter S. *et al.* (2026). System-wide extraction of cis-regulatory rules from sequence-to-function models in human neural development. BioRxiv. https://doi.org/10.64898/2026.01.14.699402](https://doi.org/10.64898/2026.01.14.699402)

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/aertslab/TF-MInDi/issues
[tests]: https://github.com/aertslab/TF-MInDi/actions/workflows/test.yaml
[documentation]: https://tf-mindi.readthedocs.io
[changelog]: https://tf-mindi.readthedocs.io/en/latest/changelog.html
[api documentation]: https://tf-mindi.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/tfmindi
