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

**TF-MINDI** is a Python package for analyzing transcription factor binding patterns from deep learning model attribution scores. It identifies and clusters sequence motifs from contribution scores, maps them to transcription factor families, and provides visualization tools for regulatory genomics analysis.

<div align="center">
   <img src="https://raw.githubusercontent.com/aertslab/TF-MINDI/main/docs/_static/tf_mindi_overview.png"
   height=700>
</div>

## Getting Started

See the [tutorials] and the [API documentation].

## Key Features

- **Seqlet Extraction**: Identifies important sequence regions from contribution scores with a choice of seqlet callers (recursive, hysteresis, local-contrast, wavelet-Otsu, finemo)
- **Motif Similarity Analysis**: Compares extracted seqlets to known motif databases using TomTom
- **TF Family Annotation**: Projects seqlets into a reference motif space and assigns them a TF family
- **Clustering & Dimensionality Reduction**: Groups similar seqlets using Leiden clustering and t-SNE visualization
- **Pattern Generation**: Creates consensus motifs from clustered seqlets with alignment
- **Visualization**: Region-level contribution plots, t-SNE embeddings, motif logos, and heatmaps

## Installation

tfmindi requires Python 3.12 and Linux.

```bash
pip install tfmindi          # CPU
pip install tfmindi[gpu]     # GPU, CUDA 12.x -- recommended
```

## GPU Acceleration

Every heavy step runs on the GPU when one is available: TomTom motif similarity, PCA, the
neighborhood graph, t-SNE, Leiden clustering, and the TF-family kNN projection. On a genome-scale
run this is **6x-7x end to end** compared to the cpu version.

The backend is auto-detected. Override it with `tm.set_backend("cpu"/"gpu")` or the
`TFMINDI_BACKEND` environment variable. Every accelerated step falls back to its CPU implementation
with a warning if the GPU path fails, so a run never breaks on a GPU problem.

If `tfmindi` can't find your GPU, import `rapids_singlecell` directly and see what errors you get.
You may have to set `LD_LIBRARY_PATH` for cuml as described [here](https://github.com/rapidsai/cuml/issues/404).

## Quick Start

TF-MINDI follows a scanpy-inspired workflow: `tm.pp` (preprocess) → `tm.tl` (analyze) → `tm.pl` (plot),
with a single AnnData carrying state between steps.

```python
import tfmindi as tm

# --- Preprocessing: seqlets -> motif similarity -> AnnData ---
collection = tm.MotifCollectionData("mcv11.refdata.tar.gz")
motifs = collection.get_motifs(20)  # 20 motifs per reference cluster

# contrib and oh are (n_regions, 4, region_width) arrays
seqlets_df, seqlet_matrices = tm.pp.extract_seqlets(contrib=contrib, oh=oh, threshold=0.05)
sim_matrix = tm.pp.calculate_motif_similarity(seqlet_matrices, motifs)
adata = tm.pp.create_seqlet_adata(
    sim_matrix,
    seqlets_df,
    oh_sequences=oh,
    contrib_scores=contrib,
    motif_collection=motifs,
)

# --- Tools: annotate, cluster, build consensus patterns ---
tm.tl.predict_tf_family_seqlets(adata, collection, cluster_resolution=5.0, n_motifs_per_reference_cluster=20)
tm.tl.embed_and_cluster(adata, resolution=2.5)
patterns = tm.tl.create_patterns(adata, annotation_col="predicted_5.0_predicted_family")

# --- Plotting ---
tm.pl.tsne(adata, color_by="predicted_5.0_predicted_family")
tm.pl.pattern_logos(patterns)
tm.pl.region_contributions(adata, annotation_col="predicted_5.0_predicted_family", example_idx=0)

tm.save_h5ad(adata, "seqlets.h5ad")  # use these, not adata.write_h5ad
```

## Release Notes

See the [changelog].

## Contact

If you found a bug, please use the [issue tracker].

## Citation

> [De Winter S. *et al.* (2026). System-wide extraction of cis-regulatory rules from sequence-to-function models in human neural development. BioRxiv. https://doi.org/10.64898/2026.01.14.699402](https://doi.org/10.64898/2026.01.14.699402)

[issue tracker]: https://github.com/aertslab/TF-MInDi/issues
[tests]: https://github.com/aertslab/TF-MInDi/actions/workflows/test.yaml
[documentation]: https://tf-mindi.readthedocs.io
[changelog]: https://tf-mindi.readthedocs.io/en/latest/changelog.html
[api documentation]: https://tf-mindi.readthedocs.io/en/latest/api.html
[tutorials]: https://tf-mindi.readthedocs.io/en/latest/tutorials.html
