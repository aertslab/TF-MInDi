# TF-MInDi
**Transcription Factor Motifs and Instances Discovery**

## Examples from paper

please see:

`paper/plot_figure_1.py`

`paper/plot_figure_2.py`

`paper/plot_figure_3.py`

## Requirements

**python packages**:
- [tangermeme](https://github.com/jmschrei/tangermeme)
- [memsuitelite](https://github.com/jmschrei/memesuite-lite)
- [scanpy](https://github.com/scverse/scanpy)

**SCENIC+ motif collection**:

Download the SCENIC+ motif collection from [aertslab resources website](https://resources.aertslab.org/cistarget/motif_collections/v10nr_clust_public).

Download the folder named `singletons`

*optionally*, download the list of sampled motifs from this repository under `paper/sampled_motifs.txt`

## Procedure

### 1. Generate nucleotide level explanations

Use your favorite method and model to generate nucleotide level explanations and save these together with the `onehot encoded regions` and  `region names` as [numpy binary files](https://numpy.org/doc/stable/reference/generated/numpy.save.html).

It's important to save the original genomic locations (`region names`) so that later you will be able to locate each predicted TF binding site in the actual genome.

For example, you can use the [CREsted](https://crested.readthedocs.io/) function `crested.tl.contribution_scores`.

### 2. Load SCENIC+ motifs and motif-to-TF family annotation

Next, we load the SCENIC+ motifs and motif-to-TF family annotations.
The motif-to-TF family annotations are acquired through the SCENIC+ motif-to-TF annotations
TF annotations are subsequently linked to DNA binding domain annoations from [*lambert et al. 2018*](https://humantfs.ccbr.utoronto.ca/index.php).

One can choose to reduce the feature set of motifs to a list of sub-sampled motifs.
In our experience using the sub-sampled list of motifs generates very similar results
with less computational requirements.

```python

from scripts.io import (
    load_motif_to_dbd,
    load_motif
)
import os
import numpy as np

MOTIF_DIR = "../../motif_collection_public/v10nr_clust_public/singletons/"

# load motifs
known_motifs = {}
for motif_file in os.listdir(MOTIF_DIR):
    known_motifs = {**known_motifs, **load_motif(os.path.join(MOTIF_DIR, motif_file))}

# optionally load list of subsampled motifs
motifs_to_keep = []
with open("../feature_reduction/sampled_motifs.txt") as f:
    for l in f:
        motifs_to_keep.append(l.strip())

# annotate motifs to tf-family names
motif_name_to_cluster_name = {}
for motif_file in os.listdir(MOTIF_DIR):
    cluster_name = motif_file.replace(".cb", "")
    m = load_motif(os.path.join(MOTIF_DIR, motif_file))
    for k in m.keys():
        motif_name_to_cluster_name[k] = cluster_name


motif_to_dbd = load_motif_to_dbd()

dbd_per_motif = np.array(
    [
        motif_to_dbd.get(
            motif_name_to_cluster_name[m], np.nan
        )
        for m in motifs_to_keep
    ]
)

```

### 3. Load nucleotide level explanations and onehote encoded sequences

Use the [numpy load](https://numpy.org/doc/stable/reference/generated/numpy.load.html) to load the data you saved in step 1.

For the purpose of this tutorial we assume the shape of these arrays are:

$(N_{sequences} \times L_{sequence} \times 4)$

with:

$N_{sequences}$: The total number of sequences

$L_{sequence}$: The length of each sequence (e.g., 500 bp)

### 4. Get seqlets

Use the [tangermeme](https://github.com/jmschrei/tangermeme) function `recursive_seqlets`
to get seqlets.

In our experience using a `threshold` value of `0.05` has high recall while still limiting the amount of false positives.

The seqlets called by this function are by default very short, for this reaon we set the
`additional_flanks` value to `3` bp

The input in this function should be of shape: $(N_{sequences} \times L_{sequence})$, adjust the `sum` axis parameter according to your data.

```python

from tangermeme.seqlet import recursive_seqlets

# contrib is a numpy array containing nucleotide level explanations
# oh is a numpy array containing the onehot encoded sequences
seqlets = recursive_seqlets(
    (contrib * oh).sum(2),
    threshold = 0.05,
    additional_flanks = 3
)

```

### 5. get contribution score of each seqlet

Next, we get the contribution score of each seqlet and if needed flip the sign so that the average contribution score is always positive. This flipping of the sign is needed because the motifs of the SCENIC+ motif collection only have positive values (as is the case with all motifs based on counts) and we want to match the part of the seqlet that has the highest average contribution to the motifs of the collection.

```python

import numpy as np

seqlet_contribs = []
seqlet_contrib_actual = []
for _, (ex_idx, start, end) in seqlets[
    ['example_idx', 'start', 'end']].iterrows():
    X = contrib[ex_idx, start:end, :].T
    O = oh[ex_idx, start:end, :].T
    X = X / abs(X).max()
    seqlet_contribs.append(X)
    seqlet_contrib_actual.append(X * O)

# flip the sign of seqlets based on the sign of the average value
unsigned_seqlet_contribs = [
    np.sign(x.mean()) * y
    for x, y in zip(
        seqlet_contrib_actual,
        seqlet_contribs
    )
]

```

This will generate a list of seqlet-contributions with shape $4 \times L_{seqlet}$

With

$L_{seqlet}$: the length of the seqlet (which is variable)

### 6. Calculate seqlet-to-motif similarity matrix

Now we will use the [memsuitelite](https://github.com/jmschrei/memesuite-lite) function `tomtom` to calculate a seqlet-to-motif similarity matrix.

For subsequent analysis we will take the $-log_{10}$ value of this matrix.

Optionally, we will generate a sparse matrix to reduce disk space when saving the results.

```python

import numpy as np
from memelite import tomtom
import scipy

sim, _, _, _, _ = tomtom(
    Qs=unsigned_seqlet_contribs,
    Ts=[known_motifs[k] for k in motifs_to_keep]
)

# Add a pseudocount of 1e-10 to avoid taking the log of 0 values
l_sim = np.nan_to_num(-np.log10(sim + 1e-10))

# Optionally, generate a sparse array
l_sim_sparse = scipy.sparse.csr_array(
    np.clip(l_sim, 0.05, l_sim.max())
)

```

### 7. Perform first round of clustering

We will use [scanpy](https://github.com/scverse/scanpy) to perform a first round of clustering using which we will identify "noisy seqlets".

```python

import scanpy as sc

seqlet_adata = sc.AnnData(
    X=l_sim_sparse, # or l_sim
)
seqlet_adata.obs = seqlets.copy()

# optionally perform tSNE for visualization
sc.tl.pca(seqlet_adata)
sc.pp.neighbors(seqlet_adata)
sc.tl.tsne(seqlet_adata)

# perform clustering
sc.tl.leiden(seqlet_adata, flavor="igraph", resolution=5)

```

We will annotate each cluster to a major TF-family. For this we obtain the motif with the highest similarity for each seqlet.

```python


seqlet_adata.obs["dbd"] = [dbd_per_motif[x] for x in seqlet_adata.X.toarray().argmax(1)]

leiden_to_dbd = seqlet_adata.obs[["leiden", "dbd"]] \
    .dropna().groupby("leiden")["dbd"].agg(lambda x: x.mode().iat[0]).to_dict()

seqlet_adata.obs["dbd_per_leiden"] = [
    leiden_to_dbd.get(cl, np.nan)
    for cl in seqlet_adata.obs["leiden"]
]

```

We strongly recommend to visualize the average contribution of each seqlet.
Noisy seqlets will have a low contribution and are often localized in the middle of the tSNE.

```python

seqlet_adata.obs["mean_contrib"] = [
    x.mean() for x in seqlet_contrib_actual
]

sc.pl.tsne(
    seqlet_adata,
    color = "mean_contrib",
    save = "_mean_contrib.png",
    cmap = "bwr"
)

```

We can also visualize the TF family annotation, noisy seqlets won't have a TF annotation and will thus be annotated as `nan`

```python

sc.pl.tsne(
    seqlet_adata, color = "dbd_per_leiden", s = 3,
    save = "_DBD_per_leiden.png"
)

```

### 8. Filter out noisy seqlets and perform second step of clustering

Now we will filter out the seqlet that were not annotated (`nan`) and perform a second step of clustering. We suggest to use a higher resolution of clustering for this step.

```python


seqlet_adata_no_na = seqlet_adata[
    seqlet_adata.obs["dbd_per_leiden"] != "nan"
].copy()

sc.tl.pca(seqlet_adata_no_na)
sc.pp.neighbors(seqlet_adata_no_na)
sc.tl.tsne(seqlet_adata_no_na)

sc.tl.leiden(
    seqlet_adata_no_na,
    flavor="igraph",
    resolution=10
)

```

Now, reannotate each seqlet.

```python

leiden_to_dbd = seqlet_adata_no_na.obs[["leiden", "dbd"]] \
    .dropna() \
    .groupby("leiden")["dbd"] \
    .agg(lambda x: x.mode().iat[0]).to_dict()

seqlet_adata_no_na.obs["dbd_per_leiden"] = [
    leiden_to_dbd.get(cl, np.nan)
    for cl in seqlet_adata_no_na.obs["leiden"]
]

```

Again, this can be visualized.

```python

sc.pl.tsne(
    seqlet_adata_no_na, color = "dbd_per_leiden", s = 3,
    save = "_DBD_per_leiden_no_na.png"
)

```

Now we are done! To view the predicted genomic locations of each TF binding site see the `seqlet_adata_no_na.obs` field.

This is a dataframe that contains the genomic location of each seqlet along with the annotation to TF-family.
Use the value in `example_idx` as an index in your `region names` to acquire the actual genomic locations.

### 9. Generate a logo per cluster

Next, we will generate a logo for each cluster of seqlets.
For this seqlets per cluster will be aligned using `tomtom`

```python

from scripts.pattern import create_pattern
cluster_to_pattern = {}
for cluster in tqdm(seqlet_adata_no_na.obs.leiden.unique()):
    idc = seqlet_adata_no_na.obs.query("leiden == @cluster").index.astype(int)
    # here the seqlets are aligned to each other
    sim_cl, _, offsets_cl, _, strands_cl = tomtom(
        Qs=[unsigned_seqlet_contribs[i] for i in idc],
        Ts=[unsigned_seqlet_contribs[i] for i in idc]
    )
    root = sim_cl.mean(0).argmin()
    p = create_pattern(
        seqlet_df=seqlets.iloc[idc],
        strands=strands_cl[root, :],
        offsets=offsets_cl[root, :],
        ohs=oh[seqlets.iloc[idc]["example_idx"]] \
            .swapaxes(1,2), # make sure the shape is (N_SEQUENCES x 4 x SEQLEN)
        contribs=contrib[seqlets.iloc[idc]["example_idx"]] \
            .swapaxes(1,2), # make sure the shape is (N_SEQUENCES x 4 x SEQLEN)
    )
    cluster_to_pattern[cluster] = p

```

These logo's can be visualized. For example, on top of the tSNE of seqlets

```python

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import logomaker
import pandas as pd

fig, ax = plt.subplots(figsize = (20, 20))
ax.scatter(
    seqlet_adata_no_na.obsm["X_tsne"][:, 0],
    seqlet_adata_no_na.obsm["X_tsne"][:, 1],
    s = 2, color = "gray", alpha=0.2
)
for cluster in tqdm(seqlet_adata_no_na.obs.leiden.unique()):
    X = seqlet_adata_no_na.obsm["X_tsne"][seqlet_adata_no_na.obs.leiden==cluster, 0].mean()
    Y = seqlet_adata_no_na.obsm["X_tsne"][seqlet_adata_no_na.obs.leiden==cluster, 1].mean()
    ax_inset = inset_axes(
        ax,
        width=0.8,
        height=0.4,
        bbox_to_anchor=(X, Y),
        bbox_transform=ax.transData,
        loc='center')
    _ = logomaker.Logo(
        pd.DataFrame(
            cluster_to_pattern[cluster].ppm * cluster_to_pattern[cluster].ic()[:, None],
            columns=["A", "C", "G", "T"]
        ),
        ax=ax_inset
    )
    ax_inset.set_axis_off()
    ax_inset.set_title(f"{len(cluster_to_pattern[cluster].seqlets)}")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("tsne1")
ax.set_ylabel("tsne2")
plt.show(fig)

```

Have fun! 🦸🏼‍♀️
