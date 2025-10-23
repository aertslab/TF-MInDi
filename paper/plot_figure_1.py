import re
import os
import numpy as np
from tqdm import tqdm
from tangermeme.seqlet import recursive_seqlets
from memelite import tomtom
import scanpy as sc
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.patches
from importlib import reload
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import logomaker
import pandas as pd
import crested
from crested.utils._seq_utils import one_hot_encode_sequence
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import seaborn as sns

import scripts.io
reload(scripts.io)
from scripts.io import load_motif, load_motif_to_dbd  # noqa: E402

import scripts.pattern # noqa: E402
reload(scripts.pattern)
from scripts.pattern import create_pattern # noqa: E402

def inch_to_mm(inch: float) -> float:
    return inch * 25.4

def mm_to_inch(mm: float) -> float:
    return mm / 25.4

FIGDIR = "figure_1"

sc.settings.figdir = os.path.join(FIGDIR, "supplement")

CONTRIB_FOLDER = "../../python_modules/crested_notebooks/modisco_results_ft_2000/"

adata = sc.read_h5ad("../../python_modules/crested_notebooks/mouse_cortex_filtered.h5ad")

mm10 = crested.Genome(
    "../../../resources/mm10/mm10.fa",
    "../../../resources/mm10/mm10.chrom.sizes"
)

oh_to_region = {
    tuple(one_hot_encode_sequence(mm10.fetch(*region), expand_dim=False).argmax(1)): region_name
    for region_name, region in tqdm(adata.var[["chr", "start", "end"]].iterrows(), total = adata.shape[1])
}

contrib = []
oh = []
classes = []

class_names = [
    re.match(r"(.+?)_oh\.npz$", f).group(1)
    for f in os.listdir(CONTRIB_FOLDER)
    if f.endswith("_oh.npz")
]

for i, c in enumerate(tqdm(class_names)):
    contrib.append(np.load(os.path.join(CONTRIB_FOLDER, f"{c}_contrib.npz"))["arr_0"])
    oh.append(np.load(os.path.join(CONTRIB_FOLDER, f"{c}_oh.npz"))["arr_0"])
    classes.append(np.repeat(c, oh[i].shape[0]))

oh = np.concatenate(oh)
contrib = np.concatenate(contrib)
classes = np.concatenate(classes)

region_names = [
    oh_to_region[tuple(o.argmax(0))]
    for o in tqdm(oh)
]

MOTIF_DIR = "../../motif_collection_public/v10nr_clust_public/singletons/"

known_motifs = {}
for motif_file in os.listdir(MOTIF_DIR):
    known_motifs = {**known_motifs, **load_motif(os.path.join(MOTIF_DIR, motif_file))}

motifs_to_keep = []
with open("../feature_reduction/sampled_motifs.txt") as f:
    for l in f:
        motifs_to_keep.append(l.strip())

motif_name_to_cluster_name = {}
for motif_file in os.listdir(MOTIF_DIR):
    cluster_name = motif_file.replace(".cb", "")
    m = load_motif(os.path.join(MOTIF_DIR, motif_file))
    for k in m.keys():
        motif_name_to_cluster_name[k] = cluster_name


motif_to_dbd = load_motif_to_dbd()

dbd_per_motif = np.array([ motif_to_dbd.get(motif_name_to_cluster_name[m], np.nan) for m in motifs_to_keep])


seqlets = recursive_seqlets(
    (contrib * oh).sum(1),
    threshold = 0.05,
    additional_flanks = 3
)

seqlet_contribs = []
seqlet_contrib_actual = []
seqlet_seq = []
for _, (ex_idx, start, end) in tqdm(seqlets[['example_idx', 'start', 'end']].iterrows(), total = len(seqlets)):
    X = contrib[ex_idx, :, start:end]
    O = oh[ex_idx, :, start:end] # noqa: E741
    X = X / abs(X).max()
    seqlet_contribs.append(X)
    seqlet_contrib_actual.append(X * O)
    seqlet_seq.append(O)

unsigned_seqlet_contribs = [
    np.sign(x.mean()) * y for x, y in zip(seqlet_contrib_actual, seqlet_contribs)
]

sim, _, _, _, _ = tomtom(
    Qs=unsigned_seqlet_contribs,
    Ts=[known_motifs[k] for k in motifs_to_keep]
)

l_sim = np.nan_to_num(-np.log10(sim + 1e-10))

l_sim_sparse = np.clip(l_sim, 0.05, l_sim.max())

import pickle
pickle.dump(l_sim_sparse, open("intermediate/l_sim_sparse.pkl", "wb"))

seqlet_adata = sc.AnnData(
    X=l_sim_sparse,
)

print("pca")
sc.tl.pca(seqlet_adata)
print("neighbors")
sc.pp.neighbors(seqlet_adata)
print("tsne")
sc.tl.tsne(seqlet_adata)
print("mean")

seqlet_adata.obs = seqlets.copy()
seqlet_adata.obs["mean_contrib"] = [x.mean() for x in seqlet_contrib_actual]

sc.pl.tsne(
    seqlet_adata,
    color = "mean_contrib",
    cmap = "bwr",
    save = "_mean_contrib.png"
)

sc.tl.leiden(seqlet_adata, flavor="igraph", resolution=3)

sc.pl.tsne(
    seqlet_adata,
    color = "leiden",
    legend_loc = "on data",
    save = "_leiden.png"
)

seqlet_adata.obs["dbd"] = [dbd_per_motif[x] for x in seqlet_adata.X.toarray().argmax(1)]

leiden_to_dbd = seqlet_adata.obs[["leiden", "dbd"]] \
    .dropna().groupby("leiden")["dbd"].agg(lambda x: x.mode().iat[0]).to_dict()

seqlet_adata.obs["dbd_per_leiden"] = [
    leiden_to_dbd.get(cl, np.nan)
    for cl in seqlet_adata.obs["leiden"]
]

sc.pl.tsne(
    seqlet_adata, color = "dbd_per_leiden", s = 3,
    save = "_DBD_per_leiden.png"
)

seqlet_adata_no_na = seqlet_adata[seqlet_adata.obs["dbd_per_leiden"] != "nan"].copy()

print("pca")
sc.tl.pca(seqlet_adata_no_na)
print("neighbors")
sc.pp.neighbors(seqlet_adata_no_na)
print("tsne")
sc.tl.tsne(seqlet_adata_no_na)

sc.tl.leiden(seqlet_adata_no_na, flavor="igraph", resolution=20)

sc.pl.tsne(
    seqlet_adata_no_na,
    color = "leiden",
    legend_loc = "on data",
    save = "_leiden_no_na.png"
)


leiden_to_dbd = seqlet_adata_no_na.obs[["leiden", "dbd"]] \
    .dropna().groupby("leiden")["dbd"].agg(lambda x: x.mode().iat[0]).to_dict()

seqlet_adata_no_na.obs["dbd_per_leiden"] = [
    leiden_to_dbd.get(cl, np.nan)
    for cl in seqlet_adata_no_na.obs["leiden"]
]

sc.pl.tsne(
    seqlet_adata_no_na, color = "dbd_per_leiden", s = 3,
    save = "_DBD_per_leiden_no_na.png"
)

cluster_to_pattern = {}
for cluster in tqdm(seqlet_adata_no_na.obs.leiden.unique()):
    idc = seqlet_adata_no_na.obs.query("leiden == @cluster").index.astype(int)
    sim_cl, _, offsets_cl, _, strands_cl = tomtom(
        Qs=[unsigned_seqlet_contribs[i] for i in idc],
        Ts=[unsigned_seqlet_contribs[i] for i in idc]
    )
    root = sim_cl.mean(0).argmin()
    p = create_pattern(
        seqlet_df=seqlets.iloc[idc],
        strands=strands_cl[root, :],
        offsets=offsets_cl[root, :],
        ohs=oh[seqlets.iloc[idc]["example_idx"]],
        contribs=contrib[seqlets.iloc[idc]["example_idx"]],
    )
    cluster_to_pattern[cluster] = p


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
fig.savefig("figure_1/supplement/tsne_logos.png")
plt.close(fig)

dbd_to_color = {
    dbd: color
    for dbd, color in zip(seqlet_adata_no_na.obs["dbd_per_leiden"].unique(), seqlet_adata_no_na.uns["dbd_per_leiden_colors"])
}

fig, ax = plt.subplots(
    figsize = (mm_to_inch(116), mm_to_inch(116)),
)
X, Y = seqlet_adata_no_na.obsm["X_tsne"][:, 0], seqlet_adata_no_na.obsm["X_tsne"][:, 1]
ax.scatter(
    X, Y,
    c = [dbd_to_color[dbd] for dbd in seqlet_adata_no_na.obs["dbd_per_leiden"]],
    s = 0.5
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig("figure_1/tSNE_dbd.png", transparent=True, dpi = 300, bbox_inches="tight", pad_inches=0)
plt.close(fig)

fig, ax = plt.subplots(figsize = (10, 2))
for dbd in dbd_to_color:
    ax.scatter([], [], label = dbd, color = dbd_to_color[dbd], s = 6)
ax.legend(ncols = len(dbd_to_color) // 2, fontsize = "xx-small", loc = "upper center")
ax.set_axis_off()
fig.savefig("figure_1/tSNE_dbd_legend.png")

seqlet_adata_no_na.obs["class"] = classes[seqlet_adata_no_na.obs["example_idx"]]


for c in tqdm(np.unique(classes)):
    idx_to_show = np.where(classes == c)[0][0]
    hits = seqlet_adata_no_na.obs.query("example_idx == @idx_to_show")[
        ["start", "end", "dbd_per_leiden", "mean_contrib"]
    ]
    x_min = hits["start"].min() - 10
    x_max = hits["end"].max() + 10
    fig, axs = plt.subplots(
        figsize = (mm_to_inch(116.25), mm_to_inch(10)),
        nrows = 2,
        sharex = True,
        gridspec_kw = dict(height_ratios = [1, 1])
    )
    ax = axs[0]
    _ = logomaker.Logo(
        pd.DataFrame(
            (contrib[idx_to_show] * oh[idx_to_show]).T,
            columns = list("ACGT")
        ),
        ax = ax,
        zorder = 1
    )
    ax.set_rasterization_zorder(2)
    ymin, ymax = ax.get_ylim()
    for i, (_, (start, end, dbd, score)) in enumerate(hits.sort_values("start").iterrows()):
        p = matplotlib.patches.Rectangle(
            xy = (start, ymin),
            width = end - start,
            height = ymax - ymin,
            facecolor = dbd_to_color[dbd],
            alpha = 0.5
        )
        axs[0].add_patch(p)
        axs[1].text(start, 0.5 * (i % 2), dbd, fontsize = 5, color = dbd_to_color[dbd])
    for ax in axs:
        ax.set_xlim(x_min, x_max)
        ax.set_axis_off()
    fig.savefig(f"figure_1/{c}_example_DE.pdf", transparent=False, dpi = 1_000, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


seqlet_adata_no_na.obs["rel_example_idx"] = (seqlet_adata_no_na.obs["example_idx"] % 2_000)

with open("intermediate/figure_1/seqlets_all_cell_types.bed", "wt") as f:
    for _, (i, start, end, cl) in seqlet_adata_no_na.obs.sort_values(["example_idx", "start"])[["example_idx", "start", "end", "leiden"]].iterrows():
        _ = f.write(f"{i}\t{start}\t{end}\t{cl}\n")

pattern_names, patterns = crested.tl.modisco.load_patterns(
    CONTRIB_FOLDER, window=1000, class_names=None
)

oh_to_idx = {
    "".join(oh[i].argmax(0)[557:1557].astype(str)): i
    for i in range(oh.shape[0])
}

modisco_seqlets = []
for pattern, name in zip(patterns, pattern_names):
    for seqlet in pattern.seqlets:
        modisco_seqlets.append(
            (
                oh_to_idx["".join(seqlet.region_one_hot.argmax(1).astype(str))],
                seqlet.start + 557,
                seqlet.end + 557,
                name,
            )
        )

df_modisco_seqlets = pd.DataFrame(modisco_seqlets)
df_modisco_seqlets.sort_values([0, 1])

df_modisco_seqlets.to_csv(
    "intermediate/figure_1/seqlets_modisco.bed", sep="\t", header=False, index=False
)

"""
!bedtools intersect \
    -a intermediate/figure_1/seqlets_all_cell_types.bed \
    -b intermediate/figure_1/seqlets_modisco.bed -wo -f 0.4 \
    > intermediate/figure_1/seqlet_modisco_intersect.tsv
"""

seqlet_intersect = pd.read_table("intermediate/figure_1/seqlet_modisco_intersect.tsv", header=None)

intersect_set = set(map(tuple, seqlet_intersect[[0, 1, 2]].drop_duplicates().values))

seqlet_adata_no_na.obs["found_by_modisco"] = seqlet_adata_no_na.obs[["example_idx", "start", "end"]] \
    .apply(tuple, axis=1).isin(intersect_set)

seqlet_found_count = pd.crosstab(
    seqlet_adata_no_na.obs["dbd_per_leiden"].values,
    seqlet_adata_no_na.obs["found_by_modisco"].values
)

sorted_idx = (seqlet_found_count[False] / seqlet_found_count.sum(1) * 100).sort_values().index

seqlet_found_count = seqlet_found_count.loc[sorted_idx]

fig, ax = plt.subplots(
    figsize = (10, 10), 
)
ax.barh(
    np.arange(seqlet_found_count.shape[0]),
    seqlet_found_count[False] / seqlet_found_count.sum(1) * 100,
    color = [dbd_to_color[dbd] for dbd in seqlet_found_count.index]
)
ax.set_yticks(
    np.arange(seqlet_found_count.shape[0]), 
    seqlet_found_count.index,
)
ax.set_ylabel("DBD")
ax.set_xlabel("Percent extra seqlets")
ax.grid()
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figure_1/counts_seqlets_v_modisco.pdf")
plt.close(fig)

np.unique(seqlet_adata_no_na.obs["dbd_per_leiden"].values)

dbd_to_abbr = {
    "C2H2 ZF; Homeodomain": "C2H2 ZF; HD",
    "CUT; Homeodomain": "CUT; HD",
    "Homeodomain; POU": "HD; POU",
    "Nuclear receptor": "NR",
}

fig = sns.clustermap(
    pd.crosstab(
        seqlet_adata_no_na.obs["dbd_per_leiden"].values,
        seqlet_adata_no_na.obs["class"].values
    ).T.drop("nan", axis = 1).rename(dbd_to_abbr, axis = 1), 
    vmin = 0,
    cmap = "Spectral_r",
    xticklabels = True,
    yticklabels = True,
    figsize = (116 / 8, 36 / 8),
    linecolor = "black",
    linewidths = 0.01,
    robust = True
)
fig.savefig("figure_1/dbd_counts_per_class.pdf")

for dbd in tqdm(seqlet_adata_no_na.obs["dbd_per_leiden"].unique()):
    cluster = seqlet_adata_no_na.obs.query("dbd_per_leiden == @dbd")["leiden"].values[0]
    pattern = cluster_to_pattern[cluster]
    fig, ax = plt.subplots(figsize = (4, 2))
    _ = logomaker.Logo(
        pd.DataFrame(
            (pattern.ppm * pattern.ic()[:, None])[slice(*pattern.ic_trim(0.2))],
            columns = list("ACGT")
        ),
        ax = ax
    )
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim((0, 2))
    ax.set_ylabel("Bits")
    fig.tight_layout()
    fig.savefig(f"figure_1/{dbd.replace('/', '_')}_logo.pdf")
    plt.close(fig)


