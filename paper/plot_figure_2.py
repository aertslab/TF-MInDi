import os
import numpy as np
import pandas as pd
import keras
import crested
from crested.utils._seq_utils import one_hot_encode_sequence
from tangermeme.seqlet import recursive_seqlets
from memelite import tomtom
import scanpy as sc
import scipy.sparse
from tqdm import tqdm
from scripts.io import (
    load_motif_to_dbd,
    load_motif
)
from scripts.pattern import create_pattern
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import logomaker
from urllib.request import urlretrieve
import pyBigWig
import matplotlib
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
sc.settings.figdir = "figure_2/supplement"


def mm_to_inch(mm: float) -> float:
    return mm / 25.4


def _load_bed(fname: str) -> list[tuple[str, int, int]]:
    records = []
    with open(fname, "rt") as f:
        for line in f:
            chrom, start, end = line.split()[0:3]
            records.append((chrom, int(start), int(end)))
    return records

def resize_region(
    chrom: str, start: int, end: int,
    input_size: int,
    genome: crested.Genome
) -> tuple[str, int, int] | None:
    mid = start + (end - start) // 2
    new_start = mid - input_size // 2
    new_end   = mid + input_size // 2
    if input_size % 2:
        new_end = new_end + 1
    if new_start < 0 or new_end > genome.chrom_sizes[chrom]:
        return None
    return (
        chrom, new_start, new_end
    )

def one_hot_to_hash(oh: np.ndarray) -> tuple[float]:
    return tuple((oh.argmax(1) + oh.sum(1)))

############
#LOAD MOTIFS
############


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

dbd_per_motif = np.array(
    [
        motif_to_dbd.get(
            motif_name_to_cluster_name[m], np.nan
        ) 
        for m in motifs_to_keep
    ]
)


#############################
#LOAD SCREEN REGIONS & RESIZE
#############################

model_path, _ = crested.get_model("DeepPBMC")
model = keras.models.load_model(
    model_path, compile=False
)

input_size  = model.input[0].shape[0]

screen_regions = _load_bed(
    "../../public_data/SCREEN_v4/GRCh38-cCREs.bed"
)

genome = crested.Genome(
    fasta="../../../resources/hg38/hg38.fa",
    chrom_sizes="../../../resources/hg38/hg38.chrom.sizes"
)

screen_regions_resized: list[tuple[str, int, int]] = []
for region in tqdm(screen_regions):
    option = resize_region(*region, input_size, genome)
    if option is None:
        continue
    screen_regions_resized.append(option)

###############
#LOAD GRADIENTS
###############

GRAD_DIR = "../shap/out"

contrib = []
oh = []
classes = []
region_idx = []

grad_file_names = [
    f for f in os.listdir(GRAD_DIR)
    if f.endswith(".npz") and "DeepPBMC" in f
]

for f in tqdm(grad_file_names):
    c = f.split(".")[-2]
    np_f = np.load(os.path.join(GRAD_DIR, f))
    contrib.append(np_f["grad"][:])
    oh.append(np_f["one_hot"][:])
    region_idx.append(np_f["idx"])
    classes.append(
        np.repeat(c, oh[-1].shape[0])
    )

oh = np.concatenate(oh)
contrib = np.concatenate(contrib)
classes = np.concatenate(classes)
region_idx = np.concatenate(region_idx)

for i in tqdm(range(region_idx.shape[0])):
    a = one_hot_encode_sequence(
        genome.fetch(
            *screen_regions_resized[region_idx[i]]
        ), 
    expand_dim=False)
    if not np.alltrue(a == oh[i]):
        raise ValueError("Mismatch")

############
#GET SEQLETS
############

seqlets = recursive_seqlets(
    (contrib * oh).sum(2),
    threshold = 0.05,
    additional_flanks = 3
)

seqlets = seqlets.reset_index(drop=True)

genomic_coords_seqlets = dict(
    chrom = np.zeros(len(seqlets), dtype="<U5"),
    g_start = np.zeros(len(seqlets), dtype=int),
    g_end = np.zeros(len(seqlets), dtype=int)
)

for i, (example_idx, start, end) in seqlets[
    ["example_idx", "start", "end"]].iterrows():
    c, s, _ = screen_regions_resized[
        region_idx[example_idx]
    ]
    genomic_coords_seqlets["chrom"][i]      = c
    genomic_coords_seqlets["g_start"][i]    = s + start
    genomic_coords_seqlets["g_end"][i]      = s + end

seqlets = pd.concat(
    [seqlets, pd.DataFrame(genomic_coords_seqlets)],
    axis = 1,
    ignore_index=False
)

seqlet_contribs = []
seqlet_contrib_actual = []
seqlet_seq = []
for _, (ex_idx, start, end) in seqlets[
    ['example_idx', 'start', 'end']].iterrows():
    X = contrib[ex_idx, start:end, :].T
    O = oh[ex_idx, start:end, :].T  # noqa: E741
    X = X / abs(X).max()
    seqlet_contribs.append(X)
    seqlet_contrib_actual.append(X * O)
    seqlet_seq.append(O)

unsigned_seqlet_contribs = [
    np.sign(x.mean()) * y 
    for x, y in zip(
        seqlet_contrib_actual,
        seqlet_contribs
    )
]

###########################################
#CALCULATE SIMILARITY MATRIX AND CLUSTERING
###########################################

sim, _, _, _, _ = tomtom(
    Qs=unsigned_seqlet_contribs,
    Ts=[known_motifs[k] for k in motifs_to_keep]
)

l_sim = np.nan_to_num(-np.log10(sim + 1e-10))

l_sim_sparse = scipy.sparse.csr_array(
    np.clip(l_sim, 0.05, l_sim.max())
)

seqlet_adata = sc.AnnData(
    X=l_sim_sparse,
)

print("pca")
sc.tl.pca(seqlet_adata)
print("neighbors")
sc.pp.neighbors(seqlet_adata)
print("tsne")
sc.tl.tsne(seqlet_adata)

seqlet_adata.obs = seqlets.copy()
seqlet_adata.obs["mean_contrib"] = [
    x.mean() for x in seqlet_contrib_actual
]

sc.pl.tsne(
    seqlet_adata,
    color = "mean_contrib",
    save = "_mean_contrib.png",
    cmap = "bwr"
)

sc.tl.leiden(seqlet_adata, flavor="igraph", resolution=5)

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


seqlet_adata_no_na = seqlet_adata[
    seqlet_adata.obs["dbd_per_leiden"] != "nan"
].copy()

print("pca")
sc.tl.pca(seqlet_adata_no_na)
print("neighbors")
sc.pp.neighbors(seqlet_adata_no_na)
print("tsne")
sc.tl.tsne(seqlet_adata_no_na)

sc.tl.leiden(
    seqlet_adata_no_na, 
    flavor="igraph", 
    resolution=10
)

sc.pl.tsne(
    seqlet_adata_no_na,
    color = "leiden",
    legend_loc = "on data",
    save = "_leiden_no_na.png"
)


leiden_to_dbd = seqlet_adata_no_na.obs[["leiden", "dbd"]] \
    .dropna() \
    .groupby("leiden")["dbd"] \
    .agg(lambda x: x.mode().iat[0]).to_dict()

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
        ohs=oh[seqlets.iloc[idc]["example_idx"]] \
            .swapaxes(1,2),
        contribs=contrib[seqlets.iloc[idc]["example_idx"]] \
            .swapaxes(1,2),
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
fig.savefig("figure_2/supplement/tsne_logos.png")
plt.close(fig)

####################
#CHIP SEQ VALIDATION
####################

hit_count_table = pd.crosstab(
    seqlet_adata_no_na.obs["example_idx"],
    seqlet_adata_no_na.obs["dbd_per_leiden"]
)

tf_to_encff = {
    "EBF1": "ENCFF810XRY",
    "IRF4": "ENCFF167KPF",
    "PAX5": "ENCFF914QGY"
}

for tf in tqdm(tf_to_encff):
    urlretrieve(
        url="https://www.encodeproject.org/files/" + \
            tf_to_encff[tf] + "/" + \
            "@@download/" + \
            tf_to_encff[tf] + ".bigWig",
        filename=f"intermediate/figure_2/{tf_to_encff[tf]}.bigWig"
    )

tf_to_ChIP = {
    tf: np.zeros(hit_count_table.shape[0], dtype = float) 
    for tf in tf_to_encff
}

for tf in tqdm(tf_to_encff):
    with pyBigWig.open(
        "intermediate/figure_2/" + \
        tf_to_encff[tf] + ".bigWig"
    ) as ChIP:
        seqlet_adata_no_na.obs[f"{tf}_ChIP"] = [
            ChIP.stats(chrom, start - 10, end + 10)[0]
            for _, (chrom, start, end) in tqdm(
                seqlet_adata_no_na.obs[
                    ["chrom", "g_start", "g_end"]].iterrows(),
                total = seqlet_adata_no_na.shape[0],
                leave = False
            )
        ]
        for i in tqdm(
            range(hit_count_table.shape[0]), leave = False
        ):
            region = screen_regions_resized[
                region_idx[hit_count_table.index[i]]
            ]
            tf_to_ChIP[tf][i] = ChIP.stats(*region)[0]

for c in seqlet_adata_no_na.obs.columns:
    if not c.endswith("_ChIP"):
        continue
    seqlet_adata_no_na.obs[f"{c}_scaled"] = \
        (seqlet_adata_no_na.obs[c] - seqlet_adata_no_na.obs[c].mean()) \
            / (seqlet_adata_no_na.obs[c].std())


sc.pl.tsne(
    seqlet_adata_no_na,
    color = [
        c for c in seqlet_adata_no_na.obs.columns
        if c.endswith("_ChIP_scaled")
    ],
    save = "_ChIP.png",
    s = 50,
    vmax = 8,
    cmap = "berlin",
    ncols = 2
)

dbd_to_color = {
    dbd: color
    for dbd, color in zip(
        seqlet_adata_no_na.obs["dbd_per_leiden"].unique(), 
        seqlet_adata_no_na.uns["dbd_per_leiden_colors"]
    )
}

fig, ax = plt.subplots(
    figsize = (mm_to_inch(114), mm_to_inch(114)),
)
X = seqlet_adata_no_na.obsm["X_tsne"][:, 0]
Y = seqlet_adata_no_na.obsm["X_tsne"][:, 1]
ax.scatter(
    X, Y,
    c = [
        dbd_to_color[dbd] 
        for dbd in seqlet_adata_no_na.obs["dbd_per_leiden"]
    ],
    s = 0.5
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "figure_2/tSNE_dbd.png",
    transparent=True, dpi = 300, 
    bbox_inches="tight", pad_inches=0)
plt.close(fig)

fig, ax = plt.subplots(figsize = (10, 2))
for dbd in dbd_to_color:
    ax.scatter([], [], label = dbd, color = dbd_to_color[dbd], s = 6)
ax.legend(ncols = len(dbd_to_color) // 2, fontsize = "xx-small", loc = "upper center")
ax.set_axis_off()
fig.savefig("figure_2/tSNE_dbd_legend.png", dpi = 300)

for tf in tqdm(["EBF1", "IRF4", "PAX5"]):
    fig, ax = plt.subplots(
        figsize = (
            mm_to_inch(40), mm_to_inch(40)
        )
    )
    X = seqlet_adata_no_na.obsm["X_tsne"][:, 0]
    Y = seqlet_adata_no_na.obsm["X_tsne"][:, 1]
    idx = np.argsort(
        seqlet_adata_no_na.obs[f"{tf}_ChIP_scaled"]
    )
    chip = seqlet_adata_no_na.obs[f"{tf}_ChIP_scaled"].values
    ax.scatter(
        X[idx], Y[idx],
        c = chip[idx],
        s = 0.1,
        zorder = 1,
        cmap = "berlin",
        vmax = 10,
        vmin = 0,
        marker = ","
    )
    #ax.set_title(tf, zorder = 2, fontdict=dict(fontsize = 5))
    ax.set_rasterization_zorder(2)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(
        f"figure_2/tSNE_{tf}.png", dpi = 400,
        transparent=True, 
        bbox_inches="tight", pad_inches=0
    )
    plt.close(fig)



tf_to_dbd = {
    "EBF1": "EBF1", 
    "IRF4": "IRF", 
    "PAX5": "Paired box"
}


for tf in tf_to_ChIP:
    tf_to_ChIP[tf] = (tf_to_ChIP[tf] - tf_to_ChIP[tf].mean()) \
        / (tf_to_ChIP[tf].std())

data = dict(
    DBD = [],
    region_contains_dbd = [],
    ChIP = []
)

for tf in tf_to_encff:
    dbd = tf_to_dbd[tf]
    c_w_dbd  = tf_to_ChIP[tf][hit_count_table[dbd] > 0]
    c_wo_dbd = tf_to_ChIP[tf][hit_count_table[dbd] == 0]
    data["DBD"].extend(
        np.repeat(dbd, len(c_w_dbd) + len(c_wo_dbd))
    )
    data["region_contains_dbd"].extend(
        np.repeat(True, len(c_w_dbd))
    )
    data["region_contains_dbd"].extend(
        np.repeat(False, len(c_wo_dbd))
    )
    data["ChIP"].extend(
        [*c_w_dbd, *c_wo_dbd]
    )

fig, ax = plt.subplots(figsize = (10, 10))
sns.boxplot(
    data=data,
    x = "DBD",
    y = "ChIP",
    hue = "region_contains_dbd",
    ax = ax,
    palette = {True: "dimgrey", False: "lightgray"}
)
ax.set_ylim(-1, 10)
ax.grid()
fig.tight_layout()
fig.savefig("figure_2/ChIP_v_DBD.pdf")

###########
#subset ETS
###########

seqlet_adata_ets = seqlet_adata_no_na[
    seqlet_adata_no_na.obs["dbd_per_leiden"] == "Ets"
]

print("pca")
sc.tl.pca(seqlet_adata_ets)
print("neighbors")
sc.pp.neighbors(seqlet_adata_ets)
print("tsne")
sc.tl.tsne(seqlet_adata_ets)
print("umap")
sc.tl.umap(seqlet_adata_ets)

del(seqlet_adata_ets.uns["leiden_colors"])
sc.tl.leiden(
    seqlet_adata_ets, 
    flavor="igraph", 
    resolution=0.15
)

sc.pl.tsne(
    seqlet_adata_ets,
    color = "leiden",
    legend_loc = "on data",
    save = "_leiden_ets.png"
)

sc.pl.umap(
    seqlet_adata_ets,
    color = "leiden",
    legend_loc = "on data",
    save = "_leiden_ets.png"
)


tf_to_encff_ets = {
    "ETS1": "ENCFF686FAA",
}

for tf in tqdm(tf_to_encff_ets):
    urlretrieve(
        url="https://www.encodeproject.org/files/" + \
            tf_to_encff_ets[tf] + "/" + \
            "@@download/" + \
            tf_to_encff_ets[tf] + ".bigWig",
        filename=f"intermediate/figure_2/{tf_to_encff_ets[tf]}.bigWig"
    )

for tf in tqdm(tf_to_encff_ets):
    with pyBigWig.open(
        "intermediate/figure_2/" + \
        tf_to_encff_ets[tf] + ".bigWig"
    ) as ChIP:
        seqlet_adata_ets.obs[f"{tf}_ChIP"] = [
            ChIP.stats(chrom, start - 10, end + 10)[0]
            for _, (chrom, start, end) in tqdm(
                seqlet_adata_ets.obs[
                    ["chrom", "g_start", "g_end"]].iterrows(),
                total = seqlet_adata_ets.shape[0],
                leave = False
            )
        ]

for c in seqlet_adata_ets.obs.columns:
    if not c.endswith("_ChIP"):
        continue
    seqlet_adata_ets.obs[f"{c}_scaled"] = \
        (seqlet_adata_ets.obs[c] - seqlet_adata_ets.obs[c].mean()) \
            / (seqlet_adata_ets.obs[c].std())

sc.pl.tsne(
    seqlet_adata_ets,
    color = "ETS1_ChIP",
    save = "_chip_ets.png",
    cmap = "berlin",
    s = 30,
    vmax = 40
)

sc.pl.umap(
    seqlet_adata_ets,
    color = "ETS1_ChIP",
    save = "_chip_ets.png",
    cmap = "berlin",
    s = 30,
    vmax = 40
)

sc.pl.pca(
    seqlet_adata_ets,
    color = "ETS1_ChIP",
    save = "_chip_ets.png",
    cmap = "berlin",
    s = 30,
    vmax = 40
)

fig, ax = plt.subplots(figsize = (10, 10))
X = seqlet_adata_ets.obsm["X_umap"][:, 0]
Y = seqlet_adata_ets.obsm["X_umap"][:, 1]
idx = np.argsort(seqlet_adata_ets.obs["ETS1_ChIP"])
chip = seqlet_adata_ets.obs["ETS1_ChIP"]
ax.scatter(
    X[idx], Y[idx],
    c = chip[idx],
    s = 30,
    vmax = 40,
    cmap = "berlin",
)
ax.set_axis_off()
fig.savefig(
    "figure_2/umap_ETS_chip.png",
    dpi = 400,
    transparent=True, 
    bbox_inches="tight", pad_inches=0
)
plt.close(fig)


cluster_to_pattern_ets = {}
for cluster in tqdm(seqlet_adata_ets.obs.leiden.unique()):
    idc = seqlet_adata_ets.obs.query("leiden == @cluster").index.astype(int)
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
            .swapaxes(1,2),
        contribs=contrib[seqlets.iloc[idc]["example_idx"]] \
            .swapaxes(1,2),
    )
    cluster_to_pattern_ets[cluster] = p

cl_to_color = {
    cl: color
    for cl, color in zip(
        seqlet_adata_ets.obs["leiden"].unique(),
        seqlet_adata_ets.uns["leiden_colors"]
    )
}

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(
    seqlet_adata_ets.obsm["X_umap"][:, 0],
    seqlet_adata_ets.obsm["X_umap"][:, 1],
    s = 2, 
    c = [
        cl_to_color[cl] for cl in seqlet_adata_ets.obs["leiden"]
    ],
    alpha=0.3,
   # z_order = 1
)
for cluster in tqdm(seqlet_adata_ets.obs.leiden.unique()):
    X = seqlet_adata_ets.obsm["X_umap"][seqlet_adata_ets.obs.leiden==cluster, 0].mean()
    Y = seqlet_adata_ets.obsm["X_umap"][seqlet_adata_ets.obs.leiden==cluster, 1].mean()
    ax_inset = inset_axes(
        ax,
        width=0.8 * 2,
        height=0.4 * 2,
        bbox_to_anchor=(X, Y),
        bbox_transform=ax.transData,
        loc='center')
    _ = logomaker.Logo(
        pd.DataFrame(
            (cluster_to_pattern_ets[cluster].ppm * \
             cluster_to_pattern_ets[cluster].ic()[:, None]
            )[slice(*cluster_to_pattern_ets[cluster].ic_trim(0.2))],
            columns=["A", "C", "G", "T"]
        ),
        ax=ax_inset,
        edgecolor = "black",
        edgewidth = 1
        #zorder = 2
    )
    ax_inset.set_axis_off()
    #ax_inset.set_title(
    #    f"{len(cluster_to_pattern_ets[cluster].seqlets)}",
    #    fontdict=dict(fontsize = 10)
    #)
   # ax.set_rasterization_zorder(2)
ax.set_axis_off()
ax.set_xlabel("tsne1")
ax.set_ylabel("tsne2")
fig.savefig(
    "figure_2/umap_logos_ets_wo_text.png",
    dpi = 400,
    transparent=True, 
    bbox_inches="tight", pad_inches=0
)
plt.close(fig)

"""
In [579]: pd.DataFrame(data).groupby(["DBD", "region_contains_dbd"]).c
        ⋮ ount()
Out[579]: 
                                 ChIP
DBD        region_contains_dbd       
EBF1       False                52564
           True                  3332
IRF        False                41940
           True                 13956
Paired box False                50372
           True                  5524

"""

seqlet_adata.write("intermediate/figure_2/seqlet_adata.h5ad")