
import numpy as np
import os
import crested
import keras
from crested.tl.modisco._tfmodisco import Seqlet, ModiscoPattern
from crested.utils._seq_utils import one_hot_encode_sequence
import pandas as pd
from tqdm import tqdm
from tangermeme.seqlet import recursive_seqlets
from memelite import tomtom
import scanpy as sc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import logomaker
import matplotlib.pyplot as plt
from pycistarget.utils import load_motif_annotations
import random
import pyBigWig
import seaborn as sns
import scipy.sparse
import gc
import matplotlib
import lda
from pycisTopic.utils import loglikelihood
import seaborn as sns
import pickle
import scipy

matplotlib.rcParams['pdf.fonttype'] = 42

sc.settings.figdir = "./plots"


def load_motif(file_name: str) -> dict[str, np.ndarray]:
    motifs: dict[str, np.ndarray] = {}
    with open(file_name) as f:
        # initialize name
        name = f.readline().strip()
        if not name.startswith(">"):
            raise ValueError(f"First line of {file_name} does not start with '>'.")
        name = name.replace(">", "")
        pwm = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # we are at the start of a new motif
                motifs[name] = np.array(pwm)
                # scale values of motif
                motifs[name] = motifs[name].T / motifs[name].sum(1)
                # reset pwm and read new name
                name = line.replace(">", "")
                pwm = []
            else:
                # we are in the middle of reading the pwm values
                pwm.append([float(v) for v in line.split()])
        # add last motif
        motifs[name] = np.array(pwm)
        # scale values of motif
        motifs[name] = motifs[name].T / motifs[name].sum(1)
    return motifs

def _load_bed(fname: str) -> list[tuple[str, int, int]]:
    records = []
    with open(fname, "rt") as f:
        for line in f:
            chrom, start, end = line.split()[0:3]
            records.append((chrom, int(start), int(end)))
    return records

def _get_one_hot(
    bed_file: str,
    input_size: int,
    genome: crested.Genome,
    idx_to_keep: list[int] | None = None
) -> np.ndarray:
    regions = _load_bed(bed_file)
    print(f"\tGenerating one hot encoded sequence for: {len(regions)} regions.")
    one_hot = np.zeros( ( len(regions), input_size, 4 ), dtype = np.float32 )
    for i, (chrom, start, end) in tqdm(enumerate(regions), total = len(regions)):
        if idx_to_keep is not None:
            if i not in idx_to_keep:
                continue
        mid = start + (end - start) // 2
        new_start = mid - input_size // 2
        new_end   = mid + input_size // 2
        if input_size % 2:
            new_end = new_end + 1
        if new_start < 0 or new_end > genome.chrom_sizes[chrom]:
            continue
        one_hot[i] = one_hot_encode_sequence(
            genome.fetch(
                chrom=chrom,
                start=new_start,
                end=new_end
            ),
            expand_dim=False
        )
    if idx_to_keep is not None:
        return one_hot[idx_to_keep]
    else:
        return one_hot


""
""


MOTIF_DIR = "../../motif_collection_public/v10nr_clust_public/singletons/"

known_motifs = {}
for motif_file in os.listdir(MOTIF_DIR):
    known_motifs = {**known_motifs, **load_motif(os.path.join(MOTIF_DIR, motif_file))}


motif_name_to_cluster_name = {}
for motif_file in os.listdir(MOTIF_DIR):
    cluster_name = motif_file.replace(".cb", "")
    m = load_motif(os.path.join(MOTIF_DIR, motif_file))
    for k in m.keys():
        motif_name_to_cluster_name[k] = cluster_name


motifs_to_keep = []
with open("../feature_reduction/sampled_motifs.txt") as f:
    for l in f:
        motifs_to_keep.append(l.strip())

motif_to_tf = load_motif_annotations("homo_sapiens", version = "v10nr_clust")

motif_to_tf = motif_to_tf.apply(lambda row: ", ".join(row.dropna()), axis = 1).str.split(", ").explode().reset_index().rename({0: "TF"}, axis = 1)

human_tf_annot = pd.read_csv("https://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv", index_col = 0)[["HGNC symbol", "DBD"]]

motif_to_tf = motif_to_tf.merge(
    right=human_tf_annot,
    how="left",
    left_on = "TF",
    right_on = "HGNC symbol"
)

motif_to_dbd = (
    motif_to_tf.dropna().groupby('MotifID')['DBD']
    .agg(lambda x: x.mode().iat[0])  # take the first mode if there's a tie
    .reset_index()
)

motif_to_dbd = motif_to_dbd.set_index("MotifID")["DBD"].to_dict()


dbd_per_motif = np.array([ motif_to_dbd.get(motif_name_to_cluster_name[m], np.nan) for m in motifs_to_keep])


""
""


GRAD_DIR = "../shap/out"
HALF_TARGET_SIZE = 250

contrib = []
oh = []
classes = []
idx = []

grad_file_names = [
    f for f in os.listdir(GRAD_DIR)
    if f.endswith(".npz")
]

for f in tqdm(grad_file_names):
    c = f.split(".")[-2]
    t = f.split(".")[-3]
    np_f = np.load(os.path.join(GRAD_DIR, f))
    size = np_f["one_hot"].shape[1]
    mid = size // 2
    contrib.append(np_f["grad"][:, mid - HALF_TARGET_SIZE: mid + HALF_TARGET_SIZE])
    oh.append(np_f["one_hot"][:,   mid - HALF_TARGET_SIZE: mid + HALF_TARGET_SIZE])
    idx.append(np_f["idx"])
    classes.append(
        np.repeat(f"{t}.{c}", oh[-1].shape[0])
    )

oh = np.concatenate(oh)
contrib = np.concatenate(contrib)
classes = np.concatenate(classes)
idx = np.concatenate(idx)

seqlets = recursive_seqlets(
    (contrib * oh).sum(2),
    threshold = 0.05,
    additional_flanks = 3
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

l_sim = np.nan_to_num(-np.log10(sim + 1e-10)).astype(np.float32)

del(sim)
gc.collect()

l_sim_sparse = scipy.sparse.csr_array(
    np.clip(l_sim, 0.05, l_sim.max())
)

seqlet_adata = sc.AnnData(
    X=l_sim_sparse,
)

seqlet_adata.obs = seqlets.copy()

seqlet_adata.write("seqlet_adata.h5ad")

print("pca")
sc.tl.pca(seqlet_adata)
print("neighbors")
sc.pp.neighbors(seqlet_adata)
print("tsne")
sc.tl.tsne(seqlet_adata)

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

sc.tl.leiden(seqlet_adata, flavor="igraph", resolution=3)

sc.pl.tsne(
    seqlet_adata,
    color = "leiden",
    legend_loc = "on data",
    save = "_leiden.png"
)

seqlet_adata.obs["dbd"] = [
    dbd_per_motif[x] for x in tqdm(seqlet_adata.X.toarray().argmax(1))
]

leiden_to_dbd = seqlet_adata.obs[["leiden", "dbd"]] \
    .dropna().groupby("leiden")["dbd"].agg(lambda x: x.mode().iat[0]).to_dict()

seqlet_adata.obs["dbd_per_leiden"] = [
    leiden_to_dbd.get(cl, np.nan)
    for cl in seqlet_adata.obs["leiden"]
]

cmap1 = matplotlib.cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin = 0, vmax = 8),
    matplotlib.cm.tab20b
)

cmap2 = matplotlib.cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin = 0, vmax = len(seqlet_adata.obs["dbd_per_leiden"].unique()) - 8),
    matplotlib.cm.tab20c
)

d, c = np.unique(seqlet_adata.obs["dbd_per_leiden"], return_counts = True)

seqlet_adata.obs["dbd_per_leiden"] = seqlet_adata.obs["dbd_per_leiden"].cat.reorder_categories(d[np.argsort(-c)])

dbd_to_color = {
    dbd: (cmap1.to_rgba(i) if i < 8 else cmap2.to_rgba(i - 8)) if dbd != "nan" else "black"
    for i, dbd in enumerate(d[np.argsort(-c)])
}

sc.pl.tsne(
    seqlet_adata, color = "dbd_per_leiden", s = 3,
    save = "_DBD_per_leiden.png",
    palette = dbd_to_color
)

seqlet_adata.obs["region_id"] = idx[seqlet_adata.obs["example_idx"]]

seqlets_dedup = seqlet_adata.obs[["region_id", "start", "end", "dbd_per_leiden", "leiden"]] \
    .drop_duplicates()

seqlets_dedup = seqlets_dedup.loc[
    seqlets_dedup["dbd_per_leiden"] != "nan"
]

count_table = pd.crosstab(
    seqlets_dedup["region_id"].values,
    seqlets_dedup["leiden"].values
)

alpha = 50
n_iter = 150
eta = 0.1

n_topics = [
    10, 15, 20, 25, 30, 35, 40, 50
]

models: dict[int, lda.LDA] = {}
for n_topic in tqdm(n_topics):
    models[n_topic] = lda.LDA(
        n_topics=n_topic,
        n_iter=n_iter,
        random_state=123,
        alpha=alpha / n_topic,
        eta=eta
    )
    models[n_topic].fit(count_table.values)

model_to_ll = {}
for model in tqdm(models.values()):
    model_to_ll[model.n_topics] = loglikelihood(model.nzw_, model.ndz_, alpha / model.n_topics , eta)

fig, ax = plt.subplots()
ax.scatter(
    list(model_to_ll.keys()),
    list(model_to_ll.values()),
    color = "black"
)
ax.grid()
fig.tight_layout()
fig.savefig("plots/model_selection.png")

model = models[40]

fig, ax = plt.subplots()
ax.scatter(
    list(range(0, n_iter, model.refresh)),
    model.loglikelihoods_,
    color = "black"
)
ax.grid()
fig.tight_layout()
fig.savefig(f"plots/model_{model.n_topics}_ll_over_iter.png")

leiden_topic = pd.DataFrame(
    model.topic_word_.T,
    index = count_table.columns.values.astype(str),
    columns = [f"Topic_{x + 1}" for x in range(model.n_topics)]
)

dbd_topic = leiden_topic.groupby(leiden_to_dbd).mean()
sorted_topic = list((dbd_topic > 0.005).sum().sort_values().index[::-1])

tmp = pd.DataFrame(dbd_topic.T.idxmax()).reset_index()
tmp["order"] = [sorted_topic.index(x) for x in tmp[0]]
sorted_dbd = list(tmp.sort_values("order")["index"])

fig = sns.clustermap(
    dbd_topic.loc[sorted_dbd, sorted_topic].T,
    figsize = (8, 10),
    xticklabels = True,
    yticklabels = True,
    row_cluster=False,
    col_cluster=False,
    lw = 0.5, edgecolor = "white",
    cmap = "viridis",
    vmax = 0.01
)
fig.savefig("plots/dbd_topic_heatmap.png")

region_topic = pd.DataFrame(
    model.doc_topic_,
    index = count_table.index.values,
    columns = [f"Topic_{x + 1}" for x in range(model.n_topics)]
)

region_adata = sc.AnnData(region_topic)

sc.tl.tsne(region_adata)

sc.pl.tsne(
    region_adata,
    color = region_adata.var_names,
    save = "_topic_region.png"
)

region_adata.obs["model"] = [
    np.unique([x.split(".")[0] for x in classes[np.where(idx == int(x))[0]]])[0]
    for x in tqdm(region_adata.obs_names)
]

sc.pl.tsne(
    region_adata,
    color = "model",
    save = "_model_region.png",
    s = 10
)

# TODO: add cell type labels from models (class to cell type) - subclass level?

seqlet_adata.write("seqlet_adata_post.h5ad")

seqlet_adata = sc.read_h5ad("seqlet_adata_post.h5ad")

oh_dedup = _get_one_hot(
    "../../public_data/SCREEN_v4/GRCh38-cCREs.bed",
    2114,
    crested.Genome(
        "../../../resources/hg38/hg38.fa",
        "../../../resources/hg38/hg38.chrom.sizes"
    ),
    idx_to_keep = np.array(sorted(region_adata.obs_names.astype(int)))
)

test = np.zeros((region_adata.shape[0], 500, 4))

for i, rel_idx in enumerate(tqdm(region_adata.obs_names.astype(int))):
    example_idx = np.where(idx == rel_idx)[0][0]
    test[i] = oh[example_idx]

np.all(test[:] == oh_dedup[:, 807:1307])

model_names = np.unique([
    c.split(".")[0] for c in classes
])

np.save("oh_dedup.npy", oh_dedup)

model_to_predictions: dict[str, pd.DataFrame] = {}

for model_name in tqdm(model_names):
    model_path, class_names = crested.get_model(model_name)
    model = keras.models.load_model(model_path, compile=False)
    input_size = model.input.shape[1]
    start = 2114 // 2 - input_size // 2
    stop  = start + input_size
    pred = pd.DataFrame(
        model.predict(
            oh_dedup[:, start: stop], 
            verbose = 1
        ),
        columns = class_names)
    model_to_predictions[model_name] = pred


for model in model_to_predictions:
    model_to_predictions[model].to_csv(
        f"{model}.pred_dedup.tsv", sep = "\t",
        header = True, index = False
    )

model_to_predictions = {
    model_name: pd.read_table(
        f"{model_name}.pred_dedup.tsv"
    )
    for model_name in model_names
}

model_name_to_abbr = {
    "DeepHumanBrain": "HB",
    "DeepHumanCortex1": "HC1",
    "DeepHumanCortex2": "HC2",
    "DeepMEL1": "MEL",
    "DeepPBMC": "PBMC"
}

for model_name in model_names:
    model_to_predictions[model_name].columns = [
        c + "." + model_name_to_abbr[model_name]
        for c in 
        model_to_predictions[model_name].columns
    ]

model_predictions = pd.concat(model_to_predictions.values(), axis = 1)

model_predictions.index = region_adata.obs_names

model_predictions_scaled = (model_predictions - model_predictions.min()) \
    / (model_predictions.max() - model_predictions.min())

pred_to_region_top_corr = np.zeros(
    (model_predictions.shape[1], region_topic.shape[1])
)
for i, cell_type in enumerate(tqdm(model_predictions.columns)):
    for j, topic in enumerate(region_topic.columns):
        rho = scipy.stats.pearsonr(
            model_predictions_scaled[cell_type].values,
            region_topic[topic].values
        ).statistic
        pred_to_region_top_corr[i, j] = rho

fig = sns.clustermap(
    pd.DataFrame(
        pred_to_region_top_corr,
        index = model_predictions.columns.values,
        columns = region_topic.columns.values
    ).T,
    vmin = 0, vmax = 0.4,
    cmap = "viridis",
    figsize = (24, 8),
    xticklabels = True,
    yticklabels = True,
    edgecolor = "white", lw = 0.5
)
fig.savefig("plots/pred_region_topic_hm.png")

region_adata.obs = pd.concat([region_adata.obs, model_predictions_scaled], axis = 1)

sc.pl.tsne(
    region_adata,
    color = [
        "Topic_28", "Topic_26",
        "MGC_2.HB", "CD14_monocyte.PBMC", "CD16_monocyte.PBMC", 
        "MGC_1.HB", "Dendritic_cell.PBMC", "MGL.HC1", "MGL.HC2"
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_mgl.png"
)

sc.pl.tsne(
    region_adata,
    color = ["Topic_4", "CD4_T_cell.PBMC", "Cytotoxic_T_cell.PBMC", "Natural_killer_cell.PBMC"],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_t_cells.png"
)


sc.pl.tsne(
    region_adata,
    color = [
        "Topic_2", "Topic_3", "Topic_30",
        "OL.HC1", "OL.HC2", "Topic_4.MEL", "OGC_1.HB", "OGC_2.HB", "OGC_3.HB",
        "OPC.HC1", "OPC.HC2", "COP.HB", "OPC.HB"
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_OPC_MEL_cells.png"
)


sc.pl.tsne(
    region_adata,
    color = [
        "Topic_21", "Topic_37",
        "Topic_7.MEL", "Topic_15.MEL"
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_MES_cells.png"
)

sc.pl.tsne(
    region_adata,
    color = [
        "Topic_18", "Topic_34", "Topic_1", "Topic_25", "Topic_31",
        "LAMP5.HB", "INH_LAMP5.HC2", "INH_LAMP5.HC1", 
        "INH_VIP.HC1", "INH_VIP.HC2", "VIP_1.HB",
        "PV_ChCs.HB", "INH_PVALB.HC1", "INH_PVALB.HC2",
        "SNCG_1.HB", "SST_1.HB"
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_inh_cells.png"
)

sc.pl.tsne(
    region_adata,
    color = [
        "Topic_20", "Topic_7",
        "FOXP2_1.HB", "D1Pu.HB",
        "MSN_2.HB", "D12NAC.HB"
        ""
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_D1_cells.png"
)

sc.pl.tsne(
    region_adata,
    color = [
        "Topic_23", "Topic_17", "Topic_14", "Topic_40", "Topic_29",
        "ITV1C_2.HB", "L6B_1.HB", "ITL23_1.HB",
        "EXC_L2_3_IT.HC1", "EXC_L2_3_IT.HC2", 
        "EXC_L4_IT.HC1",  "EXC_L4_IT.HC2",
        "EXC_L5_IT.HC1", "EXC_L5_IT.HC2",
        "EXC_L6_IT.HC1", "EXC_L6_IT.HC2",
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_EXC_cells.png"
)

seqlet_adata_no_na = seqlet_adata[
    seqlet_adata.obs["dbd_per_leiden"] != "nan"
].copy()

del(seqlet_adata)
gc.collect()


print("pca")
sc.tl.pca(seqlet_adata_no_na)
print("neighbors")
sc.pp.neighbors(seqlet_adata_no_na)
print("tsne")
sc.tl.tsne(seqlet_adata_no_na)


sc.tl.leiden(seqlet_adata_no_na, flavor="igraph", resolution=3)

sc.pl.tsne(
    seqlet_adata_no_na,
    color = "leiden",
    legend_loc = "on data",
    save = "_leiden_no_na.png"
)

seqlet_adata_no_na.obs["dbd"] = [
    dbd_per_motif[x] for x in tqdm(seqlet_adata_no_na.X.toarray().argmax(1))
]

leiden_to_dbd = seqlet_adata_no_na.obs[["leiden", "dbd"]] \
    .dropna().groupby("leiden")["dbd"].agg(lambda x: x.mode().iat[0]).to_dict()

seqlet_adata_no_na.obs["dbd_per_leiden"] = [
    leiden_to_dbd.get(cl, np.nan)
    for cl in seqlet_adata_no_na.obs["leiden"]
]

cmap1 = matplotlib.cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin = 0, vmax = 8),
    matplotlib.cm.tab20b
)

cmap2 = matplotlib.cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin = 0, vmax = len(seqlet_adata_no_na.obs["dbd_per_leiden"].unique()) - 8),
    matplotlib.cm.tab20c
)

d, c = np.unique(seqlet_adata_no_na.obs["dbd_per_leiden"], return_counts = True)

seqlet_adata_no_na.obs["dbd_per_leiden"] = seqlet_adata_no_na.obs["dbd_per_leiden"].cat.reorder_categories(d[np.argsort(-c)])

dbd_to_color = {
    dbd: (cmap1.to_rgba(i) if i < 8 else cmap2.to_rgba(i - 8)) if dbd != "nan" else "black"
    for i, dbd in enumerate(d[np.argsort(-c)])
}

sc.pl.tsne(
    seqlet_adata_no_na, color = "dbd_per_leiden", s = 3,
    save = "_DBD_per_leiden_no_na.png",
    palette = dbd_to_color
)

seqlet_adata_no_na.write("seqlet_adata_no_na.h5ad")

seqlet_adata_no_na = sc.read_h5ad(
    "seqlet_adata_no_na.h5ad"
)

###

###

seqlets_dedup = seqlet_adata_no_na.obs[["region_id", "start", "end", "dbd_per_leiden", "leiden"]] \
    .drop_duplicates()

#499_621

seqlets_dedup = seqlets_dedup.loc[
    seqlets_dedup["dbd_per_leiden"] != "nan"
]


count_table = pd.crosstab(
    seqlets_dedup["region_id"].values,
    seqlets_dedup["leiden"].values
)

# [47988 rows x 256 columns]

alpha = 50
n_iter = 150
eta = 0.1

n_topics = [
    10, 15, 20, 25, 30, 35, 40, 50
]

models: dict[int, lda.LDA] = {}
for n_topic in tqdm(n_topics):
    models[n_topic] = lda.LDA(
        n_topics=n_topic,
        n_iter=n_iter,
        random_state=123,
        alpha=alpha / n_topic,
        eta=eta
    )
    models[n_topic].fit(count_table.values)

model_to_ll = {}
for model in tqdm(models.values()):
    model_to_ll[model.n_topics] = loglikelihood(model.nzw_, model.ndz_, alpha / model.n_topics , eta)

fig, ax = plt.subplots()
ax.scatter(
    list(model_to_ll.keys()),
    list(model_to_ll.values()),
    color = "black"
)
ax.grid()
fig.tight_layout()
fig.savefig("plots/model_selection_no_na.png")

model = models[40]

fig, ax = plt.subplots()
ax.scatter(
    list(range(0, n_iter, model.refresh)),
    model.loglikelihoods_,
    color = "black"
)
ax.grid()
fig.tight_layout()
fig.savefig(f"plots/model_{model.n_topics}_ll_over_iter_no_na.png")

leiden_topic = pd.DataFrame(
    model.topic_word_.T,
    index = count_table.columns.values.astype(str),
    columns = [f"Topic_{x + 1}" for x in range(model.n_topics)]
)

dbd_topic = leiden_topic.groupby(leiden_to_dbd).mean()
sorted_topic = list((dbd_topic > 0.005).sum().sort_values().index[::-1])

tmp = pd.DataFrame(dbd_topic.T.idxmax()).reset_index()
tmp["order"] = [sorted_topic.index(x) for x in tmp[0]]
sorted_dbd = list(tmp.sort_values("order")["index"])

fig = sns.clustermap(
    dbd_topic.loc[sorted_dbd, sorted_topic].T,
    figsize = (8, 10),
    xticklabels = True,
    yticklabels = True,
    row_cluster=False,
    col_cluster=False,
    lw = 0.5, edgecolor = "white",
    cmap = "viridis",
    vmax = 0.01
)
fig.savefig("plots/dbd_topic_heatmap_no_na.png")

region_topic = pd.DataFrame(
    model.doc_topic_,
    index = count_table.index.values,
    columns = [f"Topic_{x + 1}" for x in range(model.n_topics)]
)

region_adata = sc.AnnData(region_topic)

sc.tl.tsne(region_adata)

sc.tl.pca(region_adata)
sc.pp.neighbors(region_adata)
sc.tl.umap(region_adata)

sc.pl.tsne(
    region_adata,
    color = region_adata.var_names,
    save = "_topic_region_no_na.png"
)

oh_dedup = _get_one_hot(
    "../../public_data/SCREEN_v4/GRCh38-cCREs.bed",
    2114,
    crested.Genome(
        "../../../resources/hg38/hg38.fa",
        "../../../resources/hg38/hg38.chrom.sizes"
    ),
    idx_to_keep = np.array(sorted(region_adata.obs_names.astype(int)))
)

test = np.zeros((region_adata.shape[0], 500, 4))

for i, rel_idx in enumerate(tqdm(region_adata.obs_names.astype(int))):
    example_idx = np.where(idx == rel_idx)[0][0]
    test[i] = oh[example_idx]

np.all(test[:] == oh_dedup[:, 807:1307])

np.save("oh_dedup.npy", oh_dedup)

model_to_predictions: dict[str, pd.DataFrame] = {}

for model_name in tqdm(model_names):
    model_path, class_names = crested.get_model(model_name)
    model = keras.models.load_model(model_path, compile=False)
    input_size = model.input.shape[1]
    start = 2114 // 2 - input_size // 2
    stop  = start + input_size
    pred = pd.DataFrame(
        model.predict(
            oh_dedup[:, start: stop], 
            verbose = 1
        ),
        columns = class_names)
    model_to_predictions[model_name] = pred


for model in model_to_predictions:
    model_to_predictions[model].to_csv(
        f"{model}.pred_dedup.tsv", sep = "\t",
        header = True, index = False
    )

model_to_predictions = {
    model_name: pd.read_table(
        f"{model_name}.pred_dedup.tsv"
    )
    for model_name in model_names
}

model_name_to_abbr = {
    "DeepHumanBrain": "HB",
    "DeepHumanCortex1": "HC1",
    "DeepHumanCortex2": "HC2",
    "DeepMEL1": "MEL",
    "DeepPBMC": "PBMC"
}

for model_name in model_names:
    model_to_predictions[model_name].columns = [
        c + "." + model_name_to_abbr[model_name]
        for c in 
        model_to_predictions[model_name].columns
    ]

model_predictions = pd.concat(model_to_predictions.values(), axis = 1)

model_predictions.index = region_adata.obs_names

model_predictions_scaled = (model_predictions - model_predictions.min()) \
    / (model_predictions.max() - model_predictions.min())

pred_to_region_top_corr = np.zeros(
    (model_predictions.shape[1], region_topic.shape[1])
)
for i, cell_type in enumerate(tqdm(model_predictions.columns)):
    for j, topic in enumerate(region_topic.columns):
        rho = scipy.stats.pearsonr(
            model_predictions_scaled[cell_type].values,
            region_topic[topic].values
        ).statistic
        pred_to_region_top_corr[i, j] = rho

pred_to_region_top_corr = pd.DataFrame(
    pred_to_region_top_corr,
    index = model_predictions.columns.values,
    columns = region_topic.columns.values
).T

fig = sns.clustermap(
    pred_to_region_top_corr,
    vmin = 0, vmax = 0.4,
    cmap = "viridis",
    figsize = (24, 8),
    xticklabels = True,
    yticklabels = True,
    edgecolor = "white", lw = 0.5
)
fig.savefig("plots/pred_region_topic_hm_no_na.png")

region_adata.obs = pd.concat([region_adata.obs, model_predictions_scaled], axis = 1)

cell_types = [
    "MGC_2.HB", "CD14_monocyte.PBMC", "CD16_monocyte.PBMC", 
    "MGC_1.HB", "Dendritic_cell.PBMC", "MGL.HC1", "MGL.HC2"
]
topics = pred_to_region_top_corr.index[pred_to_region_top_corr[cell_types].max(1) > 0.25]

sc.pl.tsne(
    region_adata,
    color = [
        *topics, *cell_types
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_mgl_no_na.png"
)

cell_types = [
    "CD4_T_cell.PBMC", "Cytotoxic_T_cell.PBMC", "Natural_killer_cell.PBMC"
]
topics = pred_to_region_top_corr.index[pred_to_region_top_corr[cell_types].max(1) > 0.25]


sc.pl.tsne(
    region_adata,
    color = [
        *topics, *cell_types
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_t_cells_no_na.png"
)

cell_types = [
    "OL.HC1", "OL.HC2", "Topic_4.MEL", "OGC_1.HB", "OGC_2.HB", "OGC_3.HB",
    "OPC.HC1", "OPC.HC2", "COP.HB", "OPC.HB"
]
topics = pred_to_region_top_corr.index[pred_to_region_top_corr[cell_types].max(1) > 0.25]


sc.pl.tsne(
    region_adata,
    color = [
        *topics, *cell_types
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_OPC_MEL_cells_no_na.png"
)

cell_types = [
    "Topic_7.MEL", "Topic_15.MEL"
]
topics = pred_to_region_top_corr.index[pred_to_region_top_corr[cell_types].max(1) > 0.25]

sc.pl.tsne(
    region_adata,
    color = [
        *topics, *cell_types
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_MES_cells_no_na.png"
)

cell_types = [
    "LAMP5.HB", "INH_LAMP5.HC2", "INH_LAMP5.HC1", 
    "INH_VIP.HC1", "INH_VIP.HC2", "VIP_1.HB",
    "PV_ChCs.HB", "INH_PVALB.HC1", "INH_PVALB.HC2",
    "SNCG_1.HB", "SST_1.HB"
]
topics = pred_to_region_top_corr.index[pred_to_region_top_corr[cell_types].max(1) > 0.25]


sc.pl.tsne(
    region_adata,
    color = [
        *topics, *cell_types
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_inh_cells_no_na.png"
)

cell_types = [
        "FOXP2_1.HB", "D1Pu.HB",
        "MSN_2.HB", "D12NAC.HB"
]
topics = pred_to_region_top_corr.index[pred_to_region_top_corr[cell_types].max(1) > 0.25]

sc.pl.tsne(
    region_adata,
    color = [
        *topics, *cell_types
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_D1_cells_no_na.png"
)

cell_types = [
        "ITV1C_2.HB", "L6B_1.HB", "ITL23_1.HB",
        "EXC_L2_3_IT.HC1", "EXC_L2_3_IT.HC2", 
        "EXC_L4_IT.HC1",  "EXC_L4_IT.HC2",
        "EXC_L5_IT.HC1", "EXC_L5_IT.HC2",
        "EXC_L6_IT.HC1", "EXC_L6_IT.HC2",
]
topics = pred_to_region_top_corr.index[pred_to_region_top_corr[cell_types].max(1) > 0.25]


sc.pl.tsne(
    region_adata,
    color = [
        *topics, *cell_types
    ],
    ncols = 3,
    s = 10,
    save = "_pred_region_tsne_EXC_cells_no_na.png"
)

###############
# ACTUAL FIGURE
###############

fig, ax = plt.subplots(
    figsize = (10, 10)
)
X, Y = seqlet_adata_no_na.obsm["X_tsne"][:, 0], seqlet_adata_no_na.obsm["X_tsne"][:, 1]
ax.scatter(
    X, Y,
    c = [
        dbd_to_color[dbd] for dbd in seqlet_adata_no_na.obs["dbd_per_leiden"]
    ],
    s = 2
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "figure_3/tsne_dbd.png",
    dpi = 400,
    transparent=True, 
    bbox_inches="tight", pad_inches=0
)


fig = sns.clustermap(
    pred_to_region_top_corr.T,
    vmin = 0, vmax = 0.4,
    cmap = "YlGnBu",
    figsize = (10, 20),
    xticklabels = True,
    yticklabels = True,
    edgecolor = "white", lw = 0.5,
)
fig.savefig(
    "figure_3/dbd_heatmap.pdf",
)

sorted_topic = [x.get_text() for x in fig.ax_heatmap.get_xticklabels()]
tmp = pd.DataFrame(dbd_topic.T.idxmax()).reset_index()
tmp["order"] = [sorted_topic.index(x) for x in tmp[0]]
sorted_dbd = list(tmp.sort_values("order")["index"])

fig, ax = plt.subplots(figsize = (8, 8))
sns.heatmap(
    dbd_topic.loc[sorted_dbd, sorted_topic],
    xticklabels = True,
    yticklabels = True,
    lw = 0.5, linecolor = "black",
    cmap = "RdPu",
    vmax = 0.01,
    ax = ax,
    cbar=False
)
fig.savefig(
    "figure_3/dbd_to_topic.pdf",
)

cm_red = matplotlib.colors.LinearSegmentedColormap.from_list(
    "red",
    [(0, 0, 0), (1, 1, 0)],
    N = 100
)
cmap_red = matplotlib.cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin = 0, vmax = 0.4),
    cm_red
)

cm_blue = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue",
    [(0, 0, 0), (0, 1, 1)],
    N = 100
)
cmap_blue = matplotlib.cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin = 0, vmax = 0.4),
    cm_blue
)

def add_color(a, b):
    return [
        (x + y) / 2 for x, y in zip(a, b)
    ]

fig, ax = plt.subplots(figsize = (4,4))
s_idx = np.argsort(region_adata.obs[["CD14_monocyte.PBMC", "MGC_1.HB"]].mean(1).values)
X, Y = region_adata.obsm["X_tsne"][:, 0], region_adata.obsm["X_tsne"][:, 1]
ax.scatter(
    X[s_idx], Y[s_idx],
    color = [
        add_color(
            cmap_red.to_rgba(p1), cmap_blue.to_rgba(p2)
        )
        for p1, p2 in zip(
            region_adata.obs["CD14_monocyte.PBMC"].values[s_idx],
            region_adata.obs["MGC_1.HB"].values[s_idx]
        )
    ],
    s = region_adata.obs[["CD14_monocyte.PBMC", "MGC_1.HB"]].mean(1).values * 30
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "figure_3/CD14_MGC.png",
    dpi = 400,
    transparent=True, 
    bbox_inches="tight", pad_inches=0
)

fig, ax = plt.subplots(figsize = (4,4))
s_idx = np.argsort(region_adata.obs[["SST_1.HB", "LAMP5.HB"]].mean(1).values)
X, Y = region_adata.obsm["X_tsne"][:, 0], region_adata.obsm["X_tsne"][:, 1]
ax.scatter(
    X[s_idx], Y[s_idx],
    color = [
        add_color(
            cmap_red.to_rgba(p1), cmap_blue.to_rgba(p2)
        )
        for p1, p2 in zip(
            region_adata.obs["SST_1.HB"].values[s_idx],
            region_adata.obs["LAMP5.HB"].values[s_idx]
        )
    ],
   s = region_adata.obs[["SST_1.HB", "LAMP5.HB"]].mean(1).values * 30
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "figure_3/SST_LAMP5.png",
    dpi = 400,
    transparent=True, 
    bbox_inches="tight", pad_inches=0
)

fig, ax = plt.subplots(figsize = (4,4))
s_idx = np.argsort(region_adata.obs[["Topic_4.MEL", "OGC_1.HB"]].mean(1).values)
X, Y = region_adata.obsm["X_tsne"][:, 0], region_adata.obsm["X_tsne"][:, 1]
ax.scatter(
    X[s_idx], Y[s_idx],
    color = [
        add_color(
            cmap_red.to_rgba(p1), cmap_blue.to_rgba(p2)
        )
        for p1, p2 in zip(
            region_adata.obs["Topic_4.MEL"].values[s_idx],
            region_adata.obs["OGC_1.HB"].values[s_idx]
        )
    ],
    s = region_adata.obs[["Topic_4.MEL", "OGC_1.HB"]].mean(1).values * 30
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "figure_3/Topic_4_OGC_1.png",
    dpi = 400,
    transparent=True, 
    bbox_inches="tight", pad_inches=0
)


fig, ax = plt.subplots(figsize = (4,4))
s_idx = np.argsort(region_adata.obs[["CD4_T_cell.PBMC", "Natural_killer_cell.PBMC"]].mean(1).values)
X, Y = region_adata.obsm["X_tsne"][:, 0], region_adata.obsm["X_tsne"][:, 1]
ax.scatter(
    X[s_idx], Y[s_idx],
    color = [
        add_color(
            cmap_red.to_rgba(p1), cmap_blue.to_rgba(p2)
        )
        for p1, p2 in zip(
            region_adata.obs["CD4_T_cell.PBMC"].values[s_idx],
            region_adata.obs["Natural_killer_cell.PBMC"].values[s_idx]
        )
    ],
    s = region_adata.obs[["CD4_T_cell.PBMC", "Natural_killer_cell.PBMC"]].mean(1).values * 30
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "figure_3/CD4_NK.png",
    dpi = 400,
    transparent=True, 
    bbox_inches="tight", pad_inches=0
)




fig, ax = plt.subplots(figsize = (4,4))
s_idx = np.argsort(region_adata.obs[["FOXP2_1.HB", "MSN_2.HB"]].mean(1).values)
X, Y = region_adata.obsm["X_tsne"][:, 0], region_adata.obsm["X_tsne"][:, 1]
ax.scatter(
    X[s_idx], Y[s_idx],
    color = [
        add_color(
            cmap_red.to_rgba(p1), cmap_blue.to_rgba(p2)
        )
        for p1, p2 in zip(
            region_adata.obs["FOXP2_1.HB"].values[s_idx],
            region_adata.obs["MSN_2.HB"].values[s_idx]
        )
    ],
    s = region_adata.obs[["FOXP2_1.HB", "MSN_2.HB"]].mean(1).values * 30
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "figure_3/FOXP2_MSN.png",
    dpi = 400,
    transparent=True, 
    bbox_inches="tight", pad_inches=0
)

cmap_red = matplotlib.cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin = 0, vmax = 0.7),
    cm_red
)
cmap_blue = matplotlib.cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin = 0, vmax = 0.7),
    cm_blue
)

fig, ax = plt.subplots(figsize = (4,4))
s_idx = np.argsort(region_adata.obs[["L6B_1.HB", "ITL23_1.HB"]].mean(1).values)
X, Y = region_adata.obsm["X_tsne"][:, 0], region_adata.obsm["X_tsne"][:, 1]
ax.scatter(
    X[s_idx], Y[s_idx],
    color = [
        add_color(
            cmap_red.to_rgba(p1), cmap_blue.to_rgba(p2)
        )
        for p1, p2 in zip(
            region_adata.obs["L6B_1.HB"].values[s_idx],
            region_adata.obs["ITL23_1.HB"].values[s_idx]
        )
    ],
    s = region_adata.obs[["L6B_1.HB", "ITL23_1.HB"]].mean(1).values * 30
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "figure_3/L6B_ITL23.png",
    dpi = 400,
    transparent=True, 
    bbox_inches="tight", pad_inches=0
)


fig, ax = plt.subplots(figsize = (4,4))
s_idx = np.argsort(region_adata.obs[["Topic_7.MEL", "Topic_15.MEL"]].mean(1).values)
X, Y = region_adata.obsm["X_tsne"][:, 0], region_adata.obsm["X_tsne"][:, 1]
ax.scatter(
    X[s_idx], Y[s_idx],
    color = [
        add_color(
            cmap_red.to_rgba(p1), cmap_blue.to_rgba(p2)
        )
        for p1, p2 in zip(
            region_adata.obs["Topic_7.MEL"].values[s_idx],
            region_adata.obs["Topic_15.MEL"].values[s_idx]
        )
    ],
    s = region_adata.obs[["Topic_7.MEL", "Topic_15.MEL"]].mean(1).values * 30
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "figure_3/7_15_MEL.png",
    dpi = 400,
    transparent=True, 
    bbox_inches="tight", pad_inches=0
)

topics_to_show = [
    31, 10, 14,
    9, 30, 35, 37, 13, 19
]
for topic in tqdm(topics_to_show):
    fig, ax = plt.subplots(figsize = (4, 4))
    X, Y = region_adata.obsm["X_tsne"][:, 0], region_adata.obsm["X_tsne"][:, 1]
    v = region_adata.X[:, topic - 1]
    s_idx = np.argsort(v)
    v = (v - v.min()) / (v.max() - v.min())
    ax.scatter(
        X[s_idx], Y[s_idx],
        c = v[s_idx],
        vmin = 0, vmax = 0.6,
        s = 2
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(
        f"figure_3/region_topic_{topic}.png",
        dpi = 400,
        transparent=True, 
        bbox_inches="tight", pad_inches=0
    )

cmap1 = matplotlib.cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin = 0, vmax = 9),
    matplotlib.cm.Dark2
)

cmap2 = matplotlib.cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin = 0, vmax = len(region_adata.obs["leiden"].unique()) - 9),
    matplotlib.cm.Spectral
)

sc.tl.leiden(region_adata, flavor="igraph", resolution=0.5)

d, c = np.unique(region_adata.obs["leiden"], return_counts = True)

region_adata.obs["leiden"] = region_adata.obs["leiden"].cat.reorder_categories(d[np.argsort(-c)])

leiden_to_color = {
    dbd: (cmap1.to_rgba(i) if i < 8 else cmap2.to_rgba(i - 8))
    for i, dbd in enumerate(d[np.argsort(-c)])
}

sc.pl.tsne(region_adata, color = "leiden", save = "_leiden.png", s = 30, palette = leiden_to_color)

fig, ax = plt.subplots(figsize = (4,4))
X, Y = region_adata.obsm["X_tsne"][:, 0], region_adata.obsm["X_tsne"][:, 1]
ax.scatter(
    X, Y,
    c = [leiden_to_color[l] for l in region_adata.obs["leiden"]],
    s = 30
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(
    "figure_3/region_cluster.png",
    dpi = 400,
    transparent=True, 
    bbox_inches="tight", pad_inches=0
)

