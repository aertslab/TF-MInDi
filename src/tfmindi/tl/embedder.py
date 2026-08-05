from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score,
    normalized_mutual_info_score, fowlkes_mallows_score,
    homogeneity_completeness_v_measure)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pandas import DataFrame
from anndata import AnnData
from typing import Literal
import pandas as pd
import scanpy as sc
import numpy as np

"""
    Create region embeddings using seqlets
     - count = count vector of seqlet cluster occurence per region
     - pca/vae = mean pool of pca/vae reduced seqlet embeddings per region

    region_adata = tm.tl.embed_regions(seqlet_adata, embedding)
"""


EPSILON = 10**-10
DEF_DICT = {
    'pca':50,
    'vae':10,
    'count':"cluster_dbd",
}


#### Main function --------------------------------------------------------------------------------------------

def embed_regions(
        adata: AnnData,
        embedding: Literal["count","pca","vae"] = "pca",
        secondary: int | None = None,
        class_col: str = "cell_type",
        weighted: bool = True,
        normalised: bool = True,
        noise_factor: float = 0.0,
        tsne: bool = True,
        save_path: str = None,
        TSNE_kwargs: dict | None = None,
):

    # Check and clean input
    embedding, secondary, w, n = _sanity_checks_and_fixes(adata, embedding, secondary, weighted, normalised)

    # Calculate weights
    weight_df = _calc_weights(adata, weighted)

    # Calculate embeddings
    print(f"Calculating {embedding} embeddings ", end="")
    latent = None
    example_idx = None
    match embedding:

        case "pca" | "vae":
            latent, example_idx = _mean_aggregate(adata, embedding, secondary, w, n, weight_df)

        case "count":
            latent, example_idx, _ = _count_aggragate(secondary, w, n, weight_df, noise_factor)
        
    # Create region adata object
    region_adata = AnnData(
        X=latent,
        obs=pd.DataFrame(index=example_idx).merge(
                adata.obs[['example_idx',class_col]].drop_duplicates(),
                left_index=True,
                right_on='example_idx',
                how='left'),
    )

    # Calculate TSNE reduction if asked
    if tsne:
        calculate_embedding_tsne(region_adata, TSNE_kwargs)

    # Save on disk
    if save_path:
        region_adata.write_h5ad(save_path)

    return region_adata


def calculate_embedding_tsne(region_adata: AnnData, TSNE_kwargs: dict) -> np.ndarray:
    
    _kw: dict = dict(
        n_components=2,
        perplexity=30,
        random_state=42,
        n_jobs=-1,
    )

    if TSNE_kwargs: _kw.update(TSNE_kwargs)
    tsne_obj = TSNE(**_kw)
    region_adata.obsm['TSNE'] = tsne_obj.fit_transform(region_adata.X)


def leiden_clustering(region_adata: AnnData, resolution: float = 5.0, use_rep: str = 'X'):
    sc.pp.neighbors(region_adata, use_rep=use_rep, n_neighbors=12, metric="euclidean")
    sc.tl.leiden(region_adata, resolution=resolution, key_added='leiden', flavor='igraph')
    region_adata.obs['leiden'] = [f"l{c}" for c in list(region_adata.obs['leiden'])]


def optimal_hierarchical_clustering(
        region_adata: AnnData,
        metric: Literal["ARI","AMI","FMI","homogeneity","completeness","V_measure"] = "ARI",
        class_col: str = 'cell_type',
        cluster_name: str = 'region_cluster'
) -> float:

    # Seed the embedding space with tiny leiden clusters
    print(" [clustering] Seeding embedding space with leiden clusters...")
    leiden_clustering(region_adata)

    # Calculate mean vectors for each leiden seed
    print(" [clustering] Build hierarchical tree...")
    region_vectors = pd.DataFrame(region_adata.X)
    region_vectors['leiden'] = list(region_adata.obs['leiden'])
    region_clusters = region_vectors.groupby('leiden').mean()

    # Build hierarchical tree of leiden seeds
    dist = pdist(region_clusters.to_numpy(), metric="cosine")
    Z = linkage(dist, method="average")

    # Try 100 cutting heights for the hierarchical tree
    resolutions = [i/100 for i in range(100)]
    results = []
    for res in resolutions:

        print(f"\r [clustering] Cutting at height {res}", end="", flush=True)

        # Apply resolution
        clusters = fcluster(Z, t=res, criterion="distance")
        new_clusters = dict(zip(region_clusters.index,[f"H{c-1}" for c in clusters]))
        region_adata.obs[cluster_name] = [new_clusters[c] for c in region_adata.obs['leiden']]
        
        # ARI score
        labels_pred = region_adata.obs[cluster_name]
        labels_true = region_adata.obs[class_col]
        n_clusters = labels_pred.nunique()
        ari   = adjusted_rand_score(labels_true, labels_pred)
        ami   = adjusted_mutual_info_score(labels_true, labels_pred)
        fmi   = fowlkes_mallows_score(labels_true, labels_pred)
        h, c, v = homogeneity_completeness_v_measure(labels_true, labels_pred)

        results.append({
            'resolution': res, 'n_clusters': n_clusters,
            'ARI': ari, 'AMI': ami, 'FMI': fmi,
            'homogeneity': h, 'completeness': c, 'V_measure': v
        })
    
    # Optimal resolution
    resolutions = [ari['resolution'] for ari in results]
    optimal_res = resolutions[np.argmax([ari['ARI'] for ari in results])]
    optimal_ari = np.max([ari['ARI'] for ari in results])
    print(f"\n [clustering] The optimal resolution is {optimal_res} with an ARI of {optimal_ari}")

    clusters = fcluster(Z, t=optimal_res, criterion="distance")
    new_clusters = dict(zip(region_clusters.index,[f"H{c-1}" for c in clusters]))
    region_adata.obs[cluster_name] = [new_clusters[c] for c in region_adata.obs['leiden']]
    print(f" [clustering] Saved clusters in .obs {cluster_name}")

    # Plot
    print(" [clustering] Plotting...")
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(resolutions, [ari['ARI']          for ari in results], label='ARI', color='k')
    ax1.plot(resolutions, [ari['homogeneity']  for ari in results], label='homogeneity')
    ax1.plot(resolutions, [ari['completeness'] for ari in results], label='completeness')
    ax1.plot(resolutions, [ari['V_measure']    for ari in results], label='V_measure')
    ax1.axvline(optimal_res, color='gray', linestyle='--', label=f'max ARI')
    ax1.set_ylabel('Cluster pureness metrics')
    ax1.set_xlabel('Cutting heights')
    ax1.set_title('ARI score with respect to cut height for region clusters')

    ax2 = ax1.twinx()
    ax2.plot(resolutions, [ari['n_clusters'] for ari in results], label='N', color='black', linestyle=':')
    ax2.set_ylabel('N clusters')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.show()


def get_region_profiles(
        seqlet_adata: AnnData,
        region_adata: AnnData,
        annotation_col: str = 'seqlet_cluster',
        weight_col: str = 'attribution',
        region_cluster_col: str = 'region_cluster'
) -> DataFrame:

    profile = (seqlet_adata
        .obs[['example_idx',weight_col,annotation_col]]
        .merge(region_adata.obs, on='example_idx',how='left')
        .pivot_table(index='example_idx',columns=annotation_col,values=weight_col,aggfunc='sum'))

    region_adata.uns[annotation_col] = list(profile.columns)

    region_adata.obs = region_adata.obs.merge(
        region_adata.obs[['example_idx']].merge(
            profile, on='example_idx', how='left').fillna(0),on='example_idx',how='left')

    region_adata.uns['normalised_weighted_sum'] = \
        region_adata.obs.groupby(region_cluster_col)[region_adata.uns[annotation_col]].sum().to_numpy() / \
            np.max(region_adata.obs.groupby(region_cluster_col)[region_adata.uns[annotation_col]].sum().to_numpy(), axis=1)[:,None]

    index = region_adata.obs.groupby(region_cluster_col)[region_adata.uns[annotation_col]].sum().index

    region_adata.obs.drop(columns=region_adata.uns[annotation_col], inplace=True)

    return pd.DataFrame(region_adata.uns['normalised_weighted_sum'], index=index, columns=region_adata.uns[annotation_col])

#### Helper functions -----------------------------------------------------------------------------------------

def _sanity_checks_and_fixes(
        adata: AnnData,
        embedding: Literal["count","pca","vae"] = "pca",
        secondary: int | None = None,
        weighted: bool = True,
        normalised: bool = True,
):

    w = 'weighted' if weighted else 'unweighted'
    n = 'normalised' if normalised else 'unnormalised'

    vae_keys = [k for k in adata.obsm.keys() if k.split('_')[1] == 'vae']
    
    # pca selected but not present in obsm
    if embedding == "pca": assert "X_pca" in adata.obsm, \
        (f"Must first reduce seqlet space using pca before aggregating using pca.\n"
         f"tm.tl.reduce_seqlet_space(adata)")

    # vae selected but specific latents not present
    elif embedding == "vae" and secondary is not None: assert f"X_vae_{secondary}" in adata.obsm, \
        (f"Must first reduce seqlet space using 'vae' with {secondary} latents before aggregating this embedding.\n"
         f"tm.tl.reduce_seqlet_space(adata, embedding='vae', latents={secondary})")

    # vae selected, latent not specified and vae reductions present but default latent not present
    elif embedding == "vae" and secondary is None and len(vae_keys) > 1:
        assert f"X_vae_{DEF_DICT['vae']}" in adata.obsm, \
        (f"Please specify which of {vae_keys} to reduce")
    
    # vae selected, latent not specified but no latents
    elif embedding == "vae" and secondary is None:
        assert len(vae_keys) > 0, \
        (f"Must first reduce seqlet space using 'vae' before aggregating VAE reduced seqlet embeddings\n"
         f"tm.tl.reduce_seqlet_space(adata, embedding='vae')")
    
    # No column sepcified for count vector aggregation
    if embedding == "count": assert type(secondary) is not None, \
        (f"If aggregation method 'count' is chosen, please specify with 'annotation_column' "
         f"which seqlet annotation column from adata.obs is to be used as resolution")

    if embedding == "count" and secondary is not None and secondary not in adata.obs.columns:
        raise KeyError(f"'{secondary}' not in adata.obs!")


    # Fix secondary
    if embedding is not None: 
        secondary = DEF_DICT[embedding] if secondary is None else secondary

    # vae selected and only one latent present, then update latent
    if embedding == "vae" and len(vae_keys) == 1:
        secondary = int(vae_keys[0].split('_')[2])

    # vae selected, no latent specified and default vae present, then latent is default
    if embedding == "vae" and len(vae_keys) > 1 and f"X_vae_{DEF_DICT['vae']}" in vae_keys:
        secondary = DEF_DICT['vae']
    
    if 'region_embeddings' not in adata.uns:
        adata.uns['region_embeddings'] = {}
    if embedding not in adata.uns['region_embeddings']:
        adata.uns['region_embeddings'][embedding] = {}
    if secondary not in adata.uns['region_embeddings'][embedding]:
        adata.uns['region_embeddings'][embedding][secondary] = {}
    if w not in adata.uns['region_embeddings'][embedding][secondary]:
        adata.uns['region_embeddings'][embedding][secondary][w] = {}

    return embedding, secondary, w, n


def _calc_weights(adata: AnnData, weighted: bool) -> np.ndarray:

    weight_df = adata.obs.copy()
    weight_df['weight'] = np.ones((adata.obs.shape[0],1))

    if weighted:
        print("Calculating weights")
        weight_df['attribution'] = weight_df['attribution'].fillna(0)
        weight_df['att_abs'] = weight_df['attribution'].abs()
        weight_df = weight_df.merge(
            weight_df[['example_idx','att_abs']].groupby('example_idx').agg('sum').rename(
                columns={'att_abs':'att_sum'})['att_sum'], on='example_idx')
        weight_df['att_softmax'] = weight_df.apply(lambda r: (r['attribution']+EPSILON)/(r['att_sum']+EPSILON), axis=1)
        weight_df['weight'] = np.asarray(weight_df['att_softmax'])[:, None]

    return weight_df


def _mean_aggregate(adata: AnnData, reduction: str, latent: int, w: str, n:str, weight_df: pd.DataFrame) -> dict:

    reduction_key = 'X_pca' if reduction == 'pca' else f'X_vae_{latent}'

    print(f"on obsm.{reduction_key} to uns.region_embeddings.{reduction}.{w}.{n}.{latent}")

    ### Grab n_seqlets x first 'latent' latents from reduction and multiply seqlets with weights
    region_df = pd.DataFrame(data=adata.obsm[reduction_key][:,:latent] * weight_df['weight'].values[:, None],
                            index=adata.obs['example_idx']).groupby('example_idx').agg('mean')
    
    return ((region_df.to_numpy() / np.max(np.abs(region_df.to_numpy()), axis=1)[:,None] if n == 'normalised' else region_df.to_numpy()), list(region_df.index))


def _count_aggragate(annotation_column: str, w: str, n:str, weight_df: pd.DataFrame, noise_factor: float) -> dict:

    print((f"by constructing count vectors at adata.obs['{annotation_column}'] resolution to"
           f"uns.region_embeddings.count.{w}.{n}.{annotation_column}"))

    region_df = (weight_df.groupby(["example_idx", annotation_column],
                observed=True)["weight"].sum().unstack(fill_value=0))

    noise = np.random.uniform(0,np.mean(region_df.to_numpy()),region_df.to_numpy().shape)

    return (region_df.to_numpy() / np.max(np.abs(region_df.to_numpy()) + noise * noise_factor, axis=1)[:,None] if n == 'normalised' else region_df.to_numpy(), list(region_df.index), list(region_df.columns))
