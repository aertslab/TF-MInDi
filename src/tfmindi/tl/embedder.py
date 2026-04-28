from sklearn.manifold import TSNE
from anndata import AnnData
from typing import Literal
import pandas as pd
import numpy as np


EPSILON = 10**-10
DEF_DICT = {
    'pca':50,
    'vae':10,
    'count':"cluster_dbd",
}


#### Main function --------------------------------------------------------------------------------------------

def embed_regions(
        adata: AnnData,
        aggregate: Literal["count","mean"] = "mean",
        reduction: Literal["pca", "vae"] = "pca",
        annotation_column: str = DEF_DICT['count'],
        latent: int | None = None,
        weighted: bool = False,
        tsne: bool = True,
        TSNE_kwargs: dict | None = None,
):
    
    latent = _sanity_checks_and_fixes(adata, aggregate, reduction, annotation_column, latent)

    weight_df = _calc_weights(adata, weighted)

    print(f"Calculating {aggregate} embeddings ", end="")
    reduction_uns, latent_uns = reduction, latent
    match aggregate:

        case "mean":
            adata.uns['region_embeddings'][reduction][latent] = \
                _mean_aggregate(adata, reduction, latent, weight_df)

        case "count":
            adata.uns['region_embeddings']['count'][annotation_column] = \
                _count_aggragate(adata, annotation_column, weight_df)
            reduction_uns, latent_uns = 'count', annotation_column

    if not tsne: return
    adata.uns['region_embeddings'][reduction_uns][latent_uns]['TSNE'] = \
        _tsne(adata, reduction_uns, latent_uns, TSNE_kwargs)


#### Helper functions -----------------------------------------------------------------------------------------

def _sanity_checks_and_fixes(
        adata: AnnData,
        aggregate: Literal["count","mean"] = "mean",
        reduction: Literal["pca", "vae"] = "pca",
        annotation_column: str = DEF_DICT['count'],
        latent: int | None = None
):
    vae_keys = [k for k in adata.obsm.keys() if k.split('_')[1] == 'vae']

    # pca selected but not present in obsm
    if reduction == "pca": assert "X_pca" in adata.obsm, \
        "Must first reduce seqlet space using pca before aggregating using pca."

    # vae selected but specific latents not present
    elif reduction == "vae" and latent is not None: assert f"X_vae_{latent}" in adata.obsm, \
        f"Must first reduce seqlet space using 'vae' with {latent} latents before aggregating this reduction."

    # vae selected, latent not specified and vae reductions present but default latent not present
    elif reduction == "vae" and latent is None and len(vae_keys) > 0:
        assert f"X_vae_{DEF_DICT['vae']}" in adata.obsm, \
        (f"Must first reduce seqlet space using 'vae' with default {DEF_DICT['vae']} latents "
         f"before aggregating default VAE reduced seqlet embeddings")
    
    # vae selected, latent not specified but no latents
    elif reduction == "vae" and latent is None:
        assert len(vae_keys) > 0, \
        "Must first reduce seqlet space using 'vae' before aggregating VAE reduced seqlet embeddings"
    
    # No column sepcified for count vector aggregation
    if aggregate == "count": assert annotation_column is not None, \
        (f"If aggregation method 'count' is chosen, please specify with 'annotation_column' "
         f"which seqlet annotation column from adata.obs is to be used as resolution")


    # Fix latent
    if reduction is not None: 
        latent = DEF_DICT[reduction] if latent is None else latent
 
    # Region embedding placeholder in uns
    if 'region_embeddings' not in adata.uns: adata.uns['region_embeddings'] = {'pca':{}, 'vae':{}, 'count':{}}

    # vae selected and only one latent present, then update latent
    if aggregate == "vae" and len(vae_keys) == 1: latent = int(vae_keys[0].split('_')[2])

    # vae selected, no latent specified and default vae present, then latent is default
    if aggregate == "vae" and f"X_vae_{DEF_DICT['vae']}" in vae_keys: latent = DEF_DICT['vae']

    return latent



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


def _mean_aggregate(adata: AnnData, reduction: str, latent: int, weight_df: pd.DataFrame) -> dict:

    reduction_key = 'X_pca' if reduction == 'pca' else f'X_vae_{latent}'

    print(f"on {reduction_key} in obsm to region_embeddings.{reduction}.{latent} in uns")

    ### Grab n_seqlets x first 'latent' latents from reduction and multiply seqlets with weights
    region_df = pd.DataFrame(data=adata.obsm[reduction_key][:,:latent] * weight_df['weight'],
                            index=adata.obs['example_idx']).groupby('example_idx').agg('mean')
    
    return {'latent': region_df.to_numpy(), 'example_index': list(region_df.index)}



def _count_aggragate(adata: AnnData, annotation_column: str, weight_df: pd.DataFrame) -> dict:

    print(f"by constructing count vectors at adata.obs['{annotation_column}'] resolution")

    region_df = (weight_df.groupby(["example_idx", annotation_column],
                observed=True)["weight"].sum().unstack(fill_value=0))

    return {'latent':region_df.to_numpy(),  'example_index': list(region_df.index), 'columns':list(region_df.columns)}



def _tsne(adata: AnnData, reduction_uns: str, latent_uns: int, TSNE_kwargs: dict) -> np.ndarray:
    
    print((f"Calculating TSNE reduction on region_embeddings.{reduction_uns}.{latent_uns}.latent in uns "
           f"to region_embeddings.{reduction_uns}.{latent_uns}.TSNE in uns"))
    
    _kw: dict = dict(
        n_components=2,
        perplexity=30,
        random_state=42,
        n_jobs=-1,
    )

    if TSNE_kwargs: _kw.update(TSNE_kwargs)

    tsne_obj = TSNE(**_kw)
    return tsne_obj.fit_transform(adata.uns['region_embeddings'][reduction_uns][latent_uns]['latent'])