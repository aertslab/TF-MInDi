from sklearn.manifold import TSNE
from anndata import AnnData
from typing import Literal
import pandas as pd
import numpy as np


### add option weighting
### choose PCA or VAE

EPSILON = 10**-10
DEF_DICT = {
    'pca':50,
    'vae':10,
    'count':"dbd_cluster",
}

def embed_regions(
        adata: AnnData,
        aggregate: Literal["count","mean"] = "mean",
        reduction: Literal["pca", "vae"] | None = "pca",
        annotation_column: str | None = "dbd_cluster",
        latent: int | None = None,
        weighted: bool = False,
        tsne: bool = True,
        TSNE_kwargs: dict | None = None,
):
    
    ### Input sanity checks and fixes -----------------------------------------------------------------------------------

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

    ### Calculate weights (softmax seqlet attribution per region if weighted == True, otherwise 1).  --------------------

    weights = np.ones((adata.obs.shape[0],1))
    if weighted:
        print("Calculating weights")
        wdf = adata.obs.copy()
        wdf['attribution'] = wdf['attribution'].fillna(0)
        wdf['att_abs'] = wdf['attribution'].abs()
        wdf = wdf.merge(
            wdf[['example_idx','att_abs']].groupby('example_idx').agg('sum').rename(
                columns={'att_abs':'att_sum'})['att_sum'], on='example_idx')
        wdf['att_softmax'] = wdf.apply(lambda r: (r['attribution']+EPSILON)/(r['att_sum']+EPSILON), axis=1)
        weights = np.asarray(wdf['att_softmax'])[:, None]


    ### Aggregate -------------------------------------------------------------------------------------------------------

    print(f"Calculating {aggregate} embeddings ", end="")
    match aggregate:

        case "mean":
            reduction_key = 'X_pca' if reduction == 'pca' else f'X_vae_{latent}'
            print(f"on {reduction_key} in obsm to region_embeddings.{reduction}.{latent} in uns")

            ### Grab n_seqlets x first 'latent' latents from reduction and multiply seqlets with weights
            region_df = pd.DataFrame(data=adata.obsm[reduction_key][:,:latent] * weights,
                                    index=adata.obs['example_idx']).groupby('example_idx').agg('mean')
            adata.uns['region_embeddings'][reduction][latent] = {'latent': region_df.to_numpy(), 
                                                                 'example_index': list(region_df.index)}

        case "count":
            print(f"by constructing count vectors at adata.obs['{annotation_column}'] resolution")
            raise NotImplementedError("Count vector reduction not yet implemented")


    ### TSNE embed ------------------------------------------------------------------------------------------------------

    if not TSNE: return
    print((f"Calculating TSNE reduction on region_embeddings.{reduction}.{latent}.latent in uns "
           f"to region_embeddings.{reduction}.{latent}.TSNE in uns"))
    _kw: dict = dict(
        n_components=2,
        perplexity=30,
        random_state=42,
        n_jobs=-1,
    )
    if TSNE_kwargs:
        _kw.update(TSNE_kwargs)

    tsne_obj = TSNE(**_kw)
    adata.uns['region_embeddings'][reduction][latent]['TSNE'] = \
         tsne_obj.fit_transform(adata.uns['region_embeddings'][reduction][latent]['latent'])