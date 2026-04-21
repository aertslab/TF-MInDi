from sklearn.manifold import TSNE
from anndata import AnnData
from typing import Literal
import pandas as pd
import numpy as np


### add option weighting
### choose PCA or VAE

EPSILON = 10**-10


def embed_regions(
        adata: AnnData,
        aggregate: Literal["count","mean"] = "mean",
        reduction: Literal["pca", "vae"] = "pca",
        annotation_column: str | None = "dbd_cluster",
        latent: int | None = None,
        weighted: bool = False,
        TSNE_kwargs: dict | None = None,
):
    
    ### Input sanity checks ---------------------------------------------------------------------------------------------

    # No reduction specified when 'mean' selected and more than one option is available
    if len([k for k in list(adata.obsm.keys()) if 'tsne' not in k]) > 0 and aggregate == "mean":
        assert reduction is not None and latent is not None, \
            (f"If aggregation 'mean' is chosen, please specify with 'reduction' and 'latent' "
             f"which 'X_<reduction>_<latent>' of "
             f"[{', '.join([k for k in list(adata.obsm.keys()) if 'tsne' not in k])}] "
             f" to use for embedding")
    
    # No column sepcified for count vector aggregation
    if aggregate == "count": assert annotation_column is not None, \
            (f"If aggregation 'count' is chosen, please specify with 'annotation_column' "
             f"which seqlet annotation column from adata.obs is to be used as resolution")
    

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

    if 'region_embeddings' not in adata.uns: adata.uns['region_embeddings'] = {f'RE_{reduction}_{latent}':{}}
    print((f"Calculating {aggregate} embeddings on "
           f"{annotation_column if aggregate == 'count' else f'X_{reduction}_{latent}'} "
           f"to adata.uns['region_embeddings']['RE_{reduction}_{latent}']['latent']"))
    match aggregate:
        case "mean":
            region_df = pd.DataFrame(data=adata.obsm[f'X_{reduction}_{latent}'] * weights,
                                     index=adata.obs['example_idx']).groupby('example_idx').agg('mean')
            adata.uns['region_embeddings'][f'RE_{reduction}_{latent}']['latent'] = region_df.to_numpy()
            adata.uns['region_embeddings'][f'RE_{reduction}_{latent}']['example_index'] = list(region_df.index)
        case "count":
            print("Not implemented")


    ### TSNE embed ------------------------------------------------------------------------------------------------------

    print((f"Calculating TSNE reduction on adata.uns['region_embeddings']['RE_{reduction}_{latent}'] "
           f"to adata.uns['region_embeddings'][f'RE_{reduction}_{latent}']['TSNE']"))
    _kw: dict = dict(
        n_components=2,
        perplexity=30,
        random_state=42,
        n_jobs=-1,
    )
    if TSNE_kwargs:
        _kw.update(TSNE_kwargs)

    tsne_obj = TSNE(**_kw)
    adata.uns['region_embeddings'][f'RE_{reduction}_{latent}']['TSNE'] = \
         tsne_obj.fit_transform(adata.uns['region_embeddings'][f'RE_{reduction}_{latent}']['latent'])