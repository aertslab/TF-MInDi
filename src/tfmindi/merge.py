"""TF-MInDi anndata merge functionality."""

from _collections_abc import dict_items

import anndata  # type: ignore
import numpy as np  # type: ignore
from anndata._core.merge import StrategiesLiteral  # type: ignore

_INDEX_COLS = ["example_oh_idx", "example_contrib_idx", "example_idx"]


def concat(
    adatas: list[anndata.AnnData] | dict[str, anndata.AnnData],
    idx_match: bool = False,
    index_unique: str = "-",
    merge: StrategiesLiteral | None = "same",
    **kwargs,
) -> anndata.AnnData:
    """
    Concatenate multiple TF-MInDi anndatas preserving data stored in uns['unique_examples'].

    Parameters
    ----------
    adatas
        The objects to be concatenated. If a dict is passed, keys are used for
        the keys argument and values are concatenated.
    idx_match
        Whether `example_oh_idx`, `example_contrib_idx` and `example_idx`
        refer to the same data across adatas or not.
    index_unique
        Whether to make the index unique by using the keys.
        If provided, this is the delimiter between "{orig_idx}{index_unique}{key}".
        When None, the original indices are kept.
    merge
        How elements not aligned to the axis being concatenated along are selected.
        See: anndata.concat for more info.
    **kwargs
        Extra key word arguments passed to anndata.concat
    """
    if merge is None:
        print("merge is None, vars will not be carried over to concatenated adata!")
    if not isinstance(index_unique, str):
        raise ValueError("index_unique should be a string.")

    if isinstance(adatas, dict):
        adatas_iter: dict_items[str, anndata.AnnData] | list[tuple[int, anndata.AnnData]] = adatas.items()
    else:
        adatas_iter = [(i, a) for i, a in enumerate(adatas)]

    def _has_unique_example(adata: anndata.AnnData) -> bool:
        return (
            "unique_examples" in adata.uns.keys()
            and "oh" in adata.uns["unique_examples"].keys()
            and "contrib" in adata.uns["unique_examples"].keys()
            and "example_oh_idx" in adata.obs.columns
            and "example_contrib_idx" in adata.obs.columns
        )

    if idx_match:
        # make sure same data is stored in all unique examples
        v_oh = [a.uns["unique_examples"]["oh"] for _, a in adatas_iter if _has_unique_example(a)]
        v_co = [a.uns["unique_examples"]["contrib"] for _, a in adatas_iter if _has_unique_example(a)]

        if not all(np.array_equal(v_oh[0], arr) for arr in v_oh) or not all(
            np.array_equal(v_co[0], arr) for arr in v_co
        ):
            message = (
                "All adata.uns['unique_examples']['contrib'] and adata.uns['unique_examples']['oh']"
                + "should be the same across adatas."
            )
            raise ValueError(message)

    if not idx_match:
        # the columns representing indices in adata.obs do *not* point to the
        # same data. In this case we make the indices unique across all adatas.

        # These values will be stored in the combined adata
        l_unique_examples_oh: list[np.ndarray] = []
        l_unique_examples_co: list[np.ndarray] = []

        # Dictionary to keep track of index offsets, adatas will be changed
        # in place. After concatenation the original values will be replaced
        # in the adatas.
        idx_col_offset: dict[str, int] = dict.fromkeys(_INDEX_COLS, 0)
        idx_col_offset_per_ad: dict[str | int, dict[str, int]] = {}

        for k, adata in adatas_iter:
            if _has_unique_example(adata):
                idx_col_offset_per_ad[k] = idx_col_offset
                l_unique_examples_oh.extend(adata.uns["unique_examples"]["oh"])
                l_unique_examples_co.extend(adata.uns["unique_examples"]["contrib"])

                # change indeces in place
                for col in _INDEX_COLS:
                    adata.obs[col] += idx_col_offset[col]

                # get offset for next iteration
                idx_col_offset = {col: adata.obs[col].max() + 1 for col in _INDEX_COLS}

        unique_examples_oh = np.array(l_unique_examples_oh)
        unique_examples_co = np.array(l_unique_examples_co)

    else:
        # All values in v_oh and v_co are unique, just take the first
        unique_examples_oh = v_oh[0]
        unique_examples_co = v_co[0]

    adata_concat = anndata.concat(
        adatas={str(k): adata for k, adata in adatas_iter}, index_unique=index_unique, merge=merge, **kwargs
    )

    adata_concat.uns["unique_examples"] = {}
    adata_concat.uns["unique_examples"]["oh"] = unique_examples_oh
    adata_concat.uns["unique_examples"]["contrib"] = unique_examples_co

    # 2. reset example_oh_idx and example_contrib_idx in place to original values
    if not idx_match:
        for k, adata in adatas_iter:
            if _has_unique_example(adata):
                for col in _INDEX_COLS:
                    adata.obs[col] -= idx_col_offset_per_ad[k][col]

    return adata_concat
