# Preprocessing `pp`

Preprocessing functions for seqlet extraction and motif similarity analysis.

```{eval-rst}
.. module:: tfmindi.pp
.. currentmodule:: tfmindi

.. autosummary::
    :toctree: ../generated

    pp.extract_seqlets
    pp.calculate_motif_similarity
    pp.create_seqlet_adata
```

## Accessors

Per-seqlet and per-region arrays are stored once in `adata.uns["unique_examples"]` and read
back through these functions rather than being duplicated into `adata.obs`.

```{eval-rst}
.. currentmodule:: tfmindi

.. autosummary::
    :toctree: ../generated

    pp.get_seqlet_oh
    pp.get_seqlet_ohs
    pp.get_seqlet_matrix
    pp.get_seqlet_matrices
    pp.get_example_idx
    pp.get_example_oh
    pp.get_example_contrib
```
