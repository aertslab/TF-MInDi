# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## Unreleased

### Features

- `tfmindi.pp.extract_seqlets` now supports multiple seqlet-calling algorithms through the `method` argument:
  - `"recursive_q99_abs_smooth"` (new **default**): triangular smoothing + per-region q99 normalization + the recursive caller on the *absolute* importance track. Because it calls on absolute importance, it captures both positive and negative seqlets, resolving the "positive-only" limitation of the previous algorithm noted in 1.1.0.
  - `"recursive_raw"`: the recursive caller on the raw signed track. Reproduces the previous default behaviour.
  - `"hysteresis"`: two-threshold local caller.
  - `"local_contrast"`: multi-scale sliding-window contrast caller.
  - `"wavelet_otsu"`: wavelet-denoise + Otsu-threshold caller (adds a `pywavelets` dependency).
- Method-specific hyperparameters can now be passed directly as keyword arguments, e.g. `extract_seqlets(contrib, oh, method="hysteresis", seed_z=3.0)`.

### Breaking changes

- The default seqlet caller changed from the raw recursive algorithm to `"recursive_q99_abs_smooth"`, so seqlet calls differ from previous versions by default. Pass `method="recursive_raw"` to reproduce the old behaviour.
- The seqlet DataFrame (and the resulting `adata.obs`) column `p-value` was renamed to `score`. For the recursive callers `score = -log10(p)` (higher = more significant); for the other callers it is a caller-specific confidence score, not a p-value.
- `extract_seqlets` no longer exposes `threshold` and `additional_flanks` as dedicated parameters; they are still accepted as keyword arguments (forwarded to the recursive callers). Positional use of a third argument now sets `method`.
- The log-likelihood used by `tfmindi.tl.evaluate_topic_models` was accumulating incorrectly (see Bugfixes). Now that it is fixed, the model-selection sweep can pick a **different number of topics** than previous versions did for the same data.
- `method="finemo_fit_contrib"` now writes a numeric `score` (the strongest hit coefficient of a merged seqlet) instead of a comma-joined string, restoring the shared seqlet-caller column contract. The full per-hit coefficients moved to a new `finemo_hit_coefficients` column.
- `adata.obs["seqlet_oh"]` and `adata.obs["seqlet_matrix"]` are no longer stored. Both were exact
  slices of the regions already held in `adata.uns["unique_examples"]`, and are now derived through
  the new accessors `tfmindi.pp.get_seqlet_oh` / `get_seqlet_ohs` and `get_seqlet_matrix` /
  `get_seqlet_matrices` (plural forms take an array of row positions and are the fast path).
  Consequently `create_seqlet_adata` no longer takes a `seqlet_matrices` argument.
- `adata.uns["unique_examples"]["oh"]` is stored as `uint8` instead of following the `dtype`
  argument, which now governs only the contribution scores and motif PPMs. One-hot data needs a
  single bit per entry, and this is the largest array in the object after `.X`. `Seqlet.seq_instance`
  is `uint8` as a result; its values are unchanged.
- Pattern files (`save_patterns`) store each region's one-hot once in a file-level `_regions` group
  keyed by `example_idx`, instead of repeating the full region for every seqlet of every pattern.
  `_SEQLET_SPEC` is bumped to `2.0`; `load_patterns` still reads `1.0` files.
- `tfmindi.tl.cluster_seqlets` is replaced by `tfmindi.tl.embed_and_cluster`, which does PCA →
  neighbours → t-SNE → Leiden and nothing else. The DBD annotation it used to bolt on
  (`adata.obs["seqlet_dbd"]`, `["cluster_dbd"]`, `["mean_contrib"]`) is gone; per-seqlet TF-family
  annotation is `tfmindi.tl.predict_tf_family_seqlets`' job.
- The `dbd_col` / `dbd_column` parameters are renamed to `annotation_col` and no longer default to
  `"cluster_dbd"`, in `tfmindi.tl.create_patterns`, `tfmindi.tl.run_topic_modeling`,
  `tfmindi.pl.dbd_heatmap`, `tfmindi.pl.region_contributions` and `tfmindi.pl.dbd_topic_heatmap`.
  The column written by `predict_tf_family_seqlets` carries the clustering resolution in its name
  (`predicted_<resolution>_predicted_family`), so no fixed default can be right; the functions that
  need one now raise and list the candidate columns instead of silently using a stale name.
  `tfmindi.pl.tsne` / `tsne_logos` default `color_by` to `"leiden"`.
- `tfmindi.pl.dbd_logos` and `tfmindi.pl.dbd_cluster_logos` are merged into a single
  `tfmindi.pl.pattern_logos`. `group_by="annotation"` reproduces the former (one representative logo
  per TF family) and `annotation="bHLH"` the latter (every pattern carrying that annotation);
  passing neither draws every pattern, which was not previously possible without open-coding a
  `logomaker` loop — as the analysis tutorial in fact did.
- Removed `tfmindi.pl.set_colors`, `tfmindi.pl.reset_colors`, `tfmindi.backends.get_array_module`,
  `tfmindi.backends.to_cpu`, `tfmindi.backends.to_gpu` and the unreferenced `tfmindi.pp.mappings`
  module — none had a call site in the package, the tests, the notebooks or `paper/`.
- `tfmindi.tl.run_topic_modeling` no longer adds a permanent `region_id` column to the caller's
  `adata.obs` as a side effect.
- `tfmindi.tl.embed_regions(embedding="count")` no longer falls back to a hardcoded
  `"predicted_5.0_predicted_family"` annotation column.

### GPU acceleration

- `tfmindi.pp.calculate_motif_similarity` now runs TomTom's column-distance stage and its
  alignment scoring on the GPU when the GPU backend is active, falling back to `memelite.tomtom`
  on any failure. Measured on an L40S against the full 18k-motif `v10nr_clust` collection, this is
  **~9x** faster than the 8-core CPU path (7.5 s -> 0.8 s for 326 seqlets). Against a 5,000-motif
  sampled collection -- the more common case -- it is **~7x** at 10,000 seqlets and holds a flat
  ~1.1 ms/seqlet in sustained 100k-seqlet runs. Below roughly 10^5 seqlet x motif pairs the GPU
  path is marginally slower (~0.9x); it wins from there.
- The GPU TomTom path is **bit-identical** to the CPU one -- p-values, scores, offsets, overlaps,
  strands and nearest-neighbour indices all compare equal on the full collection -- and is
  deterministic across re-runs. It needs only `cupy`, not the full RAPIDS stack.
- TomTom's p-value dynamic program, the one stage that stays on the CPU, is ~2.6x faster. Its
  scratch space is sized for the longest query in a batch and for the nominal score-bin range,
  but any one query touches a small corner of it -- and within that corner, only the true
  nonzero support of each score histogram, about half the nominal bins on real data. Confining
  every loop to the live window skips only additions of exact zeros, so every value produced is
  unchanged. The per-query nearest-neighbour selection is parallel over queries as well (~6x).
- The GPU and the dynamic program now overlap fully: each batch's dynamic program is handed to
  a worker thread the moment its score histograms exist, writing one of two alternating host
  buffers while the previous batch is scored from the other. The batch size is halved to pay
  for the second buffer, so peak memory is unchanged.
- The TomTom batch size is now capped on available host memory as well as on free GPU memory.
  A large card previously drove the host to ~9.6 GB of resident memory on a 5,000-motif run;
  the same run now peaks at ~4.3 GB.

- `tfmindi.tl.embed_regions` / `calculate_embedding_tsne` and `tfmindi.tl.leiden_clustering` now run
  on the GPU when the GPU backend is active, via cuML t-SNE and rapids-singlecell
  neighbors/Leiden. Previously the region-level pipeline was CPU-only even though the equivalent
  seqlet-level steps in `tfmindi.tl.embed_and_cluster` were already accelerated.
  `tfmindi.pl.region_topic_tsne`, which embeds every region at plot time, is accelerated too.
- `tfmindi.tl.predict_tf_family_seqlets` keeps its whole GPU path on the device: the reference
  projection is done with `cupyx` sparse (reusing the same rank-1 centering identity as the CPU
  path), and the k-NN vote is reduced on the GPU. Only the two per-seqlet result vectors are copied
  back, instead of the full (n_seqlets x n_reference_clusters) probability table that
  `cuml.KNeighborsClassifier.predict_proba` returned.
- Any failure inside a GPU step now warns and re-runs that step on the CPU. Previously an import
  error or an unsupported cuML argument aborted the call, and `predict_tf_family_seqlets` had no
  fallback at all.

Note on reproducibility: the GPU k-NN is exact (brute force), while the CPU path uses `pynndescent`,
which is approximate. The two therefore assign slightly different families for seqlets near a
cluster boundary — measured against exact brute force, `pynndescent` recall is ~1.00 for queries
sitting inside the reference cloud and degrades for queries far outside it. The GPU result is the
more accurate of the two.

### Performance

Work towards genome-wide (1M+ seqlet) runs. All changes below are output-preserving; the sparse
similarity matrices are bit-identical to previous versions and the reference projection agrees to
~1e-7 relative error.

- `tfmindi.pl.region_contributions` draws its saliency logo ~26x faster on a 500 bp region
  (1.74 s -> 0.07 s) and ~265x faster on a 2 kb one (8.0 s -> 0.03 s). `logomaker` builds a fresh
  `TextPath` for every character it draws; the four glyph outlines are now built once and placed
  with an affine transform into a single `PathCollection`. Geometry, colours and axis limits
  reproduce `logomaker.Logo`'s defaults, so the figure is unchanged: glyph vertices agree to ~1e-14
  and the rendered PNG is pixel-identical up to antialiasing. The other logo plots still use
  `logomaker`, which is faster for grids of many short logos.
- `tfmindi.pp.calculate_motif_similarity` no longer accumulates sparse-matrix coordinates in Python
  lists (~10x the memory of the equivalent numpy arrays) and no longer allocates a second full
  seqlet x motif matrix for the log transform. Its four near-duplicate code paths were merged into
  one, which also means `chunk_size` now genuinely bounds peak memory instead of accumulating every
  chunk's coordinates. The non-chunked thresholding path was additionally missing a `float32` cast
  that its three sibling paths had, so it held the full matrix in `float64`.
- `tfmindi.tl.predict_tf_family_seqlets` no longer densifies the similarity matrix. Row-centering is
  applied through the rank-1 identity `(X - mu.1^T) P == X P - outer(mu, P.sum(0))`, so `.X` stays
  sparse through the projection. Neighbour votes are tallied with a single `bincount` instead of one
  full scan per reference cluster, and the cluster-to-family lookup is built once rather than per
  seqlet.
- `tfmindi.pp.create_seqlet_adata` no longer copies the similarity matrix, the motif PPMs or the
  per-seqlet matrices when they are already the requested dtype, and derives its example indices
  with `pd.factorize` instead of a per-seqlet `iterrows()` loop.
- The projected attribution track is computed with `einsum`, avoiding a full `(n, 4, length)`
  temporary in `extract_seqlets`.
- Dropping the duplicated per-seqlet `.obs` columns and storing one-hot data as `uint8` roughly
  halves the `.h5ad` (85 MB -> 46 MB on a 11.8k-seqlet benchmark) and removes ~850 B of resident
  memory per seqlet. Because the columns no longer have to be pickled to hex strings on the way
  out, `save_h5ad` is ~50x faster and `load_h5ad` ~4x faster. Reading a cluster's arrays through
  the batch accessors is also faster than the `.loc`-per-seqlet lookups it replaces, so pattern
  creation does not regress.
- `create_patterns` resolves cluster labels to row positions once per cluster instead of calling
  `index.get_loc` per seqlet.
- `tfmindi.pp.create_seqlet_adata` aligns `motif_annotations` with a single `reindex` and maps
  `motif_to_dbd` in one pass, instead of a scalar `.loc` assignment per motif per annotation column
  into a DataFrame that grew a column at a time. On a 18k-motif collection with the four standard
  annotation columns this takes the `.var` build from **8.6 s to 0.03 s**. It also no longer walks
  the seqlet table with `iterrows()` and computes each seqlet's maximum absolute contribution once
  instead of twice.
- `tfmindi.tl.create_patterns` derives every cluster's row positions from one `groupby` pass rather
  than recomputing `adata.obs[by] == cluster` per cluster (which is O(n_clusters x n_seqlets)), and
  reads each cluster's coordinates and region indices in a single positional lookup instead of a
  scalar `.loc`/`.at` per seqlet per column. `tfmindi.pl.tsne_logos` groups the same way.
- `tfmindi.datasets.MotifCollectionData` caches the parsed metadata table, cluster annotations and
  PCA embeddings instead of re-opening the tar archive and re-parsing on every accessor call —
  `predict_tf_family_seqlets` alone triggered five archive opens, two of them re-parsing the full
  ~18k-row metadata TSV. The four copies of the open/extract/gunzip block were merged into one
  helper.
- `tfmindi.tl.loglikelihood` evaluates `scipy.special.gammaln` over whole matrices instead of
  looping `math.lgamma` per cell (~67x faster at 100k regions; agrees with the previous result to
  ~1e-11 relative, i.e. summation-order noise, far below the gaps that drive model selection).
- `tfmindi.tl.evaluate_topic_models` keeps the best model's results as it sweeps instead of
  re-fitting the winner from scratch afterwards, saving a full LDA run.
- `tfmindi.tl.optimal_hierarchical_clustering` no longer computes the AMI and Fowlkes-Mallows scores
  that nothing reads, and evaluates the 100 candidate cuts on integer cluster ids rather than
  writing a fresh string column into `region_adata.obs` at every height.
- `tfmindi.tl.predict_tf_family_seqlets` normalizes the KNN vote table after reducing it to the
  winning class per seqlet, which drops one full `n_seqlets x n_reference_clusters` allocation
  (~4 GB at 1M seqlets and 500 clusters), and formats the family label once per distinct cluster
  instead of once per seqlet.
- `@numba.njit(cache=True)` on the recursive seqlet kernel, so only the first run after an install
  pays the JIT compile.
- Dropped the unused `pybigwig` and `session-info2` dependencies, and declared `scikit-learn` and
  `tqdm`, which the package imports but only got transitively via scanpy.

### Bugfixes

- `tfmindi.tl.evaluate_topic_models`: the inner loops of the log-likelihood used `=` instead of `+=`, discarding all but the last term of each sum. Model selection was therefore based on a wrong quantity.
- `tfmindi.pp.create_seqlet_adata`: passing `oh_sequences`/`contrib_scores` without `seqlet_matrices` silently stored nothing in `uns["unique_examples"]`. The two are now independent.
- `tfmindi.tl.create_patterns`: `**kwargs` was forwarded to alignment backends that do not accept it, so passing any extra argument raised `TypeError`. `method="mafft"` now accepts `max_gap_frac` and `strategy`.
- `tfmindi.tl.create_patterns(method="kmer")`: the consensus PPM was recomputed inside the per-seqlet loop, making pattern creation quadratic in cluster size.
- `tfmindi.load_h5ad`: missing values in a restored numpy-array column were mapped to the last category instead of `None`.
- `tfmindi.tl.embed_regions(embedding="count")`: the returned region AnnData now carries the annotation categories as `var_names` instead of an anonymous range.
- Corrected a tautological assertion in `embed_regions` input validation, a duplicated entry in `tfmindi.pl.__all__`, and the spec version reported in the seqlet-version-mismatch warning.
- `tfmindi.pp.get_example_contrib` resolved its region through `adata.obs["example_oh_idx"]`, so it
  raised on an AnnData built from contribution scores alone. It now uses `example_contrib_idx`.
- `MotifCollectionData` reported the number of PCs of the wrong matrix in the "inconsistent number
  of PCs" validation error.
- `tfmindi.tl.create_patterns(method="mafft")` never reverse-complemented anything: MAFFT was called
  with `adjustdirection=False`, so the `_R_` detection that drives the reverse-complement branches
  could never fire. `adjustdirection` is now a forwardable keyword argument (still off by default).
- `tfmindi.pl` no longer reseeds the *global* `random` module when it generates a colour palette,
  which silently reset the caller's random stream.
- The former `tfmindi.pl.dbd_logos` raised `LogomakerError` whenever its grid came out one row tall
  (e.g. two annotations with the default `ncols`), because it reshaped the axes array to 2-D and
  then indexed it as 1-D. `pattern_logos` handles every grid shape.
- The documentation builds clean again under `-W`. The region-embedding tutorial was missing from
  the toctree, and `tl/embedder.py` used a non-numpydoc `Side effects` section and inline parameter
  types that Sphinx tried to resolve as classes. `tl.embed_regions`, `tl.calculate_embedding_tsne`,
  `tl.leiden_clustering`, `tl.optimal_hierarchical_clustering`, `tl.get_region_profiles`,
  `pl.region_tsne` and `pl.plot_top_motifs` are now listed in the API reference.
- Corrected the documented parameter names of `tfmindi.pl.region_contributions`
  (`annotation_to_show` → `annotations_to_show`), the `show_logos` parameter documented by
  `tfmindi.pl.tsne_logos` but never implemented, the `adata.obsm['X_topics']` key documented by
  `tfmindi.tl.run_topic_modeling` but never written, and six wrong return annotations in
  `tfmindi.tl.embedder`.
- An annotation column of `motif_annotations` that is numeric and does not cover every motif now
  lands in `.var` as a float column with `NaN` for the unannotated motifs, rather than an object
  column mixing ints and `None`. The four annotation columns produced by
  `tfmindi.datasets.load_motif_annotations` are all strings and are unaffected.

## 1.2.0

This version accompanies the human neural development paper preprint.

### Features

- updated to scverse template v0.6.0
- added parameter to select pca svd solver for running PCA instead of hardcoded "covariance_eigh"
- added functionality to save and load patterns to and from disk (`tfmindi.save_patterns` and `tfmindi.load_patterns`)
- now allows for patterns to be generated by any annotation in .obs, not only by `leiden`
- can now concatenate multiple TF-MINDI anndatas together using `tfmindi.concat`
- changed behaviour of seqlet most frequent-occurence for annotating clusters to binomial tests
- added MAFFT-based backend as option for pattern creation (much faster than TomTom or k-mer)

### Bugfixes

- fix extra 0-position being included in `Pattern.ic_trim`
- fix failure case in in `Pattern.ic_trim` when all nucleotides are above IC threshold
- plotting legends are filtered on colors in `adata.obs[color_by]`

## 1.1.0

Bugfixes, an updated seqlet calling algorithm, and new k-mer pattern tooling.
Be aware that we're not entirely satisfied with the current seqlet calling algorithm, we're working on this for the next release.

- Updated the recursive seqlet calling algorithm to match the latest version of tangermeme. This generally results in fewer but cleaner seqlets. WARNING: this algorithm now only seems to call positive seqlets (which we don't agree with). We're still working on an updated seqlet calling algorithm, but that will be for a next release. For now you can get around this by calling seqlets on absolute contribution scores.
- Added new functionality to align seqlet instances based on the hamming distance to most frequently occuring kmer. Default remains tomtom for the time being though.
- Consistent colormap keys added to anndata.uns that matches scanpy convention.
- BREAKING CHANGE: All topic modeling results are now stored in the anndata, similar to the rest of the api. Topic modeling plotting functions will now also expect the anndata as input. Tutorial has been updated to match this breaking change.
- The Pattern class now has additional functions to interact with calculated kmers (eget_unique_kmers, get_kmers, get_kmer_distances). Additionaly, the Seqlet class keeps track of the seqlet index (can be used to find back the seqlet in adata.obs).
- Added an option to filter on min_seqlets in logo_plotting (useful in case of small, noisy clusters).


## 1.0.0

Initial release
