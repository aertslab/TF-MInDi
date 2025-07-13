# TF-MInDi API Structure

## Overview
TF-MInDi follows the scverse ecosystem structure with functions organized into:
- **`tfmindi`**: Core data fetching and loading functions
- **`tfmindi.pp`**: Preprocessing functions for seqlet extraction and similarity calculation
- **`tfmindi.tl`**: Analysis tools for clustering, pattern creation, and topic modeling
- **`tfmindi.pl`**: Plotting functions for visualization

---

## `tfmindi` - Core Functions

### `fetch_motif_collection(name: str = "v10nr_clust", destination: str = ".") -> str`
**Input**: Collection name, destination directory
**Output**: Path to downloaded motif collection folder
**Purpose**: Download motif collection from aertslab resources

### `load_motif_collection(motif_dir: str) -> Dict[str, np.ndarray]`
**Input**: Directory path containing .cb motif files
**Output**: Dictionary mapping motif names to PWM matrices (4 x length)
**Purpose**: Load motif collection from directory of .cb files

### `load_motif_annotations(species: str = "homo_sapiens", version: str = "v10nr_clust") -> pd.DataFrame`
**Input**: Species name, motif collection version
**Output**: DataFrame with motif-to-TF mappings and DBD annotations
**Purpose**: Load motif annotations and DNA-binding domain information

---

## `tfmindi.pp` - Preprocessing Functions

### `extract_seqlets(contrib: np.ndarray, oh: np.ndarray, threshold: float = 0.05) -> Tuple[pd.DataFrame, List[np.ndarray]]`
**Input**: Contribution scores (n_examples, length, 4), one-hot sequences (n_examples, length, 4), importance threshold
**Output**: Seqlet coordinates DataFrame, list of processed seqlet matrices
**Purpose**: Extract, scale, and process seqlets from saliency maps

### `calculate_motif_similarity(seqlets: List[np.ndarray], known_motifs: List[np.ndarray]) -> np.ndarray`
**Input**: List of seqlet matrices, list of known motif PWMs
**Output**: Log-transformed similarity matrix (n_seqlets x n_motifs)
**Purpose**: Calculate TomTom similarity and convert to log-space for clustering

### `create_seqlet_adata(similarity_matrix: np.ndarray, seqlet_metadata: pd.DataFrame, seqlet_matrices: List[np.ndarray] = None, oh_sequences: np.ndarray = None, contrib_scores: np.ndarray = None, motif_names: List[str] = None, motif_collection: Dict[str, np.ndarray] = None, motif_annotations: pd.DataFrame = None, motif_to_dbd: Dict[str, str] = None) -> sc.AnnData`
**Input**: Similarity matrix, seqlet metadata, optional seqlet matrices, sequences, motif collection, and annotations
**Output**: AnnData object with all data needed for downstream analysis
**Purpose**: Create comprehensive AnnData object storing all seqlet and motif data for analysis pipeline

**Data Storage**:
- `.X`: Log-transformed motif similarity matrix (n_seqlets × n_motifs)
- `.obs`: Seqlet metadata and variable-length arrays stored per seqlet
  - Standard metadata: coordinates, attribution, p-values
  - `.obs["seqlet_matrix"]`: Individual seqlet contribution matrices (4 × variable_length)
  - `.obs["seqlet_oh"]`: Individual seqlet one-hot sequences (4 × variable_length)
  - `.obs["example_oh"]`: Full example one-hot sequences per seqlet (4 × example_length)
  - `.obs["example_contrib"]`: Full example contribution scores per seqlet (4 × example_length)
- `.var`: Motif names and comprehensive annotations
  - `.var["motif_pwm"]`: Individual motif PWM matrices (4 × motif_length)
  - `.var["dbd"]`: DNA-binding domain annotations
  - `.var["Direct_annot"]`: Direct TF annotations
  - Additional columns from motif_annotations DataFrame

---

## `tfmindi.tl` - Analysis Tools

### `cluster_seqlets(adata: sc.AnnData, resolution: float = 3.0) -> None`
**Input**: AnnData object (with all seqlet and motif data), clustering resolution
**Output**: None (modifies adata.obs and adata.obsm with cluster assignments and annotations)
**Purpose**: Perform complete clustering workflow including dimensionality reduction, Leiden clustering, and functional annotation

**Workflow**:
1. PCA on similarity matrix → stored in `adata.obsm["X_pca"]`
2. Compute neighborhood graph
3. Generate t-SNE embedding → stored in `adata.obsm["X_tsne"]`
4. Leiden clustering at specified resolution → stored in `adata.obs["leiden"]`
5. Calculate mean contribution scores from `adata.obs["seqlet_matrix"]` → stored in `adata.obs["mean_contrib"]`
6. Assign DBD annotations based on top motif similarity and `adata.var["dbd"]` → stored in `adata.obs["seqlet_dbd"]`
7. Map leiden clusters to consensus DBD annotations → stored in `adata.obs["cluster_dbd"]`

### `create_patterns(adata: sc.AnnData) -> Dict[str, Pattern]`
**Input**: AnnData object with cluster assignments and stored seqlet data
**Output**: Dictionary mapping cluster IDs to Pattern objects
**Purpose**: Generate aligned PWM patterns from seqlet clusters using stored data

**Workflow**:
1. For each cluster, extract seqlets belonging to that cluster
2. Use TomTom to align seqlets within the cluster using `adata.obs["seqlet_matrix"]`
3. Find consensus root seqlet (lowest mean similarity)
4. Apply strand and offset corrections using `adata.obs["example_oh"]` and `adata.obs["example_contrib"]`
5. Generate Pattern object with PWM, contribution scores, and seqlet instances

### `run_topic_modeling(adata: sc.AnnData, n_topics: int = 40, **kwargs) -> Tuple[lda.LDA, pd.DataFrame]`
**Input**: AnnData object with cluster assignments, number of topics, LDA parameters
**Output**: Fitted LDA model, region-topic matrix
**Purpose**: Discover co-occurring motif patterns using topic modeling on region-level data

**Workflow**:
1. Group seqlets by genomic regions using stored coordinates in `adata.obs`
2. Create region-cluster count matrix from leiden assignments
3. Fit LDA model to discover topics (co-occurring cluster patterns)
4. Return fitted model and region-topic assignments

### `validate_chipseq(adata: sc.AnnData, chipseq_files: Dict[str, str]) -> None`
**Input**: AnnData object with genomic coordinates, dictionary of TF:BigWig file paths
**Output**: None (adds ChIP-seq validation scores to adata.obs)
**Purpose**: Validate discovered motifs against experimental ChIP-seq data using stored coordinates

### `compare_modisco(adata: sc.AnnData, modisco_results: str) -> pd.DataFrame`
**Input**: AnnData object with seqlet results, path to MoDISco output
**Output**: DataFrame with comparison metrics (precision, recall, F1)
**Purpose**: Benchmark TF-MInDi results against TF-MoDISco using stored seqlet coordinates

---

## `tfmindi.pl` - Plotting Functions

### `tsne_logos(adata: sc.AnnData, patterns: Dict[str, Pattern], color_by: str = "dbd", figsize: Tuple[int, int] = (10, 10)) -> plt.Figure`
**Input**: AnnData with t-SNE coordinates, pattern dictionary, coloring variable, figure size
**Output**: Matplotlib figure with t-SNE plot and sequence logos
**Purpose**: Visualize seqlet clusters with sequence logos at centroids

### `dbd_heatmap(topic_matrix: pd.DataFrame, annotation_matrix: pd.DataFrame, **kwargs) -> plt.Figure`
**Input**: Topic-cluster matrix, DBD annotation matrix, plotting parameters
**Output**: Matplotlib figure with clustered heatmap
**Purpose**: Show enrichment of DNA-binding domains across topics

### `region_topics(region_adata: sc.AnnData, cell_predictions: pd.DataFrame, method: str = "tsne") -> plt.Figure`
**Input**: Region-level AnnData, cell type predictions, dimensionality reduction method
**Output**: Matplotlib figure with region-topic associations
**Purpose**: Visualize topic assignments in relation to cell type predictions

### `validation_plots(adata: sc.AnnData, validation_results: pd.DataFrame) -> plt.Figure`
**Input**: AnnData with cluster assignments, validation results DataFrame
**Output**: Matplotlib figure with validation metrics
**Purpose**: Visualize ChIP-seq validation and benchmarking results

---

## Typical Workflow

```python
import tfmindi as tm

# Load data and motifs
motifs = tm.load_motif_collection("motif_collection/")
annotations = tm.load_motif_annotations("homo_sapiens")

# Extract and process seqlets
seqlets_df, seqlet_matrices = tm.pp.extract_seqlets(contrib, oh, threshold=0.05)
similarity_matrix = tm.pp.calculate_motif_similarity(seqlet_matrices, list(motifs.values()))

# Create comprehensive AnnData object with all data
adata = tm.pp.create_seqlet_adata(
    similarity_matrix,
    seqlets_df,
    seqlet_matrices=seqlet_matrices,
    oh_sequences=oh,
    contrib_scores=contrib,
    motif_collection=motifs,
    motif_annotations=annotations,
    motif_to_dbd=tm.load_motif_to_dbd(annotations)
)

# Cluster and create patterns (clean API - only needs adata)
tm.tl.cluster_seqlets(adata, resolution=3.0)
patterns = tm.tl.create_patterns(adata)

# Topic modeling
lda_model, region_topics = tm.tl.run_topic_modeling(adata, n_topics=40)

# Validation
tm.tl.validate_chipseq(adata, chipseq_files)
comparison = tm.tl.compare_modisco(adata, "modisco_results/")

# Visualization
fig1 = tm.pl.tsne_logos(adata, patterns, color_by="dbd")
fig2 = tm.pl.dbd_heatmap(region_topics, annotations)
fig3 = tm.pl.region_topics(adata, cell_predictions)
```

---

## Function Dependencies

### Required External Packages
- `numpy`, `pandas`, `scipy` - Core data structures
- `scanpy` - Single-cell analysis framework
- `tangermeme` - Seqlet extraction
- `memelite` - TomTom motif comparison
- `lda` - Topic modeling
- `logomaker` - Sequence logo generation
- `matplotlib`, `seaborn` - Visualization
- `pyBigWig` - ChIP-seq data access

### Internal Dependencies
- `tfmindi.pp` functions must be called before `tfmindi.tl`
- `tfmindi.tl.cluster_seqlets()` must be called before `tfmindi.tl.create_patterns()`
- `tfmindi.pl` functions require results from corresponding `tfmindi.tl` functions

---

## Data Structures

### Primary Objects
- **`seqlets_df`**: DataFrame with columns [example_idx, start, end, chrom, g_start, g_end]
- **`seqlet_matrices`**: List of contribution matrices (4 x length) for each seqlet
- **`adata`**: AnnData object with similarity matrix as .X and metadata in .obs
- **`patterns`**: Dictionary mapping cluster IDs to Pattern objects with PWM and statistics

### Key AnnData Annotations
**Observations (.obs)**:
- **`adata.obs["leiden"]`**: Cluster assignments
- **`adata.obs["dbd"]`**: DNA-binding domain annotations per seqlet
- **`adata.obs["dbd_per_leiden"]`**: Consensus DBD annotations per cluster
- **`adata.obs["mean_contrib"]`**: Mean contribution scores per seqlet

**Observation matrices (.obsm)**:
- **`adata.obsm["X_pca"]`**: PCA coordinates
- **`adata.obsm["X_tsne"]`**: t-SNE coordinates
- **`adata.obsm["X_umap"]`**: UMAP coordinates

**Observation data (.obs)**:
- **`adata.obs["seqlet_matrix"]`**: Individual seqlet contribution matrices per seqlet (4 × variable_length)
- **`adata.obs["seqlet_oh"]`**: Individual seqlet one-hot sequences per seqlet (4 × variable_length)
- **`adata.obs["example_oh"]`**: Full example one-hot sequences per seqlet (4 × example_length)
- **`adata.obs["example_contrib"]`**: Full example contribution scores per seqlet (4 × example_length)

**Variables (.var)**:
- **`adata.var["motif_pwm"]`**: Individual motif PWM matrices (4 × motif_length)
- **`adata.var["dbd"]`**: DNA-binding domain annotations
- **`adata.var["Direct_annot"]`**: Direct TF annotations
- Additional motif annotation columns

---

## Configuration

### Default Parameters
```python
DEFAULT_PARAMS = {
    "seqlet_threshold": 0.05,
    "clustering_resolution": 3.0,
    "n_topics": 40,
    "lda_alpha": 50,
    "lda_eta": 0.1,
    "lda_iterations": 150,
    "similarity_threshold": 0.05,
    "chipseq_window": 20,
    "logo_ic_threshold": 0.2
}
```

### Customization
All functions accept parameter overrides through keyword arguments, allowing users to customize behavior for their specific datasets and requirements.
