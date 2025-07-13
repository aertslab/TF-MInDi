## Package Development Notes

- We are reworking some quick code for paper figures into a package
- Original code can be found in the /paper folder
- Package structure proposal is documented in TFMINDI_API.md
- Will not use crested as an external dependency; will reimplement necessary code if needed
- Package will follow the scverse template structure
- Folders have been initialized with some temporary functions that can be deleted later

## Development Workflow

- We will implement new functions one by one. Every time we do, we FIRST write some small unit tests for the expected behaviour of that function. Then we implement the function itself
- If we need custom data types, we can create them in a "types.py" module
- Same with default configurations, that goes into another module

## Environment Management

- Every time we run functions or testing, anything for which we need an environment, we use "hatch" for that. You can observe how in the pyproject.toml

## Current Implementation Status

### Completed (Preprocessing Module)
- ✅ `fetch_motif_collection()`, `load_motif_collection()`, `load_motif_annotations()` - Core data loading functions
- ✅ `extract_seqlets()` - Seqlet extraction using tangermeme
- ✅ `calculate_motif_similarity()` - TomTom-based similarity calculation
- ✅ **`create_seqlet_adata()`** - Enhanced function that creates comprehensive AnnData object

### Enhanced `create_seqlet_adata()` Design
- **Input**: similarity_matrix, seqlet_metadata, seqlet_matrices, oh_sequences, contrib_scores, motif_names
- **Storage Strategy**:
  - `.X`: similarity matrix (n_seqlets × n_motifs)
  - `.obs`: seqlet metadata
  - `.var`: motif names
  - `.uns["seqlet_matrices"]`: list of variable-length seqlet contribution matrices
  - `.uns["oh_sequences"]`: full one-hot sequences (n_examples × 4 × length)
  - `.uns["contrib_scores"]`: full contribution scores (n_examples × 4 × length)
- **Rationale**: Variable-length seqlet data stored in `.uns` follows AnnData best practices

### Next Steps (Tools Module)
- **Need to implement**: `cluster_seqlets()` - complete clustering workflow (PCA, neighbors, t-SNE, Leiden, DBD annotation)
- **Need to implement**: `create_patterns()` - pattern creation from clusters using TomTom alignment
- **API Updated**: TFMINDI_API.md reflects clean scanpy-style API where all tools only need AnnData object

### Testing
- All preprocessing functions have comprehensive unit tests
- Tests cover real data, edge cases, and error handling
- Use `hatch run test` for testing
