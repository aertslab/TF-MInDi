# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

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
