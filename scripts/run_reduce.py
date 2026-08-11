#!/usr/bin/env python3
"""Stage 3 runner — exact PCA 768->50 (metric) + UMAP 50->2 (display).

Streamed centered covariance + ``eigh``: deterministic and batch-order invariant, which the
refit-per-cutoff validation harness depends on.
"""
from brontide import cli, reduce

if __name__ == "__main__":
    sample = cli.parse_sample(__doc__)
    pca = reduce.run_pca(sample=sample)
    umap = reduce.run_umap(sample=sample)
    print(f"reduce -> pca={pca} umap={umap}")
