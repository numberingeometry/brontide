"""Stage 3 — Reduce (host).

Two independent reductions:

- **Metric space (clustering):** IncrementalPCA 768→50 on the full corpus. This
  is what regions/series operate on — NOT UMAP, which would distort the metric.
- **Display space (map only):** UMAP 50→2, fit on a ``UMAP_FIT_SAMPLE`` stratified
  sample and ``transform`` the rest (caps RAM on 2.6M). v1 ships one static map;
  Aligned/parametric UMAP for temporal stability is deferred.
"""
from __future__ import annotations

from pathlib import Path

from . import config


def run_pca(sample: int | None = None,
            embed_memmap: Path = config.EMBED_MEMMAP,
            out: Path = config.PCA_MEMMAP) -> Path:
    """IncrementalPCA 768→``PCA_DIM``, streamed in batches. Returns float32 memmap."""
    # TODO: sklearn.decomposition.IncrementalPCA(n_components=PCA_DIM);
    #       partial_fit over memmap batches, then transform -> float32 memmap.
    raise NotImplementedError("reduce.run_pca")


def run_umap(sample: int | None = None,
             pca_memmap: Path = config.PCA_MEMMAP,
             out: Path = config.UMAP_MEMMAP) -> Path:
    """UMAP ``PCA_DIM``→2, sample-fit + transform-rest. Display only. Returns memmap."""
    # TODO: umap.UMAP(n_components=2).fit(sample of PCA); .transform(all) in chunks.
    raise NotImplementedError("reduce.run_umap")
