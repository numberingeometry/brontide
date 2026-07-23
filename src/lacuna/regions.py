"""Stage 4 — Regions (host, ~30min).

MiniBatchKMeans **K=1024** microclusters on the 50-D PCA space
(~2.6k papers/region ⇒ ~11 papers/region/month — enough signal). K-means over
HDBSCAN for: full coverage (no ~40% noise class), fixed K for series power, and
centroids as the natural **leaves of the v1.5 tree**.

Then **Ward linkage on the 1024 centroids**, computed now: cut at
``N_SUPERREGIONS`` (64) for map display + labels; the full dendrogram is stored
as the v1.5 ultrametric / MinT reconciliation hook (nothing recomputed later).

**Lookahead control for validation:** when fitting for a cutoff T0, fit PCA +
k-means on papers ``<= T0`` only; assign later papers nearest-centroid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def run(sample: int | None = None,
        pca_memmap: Path = config.PCA_MEMMAP,
        fit_mask: np.ndarray | None = None,
        out_dir: Path = config.ARTIFACTS_DIR) -> Path:
    """Fit K=1024 microclusters + Ward tree. ``fit_mask`` restricts the fit set
    (papers <= T0) for validation; unmasked papers are assigned nearest-centroid.

    Writes: ``labels`` (region id per paper), ``centroids`` (1024×50),
    ``linkage`` (scipy Ward Z), ``super_region`` (region→super map at the 64 cut).
    Returns the artifacts subdir.
    """
    # TODO:
    #   1. MiniBatchKMeans(n_clusters=N_REGIONS).fit(PCA[fit_mask]); predict all.
    #   2. scipy.cluster.hierarchy.ward(centroids); fcluster at N_SUPERREGIONS.
    #   3. persist labels/centroids/linkage/super_region map.
    raise NotImplementedError("regions.run")
