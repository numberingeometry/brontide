"""Stage 4 — Regions.

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

import json
from pathlib import Path

import numpy as np

from . import config

# MiniBatchKMeans mini-batch. >> K=1024 for stable centroid updates. The whole
# PCA array is 2.7M x 50 x 4 B = 541 MB and fits in RAM, so the clustering reads
# an in-memory copy rather than re-paging the memmap each epoch.
_KM_BATCH = 10_000


def _pca_n(pca_memmap: Path) -> int:
    """Rows in the PCA memmap. Prefer the meta sidecar; fall back to file size."""
    meta = pca_memmap.with_suffix(pca_memmap.suffix + ".meta.json")
    if meta.exists():
        return int(json.loads(meta.read_text())["n"])
    size = pca_memmap.stat().st_size
    row_bytes = config.PCA_DIM * 4
    assert size % row_bytes == 0, f"pca memmap size {size} not a multiple of {row_bytes}"
    return size // row_bytes


def run(sample: int | None = None,
        pca_memmap: Path = config.PCA_MEMMAP,
        fit_mask: np.ndarray | None = None,
        k: int = config.N_REGIONS,
        out_dir: Path = config.ARTIFACTS_DIR) -> Path:
    """Fit K microclusters + Ward tree. ``fit_mask`` restricts the fit set
    (papers <= T0) for validation; unmasked papers are assigned nearest-centroid.

    ``k`` defaults to the FROZEN ``config.N_REGIONS`` (1024) — the pre-registered
    value used by the production and scored-validation runs. The K∈{256,4096}
    granularity-sensitivity check (v1 plan top-risk #5) passes an explicit ``k`` to a
    LABELLED robustness ``out_dir`` (e.g. ``artifacts/robustness/k256/``); it never
    overwrites the scored artifacts, so the frozen constant is untouched.

    Writes: ``labels`` (region id per paper), ``centroids`` (k×50),
    ``linkage`` (scipy Ward Z), ``super_region`` (region→super map at the 64 cut).
    Returns the artifacts subdir.
    """
    from scipy.cluster.hierarchy import fcluster, ward
    from sklearn.cluster import MiniBatchKMeans

    n = _pca_n(pca_memmap)
    pca = np.memmap(pca_memmap, dtype=np.float32, mode="r", shape=(n, config.PCA_DIM))
    x = np.asarray(pca)                         # 541 MB resident copy

    km = MiniBatchKMeans(n_clusters=k, batch_size=_KM_BATCH,
                         n_init=3, max_iter=100, random_state=0)
    if fit_mask is not None:
        assert fit_mask.shape[0] == n, f"fit_mask len {fit_mask.shape[0]} != N {n}"
        km.fit(x[fit_mask])                     # fit on <= T0 only ...
    else:
        km.fit(x)
    labels = km.predict(x).astype(np.int16)     # ... but assign ALL papers nearest-centroid
    centroids = km.cluster_centers_.astype(np.float32)   # (k, 50)

    # Ward linkage on the k centroids; cut to N_SUPERREGIONS. fcluster returns
    # 1..super, so subtract 1 for 0-based super ids (frees 255 as the map noise sentinel).
    z = ward(centroids)
    super_of_region = (fcluster(z, t=config.N_SUPERREGIONS, criterion="maxclust") - 1).astype(np.int16)

    region_sizes = np.bincount(labels, minlength=k)
    super_sizes = np.bincount(super_of_region[labels], minlength=config.N_SUPERREGIONS)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "region_labels.i16.npy", labels)          # per paper (big; untracked when out_dir=data/)
    np.save(out_dir / "centroids.f32.npy", centroids)
    np.save(out_dir / "linkage.f64.npy", z)
    np.save(out_dir / "super_region.i16.npy", super_of_region)  # region -> super
    # centroids.f32.npy / super_region.i16.npy / linkage.f64.npy are .npy and therefore
    # self-describing about SHAPE -- but not about which space they live in or which fit
    # numbered them. Both matter downstream: a centroid is only meaningful in the PCA
    # basis of the same build, and the ids are this fit's labels, not stable names.
    (out_dir / "regions_summary.json").write_text(json.dumps({
        "n_papers": int(n), "k_regions": k, "n_super": config.N_SUPERREGIONS,
        "pca_dim": config.PCA_DIM,
        "corpus_end": config.CORPUS_END,
        "kmeans_random_state": 0,
        "ids_are_build_scoped": ("region and super-region ids are the labels of THIS fit; "
                                 "seeded, so reproducible on identical input, but renumbered "
                                 "by any rebuild on a changed corpus"),
        "fit_on": "all" if fit_mask is None else int(fit_mask.sum()),
        "region_size_min": int(region_sizes.min()), "region_size_max": int(region_sizes.max()),
        "region_size_mean": float(region_sizes.mean()), "region_size_median": float(np.median(region_sizes)),
        "empty_regions": int((region_sizes == 0).sum()),
        "super_size_min": int(super_sizes.min()), "super_size_max": int(super_sizes.max()),
        "super_size_median": float(np.median(super_sizes)),
        "km_inertia": float(km.inertia_),
    }, indent=2))

    print(f"\nregions.run[{config.resolve_sample(sample)}] -> {out_dir}\n"
          f"  papers   : {n:,}\n"
          f"  regions  : K={k}, empty={int((region_sizes == 0).sum())}, "
          f"size min/med/max = {region_sizes.min()}/{int(np.median(region_sizes))}/{region_sizes.max()}\n"
          f"  super    : {config.N_SUPERREGIONS}, "
          f"size min/med/max = {super_sizes.min()}/{int(np.median(super_sizes))}/{super_sizes.max()}\n"
          f"  inertia  : {km.inertia_:.3e}")
    return out_dir
