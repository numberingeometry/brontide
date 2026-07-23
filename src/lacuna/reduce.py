"""Stage 3 — Reduce (host).

Two independent reductions:

- **Metric space (clustering):** IncrementalPCA 768→50 on the full corpus. This
  is what regions/series operate on — NOT UMAP, which would distort the metric.
- **Display space (map only):** UMAP 50→2, fit on a ``UMAP_FIT_SAMPLE`` stratified
  sample and ``transform`` the rest (caps RAM on 2.6M). v1 ships one static map;
  Aligned/parametric UMAP for temporal stability is deferred.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import config

# Rows per streamed batch. Sequential over the memmap, so this only bounds the
# transient float32/float64 working set, not resident memory: 50k x 768 x 4 B =
# ~150 MB, which IncrementalPCA upcasts to float64 internally (~300 MB) plus its
# SVD workspace. Comfortable on this 16 GB box.
_PCA_CHUNK = 50_000


def _embed_rows(embed_memmap: Path) -> int:
    """Full corpus N the memmap was allocated for, from the embed sidecar."""
    sidecar = embed_memmap.with_suffix(embed_memmap.suffix + ".progress.json")
    return int(json.loads(sidecar.read_text())["n_done"])


def run_pca(sample: int | None = None,
            embed_memmap: Path = config.EMBED_MEMMAP,
            out: Path = config.PCA_MEMMAP) -> Path:
    """IncrementalPCA 768→``PCA_DIM``, streamed in batches. Returns float32 memmap."""
    import numpy as np
    from sklearn.decomposition import IncrementalPCA

    # R3 — refuse a --sample run against the default output path: it would
    # truncate a completed full PCA (mode="w+") to (sample, 50).
    if sample is not None and out == config.PCA_MEMMAP:
        raise RuntimeError(
            "run_pca(sample=...) with the default out would overwrite the full "
            "pca50 memmap. Pass an explicit out=... for sample runs."
        )

    emb_rows = _embed_rows(embed_memmap)          # embed memmap is always full-corpus sized
    n = emb_rows if sample is None else min(sample, emb_rows)

    # R1 — never reduce a half-written embedding memmap. Unwritten rows are
    # exactly 0.0 (finite), so nothing would error; this is the only guard.
    if sample is None and emb_rows != n:
        raise RuntimeError(f"embed incomplete: sidecar n_done={emb_rows}, expected {n}")

    emb = np.memmap(embed_memmap, dtype=np.float16, mode="r", shape=(emb_rows, config.EMBED_DIM))

    ipca = IncrementalPCA(n_components=config.PCA_DIM)
    t0 = time.perf_counter()
    for lo in range(0, n, _PCA_CHUNK):
        hi = min(lo + _PCA_CHUNK, n)
        if hi - lo < config.PCA_DIM:               # IncrementalPCA needs batch >= n_components
            break                                  # tiny trailing remainder: skip fit, still transformed below
        ipca.partial_fit(np.asarray(emb[lo:hi], dtype=np.float32))
    fit_s = time.perf_counter() - t0

    # Transform to a temp file, then atomically replace, so an interrupted
    # transform never leaves a truncated pca50 that a consumer would mis-join.
    tmp = out.with_suffix(out.suffix + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    outmm = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(n, config.PCA_DIM))
    for lo in range(0, n, _PCA_CHUNK):
        hi = min(lo + _PCA_CHUNK, n)
        outmm[lo:hi] = ipca.transform(np.asarray(emb[lo:hi], dtype=np.float32)).astype(np.float32)
    outmm.flush()
    del outmm
    os.replace(tmp, out)

    evr = ipca.explained_variance_ratio_
    total_s = time.perf_counter() - t0
    # R4 — headerless memmap: record N/dim so consumers can assert on join width.
    (out.with_suffix(out.suffix + ".meta.json")).write_text(json.dumps({
        "n": int(n), "dim": config.PCA_DIM, "dtype": "float32",
        "source_rows": int(emb_rows), "method": "IncrementalPCA",
        "explained_variance_ratio_top10": [float(x) for x in evr[:10]],
        "total_explained": float(evr.sum()),
        "expected_bytes": int(n) * config.PCA_DIM * 4,
    }, indent=2))

    print(f"\nreduce.run_pca[{config.resolve_sample(sample)}] -> {out}\n"
          f"  shape/dtype : ({n:,}, {config.PCA_DIM}) float32\n"
          f"  size        : {out.stat().st_size / 1e6:.1f} MB\n"
          f"  EVR[0]      : {evr[0]:.4f}   top-{config.PCA_DIM} total: {evr.sum():.4f}\n"
          f"  timing      : fit {fit_s:.0f}s, total {total_s:.0f}s")
    return out


def run_umap(sample: int | None = None,
             pca_memmap: Path = config.PCA_MEMMAP,
             out: Path = config.UMAP_MEMMAP) -> Path:
    """UMAP ``PCA_DIM``→2, sample-fit + transform-rest. Display only. Returns memmap."""
    # TODO: umap.UMAP(n_components=2).fit(sample of PCA); .transform(all) in chunks.
    raise NotImplementedError("reduce.run_umap")
