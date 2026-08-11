"""Stage 3 — Reduce (host).

Two independent reductions:

- **Metric space (clustering):** exact PCA 768→50 (streamed centered-covariance +
  ``eigh``) on the full corpus. This is what regions/series operate on — NOT UMAP,
  which would distort the metric.
- **Display space (map only):** UMAP 50→2, fit on a ``UMAP_FIT_SAMPLE`` stratified
  sample and ``transform`` the rest (caps RAM on 2.6M). v1 ships one static map;
  Aligned/parametric UMAP for temporal stability is deferred.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from . import config

# Rows per streamed batch. Sequential over the memmap, so this only bounds the
# transient float64 working set, not resident memory: a 50k x 768 float64 chunk is
# ~300 MB; the accumulated 768 x 768 Gram is ~4.7 MB. Comfortable on this 16 GB box.
_PCA_CHUNK = 50_000


def _embed_rows(embed_memmap: Path) -> int:
    """Full corpus N the memmap was allocated for, from the embed sidecar."""
    sidecar = embed_memmap.with_suffix(embed_memmap.suffix + ".progress.json")
    return int(json.loads(sidecar.read_text())["n_done"])


def run_pca(sample: int | None = None,
            embed_memmap: Path = config.EMBED_MEMMAP,
            fit_mask: np.ndarray | None = None,
            out: Path = config.PCA_MEMMAP) -> Path:
    """Exact streaming PCA 768→``PCA_DIM``. Returns a float32 (N, PCA_DIM) memmap.

    Solver (CPU-only environment-directed 2026-07-23, replaces IncrementalPCA — see the
    validation.md clarification): accumulate the row count, first moment Σx and
    second moment Σxxᵀ in float64 over the *fit set* in streamed batches, form the
    centered covariance ``C = Σxxᵀ/n_fit − μμᵀ`` (μ = Σx/n_fit), and take the top
    ``PCA_DIM`` eigenvectors via ``eigh``. This is EXACT (not IncrementalPCA's
    order-dependent streaming approximation), DETERMINISTIC (batch-order invariant —
    load-bearing for the refit-per-cutoff validation), and faster; d=768 ≪ N makes
    the 768×768 covariance trivial. A fixed sign convention (each component's
    max-abs loading forced positive) keeps raw coordinates reproducible across runs.
    NB: ``eigh``/GEMM here are the LAPACK/BLAS calls the host's MKL_THREADING_LAYER=TBB
    fix protects — launch via ``conda run -n brontide``, never a hard interpreter path.

    ``fit_mask`` (bool, length = processed N) restricts the fit set to papers
    ``<= T0`` for the masked validation run — the mean AND covariance are both
    accumulated on the masked rows only, then ALL rows are centered by the fit-set
    mean and projected. ``None`` = fit on all (production run A). Mirrors
    ``regions.run``'s fit_mask; the pre-registered lookahead control requires it.
    """
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
    if fit_mask is not None and fit_mask.shape[0] != n:
        raise RuntimeError(f"fit_mask len {fit_mask.shape[0]} != processed N {n}")

    emb = np.memmap(embed_memmap, dtype=np.float16, mode="r", shape=(emb_rows, config.EMBED_DIM))

    # --- fit: exact centered covariance over the (optionally masked) fit set ----
    d = config.EMBED_DIM
    n_fit = 0
    s = np.zeros(d, dtype=np.float64)             # Σx over the fit set
    gram = np.zeros((d, d), dtype=np.float64)     # Σxxᵀ over the fit set
    t0 = time.perf_counter()
    for lo in range(0, n, _PCA_CHUNK):
        hi = min(lo + _PCA_CHUNK, n)
        xb = np.asarray(emb[lo:hi], dtype=np.float64)
        if fit_mask is not None:
            xb = xb[fit_mask[lo:hi]]
        if xb.shape[0] == 0:
            continue
        n_fit += xb.shape[0]
        s += xb.sum(axis=0)
        gram += xb.T @ xb
    if n_fit <= config.PCA_DIM:
        raise RuntimeError(f"fit set too small: {n_fit} rows <= PCA_DIM {config.PCA_DIM}")
    mu = s / n_fit
    cov = gram / n_fit - np.outer(mu, mu)
    cov = (cov + cov.T) * 0.5                      # symmetrize against fp drift
    evals, evecs = np.linalg.eigh(cov)            # ascending eigenvalues
    order = np.argsort(evals)[::-1][:config.PCA_DIM]
    comps = evecs[:, order].copy()                # (768, PCA_DIM), principal axes
    top_evals = evals[order]
    for j in range(comps.shape[1]):               # deterministic sign convention
        if comps[np.argmax(np.abs(comps[:, j])), j] < 0.0:
            comps[:, j] = -comps[:, j]
    total_var = float(evals.sum())                # trace(C) = total variance
    evr = top_evals / total_var
    fit_s = time.perf_counter() - t0

    # Transform ALL rows (centered by the fit-set mean) to a temp file, then
    # atomically replace, so an interrupted transform never leaves a truncated
    # pca50 that a consumer would mis-join.
    tmp = out.with_suffix(out.suffix + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    outmm = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(n, config.PCA_DIM))
    for lo in range(0, n, _PCA_CHUNK):
        hi = min(lo + _PCA_CHUNK, n)
        xb = np.asarray(emb[lo:hi], dtype=np.float64)
        outmm[lo:hi] = ((xb - mu) @ comps).astype(np.float32)
    outmm.flush()
    del outmm
    os.replace(tmp, out)

    total_s = time.perf_counter() - t0
    fit_desc = "all" if fit_mask is None else f"{n_fit:,}"
    # R4 — headerless memmap: record N/dim so consumers can assert on join width.
    (out.with_suffix(out.suffix + ".meta.json")).write_text(json.dumps({
        "n": int(n), "dim": config.PCA_DIM, "dtype": "float32",
        "source_rows": int(emb_rows), "method": "exact-cov-eigh",
        "fit_on": "all" if fit_mask is None else int(n_fit),
        "explained_variance_ratio_top10": [float(x) for x in evr[:10]],
        "total_explained": float(evr.sum()),
        "expected_bytes": int(n) * config.PCA_DIM * 4,
    }, indent=2))

    print(f"\nreduce.run_pca[{config.resolve_sample(sample)}] -> {out}\n"
          f"  shape/dtype : ({n:,}, {config.PCA_DIM}) float32\n"
          f"  size        : {out.stat().st_size / 1e6:.1f} MB\n"
          f"  fit_on      : {fit_desc} rows (exact cov eigh, deterministic)\n"
          f"  EVR[0]      : {evr[0]:.4f}   top-{config.PCA_DIM} total: {evr.sum():.4f}\n"
          f"  timing      : fit {fit_s:.0f}s, total {total_s:.0f}s")
    return out


def run_umap(sample: int | None = None,
             pca_memmap: Path = config.PCA_MEMMAP,
             out: Path = config.UMAP_MEMMAP) -> Path:
    """UMAP ``PCA_DIM``→2, sample-fit + transform-rest. Display only. Returns memmap.

    Display-only (the metric space is PCA-50; this is just the map). On this 16 GB
    box the ~7.4 GB spectral-init peak is the pipeline's RAM ceiling AND a silent-
    failure hotspot (umap catches ArpackError and quietly returns a random layout),
    so init="random" is used deliberately — lower, predictable RAM; UMAP's SGD
    recovers structure regardless for a static v1 map. Fit on a super-region-
    stratified ``UMAP_FIT_SAMPLE``; ``transform`` the rest in chunks.
    """
    import umap

    meta = json.loads((pca_memmap.with_suffix(pca_memmap.suffix + ".meta.json")).read_text())
    n = int(meta["n"]) if sample is None else min(sample, int(meta["n"]))
    pca = np.memmap(pca_memmap, dtype=np.float32, mode="r", shape=(int(meta["n"]), config.PCA_DIM))

    # super-region-stratified fit sample (proportional; structural fidelity for the map)
    labels = np.load(config.ARTIFACTS_DIR / "region_labels.i16.npy").astype(np.int32)[:n]
    super_of_region = np.load(config.ARTIFACTS_DIR / "super_region.i16.npy").astype(np.int32)
    paper_super = super_of_region[labels]
    rng = np.random.default_rng(0)
    n_fit = min(config.UMAP_FIT_SAMPLE, n)
    picks = []
    for s in range(config.N_SUPERREGIONS):
        idx = np.where(paper_super == s)[0]
        if idx.size == 0:
            continue
        take = max(1, round(n_fit * idx.size / n))
        picks.append(rng.choice(idx, min(take, idx.size), replace=False))
    fit_idx = np.sort(np.concatenate(picks))
    x_fit = np.asarray(pca[fit_idx], dtype=np.float32)

    t0 = time.perf_counter()
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", init="random", low_memory=True, verbose=True)
    reducer.fit(x_fit)
    fit_s = time.perf_counter() - t0

    tmp = out.with_suffix(out.suffix + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    outmm = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(n, 2))
    CH = 100_000
    for lo in range(0, n, CH):
        hi = min(lo + CH, n)
        outmm[lo:hi] = reducer.transform(np.asarray(pca[lo:hi], dtype=np.float32)).astype(np.float32)
    outmm.flush()
    del outmm
    os.replace(tmp, out)

    (out.with_suffix(out.suffix + ".meta.json")).write_text(json.dumps({
        "n": int(n), "dim": 2, "dtype": "float32", "init": "random",
        "fit_sample": int(fit_idx.size), "stratified_by": "super_region",
        "expected_bytes": int(n) * 2 * 4,
    }, indent=2))
    print(f"\nreduce.run_umap[{config.resolve_sample(sample)}] -> {out}\n"
          f"  shape/dtype : ({n:,}, 2) float32   fit_sample={fit_idx.size:,} (stratified)\n"
          f"  timing      : fit {fit_s:.0f}s, total {time.perf_counter() - t0:.0f}s")
    return out
