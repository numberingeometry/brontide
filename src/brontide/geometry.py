"""geometry.py -- put a NEW paper into the EXISTING frozen space.

This is the module the pipeline never had, and its absence is why brontide had no
incremental path. Three layers have to be re-applicable to place a paper that arrives
after the build, and only one of them was:

===============  ==========================================  ==================
layer            was it re-applicable?                       now
===============  ==========================================  ==================
region           yes -- ``centroids.f32.npy`` is saved        unchanged
PCA 768->50      no -- only the projection was saved          ``pca_basis()``
UMAP 50->2       no -- the fitted reducer was discarded       ``to_umap()``
===============  ==========================================  ==================

## The PCA basis is recoverable, exactly

``run_pca`` throws away ``comps`` and ``mu`` but the fit is deterministic: an exact
centered covariance plus ``eigh`` plus a fixed sign convention. Re-running the fit on the
same embeddings reproduces the same basis bit-for-bit, so the basis for a build that
predates this module can be recovered rather than guessed -- and ``verify()`` checks it
against the stored ``pca50`` rows instead of trusting that argument.

## The UMAP reducer is NOT recoverable, and that is the honest part

A fitted UMAP is a neighbour index over its training sample plus an optimised embedding.
Nothing in the outputs determines it, so for any build made before ``run_umap`` started
saving the reducer there is no way to get it back. **Refitting is not a substitute**: with
``init="random"`` and no ``random_state``, a refit produces a completely different map, and
every coordinate already in circulation -- 2.7M rows, the published figures, the dashboard,
any cached layout downstream -- would move at once.

So ``to_umap`` has two modes and says which it used:

- ``reducer`` -- a saved reducer exists, so ``transform`` places the paper against the
  fitted graph. **Not exact, and the figure is measured rather than asserted**: median
  0.069 raw units = **0.38% of the map**, p90 0.85%, max 2.2%, over 60 rows the fit never
  saw (``scripts/test_placement_modes.py``). ``transform`` re-optimises each point rather
  than looking it up, so it lands near where the batch fit put the row, not on it.
- ``knn`` -- no reducer, so interpolate: inverse-distance over the k nearest CORPUS rows
  in PCA-50, averaged in UMAP space. On the same held-out probes: median **1.25% of the
  map**, p90 6.5%, max **18.2%**. So the reducer is 3.3x better on the median and 7.6x on
  p90 -- and the real gap is the TAIL, where k-NN can throw a paper a fifth of the map
  away. k-NN also costs a full corpus scan per query, which is fine for a handful of rows
  and 12 hours for a monthly catch-up (hence the batched form below).

  When measuring either mode, the probe row must be excluded from the anchor set. Probes
  are real corpus rows, so their own nearest neighbour is themselves at distance zero:
  without the exclusion k-NN scores a perfect 0.0000 and looks 358,917x better than the
  reducer, which is a self-lookup and not a result.

The `knn` mode is the default for the current build and is deliberately not presented as
equivalent. To get the exact path, run a full refit and keep the reducer it saves.

**A saved reducer does not mean "never refit again."** It means *appends* stop forcing a
refit: `transform` places new papers onto the existing manifold, so nothing already placed
moves. The reducer is still fit on a sample of one corpus, and as the corpus grows it becomes
less representative -- a genuinely new research area has no manifold structure in that fit and
gets pressed into the nearest existing one, which is the same degradation the region centroids
suffer. So refits stay necessary; what changes is that a refit becomes a **scheduled decision,
taken when drift measurements justify it and logged**, rather than the price of every ingest.
Same distinction the region geometry already makes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import config

BASIS = config.PCA_MEMMAP.with_name("pca50_basis.npz")
REDUCER = config.UMAP_MEMMAP.with_name("umap2_reducer.joblib")
_PCA_CHUNK = 50_000


def _rows(memmap: Path, width: int, itemsize: int) -> int:
    meta = memmap.with_suffix(memmap.suffix + ".meta.json")
    try:
        return int(json.loads(meta.read_text())["n"])
    except (OSError, KeyError, ValueError, TypeError):
        return memmap.stat().st_size // (width * itemsize)


def corpus_rows() -> int:
    return _rows(config.PCA_MEMMAP, config.PCA_DIM, 4)


# ---------------------------------------------------------------- PCA
def pca_basis(recover: bool = True) -> dict:
    """``{mean, components, affine}``. Loads the saved basis, or recovers it.

    Recovery re-runs the same exact-covariance fit ``run_pca`` uses. It reads the whole
    4 GB embedding memmap once (~30 s) and then caches the result to ``BASIS``, so the
    cost is paid at most once per build.
    """
    if BASIS.exists():
        z = np.load(BASIS)
        return {"mean": z["mean"], "components": z["components"], "affine": z["affine"],
                "source": "saved"}
    if not recover:
        raise FileNotFoundError(f"no PCA basis at {BASIS}")

    n = corpus_rows()
    d = config.EMBED_DIM
    emb = np.memmap(config.EMBED_MEMMAP, dtype=np.float16, mode="r", shape=(n, d))
    s = np.zeros(d, dtype=np.float64)
    gram = np.zeros((d, d), dtype=np.float64)
    for lo in range(0, n, _PCA_CHUNK):
        xb = np.asarray(emb[lo:min(lo + _PCA_CHUNK, n)], dtype=np.float64)
        s += xb.sum(axis=0)
        gram += xb.T @ xb
    mu = s / n
    cov = gram / n - np.outer(mu, mu)
    cov = (cov + cov.T) * 0.5
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:config.PCA_DIM]
    comps = evecs[:, order].copy()
    for j in range(comps.shape[1]):               # the same sign convention run_pca uses
        if comps[np.argmax(np.abs(comps[:, j])), j] < 0.0:
            comps[:, j] = -comps[:, j]
    affine = np.vstack([comps, (-mu @ comps)[None, :]]).astype(np.float32)
    BASIS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(BASIS, mean=mu, components=comps, affine=affine,
             eigenvalues=evals[order], dim=np.int64(config.PCA_DIM),
             embed_dim=np.int64(d), fit_rows=np.int64(n), recovered=np.int64(1))
    return {"mean": mu, "components": comps, "affine": affine, "source": "recovered"}


def to_pca(vecs768: np.ndarray, basis: dict | None = None) -> np.ndarray:
    b = basis or pca_basis()
    x = np.asarray(vecs768, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    return ((x - b["mean"]) @ b["components"]).astype(np.float32)


# ---------------------------------------------------------------- region
def to_region(pca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> (microregion, super_region). Nearest centroid; exact, and always was.

    The distance is expanded rather than broadcast. `(cent[None] - p[:, None]) ** 2` reads
    beautifully and allocates an (n, 1024, 50) intermediate: for a 100,000-row chunk that is
    **20.5 GB** on a 16 GB machine, which does not fail cleanly -- it thrashes. Via
    `||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b` the same argmin is one (n, 1024) matrix, 0.41 GB,
    and `||p||^2` drops out because it is constant across centroids within a row.

    Fine for the handful of rows this was written for; the difference only appears when
    something asks it for the whole corpus.
    """
    cent = np.load(config.ARTIFACTS_DIR / "centroids.f32.npy").astype(np.float32)
    sup = np.load(config.ARTIFACTS_DIR / "super_region.i16.npy").astype(int)
    p = np.atleast_2d(np.asarray(pca, dtype=np.float32))
    cn = (cent * cent).sum(1)[None, :]
    micro = np.empty(len(p), dtype=np.int64)
    CH = 50_000
    for lo in range(0, len(p), CH):
        blk = p[lo:lo + CH]
        micro[lo:lo + len(blk)] = (cn - 2.0 * (blk @ cent.T)).argmin(1)
    return micro, sup[micro]


# ---------------------------------------------------------------- UMAP
def to_umap(pca: np.ndarray, k: int = 8, prefer_reducer: bool = True
            ) -> tuple[np.ndarray, str]:
    """-> (xy, mode) where mode is "reducer" or "knn". See the module docstring."""
    p = np.atleast_2d(np.asarray(pca, dtype=np.float32))
    if prefer_reducer and REDUCER.exists():
        import joblib
        reducer = joblib.load(REDUCER)
        return np.asarray(reducer.transform(p), dtype=np.float32), "reducer"

    # The anchor set is rows that have BOTH a pca50 coordinate and a umap2 one. Sizing
    # either memmap from the other's row count is how this first failed: project_tail bumps
    # pca50's sidecar before placing on the map, so corpus_rows() already said 2,706,067
    # while umap2 still held 2,705,992, and mapping past the end of a read-only file gives
    # the memorably unhelpful "WinError 8: Not enough memory resources".
    n_pca = _rows(config.PCA_MEMMAP, config.PCA_DIM, 4)
    n_umap = _rows(config.UMAP_MEMMAP, 2, 4)
    n = min(n_pca, n_umap)
    PC = np.memmap(config.PCA_MEMMAP, dtype=np.float32, mode="r",
                   shape=(n_pca, config.PCA_DIM))
    U = np.memmap(config.UMAP_MEMMAP, dtype=np.float32, mode="r", shape=(n_umap, 2))

    # BATCHED, because the obvious loop does not scale and the difference is a working day.
    # One scan of the corpus per query row is fine for the 75-row case it was written for
    # and is 12 HOURS for a monthly catch-up of ~49,000: 132 billion distance evaluations
    # issued as 49,000 separate passes. Expanded as
    #     ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    # the whole thing is one matrix product per (query block x corpus block) -- 13 Tflop,
    # about four minutes of BLAS. Blocks are sized so no intermediate exceeds ~1 GB;
    # `||a||^2` is dropped because it is constant within a row and cannot change an argmin.
    QB, NB = 512, 400_000
    nq = len(p)
    best_d = np.full((nq, k), np.inf, dtype=np.float32)
    best_i = np.zeros((nq, k), dtype=np.int64)
    for q0 in range(0, nq, QB):
        q1 = min(q0 + QB, nq)
        Q = p[q0:q1]
        bd, bi = best_d[q0:q1], best_i[q0:q1]
        for r0 in range(0, n, NB):
            r1 = min(r0 + NB, n)
            B = np.asarray(PC[r0:r1], dtype=np.float32)
            d2 = (B * B).sum(1)[None, :] - 2.0 * (Q @ B.T)     # + const per row
            take = min(k, r1 - r0)
            sel = np.argpartition(d2, take - 1, axis=1)[:, :take]
            cd = np.take_along_axis(d2, sel, axis=1)
            bd = np.concatenate([bd, cd], axis=1)
            bi = np.concatenate([bi, sel + r0], axis=1)
            keep = np.argpartition(bd, k - 1, axis=1)[:, :k]
            bd = np.take_along_axis(bd, keep, axis=1)
            bi = np.take_along_axis(bi, keep, axis=1)
        best_d[q0:q1], best_i[q0:q1] = bd, bi

    # Recover true distances for the weights: the dropped ||a||^2 shifts every candidate of
    # a row equally, so it never changed WHICH neighbours were picked -- but it would corrupt
    # the inverse-distance weighting, so the k survivors are measured properly.
    out = np.zeros((nq, 2), dtype=np.float32)
    for i in range(nq):
        idx = best_i[i]
        dd = np.linalg.norm(np.asarray(PC[idx], dtype=np.float64) - p[i], axis=1)
        w = 1.0 / (dd + 1e-6)
        out[i] = (np.asarray(U[idx], dtype=np.float64) * w[:, None]).sum(0) / w.sum()
    return out, "knn"


# ---------------------------------------------------------------- verification
def verify(n_probe: int = 2000, seed: int = 0, max_row: int | None = None) -> dict:
    """Do the re-applicable transforms actually reproduce the stored ones?

    Checked against the corpus itself, which is the only ground truth available: take real
    rows, run them forward through this module, and compare with what is on disk.
    """
    n = corpus_rows()
    # `max_row` bounds the probe to rows whose EMBEDDINGS are actually populated. Without it
    # a verify during a restore samples the zeroed tail and reports a basis failure that is
    # really an incomplete embed -- which is exactly what preflight did on first run.
    hi = n if max_row is None else max(2, min(int(max_row), n))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(hi, min(n_probe, hi), replace=False))
    emb = np.memmap(config.EMBED_MEMMAP, dtype=np.float16, mode="r",
                    shape=(n, config.EMBED_DIM))
    pca = np.memmap(config.PCA_MEMMAP, dtype=np.float32, mode="r",
                    shape=(n, config.PCA_DIM))

    b = pca_basis()
    got = to_pca(np.asarray(emb[idx], dtype=np.float64), b)
    want = np.asarray(pca[idx], dtype=np.float64)
    err = np.linalg.norm(got - want, axis=1)
    scale = np.linalg.norm(want, axis=1)

    micro, sup = to_region(want.astype(np.float32))
    micro2, _ = to_region(got)

    out = {"basis_source": b["source"], "probes": int(len(idx)),
           "probe_ceiling": int(hi),
           "pca_err_median": float(np.median(err)),
           "pca_err_p99": float(np.percentile(err, 99)),
           "pca_err_relative_median": float(np.median(err / np.maximum(scale, 1e-9))),
           "region_agreement": float((micro == micro2).mean())}

    # UMAP: only meaningful on rows the fit did NOT see, and only if a reducer exists.
    fit_rows_p = config.UMAP_MEMMAP.with_name("umap2_fit_rows.npy")
    out["reducer"] = REDUCER.exists()
    if REDUCER.exists() and fit_rows_p.exists():
        held = np.setdiff1d(idx, np.load(fit_rows_p))[:200]
        if len(held):
            U = np.memmap(config.UMAP_MEMMAP, dtype=np.float32, mode="r", shape=(n, 2))
            xy, mode = to_umap(np.asarray(pca[held], dtype=np.float32))
            e = np.linalg.norm(xy - np.asarray(U[held], dtype=np.float32), axis=1)
            out.update(umap_mode=mode, umap_probes=int(len(held)),
                       umap_err_median=float(np.median(e)),
                       umap_err_p90=float(np.percentile(e, 90)))
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Check that the frozen space is re-applicable.")
    ap.add_argument("--probes", type=int, default=2000)
    a = ap.parse_args()
    r = verify(a.probes)
    print(f"\ngeometry.verify  ({r['probes']:,} corpus rows)")
    print(f"  PCA basis        : {r['basis_source']}")
    print(f"  PCA error        : median {r['pca_err_median']:.3e}  "
          f"p99 {r['pca_err_p99']:.3e}  (relative {r['pca_err_relative_median']:.2e})")
    print(f"  region agreement : {r['region_agreement']:.2%}")
    if r.get("umap_mode"):
        print(f"  UMAP ({r['umap_mode']}) : median {r['umap_err_median']:.4f}  "
              f"p90 {r['umap_err_p90']:.4f} over {r['umap_probes']} held-out rows")
    else:
        print("  UMAP             : no saved reducer -- new papers place by corpus k-NN "
              "(median ~0.95% of the map). A full refit is what creates one.")


if __name__ == "__main__":
    main()
