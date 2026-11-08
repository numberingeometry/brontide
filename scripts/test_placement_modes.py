"""How good is `transform` really, against k-NN, on the same held-out rows?

The saved-reducer path was documented as "exact". It is not, and this measures by how much
rather than repeating the claim. UMAP's `transform` is a fresh optimisation of a new point
against the fitted graph, not a lookup, so it lands near where the batch fit put the row
rather than exactly on it.

Ground truth is the stored `umap2` coordinate. Probes are rows the fit did NOT see -- rows in
the fit sample were placed by the fit itself, so asking `transform` to reproduce them measures
the wrong thing and flatters the result.

    conda run -n brontide python scripts/test_placement_modes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from brontide import config, geometry  # noqa: E402


def main() -> int:
    n = geometry.corpus_rows()
    pca = np.memmap(config.PCA_MEMMAP, dtype=np.float32, mode="r",
                    shape=(n, config.PCA_DIM))
    U = np.memmap(config.UMAP_MEMMAP, dtype=np.float32, mode="r", shape=(n, 2))

    fit_rows = np.load(config.UMAP_MEMMAP.with_name("umap2_fit_rows.npy"))
    held = np.setdiff1d(np.arange(n), fit_rows)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(held, 60, replace=False))
    P = np.asarray(pca[idx], dtype=np.float32)
    truth = np.asarray(U[idx], dtype=np.float64)

    Uall = np.asarray(U, dtype=np.float32)
    span = float(max(np.ptp(Uall[:, 0]), np.ptp(Uall[:, 1])))
    print(f"corpus {n:,}   fit sample {len(fit_rows):,}   held-out probes {len(idx)}")
    print(f"map span {span:.2f} raw units\n")

    # k-NN MUST exclude the probe row itself. The probes are real corpus rows, so the
    # nearest neighbour of a probe is the probe, at distance zero, and `to_umap` returns its
    # own stored coordinate -- a perfect 0.0000 that measures a self-lookup and nothing else.
    # (Reported "k-NN is 358,917x better than the reducer" before this was caught.) In
    # production the query is a paper that is NOT in the corpus, so self-exclusion is what
    # reproduces the real situation.
    def knn_excluding_self(k: int) -> np.ndarray:
        PC = np.memmap(config.PCA_MEMMAP, dtype=np.float32, mode="r",
                       shape=(n, config.PCA_DIM))
        out = np.zeros((len(idx), 2), dtype=np.float64)
        NB = 400_000
        for i, (row, v) in enumerate(zip(idx, P)):
            bd = np.full(k, np.inf); bi = np.zeros(k, dtype=np.int64)
            for r0 in range(0, n, NB):
                r1 = min(r0 + NB, n)
                B = np.asarray(PC[r0:r1], dtype=np.float32)
                d = np.linalg.norm(B - v, axis=1)
                if r0 <= row < r1:
                    d[row - r0] = np.inf                  # the row itself
                take = min(k, r1 - r0)
                sel = np.argpartition(d, take - 1)[:take]
                bd = np.concatenate([bd, d[sel]]); bi = np.concatenate([bi, sel + r0])
                keep = np.argsort(bd)[:k]; bd, bi = bd[keep], bi[keep]
            w = 1.0 / (bd.astype(np.float64) + 1e-6)
            out[i] = (np.asarray(U[bi], dtype=np.float64) * w[:, None]).sum(0) / w.sum()
        return out

    rows = []
    xy, mode = geometry.to_umap(P, prefer_reducer=True)
    e = np.linalg.norm(np.asarray(xy, dtype=np.float64) - truth, axis=1)
    rows.append(("reducer (transform)", "reducer", np.median(e),
                 np.percentile(e, 90), e.max()))
    for k in (8, 1):
        xy = knn_excluding_self(k)
        e = np.linalg.norm(xy - truth, axis=1)
        rows.append((f"corpus k-NN, k={k}", f"knn{k}", np.median(e),
                     np.percentile(e, 90), e.max()))

    print(f"  {'mode':<22} {'median':>18} {'p90':>16} {'max':>16}")
    for label, mode, med, p90, mx in rows:
        print(f"  {label:<22} {med:8.4f} ({med / span:5.2%}) "
              f"{p90:7.3f} ({p90 / span:5.2%}) {mx:7.3f} ({mx / span:5.2%})")

    best = min(rows, key=lambda r: r[2])
    print(f"\n  best median: {best[0]}")
    r_med = rows[0][2]
    k_med = rows[1][2]
    print(f"  reducer vs k-NN(8): {k_med / r_med:.2f}x better median"
          if r_med < k_med else
          f"  k-NN(8) is {r_med / k_med:.2f}x better median than the reducer")
    print("\nNote: `transform` re-optimises each point against the fitted graph, so it is\n"
          "close-but-not-identical to what the batch fit produced. The real advantage over\n"
          "k-NN is not accuracy alone -- it is that appends need no corpus scan and stay\n"
          "consistent as the library grows past the anchors k-NN depends on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
