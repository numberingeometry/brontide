"""Does the batched k-NN placement agree with the naive per-row loop?

The batched form drops ``||a||^2`` from the distance (constant within a row, so it cannot
change an argmin) and partitions over both query and corpus blocks. Both are the kind of
optimisation that is easy to get subtly wrong -- off-by-one in a block offset places a paper
next to the wrong neighbours and nothing downstream complains. So it is checked against the
straightforward implementation on real corpus rows, where the correct answer is knowable.

    conda run -n brontide python scripts/test_knn_batched.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from brontide import config, geometry  # noqa: E402


def naive(p: np.ndarray, PC: np.ndarray, U, k: int) -> np.ndarray:
    out = np.zeros((len(p), 2), dtype=np.float32)
    for i, v in enumerate(p):
        d = np.linalg.norm(PC - v, axis=1)
        near = np.argpartition(d, k)[:k]
        near = near[np.argsort(d[near])]
        w = 1.0 / (d[near].astype(np.float64) + 1e-6)
        out[i] = (np.asarray(U[near], dtype=np.float64) * w[:, None]).sum(0) / w.sum()
    return out


def main() -> int:
    n_pca = geometry._rows(config.PCA_MEMMAP, config.PCA_DIM, 4)
    n_umap = geometry._rows(config.UMAP_MEMMAP, 2, 4)
    n = min(n_pca, n_umap)
    PCmm = np.memmap(config.PCA_MEMMAP, dtype=np.float32, mode="r",
                     shape=(n_pca, config.PCA_DIM))
    Umm = np.memmap(config.UMAP_MEMMAP, dtype=np.float32, mode="r", shape=(n_umap, 2))

    # A real subset as the corpus, so the naive comparison is affordable.
    SUB = 300_000
    PC = np.asarray(PCmm[:SUB], dtype=np.float32)
    U = np.asarray(Umm[:SUB], dtype=np.float32)

    rng = np.random.default_rng(0)
    q = np.asarray(PCmm[rng.integers(SUB, n, size=120)], dtype=np.float32)
    print(f"corpus subset {SUB:,}   queries {len(q)}   k=8")

    t0 = time.perf_counter()
    want = naive(q, PC, U, 8)
    t_naive = time.perf_counter() - t0

    # Point the module at the same subset by monkeypatching its row lookups.
    real_rows = geometry._rows
    geometry._rows = lambda path, w, i: SUB                       # noqa: ARG005
    try:
        t0 = time.perf_counter()
        got, mode = geometry.to_umap(q, k=8, prefer_reducer=False)
        t_batch = time.perf_counter() - t0
    finally:
        geometry._rows = real_rows

    err = np.linalg.norm(got - want, axis=1)
    span = float(max(np.ptp(U[:, 0]), np.ptp(U[:, 1])))
    print(f"  naive {t_naive:.2f}s   batched {t_batch:.2f}s   "
          f"speedup {t_naive / max(t_batch, 1e-9):.1f}x   mode={mode}")
    print(f"  placement difference: median {np.median(err):.3e}  max {err.max():.3e}"
          f"  ({err.max() / span:.4%} of the map)")
    exact = int((err < 1e-4).sum())
    print(f"  identical placements: {exact}/{len(q)}")

    # Ties are legitimate: equidistant neighbours may be picked in either order, which moves
    # nothing meaningfully. A real indexing bug shows as a large max, not a small tail.
    ok = err.max() < 0.01 * span
    print(f"\n{'PASS' if ok else 'FAIL'} -- batched agrees with naive")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
