"""Does the matmul form of to_region pick the same centroids as the broadcast form?

`to_region` originally read `((cent[None] - p[:, None]) ** 2).sum(-1).argmin(1)` -- correct,
and it allocates an (n, 1024, 50) intermediate. That is invisible for the handful of rows it
was written for and **20.5 GB** when something asks it for a 100,000-row chunk, which on a
16 GB box does not fail cleanly, it thrashes.

The replacement expands the distance and drops the per-row constant. Dropping a term from a
distance is exactly the kind of change that is either exactly right or quietly wrong, so it
is checked against the original on real corpus rows.

    conda run -n brontide python scripts/test_to_region.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from brontide import config, geometry  # noqa: E402


def main() -> int:
    n = geometry.corpus_rows()
    pca = np.memmap(config.PCA_MEMMAP, dtype=np.float32, mode="r",
                    shape=(n, config.PCA_DIM))
    cent = np.load(config.ARTIFACTS_DIR / "centroids.f32.npy").astype(np.float32)

    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n, 6000, replace=False))
    P = np.asarray(pca[idx], dtype=np.float32)
    print(f"corpus {n:,}   probes {len(idx):,}   centroids {len(cent)}")

    t0 = time.perf_counter()
    ref = ((cent[None, :, :] - P[:, None, :]) ** 2).sum(-1).argmin(1)
    t_ref = time.perf_counter() - t0
    peak = len(P) * len(cent) * config.PCA_DIM * 4 / 1e9

    t0 = time.perf_counter()
    got, sup = geometry.to_region(P)
    t_new = time.perf_counter() - t0

    same = int((ref == got).sum())
    print(f"  broadcast {t_ref:.2f}s (peak intermediate {peak:.2f} GB)   "
          f"matmul {t_new:.2f}s (peak {len(P) * len(cent) * 4 / 1e9:.3f} GB)")
    print(f"  identical microregions: {same:,}/{len(idx):,}  ({same / len(idx):.4%})")
    print(f"  super_region shape ok  : {sup.shape == got.shape}")
    ok = same == len(idx)
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
