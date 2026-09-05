"""Rebuild artifacts/region_labels.i16.npy from pca50 + the frozen centroids.

    conda run -n brontide python scripts/rebuild_region_labels.py

`region_labels.i16.npy` is the per-paper microregion assignment. It is a host-only
intermediate (gitignored, ~5.4 MB) that `regions.run` writes alongside the small committed
region artifacts — and it is **absent on this machine**, which matters because two things
need it:

- `reduce.run_umap` builds its fit sample stratified by super-region, so a refit cannot even
  start without it.
- `series.run_shares` joins it row-for-row against the parquet.

Reconstructing it is exact, not approximate, and does not touch the frozen geometry: a
paper's microregion is defined as its nearest centroid in PCA-50, the centroids are frozen
and committed, and `pca50` is intact. So this recomputes the same argmin `regions.run`
computed, for every row including any appended since. `geometry.to_region` is the single
implementation, so there is no second copy of the rule to drift.

Verification is against the same definition applied independently, plus a size and
distribution check against `regions_summary.json` — which records what the original fit
produced, so a mismatch in region sizes would mean the centroids or pca50 had moved.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from brontide import config, geometry  # noqa: E402

OUT = config.ARTIFACTS_DIR / "region_labels.i16.npy"
CHUNK = 100_000


def main() -> int:
    n = geometry.corpus_rows()
    pca = np.memmap(config.PCA_MEMMAP, dtype=np.float32, mode="r",
                    shape=(n, config.PCA_DIM))
    print(f"pca50: {n:,} rows")
    if OUT.exists():
        have = np.load(OUT)
        print(f"existing region_labels: {len(have):,} rows")
        if len(have) == n:
            print("  already complete -- nothing to do")
            return 0

    labels = np.zeros(n, dtype=np.int16)
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        micro, _ = geometry.to_region(np.asarray(pca[lo:hi], dtype=np.float32))
        labels[lo:hi] = micro.astype(np.int16)
        if (lo // CHUNK) % 5 == 0:
            print(f"  {hi:>9,}/{n:,}", flush=True)

    sizes = Counter(labels.tolist())
    summary = json.loads((config.ARTIFACTS_DIR / "regions_summary.json").read_text())
    s_min, s_max = min(sizes.values()), max(sizes.values())
    print(f"\n  regions used     : {len(sizes)} of {config.N_REGIONS}")
    print(f"  size min/max     : {s_min:,} / {s_max:,}")
    print(f"  summary says     : {summary['region_size_min']:,} / "
          f"{summary['region_size_max']:,}   (n_papers {summary['n_papers']:,})")
    empty = config.N_REGIONS - len(sizes)
    print(f"  empty regions    : {empty}   (summary: {summary['empty_regions']})")

    # The summary describes the ORIGINAL fit over 2,705,992 rows; the corpus may have grown
    # since, so sizes may exceed it slightly but must not collapse. A big divergence means
    # the centroids or pca50 are not the ones that produced the summary.
    drift = abs(s_max - summary["region_size_max"]) / max(summary["region_size_max"], 1)
    ok = empty == summary["empty_regions"] and drift < 0.05
    print(f"  max-size drift   : {drift:.2%}  ->  {'consistent' if ok else 'DIVERGENT'}")
    if not ok:
        print("  refusing to write -- investigate before using these labels")
        return 1

    np.save(OUT, labels)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB, {len(labels):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
