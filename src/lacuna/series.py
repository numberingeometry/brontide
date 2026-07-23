"""Stage 5 — Series (host, minutes).

Two signals per (micro-region × month), ``CORPUS_START``→present (~235 pts):

1. **Primary target — EB-smoothed region share** of monthly submissions
   (controls the arXiv-growth confound). Region share shrunk toward its
   super-region's share (shrinkage weight from region size); add-``ADD_ALPHA``
   smoothing before any log/logit so zero months are defined. **No small-region
   deletion** — small regions stay in the ranking universe (EB handles noise);
   ``ACTIVITY_FLOOR_TRAILING12M`` is a *reporting* filter only. Region set is
   **frozen per forecast origin**.

2. **Novelty stream** — per month, count + fraction of new papers farther than a
   calibrated distance threshold from *every* existing centroid (genuinely-new-
   topic influx), per super-region and global. This is the emergence signal that
   nearest-centroid assignment would otherwise smear away.
"""
from __future__ import annotations

from pathlib import Path

from . import config


def run_shares(sample: int | None = None,
               out: Path = config.ARTIFACTS_DIR / "region_shares.parquet") -> Path:
    """(region × month) counts → EB-smoothed shares (long parquet). ~2 MB committed."""
    # TODO:
    #   1. groupby(region, month).size() over the frozen region set.
    #   2. share = (count + ADD_ALPHA) / (monthly_total + ADD_ALPHA*N_REGIONS).
    #   3. EB shrink toward super-region share, weight ~ region size.
    raise NotImplementedError("series.run_shares")


def run_novelty(sample: int | None = None,
                out: Path = config.ARTIFACTS_DIR / "novelty_stream.parquet") -> Path:
    """Per-month new-topic influx: fraction beyond a calibrated centroid distance."""
    # TODO: nearest-centroid distance per paper; threshold via a held-early calibration;
    #       aggregate count+fraction per (super-region, month) and global.
    raise NotImplementedError("series.run_novelty")
