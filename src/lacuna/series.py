"""Stage 5 — Series (host, minutes).

Two signals per (micro-region × month), ``CORPUS_START``→``CORPUS_END`` (the last
COMPLETE month; the trailing partial month is dropped — its raw count deflates the
eligibility floor):

1. **Primary target — EB-smoothed region share** of monthly submissions (controls
   the arXiv-growth confound). Each micro-region's share ``p_r`` is shrunk toward its
   super-region's **per-region share level ``P_S/n_S``** (NOT the raw super total
   ``P_S`` — that is ~16× larger, breaks scale, and inverts the ranking), with weight
   ``w_r = n_r/(n_r+κ)`` where ``n_r`` is the region's **trailing-12m** count fixed
   per origin (NOT the raw monthly count, which collapses ĝ to the super ranking).
   ``κ`` is empirical-Bayes **estimated** (symmetric Dirichlet-Multinomial
   concentration, method-of-moments, one global pool), refit on ``<= T0`` per origin.
   add-``ADD_ALPHA`` before any log/logit so zero months are defined. **No small-
   region deletion** — ``ACTIVITY_FLOOR_TRAILING12M`` is a *reporting* filter only.
   Region set **frozen per forecast origin**. Full spec: config.py EB block +
   artifacts/validation.md (2026-07-23 clarification).

2. **Novelty stream** — per month, count + fraction of new papers whose nearest-
   centroid distance exceeds ``NOVELTY_FIT_QUANTILE`` of the *fit-set's* nearest-
   centroid distances (per super-region and global). Leak status follows the fit
   set: leak-free ``<= T0`` in the masked run; "novelty vs the final frozen map"
   (display-only, like UMAP) in the unmasked production run.
"""
from __future__ import annotations

from pathlib import Path

from . import config


def run_shares(sample: int | None = None,
               cutoff: str | None = None,
               regions_dir: Path = config.ARTIFACTS_DIR,
               out: Path = config.ARTIFACTS_DIR / "region_shares.parquet") -> Path:
    """(region × month) counts → EB-smoothed shares (long parquet). ~2 MB committed.

    ``cutoff`` = forecast origin T0 (``"YYYY-MM"``); ``None`` = production origin =
    ``config.CORPUS_END``. Months used = ``CORPUS_START..T0`` (masking for run B).
    ``regions_dir`` supplies the cutoff-specific labeling from ``regions.run``
    (``region_labels``, ``super_region`` map, ``centroids``) — the source of the
    region set + n_S, so a masked validation origin points here at its own
    ``artifacts/cutoffs/<T0>/`` dir, never the production one.
    """
    # TODO (EB target disambiguated 2026-07-23 — see config.py EB block + validation.md):
    #   0. T0 = cutoff or config.CORPUS_END; keep papers with month <= T0.
    #      Load region_labels + super_region map (region->super, gives n_S per super)
    #      from regions_dir.
    #   1. Per (region r, month t): count c_{r,t}; monthly total M_t; add-alpha share
    #      p_{r,t} = (c_{r,t} + ADD_ALPHA) / (M_t + ADD_ALPHA*N_REGIONS).
    #   2. Super per-region mean m_{r,t} = P_{S,t}/n_S, where P_{S,t} = sum_{r' in S} p_{r',t}
    #      (over the SAME add-alpha shares) and n_S = frozen # micro-regions in S.  <-- P_S/n_S, NOT P_S.
    #   3. kappa: ONE global value, symmetric Dirichlet-Multinomial concentration by
    #      method-of-moments over within-super counts on the trailing-12m window ending at T0,
    #      papers <= T0 only. Refit per origin (per CV fold: on <= that fold's train end).
    #   4. weight w_r = n_r/(n_r+kappa) with n_r = region r's TRAILING-12m count at T0,
    #      FIXED across the origin's months (do NOT use the monthly count -> w->0 -> ranking
    #      collapses to super-region level). Per-month signal enters via p_{r,t} and P_{S,t}.
    #   5. p_hat_{r,t} = w_r*p_{r,t} + (1-w_r)*m_{r,t}. Long parquet: region, month, share_smoothed.
    raise NotImplementedError("series.run_shares")


def run_novelty(sample: int | None = None,
                cutoff: str | None = None,
                regions_dir: Path = config.ARTIFACTS_DIR,
                out: Path = config.ARTIFACTS_DIR / "novelty_stream.parquet") -> Path:
    """Per-month new-topic influx: fraction beyond the fit-set distance quantile.

    ``cutoff``/``regions_dir`` as in ``run_shares``. The distance threshold is
    ``NOVELTY_FIT_QUANTILE`` of the FIT-SET nearest-centroid distances, so it inherits
    the fit set's leak profile (leak-free <= T0 in run B; vs-final-map in run A).
    """
    # TODO:
    #   1. nearest-centroid distance per paper against the regions_dir centroids
    #      (PCA-50 space; the SAME <= T0 centroids in run B).
    #   2. tau = NOVELTY_FIT_QUANTILE quantile of the fit-set (<= T0) distances.
    #   3. novel = dist > tau; aggregate count + fraction per (super-region, month) and global.
    #   run A (cutoff=None): centroids see all data -> "novelty vs the final frozen map",
    #   display-only; emphasize the leading-edge frontier, not the degenerate 2007->2026 curve.
    raise NotImplementedError("series.run_novelty")
