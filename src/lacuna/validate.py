"""Stage 8 — Out-of-time validation (CPU-only environment). THE headline artifact.

Three replicates at ``config.VALIDATION_CUTOFFS`` = (2017-12, 2020-12, 2022-12):
one quiet window + two regime-shift stress tests (COVID, the LLM explosion).

**Lookahead control:** PCA + k-means fit on papers ≤ T0 only; later papers
assigned nearest-centroid; forecasts from ≤ T0 data; novelty stream reported per
window. κ and the novelty quantile τ are ALSO fit on ≤ T0 (per fold-train-end).
Accepted-and-disclosed leakages: SPECTER2 weights trained on post-T0 literature;
UMAP display fit on all data (display only).

**Scoring** against realized *smoothed*-share growth over (T0, T0+24m] (same
stabilization both sides): **precision@50** vs top-decile realized growth +
**Spearman**(ĝ, realized) + **CSI** (critical success index — K&B-comparable) —
**headline comparison vs the SmoothedMomentum baseline** (persistence and
seasonal-naive also reported).

**Pre-registered negative criterion:** if the roster fails to beat smoothed
momentum on precision@50 in ≥ ``NEGATIVE_CRITERION_MIN_WINS`` of 3 replicates,
the writeup says so plainly ("region-level emergence is momentum-dominated at
24m") — the harness, map, and novelty stream remain the résumé artifact.

Sprint 1 ships the pure scoring functions below (unit-tested). ``run`` (the masked
Run B orchestration) is Sprint 2 — it needs the host (the ≤T0 re-fits require the
embeddings memmap).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


# --- scoring primitives (pure, unit-tested) ---------------------------------
def precision_at_k(pred_ranked: np.ndarray, realized_positive: set, k: int = config.TOP_N) -> float:
    """Fraction of the top-``k`` predicted regions that are in the realized-positive
    set (e.g. the realized top-decile of growth). ``pred_ranked`` = region ids sorted
    most-emerging first."""
    top = pred_ranked[:k]
    if len(top) == 0:
        return 0.0
    return float(sum(1 for r in top if r in realized_positive) / len(top))


def csi(pred_set: set, realized_set: set) -> float:
    """Critical Success Index = hits / (hits + misses + false alarms) = |P∩R| / |P∪R|
    (K&B-comparable). 1.0 = perfect, 0.0 = disjoint. Empty ∪ → 0.0."""
    p, r = set(pred_set), set(realized_set)
    union = p | r
    return float(len(p & r) / len(union)) if union else 0.0


def spearman_growth(pred_growth: np.ndarray, realized_growth: np.ndarray) -> float:
    """Spearman correlation of predicted vs realized growth over the scored regions."""
    from scipy.stats import spearmanr
    m = np.isfinite(pred_growth) & np.isfinite(realized_growth)
    if m.sum() < 3:
        return 0.0
    val = spearmanr(pred_growth[m], realized_growth[m]).statistic
    return float(val) if np.isfinite(val) else 0.0


def top_decile(growth: np.ndarray, eligible: np.ndarray | None = None) -> set:
    """Region ids in the top decile of ``growth`` (restricted to ``eligible`` if given).
    The realized-positive reference set for precision@50."""
    idx = np.arange(len(growth))
    if eligible is not None:
        idx = idx[eligible]
    g = growth[idx]
    finite = np.isfinite(g)
    idx, g = idx[finite], g[finite]
    if len(idx) == 0:
        return set()
    thresh = np.quantile(g, 0.9)
    return set(int(r) for r in idx[g >= thresh])


def evaluate_negative_criterion(model_wins: int, n_replicates: int = 3,
                                min_wins: int = config.NEGATIVE_CRITERION_MIN_WINS) -> dict:
    """Did the roster beat SmoothedMomentum on precision@50 in ≥ ``min_wins`` of
    ``n_replicates`` cutoffs? Returns the verdict + the pre-registered sentence."""
    beat = model_wins >= min_wins
    return {
        "wins": int(model_wins), "of": int(n_replicates), "min_wins": int(min_wins),
        "beats_baseline": bool(beat),
        "verdict": ("roster beats SmoothedMomentum" if beat
                    else "region-level emergence is momentum-dominated at 24m"),
    }


def run(cutoffs: tuple[str, ...] = config.VALIDATION_CUTOFFS,
        out: Path = config.ARTIFACTS_DIR / "validation.md") -> Path:
    """Run the full out-of-time harness across cutoffs, append the results table to
    the (already pre-registered) validation.md, and evaluate the negative criterion.

    IMPORTANT: the thresholds section of validation.md must already be committed
    (dated, content-hashed) BEFORE this scoring run — pre-registration discipline.

    Sprint 2 (host-executed — the ≤T0 re-fits need the embeddings memmap). Per cutoff T0:
      1. fit_mask = month <= T0; reduce.run_pca(fit_mask, out=cutoffs/<T0>/pca50);
         regions.run(pca_memmap=<T0>, fit_mask, out_dir=cutoffs/<T0>); series.run_shares
         + run_novelty(cutoff=T0, regions_dir=cutoffs/<T0>, pca_memmap=<T0>).
         --> ALL cutoff-scoped paths; κ + τ refit on <=T0 (per fold-train-end in CV).
      2. forecast.run(cutoff=T0, shares=<T0>); emergence.run(cutoff=T0) -> ĝ ranking.
      3. realized smoothed-share growth over (T0, T0+24m]; precision@50 (top_decile),
         spearman_growth, csi -- for the roster AND SmoothedMomentum/Naive/SeasonalNaive.
      4. tally roster-beats-baseline on precision@50; evaluate_negative_criterion.
      5. append the results table to validation.md (below the frozen thresholds).
    Blocked on: series.run_novelty needs a pca_memmap param; validate must never reuse
    the production all-data memmaps (leak-safety).
    """
    raise NotImplementedError("validate.run (Sprint 2, host-executed)")
