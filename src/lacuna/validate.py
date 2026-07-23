"""Stage 8 — Out-of-time validation (CPU-only environment). THE headline artifact.

Three replicates at ``config.VALIDATION_CUTOFFS`` = (2017-12, 2020-12, 2022-12):
one quiet window + two regime-shift stress tests (COVID, the LLM explosion).

**Lookahead control:** PCA + k-means fit on papers ≤ T0 only; later papers
assigned nearest-centroid; forecasts from ≤ T0 data; novelty stream reported per
window. Accepted-and-disclosed leakages: SPECTER2 weights trained on post-T0
literature; UMAP display fit on all data (display only).

**Scoring** against realized *smoothed*-share growth over (T0, T0+24m] (same
stabilization both sides): **precision@50** vs top-decile realized growth +
**Spearman**(ĝ, realized) + **CSI** (critical success index — K&B-comparable) —
**headline comparison vs the SmoothedMomentum baseline** (persistence and
seasonal-naive also reported).

**Pre-registered negative criterion:** if the roster fails to beat smoothed
momentum on precision@50 in ≥ ``NEGATIVE_CRITERION_MIN_WINS`` of 3 replicates,
the writeup says so plainly ("region-level emergence is momentum-dominated at
24m") — the harness, map, and novelty stream remain the résumé artifact.
"""
from __future__ import annotations

from pathlib import Path

from . import config


def run(cutoffs: tuple[str, ...] = config.VALIDATION_CUTOFFS,
        out: Path = config.ARTIFACTS_DIR / "validation.md") -> Path:
    """Run the full out-of-time harness across cutoffs, append the results table to
    the (already pre-registered) validation.md, and evaluate the negative criterion.

    IMPORTANT: the thresholds section of validation.md must already be committed
    (dated, content-hashed) BEFORE this scoring run — pre-registration discipline.
    """
    # TODO per cutoff T0:
    #   1. fit_mask = month <= T0; regions.run(fit_mask=...); series/forecast on <= T0.
    #   2. emergence.run(cutoff=T0) -> ĝ ranking.
    #   3. realized smoothed-share growth over (T0, T0+24m]; precision@50, Spearman, CSI.
    #   4. same for SmoothedMomentum / Naive / SeasonalNaive.
    #   5. tally wins vs baseline; write results table; flag negative criterion.
    raise NotImplementedError("validate.run")
