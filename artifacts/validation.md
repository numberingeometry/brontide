# Pre-registration — out-of-time emergence validation

> **PRE-REGISTERED 2026-07-22** (repo scaffold; before any data touched). Every threshold below is frozen. This file is committed **before** the mid-August scoring run; the results table at the bottom is appended by `validate.run()` at run time. Do not edit the thresholds after the first scoring commit — if a knob must change, that is a new, dated pre-registration, disclosed as such.

Mirror of the frozen constants in `src/lacuna/config.py` (that module is the code's source of truth; this file is the dated human record).

## Target

Per (micro-region × month), the **empirical-Bayes-smoothed region share** of monthly arXiv submissions (add-α = 1.0; shrunk toward super-region share, weight ∝ region size). Region set **frozen per forecast origin**. Corpus from 2007-01.

## Emergence statistic

At cutoff T0, rank regions by
`ĝ(r) = forecast smoothed-share (avg of horizon months 19–24) / smoothed-share(T0, trailing-12m avg)`.

| Label | Rule |
|---|---|
| **Emerging** | top-50 by ĝ with ĝ ≥ **1.5**, eligibility ≥ **20** papers in trailing-12m at T0, ties by region size |
| **Declining** | bottom-50 with ĝ ≤ **0.75** (share decline, not saturation) |
| **Saturating** (separate) | trailing-24m growth ≥ **1.3** AND **0.9 ≤ ĝ ≤ 1.1** |

## Forecast protocol

- Roster: Naive, SeasonalNaive, **SmoothedMomentum** (baseline), AutoETS, AutoARIMA, AutoTheta (statsforecast, freq = MS).
- Horizon **24 months**; rolling-origin CV: 24-month test windows, 4 origins, ≥120-month min train; scored overall AND on horizon months 19–24.
- Co-primary selection metrics: per-series **MASE** (not pooled) + **Spearman** of forecast-growth vs realized-growth ranking on CV folds.
- v1 point forecasts EB-shrunk toward the 64 super-regions (Ward tree).
- Disclosed: shares are compositional; 1024 univariate forecasts ignore the sum-to-1 constraint (tolerable for ranking).

## Out-of-time evaluation

- Cutoffs **T0 ∈ {2017-12, 2020-12, 2022-12}** (one quiet + two regime-shift stress tests: COVID, LLM explosion).
- Lookahead control: PCA + k-means fit on papers ≤ T0 only; later papers nearest-centroid; forecasts from ≤ T0 data.
- Score vs realized smoothed-share growth over (T0, T0+24m]: **precision@50** (vs top-decile realized), **Spearman**(ĝ, realized), **CSI** (critical success index — K&B-comparable). Headline comparison = vs **SmoothedMomentum**; persistence + seasonal-naive also reported.
- Accepted-and-disclosed leakages: SPECTER2 weights trained on post-T0 literature; UMAP display fit on all data (display only).

## Pre-registered negative criterion

If the model roster fails to beat SmoothedMomentum on **precision@50 in ≥ 2 of 3** replicates, the writeup states plainly: *"region-level emergence is momentum-dominated at 24m."* The harness, map, and novelty stream remain the résumé artifact regardless.

## Predecessors (cited, no "first/only" language)

Klavans, Boyack & Murdick (PLOS ONE 2020) + CSET/ETO Map of Science — deployed out-of-time region-growth forecasting at 38–43M documents. This project adds the share formulation + statistical TS forecasts + the pre-registered multi-cutoff harness; it does not claim priority or scale.

---

## Results

_(appended by `validate.run()` at scoring time — mid-August. Empty until then.)_
