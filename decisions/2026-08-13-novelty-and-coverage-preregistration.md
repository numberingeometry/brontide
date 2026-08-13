---
type: adr
status: accepted
opened: 2026-08-13
affects:
  - artifacts/novelty_stream.parquet
  - artifacts/cutoffs/*/novelty_stream.parquet
  - artifacts/cutoffs/*/forecasts.parquet
  - scripts/verify_coverage.py
---

# 2026-08-13 — pre-registration: novelty-stream incremental skill, and 80% interval coverage / PIT

**Frozen before any number below exists.** Two signals have been sitting computed in the
committed artifacts since 2026-07-23 and have never been scored. `README.md` records that a
clean pre-registration was still available for them. This ADR takes it. It is committed before
`scripts/verify_coverage.py` is written and before either signal is evaluated.

## What was inspected before this commit, and why

Two structural facts about the artifacts were established first, because they determine *what
the estimand is* rather than what its value is. Both are disclosed here and neither carries a
threshold:

1. **`novelty_stream.parquet` is keyed by `super_region`, not by micro-region** (64 supers plus
   a `-1` corpus-total row). Its counts agree exactly with `region_shares.parquet` aggregated to
   supers — 0 mismatching rows out of 8,448 / 10,752 / 12,288 at the three cutoffs, and the
   corpus totals agree to the paper (942,716 / 1,417,285 / 1,784,871). Each cutoff's stream
   stops exactly at its own `T0`, so it is leak-free for scoring anything after `T0`.
2. **In `forecasts.parquet` the published `mean` is EB-shrunk while `lo`/`hi` are not.**
   `forecast.py` writes `shrunk[r,j]` as the mean and the raw `statsforecast` bounds as the
   interval, so the interval is not centred on the point forecast it ships with. On AutoETS at
   2017-12 the median relative gap between `mean` and `(lo+hi)/2` is 1.5%, and in 0.19% of rows
   the published mean lies **outside its own published interval**. This is a defect of the
   artifact, it is reported as a diagnostic with no pass/fail, and it is the reason the coverage
   test below is specified in two variants rather than one.

## Signal A — does the novelty stream carry incremental skill?

**Question.** At a forecast origin, does a region's recent *novelty rate* — the fraction of its
papers that land far from every centroid of the frozen map — predict its subsequent share growth,
over and above what momentum already says?

**Unit of analysis.** The **super-region**, at each cutoff `T0 ∈ {2017-12, 2020-12, 2022-12}`,
using that cutoff's own leak-free replica throughout (its map, its supers, its shares, its
novelty stream). 64 units per cutoff. Scoring at micro-region resolution is forbidden: the
signal exists only at super resolution, and broadcasting it to 1,024 leaves would inflate `n`
sixteenfold without adding information.

**Feature (frozen).** Trailing-12-month novelty fraction at the origin,
```
nov(S) = ( Σ_{t = T0-11..T0} n_novel(S,t) ) / ( Σ_{t = T0-11..T0} n_papers(S,t) ) ,
```
undefined (unit dropped, and the drop reported) if the denominator is 0.

**Control (frozen).** Trailing momentum on the same shares the target uses,
```
mom(S) = mean_{t = T0-11..T0} s(S,t)  /  mean_{t = T0-23..T0-12} s(S,t) ,
```
where `s(S,t) = Σ_{r ∈ S} share_smoothed(r,t)` from that cutoff's `region_shares.parquet`.
Momentum is the control because `SmoothedMomentum` is the model that actually won the frozen
bake-off; beating a control the instrument already lost to would be meaningless.

**Target (frozen).** The same growth functional the project's frozen protocol uses, lifted to
supers:
```
G(S) = mean_{t = T0+19..T0+24} s(S,t)  /  mean_{t = T0-11..T0} s(S,t) .
```

**Eligibility (frozen).** A super is scored iff it has ≥ 20 papers in the trailing 12 months at
`T0` — the same absolute floor `validation.md` freezes for micro-regions, applied at the unit of
analysis. Exclusions are counted and reported.

**Primary statistic (frozen).** The **first-order partial Spearman correlation** between
`nov` and `G` controlling for `mom`: rank-transform all three vectors, then
`ρ_partial = (ρ_ng − ρ_nm ρ_mg) / sqrt((1−ρ_nm²)(1−ρ_mg²))` on those ranks. Two-sided p-value
from `t = ρ_partial sqrt((n−3)/(1−ρ_partial²))` on `n−3` degrees of freedom.

**Frozen criterion.**
> Incremental novelty skill is **established** iff `ρ_partial > 0` with `p < 0.05` at **≥ 2 of the
> 3 cutoffs**. Otherwise it is **not established**, and that is the reported result.

The sign is part of the criterion. A significant *negative* partial correlation at ≥ 2 of 3 does
**not** count as skill; it is reported prominently as a sign-reversed finding, because "recent
novelty predicts subsequent *decline*" is a substantive claim and would need saying, not burying.

**Secondaries (reported, no threshold).** The raw Spearman `ρ(nov, G)` with no control; the
partial controlling instead for the deployed forecast `ĝ` aggregated to supers; and the same
tests with the novelty *trend* `nov_trend(S) = nov(S) − nov_{prev12}(S)` in place of `nov`.

## Signal B — are the published 80% intervals calibrated?

**Object.** The `lo`/`hi` columns of `artifacts/cutoffs/<T0>/forecasts.parquet` for the three
models that carry intervals — AutoETS, AutoARIMA, AutoTheta — at all three cutoffs. Nominal
level 80% (`forecast.py` calls `sf.forecast(..., level=[80])`). 1,024 regions × 24 horizon
months × 3 models × 3 cutoffs = 221,184 interval evaluations.

**Realised value (frozen).** `share_smoothed(r,t)` from the **same cutoff replica's**
`region_shares.parquet`, which extends to `T0+24m`. Not the production artifact: the replica is
the leak-free object the forecast was made on.

**Coverage, primary (frozen).** Pooled empirical coverage `P(lo ≤ y ≤ hi)` per (model, cutoff)
cell — 9 cells.

> The published intervals are **calibrated** iff at least **7 of the 9** cells have empirical
> coverage in **[0.75, 0.85]**. Otherwise calibration **fails**, and the nine numbers are
> reported either way.

**Coverage, secondary (frozen, reported alongside).** The same statistic on the interval
*recentred* on the published mean, `[lo + c, hi + c]` with `c = mean − (lo+hi)/2`, which is the
interval a reader gets if they assume — as the artifact's own column names invite — that the
interval is centred on the point forecast. Both variants are reported; the primary is the
interval **exactly as published**, because that is what a consumer of the artifact actually has.

**Dependence (frozen).** The 221,184 evaluations are not independent: 24 months share a region
and all regions share a monthly denominator. No p-value is computed on pooled coverage. Instead
the **region-clustered** distribution is reported: coverage per region (mean over its 24 months),
summarised by its median and its 10th/90th percentiles across the 1,024 regions.

**PIT (frozen).** `statsforecast` intervals for ETS/ARIMA/Theta are normal-theory, so a
Gaussian-implied PIT is available from the two published bounds:
```
σ̂ = (hi − lo) / (2 z),  z = Φ⁻¹(0.9) = 1.2815515655446004,
u = Φ( (y − centre) / σ̂ ),   centre = published mean (primary) or (lo+hi)/2 (secondary).
```
This **assumes normality of the predictive distribution** — it is the models' own assumption,
not an extra one, but the write-up must say so.

Reported per cell: mean PIT, the fraction below 0.1 and above 0.9, a 10-bin histogram, and the
Kolmogorov–Smirnov statistic against U(0,1).

> PIT uniformity is **established** iff the KS statistic is **< 0.05** in ≥ **7 of the 9** cells.

The criterion is on the KS *statistic*, not on a p-value, and deliberately so: at n = 24,576 per
cell any p-value criterion rejects uniformity for deviations far too small to matter, so a
p-value here would measure sample size rather than calibration.

## Commitment to report either way

Both signals are reported with their frozen criteria and their full tables whatever comes out.

- Novelty skill **not established** is the expected outcome given the instrument's own null
  (`region-level emergence is momentum-dominated at 24m`), and it is a result: a signal that has
  been computed, shipped in a public artifact, and displayed in a dashboard, and that carries no
  incremental information about the thing the artifact is for.
- Intervals **not calibrated** is likewise reported plainly, together with the direction of the
  miscalibration.
- Neither outcome may be dropped, softened, or relabelled exploratory after the fact. If a
  criterion is met, the paper says established; if not, the paper says not established.

## What these tests cannot establish

- Signal A tests *this* novelty definition (distance to the frozen map's centroids, thresholded
  at the 0.99 fit-set quantile) at *this* horizon. A null does not retire novelty as a construct.
- Signal B tests the intervals shipped by `statsforecast` under its own distributional
  assumptions applied to an EB-smoothed share series. Miscalibration is a fact about this
  pipeline's uncertainty quantification, not about ETS/ARIMA/Theta in general.
- Neither signal is a model comparison. Nothing here changes the frozen 18-row results table.

## Verification

`scripts/verify_coverage.py` recomputes every number above from the committed artifacts, prints
each frozen criterion with its PASS/FAIL, and exits non-zero if it cannot reproduce its own
reported values. It re-derives the EB anchor (`κ = 3.147960961354892`,
`max|p̂ − p̂_committed| = 0`) before trusting any derived number, as every verifier in this repo
does.
