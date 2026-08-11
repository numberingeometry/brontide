# Limitations and defect disclosure

> Written 2026-07-24, after scoring, and published unedited alongside the frozen result. Nothing here revises the pre-registered verdict in [`artifacts/validation.md`](artifacts/validation.md); it states what that verdict measured, and the three defects in the harness that produced it.
>
> The pre-registration text itself is never edited — `validate.py` hashes everything above the `## Results` marker, so any change would break the published content hash `dea6cc0db2ae1e83`.

## The frozen verdict, verbatim

> "Roster beat SmoothedMomentum on precision@50 in **0/3** cutoffs (need ≥2).
> Verdict: **region-level emergence is momentum-dominated at 24m**."

That sentence is reported as written. Its over-breadth is flagged below rather than revised.

## Scope note

This criterion compared the CV-selected member of a six-model roster — all six consuming the identical univariate own-share path — against a benchmark that was itself a roster member, and which was selected at two of three cutoffs. **It measures the pre-registered pipeline, not the phenomenon.** The evidence supports the narrower claim that *within the univariate own-share information set, momentum was not improved upon at $h=24$ by this selector*.

## The three structural defects

**1. The baseline is inside the candidate set.** At 2020-12 and 2022-12 the CV selected `SmoothedMomentum` itself, so `validate.py:211-213` evaluated `0.240 > 0.240` and `0.300 > 0.300` — $x > x$, false by identity. Because SmoothedMomentum is also the ex-post argmax at 2017-12 (0.760 vs AutoETS 0.700), the **maximum attainable win count was 2 — exactly the bar.** Zero margin.

**2. At 2017-12 the baseline was never on the ballot.** `forecast._rolling_cv` breaks when `test_start < CV_MIN_TRAIN_MONTHS = 120`. With $T = 132$ months, $w=0$ gives `test_start = 108` → zero folds for all three numpy models. Confirmed directly: `artifacts/cutoffs/2017-12/forecast_cv.parquet` holds 12 rows covering only AutoARIMA / AutoETS / AutoTheta. Meanwhile `statsforecast.cross_validation` carries no such guard and ran 4 windows on trains of 108/84/60/36 months — each violating the registered ≥120-month minimum. Fold counts across the run: auto-models 4/4/4, numpy baselines **0/2/3**.

The protocol is in fact arithmetically infeasible at that cutoff under its own rules: 24-month windows × 4 origins with a ≥120-month floor needs 144 months for one fold and 216 for four. There were 132. A correctly symmetric guard makes 2017-12 unscorable.

**3. The two CV paths compute different statistics, then get pooled and argmax'd.** `_rolling_cv` bases growth on `train[:, -12:]` (trailing 12m of *train*, matching deployed $\hat g$) and MASE on seasonal-naive-on-train. `_sf_cv_metrics` bases growth on the first 12 months of the *test* window — post-origin realized data — and MASE on a first-difference of the test window. Measured effect: this flips SeasonalNaive's mean CV-Spearman from $+0.203 \to -0.093$ and Naive's from $+0.039 \to -0.083$, while leaving SmoothedMomentum nearly invariant ($0.297 \to 0.309$). Momentum's apparent CV superiority at the two later cutoffs is therefore **confounded with a bug**.

## "No skill" vs "selection failed" — the data separates these

Chance precision@50 is **0.1007** (103 realized-positive regions out of 1021–1023 eligible micro-regions; the top-decile threshold is inclusive). Against that reference:

| Model | 2017-12 | 2020-12 | 2022-12 | mean | sd | × chance |
|---|---|---|---|---|---|---|
| **AutoETS** | 0.700 | 0.540 | 0.600 | **0.613** | **0.081** | 6.1× |
| AutoARIMA | 0.540 | 0.460 | 0.360 | 0.453 | 0.090 | 4.5× |
| **SmoothedMomentum** *(baseline)* | **0.760** | 0.240 | 0.300 | 0.433 | **0.285** | 4.3× |
| *CV-selected policy (as scored)* | *0.700* | *0.240* | *0.300* | *0.413* | *0.250* | *4.1×* |
| AutoTheta | 0.340 | 0.360 | 0.380 | 0.360 | 0.020 | 3.6× |
| SeasonalNaive | 0.260 | 0.180 | 0.160 | 0.200 | 0.053 | 2.0× |
| Naive | 0.020 | 0.160 | 0.140 | 0.107 | 0.076 | 1.1× |

Every non-Naive method clears chance at every cutoff — **the models have skill.** What failed is the step that picks among them. The CV-selected policy (0.413) is *worse than always using the baseline* (0.433) and 0.200 below a fixed AutoETS (0.613). At both 2020-12 and 2022-12 the CV picked the model whose realized Spearman was **negative** ($-0.034$, $-0.016$) while AutoETS ran $+0.264$ and $+0.361$ and AutoTheta reached $+0.467$.

The failure is a **selection-objective mismatch**: the harness selected on point accuracy (MASE + Spearman) and was scored on the top 5% of a ranking. Those are not the same loss.

Paired testing (the top-50 lists overlap 24/50, 17/50, 15/50, so independent binomials do not apply). AutoETS vs SmoothedMomentum on discordant regions: 2017-12 $p \approx 0.69$ (35 vs 38 hits — a tie; the "momentum win" is noise); 2020-12 $p \approx 0.0003$; 2022-12 $p \approx 0.001$. On Spearman, AutoETS beats the baseline by $>3\sigma$ (Fisher $z$) at all three cutoffs. On CSI it does **not** beat the baseline reliably anywhere ($z = -1.05, +1.61, +0.80$).

## Reading dependence of the frozen sentence

The same frozen sentence admits four defensible operationalizations:

| Reading | Tally | Verdict |
|---|---|---|
| CV-selected vs baseline (**implemented, scored**) | 0/3 | fires |
| Best roster member (the literal "the roster") | 2/3 | **does not fire** |
| CV-selected, baseline removed from pool | 0/3 | fires |
| CV-selected, symmetric ≥120 min-train | 0/2 | fires |

Three of four preserve the null. The one that flips it is statistically indefensible: best-of-5 fires ~97% of the time under a coin-flip null ($1 - 0.5^5$, assuming independence). **The conservative reading was the one scored — the one that cost the win.** The prose→code narrowing ("the roster" → "the CV-selected member of the roster") was committed 1h41m *before* scoring, so this is a documentation gap rather than a post-hoc choice, and the git history proves the ordering.

## What the 0/3 does and does not license

**Licensed:** "This end-to-end pipeline — rolling-origin CV selection on co-primary MASE + Spearman over a six-model roster — failed to beat a trailing-momentum benchmark on precision@50 at any of three out-of-time cutoffs, $h=24$."

**Not licensed:**

- *"Region-level emergence is momentum-dominated at 24m."* Contradicted by the study's own pre-registered table on 2 of 3 cutoffs, on the mean, and on the co-primary ranking metric.
- *"Region-level emergence is not forecastable at 24m."* Refuted at 6× chance.
- Anything about other horizons. `HORIZON_MONTHS` was never varied.
- Any error-rate claim. The three cutoffs are **nested** (2007-01→2017-12 ⊂ →2020-12 ⊂ →2022-12), and a ≥2/3 majority rule passes **50%** of the time under a coin-flip null. This is a decision rule, not a hypothesis test.

## Accepted and disclosed leakages

- SPECTER2 weights were trained on post-$T_0$ literature.
- UMAP display coordinates are fit on all data (display only; never scored).
- The production novelty threshold is computed against the final frozen map — display-only, not scored. The masked per-cutoff novelty streams are leak-free but have never been scored.
- $\kappa$ is refit once per forecast origin rather than per CV fold; this affects model *selection* only, not the scored $\hat g$.

## Status of everything above

Everything in this file is **exploratory and post-hoc** except the verdict, the full 6 × 3 × {precision@50, Spearman, CSI} table (all three metrics were registered as reported out-of-time metrics, so publishing the whole table discharges an obligation rather than mining it), and the 0.1007 chance rate (definitional). AutoETS was named *after* seeing 18 model × cutoff cells — that multiplicity applies to every comparison involving it.

No repaired tally is a "pre-specified secondary criterion." No secondary criterion was registered.

## What a corrected v1.5 would change

Pre-registered separately, before any re-run — never as a replacement for the frozen result:

1. Remove the baseline from the decision set (fixed comparator, à la Diebold–Mariano / Hansen SPA).
   On this data it changes nothing.
2. Enforce a common fold set across both CV paths; unify the growth denominator and MASE scaler; assert equality at run time. A straight bug fix, not a tuning choice.
3. Select on the **deployed loss** (CV precision@50 / CSI on ≤$T_0$ folds), not MASE + Spearman.
4. Move the unit of replication from cutoff to **region** — paired per-region loss differentials with a block bootstrap over ~1000 regions × cutoffs. Three binary cutoffs will never have power.
5. Add an equal-weight combination to the roster. Winner-take-all selection is the design the M-competitions found inferior, and here it demonstrably was.
