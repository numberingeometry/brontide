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

## Clarification — 2026-07-23 (disambiguation of frozen intent; NOT a threshold change; Results still empty)

No frozen threshold moves (K=1024, 64 supers, cutoffs 2017-12/2020-12/2022-12, ĝ 1.5 / 0.75, TOP_N=50, eligibility 20, add-α 1.0, saturating band, horizon 24m, negative criterion ≥2/3 — all intact). The items below pin ambiguous wording and register two previously-unregistered nuisance knobs, **before any scoring commit**, so the scoring run has zero free choices. Independently derived and adversarially verified.

1. **EB shrinkage target (Target §, Forecast §).** "Shrunk toward super-region share" = toward the super's *per-region share level* **P_S/n_S** (n_S = frozen # of micro-regions in that super, from the 64-cut map; same ~1/1024 scale as the region share) — **not** the raw super total P_S (~1/64, ~16× larger), which breaks scale and *inverts* the emerging/declining ranking. "Weight ∝ region size" = **w_r = n_r/(n_r+κ)** with n_r the region's **trailing-12m** count fixed per origin (not the raw monthly count, which drives w→0 and collapses ĝ to the super-region ranking). Identical semantics for the v1 point-forecast shrink toward the 64 super-regions.

2. **κ (EB shrinkage strength) — a fitted nuisance parameter, not a threshold.** κ is empirical-Bayes **estimated**, never hardcoded (config.py holds no κ by design). Estimator, pinned: symmetric **Dirichlet-Multinomial concentration by method-of-moments**, a **single global pool** across the 64 supers, over within-super counts on the trailing-12m window. **Exact closed form** (ref impl `series._estimate_kappa`, verified 2026-07-23 to recover known κ on synthetic DM data): per super S, the Pearson-vs-uniform statistic `G_S = Σ_i (x_i − N_S/n_S)²/(N_S/n_S)` is matched to its DM expectation `E[G_S] = (n_S−1)(N_S + κ·n_S)/(κ·n_S + 1)`; the single global κ solves `Σ_S G_S = Σ_S E[G_S](κ)` (monotone decreasing → unique root). Bound by the lookahead control — refit on **≤ (fold-train-end)** data at every rolling-origin CV fold and every validation cutoff, so no future data enters model selection or the scored ĝ. Not on the accepted-leakage list (it is masked, like PCA/k-means). Run A measured **κ = 3.15** (a gentle shrink — median w_r = 0.988 — reflecting genuinely high within-super dispersion, so the accepted small-breakout false-negative risk is mild in practice). v1.5, disclosed/deferred: per-super κ_S shrunk toward the global κ.

3. **Novelty threshold — registered.** The previously-"calibrated" threshold is **NOVELTY_FIT_QUANTILE = 0.99** of the *fit-set* nearest-centroid distance distribution (dimensionless; leak status = fit set's). Masked run: centroids + quantile from ≤T0 → leak-free, reported per window. Production run: vs the final frozen map → **display-only** (same status as UMAP-on-all-data); not scored and **not** added to the accepted-leakage list.

4. **PCA solver.** "PCA 768→50" is computed by exact streaming **centered-covariance + eigh** (deterministic, batch-order invariant), replacing IncrementalPCA (an order-dependent approximation). Same mathematical object (verified to match full PCA exactly); measured top-50 EVR ≈ 0.688 — a reported diagnostic, never a scored threshold. Determinism is load-bearing for the refit-per-cutoff harness.

5. **Data window — trailing partial month dropped.** The production series ends at **CORPUS_END** = last complete calendar month (a data-window marker mirroring CORPUS_START, not a scored threshold); the partial current month is excluded because its **raw** count would deflate the eligibility floor (shares are ~completeness-invariant; the floor is a raw count). The December validation cutoffs (T0+24m all past) are unaffected. Invariant: CORPUS_END ≥ latest cutoff T0+24m.

6. **Lookahead control — extended.** The frozen control ("PCA + k-means fit ≤T0; later papers nearest-centroid; forecasts from ≤T0") now explicitly also binds **κ and the novelty quantile τ** to ≤T0 (both are fit-set statistics). Realized-side scoring uses the **same ≤T0 κ** as the forecast side (symmetric stabilization). Leak-safety requires the masked run to read/write **cutoff-scoped paths**, never the production all-data memmaps/centroids. The eligibility floor is a **raw** trailing-12m count.

---

## Results

_(appended by `validate.run()` at scoring time — mid-August. Empty until then.)_
