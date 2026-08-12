---
type: adr
register: dated
status: accepted
opened: 2026-08-12
updated: 2026-08-12
affects: [decisions/2026-08-12-scale-spectrum-preregistration.md, artifacts/region_shares.parquet]
supersedes: []
---

<!-- AMENDMENT to a pre-registration. Supersedes ONE clause (S2b) of the same day's
     freeze; S1, S2a and S3 are untouched and still bind as written. Disclosed as a
     new dated pre-registration rather than an edit, per that file's own rule. -->

# 2026-08-12 — S2b is re-specified on the realized growth ratio

## Context

Earlier today [`2026-08-12-scale-spectrum-preregistration.md`](2026-08-12-scale-spectrum-preregistration.md)
froze three scored quantities. Its S2b clause reads:

> the share-based emergence statistic `ĝ` (`emergence.g_hat` on `share_smoothed`) **does** change.
> Scored as the top-50 symmetric-difference fraction …

Writing the runner exposed a gap in that wording. `emergence.g_hat(forecast_means,
trailing12m_share)` takes **forecast** means as its numerator. Restricting the corpus to a
subcomposition changes the whole share series, so a faithful recomputation would have to **refit
the forecasts under every restriction** — 64 restrictions × 1,024 series, and `statsforecast` is not
installed in the working environment.

The obvious shortcut — hold the committed forecast means fixed and renormalize only the
trailing-12m denominator — is **not** conservative in a known direction. Under restriction the
numerator's horizon months and the denominator's trailing window are renormalized by *different*
monthly factors, so freezing one side can inflate or deflate the churn depending on the region.
An unsigned bias is not something to disclose and move past; it is a reason to score a different
quantity.

S1 was scored before this amendment was written; its result is recorded in the session note and is
unaffected either way. **No S2 quantity had been computed when this file was committed.**

## Decision

**S2b is re-specified on the realized growth ratio, and only S2b.** S1, S2a and S3 stand exactly as
frozen.

For the production origin, define for each micro-region `r`

```
g_real(r) = mean of share_smoothed over horizon months 19..24 after T0
            / trailing-12m mean of share_smoothed at T0
```

— the same functional form as `ĝ`, with the model's forecast replaced by what actually happened.
`emergence.g_hat`'s horizon window (`EMERGENCE_HORIZON_MONTHS = (19, 24)`), the trailing-12m
denominator, and the eligibility floor (`ELIGIBILITY_FLOOR` on the raw trailing-12m count) are
unchanged; this is `validate._score_cutoff`'s `realized_g`, which is already implemented and already
part of the scored harness.

`T0` is set so the realized window exists inside the production series: `T0 = 2024-06`, giving
horizon months through 2026-06 = `CORPUS_END`.

Restrictions, churn statistics and the decision threshold are otherwise **unchanged from the
original clause**: `C` ranges over the 64 super-regions; both top-50s are taken over surviving
eligible regions only; the scored quantity is the top-50 symmetric-difference fraction
`|T_full Δ T_restricted| / 50` plus Kendall's tau-b; and

> **S2 counts as a finding only if** S2a holds at ≤1e-12 **and** the median over the 64 restrictions
> of the symmetric-difference fraction is **≥ 0.10**.

The median and the full 64-value distribution are reported either way.

## Alternatives rejected

- **Disclosed convention: hold forecasts fixed, renormalize the denominator.** Rejected on the
  unsigned-bias argument above. Disclosing a bias whose *direction* is unknown does not let a reader
  correct for it, so the disclosure would not do the work disclosure is for.
- **Install `statsforecast` and refit under all 64 restrictions.** Rejected on cost-versus-question,
  not on correctness — it is the most faithful reading of the original wording. But it turns the
  cheapest probe into the most expensive one, and it answers a *worse* question: refitting entangles
  the invariance of the statistic with the behaviour of a particular forecaster. **Revive if the
  claim is ever made about a deployed forecasting pipeline rather than about the statistic**, which
  is what a follow-up on PreScience-style targets would require.
- **Silently editing the original clause.** Rejected: that file's own rule is that a knob change is a
  new dated pre-registration, disclosed as such. The whole line of work argues that machine-checkable
  ordering is what makes a freeze mean anything, so quietly rewriting one would be self-refuting.
- **Dropping S2b and keeping only the S2a theorem check.** Rejected: S2a alone is an implementation
  test that must pass by construction. The empirical half — that the *practised* statistic moves when
  the corpus boundary moves — is the entire transportability claim.

## Consequences

- S2b now needs no model and no GPU: it runs from `artifacts/region_shares.parquet` alone.
- The claim S2b can support narrows, and correctly so. It is now about **the emergence statistic
  itself**, not about brontide's forecasting pipeline — "this measurement is corpus-dependent",
  not "these forecasts are corpus-dependent". Any paper must use the narrower wording.
- Because `g_real` uses realized rather than forecast shares, the invariance question is isolated
  from forecaster error, which is what makes the result transferable to other groups' pipelines —
  the reason the claim generalizes at all.
- The freeze now spans two files. Anything citing it must cite both, and the S2b threshold lives
  here.

## Verification

- `python scripts/verify_results.py` → `OK: all 18 published rows reproduce exactly`;
  `python blueprint/check.py` → exit 0. This file touches no published number.
- `git log --follow` on this file must show its commit **before** the first commit carrying any S2
  number, and **after** the S1 result was recorded — the ordering this amendment relies on.
- `artifacts/validation.md` untouched above its `## Results` marker; `dea6cc0db2ae1e83` still verifies.
