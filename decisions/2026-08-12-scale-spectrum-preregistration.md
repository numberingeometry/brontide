---
type: adr
register: dated
status: accepted
opened: 2026-08-12
updated: 2026-08-12
affects: [artifacts/linkage.f64.npy, artifacts/region_shares.parquet, artifacts/cutoffs, decisions]
supersedes: []
---

<!-- PRE-REGISTRATION. Frozen before any of the three scored quantities below was
     computed. Immutable once accepted: if a knob must change, that is a NEW dated
     pre-registration, disclosed as such — never an edit to this file. -->

# 2026-08-12 — pre-registration: the scale spectrum, corpus transportability, and clustering stability

## Context

A new line of work reuses this repo's committed artifacts as an **instrument** rather than
re-opening its forecasting result. Three quantities are to be computed and reported. All three are
scored claims, so all three are frozen here first.

**Nothing scored below has been computed.** What *has* been computed, and is reported here as the
motivation, is a construction check and a structural property of the tree — neither is a scored
quantity:

1. **The balance basis is valid.** The 1,023 ilr balance coordinates built from `linkage.f64.npy`
   (sequential-binary-partition construction, Egozcue–Pawlowsky-Glahn; PhILR's transport of it to a
   dendrogram) satisfy `Psi Psi^T = I` to ≤1.3e-15, `Psi 1 = 0` to ≤2.8e-15, scale-invariance to
   ≤1.1e-14, and reconstruct the closed composition to ≤6.3e-15 relative, on **all four origins**
   (production + the three masked cutoffs). There are **zero non-positive shares** anywhere — add-α
   plus EB shrinkage guarantees strict positivity — so the zero-handling problem that usually
   attends log-ratio methods does not arise on this data.

2. **The Ward tree is strongly unbalanced, so combinatorial depth is not a scale.** A balanced binary
   tree on 1,024 leaves has depth 10; these run 0–14 (production) and 0–16 / 0–15 / 0–14 (masked). At
   production depth 5 the subtree sizes span 3→140 (47×); on the 2017-12 tree, 2→217 (108×).
   `corr(depth, log2 subtree size)` is only −0.699 and −0.584. Two nodes at equal depth are therefore
   not at comparable scale, **and the depth ranges differ across origins, so a depth-indexed exponent
   would not be comparable across the masked cutoffs** — which is exactly the cross-origin stability
   check this design depends on.

Fact 2 is why the scale variable is specified below rather than assumed, and why all three
parameterizations are frozen and reported together instead of one being chosen after the fact.

## Decision

### Common definitions

- **Origins.** All quantities are computed on all four origins: `production` and the masked cutoffs
  `2017-12`, `2020-12`, `2022-12`. Each uses **its own** `linkage.f64.npy`, `region_shares.parquet`
  and `centroids.f32.npy` from its own artifacts directory. The production tree saw all data and its
  results are labelled display-only in this repo's existing sense; **the three masked origins are the
  leak-free anchors** and any claim must hold on them.
- **Balances.** `b_v(t) = Psi_v · log x(t)` with `Psi` the SBP contrast matrix above and `x(t)` the
  vector of 1,024 `share_smoothed` values in month `t`. Because `Psi 1 = 0`, `b` is invariant to the
  monthly total.
- **Increments.** `Δb_v(t) = b_v(t) − b_v(t−1)`. **The scored dispersion is the variance of
  increments, not of levels** — a diffusion coefficient is an increment variance, and balance levels
  are trending and non-stationary, so a level variance would measure trend, not reallocation.
- **Per-node dispersion.** `D_v = Var_t[Δb_v(t)]` over all available months at that origin.

### S1 — the scale spectrum and its exponent

Three parameterizations, **all three reported for every origin, regardless of outcome**:

| | scale variable | model | exponent |
|---|---|---|---|
| **Primary** | merge height `h_v = Z[v,2]` (the ultrametric radius at which the node's two clades join) | `log2 D_v = c − α_h · log2 h_v` | `α_h` |
| Secondary A | combinatorial depth `d_v` (root = 0) | `log2 D_v = c − α_d · d_v` | `α_d` |
| Secondary B | subtree size `s_v` (leaves under the node) | `log2 D_v = c + α_s · log2 s_v` | `α_s` |

The primary is the merge height because in the Avetisov–Bikulov–Kozyrev–Osipov class of ultrametric
diffusions the transition rate is a function of the **ultrametric distance**, and a dendrogram's merge
height *is* that distance. Depth is retained as a secondary only because it is the parameterization
the originating sketch proposed; §Context fact 2 is the reason it is not primary.

Fits are ordinary least squares over the 1,023 nodes, unweighted.

**S1 counts as a finding only if all four hold:**

1. Primary `R² ≥ 0.30` on **all four** origins.
2. `sign(α_h)` identical on all four origins.
3. The three **masked** `α_h` estimates lie within ±25% of their own mean.
4. The observed production `α_h` falls outside the central 95% of the matched-N null (below).

If any fails, the reported result is "no stable scale exponent at this resolution", stated plainly,
and S1 does not appear as a headline in any paper.

**The matched-N null (200 replicates).** The confound to defeat is Murtagh's: ultrametricity and
dispersion both rise with sparsity, and the early years are thin, so a spectrum could be pure
sampling noise. For each replicate, and for each month `t`, redraw the 1,024 region counts from
`Multinomial(N_t, q)` where `N_t` is that month's **observed** total and `q` is the region's
**time-averaged** share over the origin's window. This preserves the corpus-size trajectory and the
region-size distribution while destroying all real dynamics. Then run the identical pipeline —
add-α, EB shrinkage with κ re-estimated by `series._estimate_kappa` on the resampled counts, the same
`Psi`, the same increments, the same regression — and record `α_h`. Seed `0..199`.

### S2 — corpus transportability

For the production origin, let `C` range over the 64 super-regions. Dropping `C` means restricting the
composition to the 1,024 − |C| surviving micro-regions and recomputing every downstream quantity on
that subcomposition.

- **S2a (theorem check, expected to hold exactly).** Every balance `b_v` whose contrast support is
  disjoint from `C` is numerically unchanged by the restriction, to ≤1e-12 relative. This is
  subcompositional coherence; a failure means the implementation is wrong, not that the theorem is.
- **S2b (the empirical claim).** For the *same* surviving regions, the share-based emergence
  statistic `ĝ` (`emergence.g_hat` on `share_smoothed`) **does** change. Scored as the top-50
  symmetric-difference fraction `|T_full Δ T_restricted| / 50`, where both top-50s are taken over
  surviving eligible regions only, and as Kendall's tau-b between the two rankings.

**S2 counts as a finding only if:** S2a holds at ≤1e-12, **and** the median over the 64 restrictions
of the S2b symmetric-difference fraction is **≥ 0.10**. Both the median and the full 64-value
distribution are reported either way.

*Why this is expected to be non-zero and is nonetheless worth scoring:* the idealized share-ratio
ranking is invariant to the corpus total, but this pipeline's stabilizations are not.
`series.py:163` forms add-α shares `p = (C+α)/(M_t+αK)` — `K` changes under restriction — and
`series.py:176` returns `p̂ = w·p + (1−w)·m` with `m = P_S/n_S` a super-region mean and `w` fixed per
origin. Both couple a region's value to which other regions are in the corpus. The size of the effect
is not known in advance, and a small median would make this a footnote rather than a claim.

### S3 — clustering stability

Production centroids, `centroids.f32.npy` (1,024 × 50), perturbed at **the sampling scale actually
present in the data**: for region `r` with trailing-12m count `n_r`, add Gaussian noise of per-
coordinate scale `σ_r / sqrt(n_r)`, with `σ_r` the within-region standard deviation of the PCA-50
coordinates of that region's own papers (computed once from `data/pca50.f32.memmap`). 100 replicates,
seeds `0..99`.

For each replicate, recompute the hierarchy under **Ward** and under **single linkage**, and score the
normalized cophenetic distortion `max_{i,j} |u'(i,j) − u(i,j)| / max u`, plus the survival fraction of
the 64-cluster cut and of the top-50 emerging set.

**S3 counts as a finding only if:** Ward's **median** normalized cophenetic distortion is **≥ 0.10**.
Both linkages' full distributions are reported either way, including the case where single linkage is
*worse* on this data — Carlsson–Mémoli's theorem is about stability under perturbation of the input
metric and does not promise that single linkage produces a more useful map here.

### Reporting commitment

All three results are reported with the numbers above, in whichever direction they come out, in any
paper drawing on this instrument. No quantity defined here may be replaced by a variant chosen after
seeing its value; a variant may be *added*, labelled exploratory, and must not be given a headline.

## Alternatives rejected

- **Depth as the primary scale variable** — rejected on §Context fact 2: the trees are unbalanced
  (47×–108× size spread at fixed depth) and their depth ranges differ across origins, so `α_d` is not
  comparable across the masked cutoffs. Retained as a reported secondary, not dropped, because it is
  what the originating sketch proposed and hiding it would be a silent substitution.
- **Variance of balance levels rather than increments** — rejected: the levels trend, so their
  variance measures trend rather than reallocation, and no diffusion reading is available from it.
- **Choosing one parameterization after seeing all three** — rejected; that is precisely the analyst
  degree of freedom this file exists to remove. All three are primary-or-secondary by name in advance.
- **A permutation null over months** — rejected: shuffling months preserves the marginal dispersion
  that the confound lives in, so it would not test the sparsity artifact. The matched-N multinomial
  null does.
- **Bootstrapping papers and refitting k-means for S3** — rejected for cost, not correctness: it needs
  a full k-means refit per replicate over 2.7M points. Centroid perturbation at the measured sampling
  scale is the tractable proxy and is stated as a proxy. Revive if the tractable version lands near
  the 0.10 threshold, where the proxy's looseness would matter.

## Consequences

- The three scored quantities now have zero free parameters. Anything computed differently is a new
  pre-registration, dated and disclosed.
- Reporting all three parameterizations costs paper space and may produce a mixed picture — accepted,
  because a mixed picture reported is worth more than a clean one selected.
- The S1 criteria are strict enough that a null is a live outcome. That is intended: the fallback
  direction (selection-consistency of benchmarks) needs no empirical luck.
- This freeze is only as strong as its ordering evidence. It is committed **before** the scoring run;
  `git log --follow` on this file must show that commit strictly earlier than the first commit
  carrying any S1/S2/S3 number. Until the repo is pushed, the freeze is local-only and the honest
  wording is "frozen before scoring", not an unqualified "pre-registered".

## Verification

- `python scripts/verify_results.py` → `OK: all 18 published rows reproduce exactly`, and
  `python blueprint/check.py` → exit 0. Neither may change; this file touches no published number.
- `git log --follow --format='%h %ad %s' --date=short -- decisions/2026-08-12-scale-spectrum-preregistration.md`
  establishes the freeze commit.
- `artifacts/validation.md` is untouched above its `## Results` marker; content hash
  `dea6cc0db2ae1e83` still verifies.
