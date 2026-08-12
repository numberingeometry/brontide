---
type: adr
register: dated
status: accepted
opened: 2026-08-12
updated: 2026-08-12
affects: [artifacts/linkage.f64.npy, artifacts/region_shares.parquet, data/papers.parquet, decisions]
supersedes: []
---

<!-- PRE-REGISTRATION. Frozen before any of the scored quantities below was computed.
     Immutable once accepted: a knob change is a NEW dated pre-registration. -->

# 2026-08-12 — pre-registration: the human-hierarchy control

## Context

Two results in this line of work say that hierarchy-indexed statistics inherit their hierarchy's
instability: the Carlsson–Mémoli argument (Ward is not Gromov–Hausdorff stable; measured cophenetic
distortion **0.9737** under perturbation at the data's own sampling scale) and the transport result
(depth-indexed profiles are provably not invariants; swapping the linkage rule moved a coarse-scale
share by **121×**).

Both are demonstrated only against hierarchies derived from the *same embedding* — a perturbed Ward
tree, or single linkage on the same centroids. Neither answers the question a reader will actually
ask: **does the verdict change when the hierarchy is replaced by one that people built?**

That control is available at zero cost and has not been used. `data/papers.parquet` carries
`categories`: **152 primary arXiv categories in 20 top-level groups, 100% coverage, 50.8% of the
2,705,992 papers carrying more than one.** It is human-curated, hierarchical, overlapping, and
defined over exactly the same documents.

**Nothing scored below has been computed.** What is established beforehand, and is descriptive rather
than scored, is the taxonomy's shape as just stated.

## Decision

### Common construction

The leaf set is held fixed at the **existing 1,024 micro-regions**, so that both hierarchies are
comparable objects on the same leaves and the O5 result (two SBP bases differ by an orthogonal $Q$)
applies. Only the tree over those leaves changes.

- **Ward hierarchy $T_W$** — the committed `artifacts/linkage.f64.npy`, unchanged.
- **Category hierarchy $T_C$** — built from the human taxonomy: each micro-region is described by its
  distribution over the 152 primary categories (counted over that region's own papers, using the
  first listed category per paper); regions are then joined by average linkage on the Aitchison
  distance between those category profiles. The 20 top-level arXiv groups are *not* imposed as a cut;
  they are reported as a diagnostic only, so that $T_C$ is derived from the taxonomy rather than
  from a second clustering choice of ours.

Both hierarchies are cut at **64** super-regions, and the full EB pipeline is re-run under each, so
that the shrinkage target $m$, the concentration $\kappa$, the smoothed shares and hence the
emergence statistic all inherit the hierarchy under test. $T_0 = $ 2024-06, horizon months 19–24,
trailing 12 months, eligibility floor on raw trailing counts — all as in the existing harness.

### H1 — how different are the two hierarchies (descriptive, not a criterion)

Report the normalised cophenetic distortion $\max_{i,j}|u_{T_C}(i,j) - u_{T_W}(i,j)| / \max u_{T_W}$,
alongside the $0.9737$ produced by sampling-scale perturbation of $T_W$, and the 64-cut adjusted Rand
index between the two. Reported for context; no threshold attaches.

### H2 — does replacing the hierarchy change the verdict? (scored)

For $k \in \{5, 10, 25, 50, 100, 200, 400, 600\}$, compute the top-$k$ symmetric-difference fraction
$|T^W_k \,\triangle\, T^C_k| / 2k$ between the verdict under $T_W$ and under $T_C$, over regions
eligible on both sides.

**H2 counts as establishing hierarchy-dependence iff the median over $k$ is $\ge 0.10$.** The full
$k$-profile is reported either way.

### H3 — does the margin account survive the change of hierarchy? (scored)

On $T_C$, repeat the margin measurement: the boundary-band count
$N(k) = \#\{r : |g(r) - g_{(k)}| < \operatorname{median}|\Delta g|\}$ against the observed churn under
the same centroid perturbation used for S3 (per-coordinate $\sigma_r/\sqrt{n_r}$, seeds 0–9, all
2,705,992 papers reassigned).

**H3 counts as a replication iff** the Pearson correlation between $N(k)$ and observed churn across
the sweep is $\ge 0.70$ **and** churn at $k=600$ exceeds churn at $k=5$ by at least a factor of 5.
Both numbers reported either way.

### Reporting commitment

All three are reported with the numbers above, in whichever direction they come out. No quantity
defined here may be replaced by a variant chosen after seeing its value; a variant may be added,
labelled exploratory, and given no headline.

## Alternatives rejected

- **Using the 152 categories directly as the region set.** Rejected: it changes the leaf set as well
  as the tree, so any difference confounds "different hierarchy" with "different regions", and the
  O5 orthogonal-change-of-basis result no longer applies. Holding the leaves fixed is what makes the
  comparison a measurement of hierarchy-dependence specifically.
- **Imposing the 20 arXiv groups as the 64-cut.** Rejected: 20 ≠ 64, and forcing the cut would make
  the two hierarchies differ in granularity as well as shape. Groups are reported as a diagnostic.
- **Using all listed categories rather than the first.** Rejected for the frozen statistic because
  50.8% of papers are multi-labelled and a membership weighting is a free parameter; the
  multi-label version may be added later as an explicitly exploratory secondary.
- **Deferring until the OpenAlex join lands.** Rejected: the arXiv taxonomy is already local, costs
  nothing, and answers the question on its own. OpenAlex adds a second, finer, cross-field hierarchy
  and a real coverage boundary, and will carry its own pre-registration.

## Consequences

- The scored quantities have no free parameters left. Anything computed differently is a new,
  dated pre-registration.
- A null on H2 would be a substantive result, not a failure: it would say the verdict is robust to
  the hierarchy in a way the Carlsson–Mémoli argument does not guarantee, and would need explaining.
- H3 is the load-bearing generalisation. If the margin account holds on a human hierarchy too, it is
  a property of top-$k$ decisions rather than of this construction.
- The comparison inherits every limitation of the arXiv taxonomy — it is a submission-time
  self-assignment, not a curated post-hoc indexing like MeSH, and authors game it.

## Verification

- `python scripts/verify_results.py` → `OK: all 18 published rows reproduce exactly`;
  `python blueprint/check.py` → exit 0. This file touches no published number.
- Any script computing H1–H3 must first reimplement the EB pipeline and assert
  $\kappa = 3.147960961354892$ with `max|share_smoothed − committed| = 0.0`.
- `git log --follow` on this file must show its commit strictly before the first commit carrying an
  H1/H2/H3 number.
- `artifacts/validation.md` untouched above its `## Results` marker; `dea6cc0db2ae1e83` still verifies.
