---
type: adr
status: accepted
opened: 2026-08-13
affects:
  - scripts/verify_selectionconsistency.py
  - scripts/verify_rank_relevant.py
---

# 2026-08-13 — novelty gate on the selection/decision mathematics

The project's rule is **no novelty claim without the search, and record what was searched and what
was found.** This is that record for the four mathematical claims the paper carries. It is written
against the register `RELATED-WORK.md` uses: `owned` = established elsewhere, cite and never
re-claim; `open` = no hit across repeated searches, weaker than proof of absence; `occupied` = a
claim killed by the gate.

**Outcome up front: three of the four are `owned` or `occupied`, and one name collides with an
established term. The mathematics must be presented as a restatement with full hypotheses and full
attribution, not as new. What survives as contribution is the conjunction, the domain, and the
measurement.**

## What was searched

Web search, 2026-08-13, four queries:

1. `"selection consistency" benchmark model selection score consistent for different functional than
   reported leaderboard`
2. `Osband principle elicitable functional scoring rule consistent for T gives no guarantee for
   different functional reported evaluation mismatch`
3. `decomposition model selection regret "value of adaptivity" versus selection loss best fixed
   model oracle per-fold identity benchmark`
4. Earlier, in the 2026-08-12 literature review: top-k elicitability, Gneiting consistency,
   Dietterich on cross-validated comparison.

Every arXiv identifier below was verified against the arXiv API (title, date, authors read back),
not quoted from a search snippet.

## Claim 1 — the alignment condition: `owned`

*Claim.* Selection is guaranteed non-harmful iff the reported functional is a non-decreasing
function of the selection functional.

**Owned by the elicitation literature.** This is the order-theoretic content of consistency:

- **Gneiting (2011)**, *Making and Evaluating Point Forecasts* (arXiv:0912.0902) — choose the
  functional first, then a loss strictly consistent for it; other losses "may lead to misguided
  conclusions".
- **Fissler & Ziegel (2016)**, *Higher order elicitability and Osband's principle*
  (arXiv:1503.08123, *Ann. Statist.* 44) — **verified**: published 2015-03-27, Fissler and Ziegel.
  Osband's principle gives necessary conditions for a scoring function to be strictly consistent
  for a functional `T`; consistency for `T` carries no implication for a different functional.
- **Osband (1985)** — the revelation-principle observation that validity requires a *bijection*
  between the reported object and the elicited one, which is the same requirement as ours.
- **Frongillo & Kash (2015)** — separability of scoring rules; the answer is in general negative.

**Nothing here is ours.** The paper must state the condition, attribute it, and use it. What is
ours is only the *instantiation*: an explicit witness at the instrument's own dimensions showing
that the pair actually deployed, (global rank correlation, precision@50) at `N = 1023`, `k = 50`,
violates it.

## Claim 2 — the name "selection-consistency": `occupied` (name collision)

**`model selection consistency` is an established term with a different meaning.** Weinstein &
Miller, *Bayesian data selection* (arXiv:2109.02712) — **verified**: published 2021-09-06 —
formalises "data selection consistency, nested data selection consistency, model selection
consistency, and nested model selection consistency" in the classical sense: the procedure
recovers the true model asymptotically.

That is not what this paper means. Using the bare phrase would collide with a term of art in
statistics.

**Decision.** Always qualify it in the paper as **selection–report consistency**, defined
explicitly as a property of a *pair* `(S_sel, T_rep)`, with a footnote naming the collision. Never
write "selection consistency" unqualified.

## Claim 3 — the `A − Λ` decomposition: `occupied`, and recently

*Claim.* `E[R(m̂)] − max_m E[R(m)] = A − Λ`, where `A` is the value of adaptivity (what choosing
per-cutoff could win) and `Λ` is the selection loss, so selection pays iff `Λ < A`.

**Occupied.** Shihab, Al Ahsan & Swaqeeb, *Opportunity Is Not Realizability: Selection-Valid
Diagnostics for Multi-LLM Routing* (arXiv:2608.08265) — **verified**: published 2026-08-08 —
operationalises exactly this decomposition in a different domain. It benchmarks against a
train-selected **best fixed model**, defines a **dataset-level oracle** and an **item-level
oracle**, and reports a **rescue/loss decomposition** relative to the fixed baseline whose net gain
is rescue minus loss. That is `A − Λ` under other names, five days before this ADR.

**Consequence.** The identity is arithmetic and cannot be owned by anyone, but the *framing* is now
demonstrably not new. The paper must cite arXiv:2608.08265 as a concurrent, independent instance in
another domain, and drop any language implying the decomposition is introduced here. It is used
here because it is the right tool, and because it is arithmetic on a published table rather than a
model fit — which remains a fair thing to say.

**A neighbour, also recent.** Li, *UpliftBench: Revealing Outcome-Regime and Objective Mismatch in
Uplift Evaluation* (arXiv:2608.00915) — **verified**: published 2026-08-02 — is "objective
mismatch in evaluation" in the uplift-modelling domain: the same shape of problem, independently
found. Cite it; do not claim the shape.

## Claim 4 — the instrument/decision dichotomy: `open`, but stated narrowly

*Claim.* Perturbations of a cluster-indexed, share-derived reported statistic fall into two classes
governed by different theorems: instrument perturbations act only through the region-varying
residual `ε` and are therefore margin-bounded however large `‖δ‖` is; decision perturbations
substitute the map itself, no margin constrains them, and the reachable damage is the spread of the
candidate class, attained.

No hit across repeated searches for this *as a classification of robustness criteria for
cluster-indexed bibliometric claims*. But its two halves are separately unremarkable: the
"a common factor cannot reorder" half is the gauge argument any log-ratio analyst would make, and
the "spread of the candidate class is attained" half is immediate from Claim 1. **Its status is
`open` in the weak sense of no hit found, and its value is not novelty but that it explains a
measured inversion:** three criteria violated at 11×–97× that move nothing, and one that consumes
the whole spread.

The honest register for the paper is the one `RELATED-WORK.md` already uses: *the contribution is
the conjunction and the domain*, plus the measurement. Not the theorems.

## What this changes in the draft

1. Rename to **selection–report consistency** throughout, with the collision footnoted.
2. Attribute the alignment condition to Gneiting / Osband / Fissler–Ziegel / Frongillo–Kash, and
   present it as a restatement with hypotheses made explicit for this setting.
3. Cite arXiv:2608.08265 for the `A − Λ` decomposition as concurrent work, and remove any
   introduction-shaped framing of it.
4. Cite arXiv:2608.00915 as the neighbouring "objective mismatch" result.
5. Keep the dichotomy, stated as an organising theorem for this claim class, with no novelty claim
   attached to either half.
