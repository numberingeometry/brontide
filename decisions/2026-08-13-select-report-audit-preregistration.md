---
type: adr
status: accepted
opened: 2026-08-13
affects:
  - artifacts/audit/
  - scripts/audit_fetch.py
  - scripts/verify_audit.py
---

# 2026-08-13 — pre-registration: a coded audit of select/report functional mismatch

**Frozen before any methods section was read.** This file is committed before
`artifacts/audit/coding.json` exists and before any full text is fetched. Everything below —
the sample frame, the screen, the coding rubric, the estimand, and the reporting commitment —
is fixed at this commit. If any of it must change, that is a new dated ADR that says so, and
the change is disclosed in the paper.

## Context

The working claim is a **mechanism**: a benchmark that selects among candidate models using a
score strictly consistent for functional `T_sel`, and reports accuracy on a different functional
`T_rep`, carries no guarantee on the reported quantity, and the reachable damage is the spread
of the candidate class rather than anything margin-bounded.

The evidence to date is *one* verified instance (this project's own harness: select on Spearman
rank correlation, report precision@50) plus a set of unverified candidates in the emerging-topic
literature. The current draft concedes this in writing: *"establishing a rate requires coding
methods sections across a defined sample, which we have not done."*

This ADR defines that sample and that coding, so the concession can be replaced by a measured
rate over a stated frame — or so the claim can be withdrawn if the rate is low.

## What has already been looked at, and what has not

Honesty about the pre-registration boundary matters more than the appearance of purity.

- **Already seen at this commit:** the arXiv API result *counts* per query, and the size of the
  deduplicated union (225). Nothing else. No title, no abstract, no full text was read.
- **Not seen:** any candidate paper's title, abstract, methods section, or evaluation protocol.

The distinction that matters: a mismatch call requires reading a methods section. Counts of
search hits carry no information about the rate, so freezing the rubric after seeing them
cannot have tuned it.

## Sample frame — Stratum A (the rate)

**Source.** The arXiv API (`export.arxiv.org/api/query`). Chosen over Scopus / Web of Science /
Dimensions, which cover the scientometrics literature far better, because the arXiv API is
open, its results are reproducible by a reader with no subscription, and its full texts can be
stored so that **every coded quote is machine-checkable against the source**. This is a
convenience frame and every statement of the resulting rate must say so.

**Queries (frozen, verbatim).** Nine `search_query` strings, each conjoined with the date
window `submittedDate:[201901010000 TO 202608130000]`:

1. `abs:"emerging topic detection"`
2. `abs:"emerging topics" AND abs:"prediction"`
3. `abs:"emerging research topics"`
4. `abs:"scientific trends" AND abs:"predict"`
5. `abs:"emerging technology" AND abs:"forecasting"`
6. `abs:"science of science" AND abs:"forecast"`
7. `abs:"topic popularity" AND abs:"predict"`
8. `abs:"research topics" AND abs:"forecasting"`
9. `abs:"emerging trends" AND abs:"scientific"`

**Pool construction (deterministic).** Take up to 200 results per query; strip version suffixes;
take the set union; sort ascending as strings by arXiv identifier. Measured union size at this
commit: **225**. The realised pool is written to `artifacts/audit/candidates.json` with its
retrieval date and the per-query counts.

**Screen (title + abstract only).** A paper is **IN** iff all four hold:

- **S1 — object.** The units being detected, ranked, forecast or classified are *research
  topics, fields, clusters, concepts, technologies, or keywords* — not individual papers,
  authors, venues, citations of a single paper, or social-media content.
- **S2 — corpus.** The data are *scholarly documents* (papers, patents, grants, preprints).
  Social-media, news and web corpora are OUT.
- **S3 — quantitative evaluation.** The paper reports at least one numerical evaluation of its
  own output against some reference — held-out data, later data, expert labels, or a curated
  ground-truth list. Purely descriptive mapping/visualisation papers are OUT.
- **S4 — not a survey.** Reviews, surveys, position papers, and dataset-only papers are OUT.

Screening decisions are recorded for **every one of the 225** candidates, with the failing
criterion named. No candidate is dropped silently.

**Coding cap.** If more than **25** papers pass the screen, code a deterministic pseudo-random
subsample of 25: `random.Random(20260813).sample(sorted(included_ids), 25)`. The subsampling and
the number passed over are reported. If 25 or fewer pass, code all of them.

**Attrition.** A paper whose full text cannot be retrieved (no arXiv HTML, no ar5iv HTML, no
extractable PDF text) is coded `NO-TEXT`, excluded from every rate, and counted in the reported
attrition. Full texts are stored under `artifacts/audit/text/`.

## Sample frame — Stratum B (supplementary, NOT part of the rate)

The five systems this project's related-work section already names, coded with the same rubric
and reported **separately and explicitly outside the rate**, because they were selected by
relevance rather than by the frame: CSET/ETO Map of Science line (Klavans, Boyack & Murdick
2020), PreScience v2 (arXiv:2602.20459), AutoCluster/AutoTrendAnalysis (arXiv:2607.22641),
the PeerJ CS 2875 (2025) emerging-topic evaluation, and the JOI 2019 emerging-research-topics
line. Any of these that also appears in Stratum A stays in Stratum A and is not double-counted.

## Coding rubric

Read the **methods / experimental-setup / evaluation section**. Abstracts and introductions are
insufficient and may not be used to make a call.

For each paper record five fields, each with a **verbatim quote** and a locator (section name
or heading) that a reader can check:

1. **`has_selection`** — Does the paper choose among two or more candidate configurations
   (models, algorithms, hyperparameters, thresholds, embeddings) using a *quantitative
   criterion computed from data*, and then carry the winner forward into the headline result?
   - `yes` — an explicit selection rule is stated.
   - `no` — a single fixed method is evaluated; no choice is made from data.
   - `unstated` — the paper reports results for a configuration that must have been chosen
     (e.g. "we set k = 20", "our best model"), but never says on what basis.
2. **`T_sel`** — the functional the selection criterion targets, quoted. Recorded at the level
   of *what the criterion is a function of*: a point-accuracy loss (MAE, RMSE, MAPE, MASE), a
   rank/ordering criterion (Spearman, Kendall, NDCG), a set criterion (precision@k, recall@k,
   F1, accuracy), a threshold-free classification criterion (AUC, AP), a likelihood/probabilistic
   score (log-loss, CRPS, perplexity), or a qualitative/manual criterion.
3. **`T_rep`** — the functional the *headline* result reports, quoted. The headline is the
   quantity in the abstract or the paper's summary table; if several, the first one stated.
4. **`mismatch`** — the call:
   - `aligned` — `T_rep` is the same functional as `T_sel`, or is a strictly monotone function
     of it, so the selection criterion is consistent for what is reported.
   - `mismatch` — `T_sel` and `T_rep` are different functionals of the predictive distribution
     and neither determines the other. *This is the alignment condition of the theorem, not a
     judgement about whether the mismatch mattered.*
   - `undetermined` — the methods section does not permit the call.
5. **`quote`** — the verbatim sentence(s) justifying 1–4, copied character-for-character from
   the stored text, plus the locator.

**Two ambiguity rules, frozen now because they will otherwise be decided by whichever answer
looks better:**

- *Multiple reported metrics.* If a paper reports several metrics and one of them equals
  `T_sel`, the paper is **`aligned`**. The claim is about a guarantee existing, not about
  emphasis. This is the conservative direction: it can only lower the measured rate.
- *Selection on a validation split of the same metric.* Selecting on metric `M` computed on a
  validation split and reporting `M` on a test split is **`aligned`**. Selection-consistency is
  about the functional, not about the split.

## Double coding

Every coded paper is coded **twice, independently**:

- **Coder 1** — the primary coder, working from the stored full text.
- **Coder 2** — an independent agent given only (i) this rubric and (ii) the stored full text.
  It does not see coder 1's call, and it is not told what the paper's thesis is or what result
  is wanted.

Agreement on `mismatch` is computed over the doubly-coded set and **reported as a number**
(raw agreement and Cohen's κ). Disagreements are adjudicated in a third pass that sees both
codes and the text; the adjudicated code is the one used, and every adjudication is recorded
with its reason.

## Estimand and uncertainty

**Primary estimand.** Among papers that have a selection step
(`has_selection == "yes"`) *and* a determinable call
(`mismatch != "undetermined"`), the fraction coded `mismatch`:

```
rate = #mismatch / (#mismatch + #aligned)
```

Reported with a **Wilson 95% interval**, because the denominator is small and the normal
approximation is unusable near 0 or 1.

**Secondary counts, all reported:** `has_selection == "no"`, `has_selection == "unstated"`,
`mismatch == "undetermined"`, and `NO-TEXT` attrition. The `unstated` count is a finding in its
own right: a paper that reports a chosen configuration without saying how it was chosen cannot
be checked for selection-consistency by any reader.

**Scope sentence, frozen.** Any statement of the rate must name the frame:
*"in a sample of arXiv-indexed papers, 2019–2026, matching nine frozen queries on
emerging-topic detection and forecasting"* — never *"in the field"*, never *"in the
literature"*, and never with the arXiv frame silently dropped.

## Commitment to report either way

The rate is reported whatever it is.

- A **high** rate supports the mechanism being common in this frame, and is reported with the
  frame named.
- A **low** rate (say below 0.25) is reported as a **negative result** in the same sentence
  that introduces the mechanism, and the paper's claim reverts to what one measured instance
  supports: that the failure mode is *reachable and unbounded when it occurs*, not that it is
  *prevalent*. The dichotomy theorem is unaffected either way — it is about what a class of
  perturbation can do, not about how often anyone does it.
- A denominator too small to say anything (fewer than 8 codeable papers) is reported as
  **inconclusive**, with the counts, and the draft's existing concession stands.

No outcome licenses dropping the audit from the paper.

## What this audit cannot establish

- It cannot establish that a mismatch *harmed* any audited paper's conclusions. That requires
  each paper's data. The audit measures the **antecedent** of the theorem, not its consequence.
- It cannot speak for the scientometrics journals (Scientometrics, JOI, QSS), which the frame
  does not reach.
- It cannot distinguish "the authors chose badly" from "the authors chose well and reported a
  different thing for good reasons". Alignment is a property of the *pair*, not a verdict on
  anyone's competence, and the write-up must not read as one.

## Verification

`scripts/verify_audit.py` re-derives every reported number from `artifacts/audit/coding.json`
and, for each coded paper, asserts the recorded quote appears **byte-for-byte** in the stored
text file. A quote that does not appear is a hard failure. The rate, the Wilson interval, and
the inter-coder agreement are recomputed from the coding file rather than read from prose.
