---
type: adr
status: accepted
opened: 2026-08-13
supersedes: none
amends: 2026-08-13-select-report-audit-preregistration.md
affects:
  - artifacts/audit/coding.json
  - scripts/verify_audit.py
---

# 2026-08-13 — amendment: two cases the audit rubric did not cover

**Written after reading the 22 methods sections, and before `artifacts/audit/coding.json` was
committed.** That ordering is disclosed here rather than hidden: this is an amendment forced by
contact with the material, in the same register as
`2026-08-12-s2b-respecified-on-realized-growth.md`, which was likewise written after the runner
exposed a specification gap. The frozen sample frame, the screen, and the estimand are **not**
touched. What changes is how two situations the rubric is silent about are coded.

**Both changes move the measured mismatch rate DOWN or leave it unchanged. Neither can raise it.**
That is stated up front because an amendment written after seeing the data is only defensible if
its direction is against the author's interest.

## Gap 1 — a paper with several selection steps

The rubric says to record "`T_sel` — the functional the selection criterion targets", in the
singular. Several audited papers have more than one.

The clearest case is arXiv:2606.27394, which selects a **clustering algorithm** by *silhouette
score*, then a **topic model** by *coherence*, then a **forecasting model** by *RMSE*, and reports
RMSE. Under the theory all three selections propagate into the headline number, and the first two
are plainly not consistent for it.

**Rule adopted.** Code the **terminal** selection — the last selection step before the headline
result — as `T_sel`. Record any earlier ones in a new field `upstream_selection`, reported as a
secondary count and never folded into the rate.

*Direction.* The terminal selection is the one nearest the reported quantity and therefore the one
most likely to be aligned with it. Coding it rather than the worst upstream one can only **lower**
the measured rate.

## Gap 2 — a paper whose headline output is never evaluated

The rubric's three calls are `aligned`, `mismatch`, `undetermined`, where `undetermined` means "the
methods section does not permit the call". None of them fits a paper that **selects a configuration
using a quantitative criterion and then ships a headline output that is never scored against
anything**.

Two audited papers are in exactly this position:

- **arXiv:2209.10224** selects a per-patent classifier with auto-sklearn to maximise classification
  score, reports its accuracy (0.6737), and then delivers a **ranked list of emerging technologies
  and leading companies** built by aggregating those per-patent predictions. That ranking — the
  paper's actual product — is compared with no reference at all.
- **arXiv:2411.05930** (BERTrend) tunes BERTopic hyperparameters and merge thresholds, and states
  plainly that no quantitative evaluation exists: *"there are currently no established metrics for
  comparing weak signal detection performance within large volumes of data."*

Reading these as `aligned` because an internal component's metric happens to match itself would
make the audit vacuous. Reading them as `mismatch` would assert a disagreement between two
functionals when only one exists.

**Rule adopted.** A fourth call, **`no_reported_functional`**: the paper has a selection step, but
its headline output carries no evaluated functional to compare the selection criterion against.
These are **excluded from the rate denominator** and reported as their own count.

*Direction.* If these papers were instead coded `mismatch` — the reading the theory would most
naturally invite — the rate would rise. Excluding them **lowers** it.

*Why this count matters on its own.* A `no_reported_functional` paper is not a milder problem than
a mismatch. It is a paper whose central claim **cannot be checked for selection-consistency by any
reader**, because there is no reported functional to check against. The audit should say so in
those words.

## Gap 3 (bookkeeping) — screen criteria resolved on the full text

The screen's uncertainty rule, recorded in `artifacts/audit/screen.json` before any full text was
read, includes borderline papers and defers the call to the coding stage. Coding therefore records
a `screen_on_full_text` field per paper: `confirmed`, or the criterion that in fact fails. Papers
that fail a screen criterion on the full text are **excluded from the rate** and counted as their
own category. This is the pre-specified rule being executed, not a new one; the field is added so
the exclusion is visible rather than silent.

## What does not change

- The nine frozen queries, the date window, the pool of 225, the deterministic ordering.
- The four screen criteria and the 22/203 screen decisions already recorded.
- The estimand: `rate = #mismatch / (#mismatch + #aligned)` with a Wilson 95% interval.
- The two frozen ambiguity rules (multiple reported metrics → `aligned`; validation-split selection
  on the same metric → `aligned`).
- The commitment to report the rate either way, including the explicit low-rate branch and the
  "fewer than 8 codeable papers → inconclusive" branch.
- The frozen scope sentence.
