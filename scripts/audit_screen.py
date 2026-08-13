#!/usr/bin/env python3
"""audit_screen.py -- record the abstract-level screen for all 225 candidates.

The screen itself is human judgement applied to title+abstract under the four
criteria frozen in decisions/2026-08-13-select-report-audit-preregistration.md.
This script is the *record* of that judgement, not a re-derivation of it: the
include list and the per-candidate failure code are written here explicitly so
that a reader can check any single call against the stored abstract.

  python scripts/audit_screen.py   ->  artifacts/audit/screen.json
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "audit"

# --------------------------------------------------------------------------- #
# Operational clarification of the screen, fixed BEFORE any full text was read
# and disclosed in the artifact (same register as validation.md's 2026-07-23
# clarification). It resolves what to do when the abstract is silent:
#
#   * S1 (object), S3 (quantitative evaluation), S4 (not a survey) uncertain
#       -> INCLUDE. The coding stage sees the full text and can record
#          has_selection="no"; a paper with no selection step never enters the
#          rate denominator, so including it cannot inflate the rate.
#   * S2 (scholarly corpus) uncertain
#       -> INCLUDE, and resolve from the full text at the coding stage, recording
#          the resolution.
#
# Direction of the rule is conservative: it can only add work, never raise the
# measured rate.
# --------------------------------------------------------------------------- #
CLARIFICATION = (
    "Uncertainty rule, fixed before any full text was read: if S1, S3 or S4 cannot be "
    "settled from the abstract, INCLUDE and let the full-text coding stage settle it "
    "(a paper with no selection step never enters the rate denominator, so this cannot "
    "raise the rate); if S2 (scholarly vs non-scholarly corpus) cannot be settled from "
    "the abstract, INCLUDE and record the resolution at the coding stage."
)

INCLUDE = {
    "1907.09395": "topics/concepts: predicts future links in a keyword co-occurrence network built from published literature (literature-based discovery). Borderline on S1 (units are concept pairs, not single topics); included under the uncertainty rule.",
    "1911.02890": "topics/fields: embeds PACS codes and predicts couplings of physics domains from APS publications. S3 not settled from the abstract; included under the uncertainty rule.",
    "2106.12875": "topics: detection, analysis and forecasting of research topics over a scientific knowledge graph.",
    "2109.06675": "topics: newly added MeSH terms as emergent biomedical topics, with a stated prediction question about their future trend.",
    "2111.03496": "topics: detection of slowly emerging new topics from embedding geometry. S2 (corpus type) not settled from the abstract; included under the uncertainty rule.",
    "2207.12093": "topics/technologies: keyword trajectories in cloud computing publications 2004-2021, with comparative growth statements across topics.",
    "2209.10224": "technologies: ranks patent clusters as emerging technologies, patents being scholarly documents under S2.",
    "2211.01348": "topics/technologies: indicator-based emergent-topic detection in AI, with an explicit evaluation of emergence.",
    "2211.15334": "technologies: technology forecasting with RNNs, explicitly compared against S-curve and other forecasting approaches.",
    "2305.04133": "topics: forecasts scientific topic popularity levels and changes from PubMed publication series and patents.",
    "2403.16157": "topics/keywords: keyword co-occurrence trend analysis toolkit over academic corpora. S3 not settled from the abstract; included under the uncertainty rule.",
    "2408.06839": "topics: hierarchical model of knowledge diffusion paths with stated prediction of knowledge trends. Borderline on S1 (single theory as the case); included under the uncertainty rule.",
    "2411.05930": "topics: neural topic modelling for emerging-trend and weak-signal detection. S2 not settled from the abstract; included under the uncertainty rule.",
    "2503.09894": "topics/fields: knowledge-graph plus LLM mapping of scientific research. S3 not settled from the abstract; included under the uncertainty rule.",
    "2503.23285": "topics/fields: diachronic periodical embeddings used to track evolution of research topics and identify rapidly developing fields.",
    "2504.08743": "topics: dynamic topic analysis of academic journals with a stated quantitative objective (consistency, sparsity, interpretability).",
    "2507.21125": "technologies: LLM pipeline extracting technologies from scientific literature, evaluated on recall/precision of candidate generation.",
    "2508.20514": "topics: topic discovery in scientific literature, evaluated quantitatively against existing topic-discovery methods.",
    "2508.21329": "technologies: predictive framework for technological maturity with S-curve forecasting of indicators.",
    "2510.15801": "technologies: predictive methodology anticipating future disruptive innovations. S2 not settled from the abstract; included under the uncertainty rule.",
    "2601.16990": "topics/keywords: network-based bibliometric analysis library including keyword co-occurrence trends. S3 not settled from the abstract; included under the uncertainty rule.",
    "2606.27394": "topics: clustering + topic modelling + time-series forecasting of technological directions over 127,820 abstracts -- the same pipeline shape as the instrument in this paper.",
}

# Science-of-science-adjacent candidates that still fail a named criterion.
# Everything not listed here and not in INCLUDE fails S1 as a self-description
# (see DEFAULT_OUT below).
NAMED_OUT = {
    "1905.10022": ("S1", "forecasts forward citations of individual patents; the unit is a document, not a topic."),
    "2006.08486": ("S3", "scientometric overview of the literature-based-discovery field; descriptive mapping, no evaluation against a reference."),
    "2009.13059": ("S3", "visual exploration of biomedical 'dark data'; no quantitative evaluation of a detection or forecast."),
    "2011.01103": ("S1", "knowledge-graph construction from scholarly text; the evaluated units are extracted entities and relations, not topics being ranked or forecast."),
    "2103.09431": ("S3", "descriptive bibliometrics of a Master's programme; no evaluation against a reference."),
    "2106.13537": ("S3", "descriptive bibliometric study of one topic (heat waves); no detection or forecasting task is evaluated."),
    "2112.06267": ("S2", "information diffusion on online social networks, not a scholarly corpus."),
    "2302.02267": ("S4", "meta-review of machine-learning methods in crisis management."),
    "2305.11204": ("S4", "literature review of exoplanet habitability methods."),
    "2307.01420": ("S2", "tag prediction on StackExchange community QA posts, not a scholarly corpus."),
    "2312.03769": ("S1", "evaluates LLM-written scientific reviews; the units are reviews, not topics."),
    "2312.06382": ("S1", "curated dataset of scientific software repositories; the units are software projects."),
    "2404.04707": ("S1", "measures gender composition of author teams after a topic emerges; the units are authors, and topic emergence is an input rather than the output."),
    "2409.01739": ("S3", "bibliometric review of the EU ETS literature; descriptive, no evaluation against a reference."),
    "2502.08648": ("S3", "co-word bibliometric map of AI in communication; descriptive, no evaluation against a reference."),
    "2504.13685": ("S1", "fine-tuned LLMs for extracting qualitative insights in a literature review; the units are annotations of documents."),
    "2504.16063": ("S2", "reconstructs full-text news articles from GDELT; a news corpus."),
    "2506.22729": ("S1", "career trajectories of scientists across a paradigm shift; the units are researchers."),
    "2509.07801": ("S1", "benchmark for entity and relation extraction; the units are entities and relations, not topics being detected or forecast."),
    "2509.16323": ("S3", "visualisation tool for science-funding impact; no evaluation of a detection or forecast against a reference."),
    "2604.17421": ("S1", "forecasts the cost of a technology from deployment data; the unit is a cost curve and the corpus is not scholarly documents."),
    "2605.27404": ("S1", "measures research-team size and composition under AI-assisted writing; the units are teams."),
    "2606.12413": ("S4", "pedagogical framework paper describing a platform; no quantitative evaluation of a topic-detection output."),
}

DEFAULT_OUT = (
    "S1",
    "the paper's subject is not the detection, ranking or forecasting of research topics; "
    "the query matched because 'emerging research topic' (or 'research topic') appears in the "
    "abstract as a self-description of the paper's own subfield.",
)


def main() -> int:
    doc = json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))
    cands = doc["candidates"]
    decisions = []
    for c in cands:
        pid = c["id"]
        if pid in INCLUDE:
            decisions.append({
                "id": pid, "title": c["title"], "decision": "IN",
                "criterion": "S1-S4 all pass", "reason": INCLUDE[pid],
            })
        else:
            crit, reason = NAMED_OUT.get(pid, DEFAULT_OUT)
            decisions.append({
                "id": pid, "title": c["title"], "decision": "OUT",
                "criterion": f"{crit} fails", "reason": reason,
            })

    missing = sorted(set(INCLUDE) - {c["id"] for c in cands})
    if missing:
        raise SystemExit(f"include list names ids not in the pool: {missing}")

    n_in = sum(1 for d in decisions if d["decision"] == "IN")
    out = {
        "preregistration": "decisions/2026-08-13-select-report-audit-preregistration.md",
        "clarification": CLARIFICATION,
        "screened": len(decisions),
        "included": n_in,
        "excluded": len(decisions) - n_in,
        "coding_cap": 25,
        "subsampled": n_in > 25,
        "decisions": decisions,
    }
    (OUT / "screen.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"screened {len(decisions)}; IN {n_in}; OUT {len(decisions)-n_in}; cap 25 "
          f"-> subsampling {'REQUIRED' if n_in > 25 else 'not required'}")
    for d in decisions:
        if d["decision"] == "IN":
            print(f"  IN  {d['id']}  {d['title'][:78]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
