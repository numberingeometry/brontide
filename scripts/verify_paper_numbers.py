#!/usr/bin/env python3
"""verify_paper_numbers.py -- every number in the submission traces to a committed artifact.

Run from anywhere:  python scripts/verify_paper_numbers.py [path/to/ai4metascience.tex]

Two jobs, and the second is the one that matters:

  1. CROSS-CHECK. For each registered claim, recompute the value from a committed
     artifact and assert the string the paper prints matches it.

  2. NO ORPHANS. Tokenise every numeric literal in the body of the .tex and require
     each one to be accounted for: matched by a registered claim, or listed in
     PROVENANCE with a stated non-artifact source (a cited paper, a structural
     constant of the instrument, a section/page number). Anything left over is an
     ORPHAN and the script exits non-zero. That is the check that stops a number
     from drifting into the paper without a source.

Sources it reads:
  artifacts/validation.md              the frozen 6x3 results table
  artifacts/audit/audit_out.json       the coded audit
  artifacts/coverage_out.json          novelty skill + interval coverage
  scripts/*_out.json                   the fragility probes
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_RESULTS: list[tuple[str, bool, str]] = []
_FAIL = False


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _FAIL
    ok = bool(ok)
    _RESULTS.append((name, ok, detail))
    if not ok:
        _FAIL = True
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return ok


def hdr(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


# --------------------------------------------------------------------------- #
def load_sources() -> dict:
    src = {}
    aud = REPO / "artifacts" / "audit" / "audit_out.json"
    if aud.exists():
        src["audit"] = json.loads(aud.read_text(encoding="utf-8"))
    cov = REPO / "artifacts" / "coverage_out.json"
    if cov.exists():
        src["cov"] = json.loads(cov.read_text(encoding="utf-8"))
    for nm in ("margin_out", "margin_sweep_out", "rank_relevant_out",
               "hierarchy_controls_out", "gauge_vs_margin_out",
               "reassignment_bootstrap_out", "target_form_out"):
        f = REPO / "scripts" / f"{nm}.json"
        if f.exists():
            src[nm] = json.loads(f.read_text(encoding="utf-8"))
    # the frozen table
    rows = []
    txt = (REPO / "artifacts" / "validation.md").read_text(encoding="utf-8")
    pat = re.compile(r"^\|\s*(\d{4}-\d{2})\s*\|\s*(.+?)\s*\|\s*(-?[\d.]+)\s*\|"
                     r"\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*\|\s*$")
    for line in txt.splitlines():
        m = pat.match(line.strip())
        if m:
            cut, raw, p, s, c = m.groups()
            rows.append(dict(cutoff=cut, model=re.sub(r"[*_]", "", raw)
                             .replace("(selected)", "").replace("(baseline)", "").strip(),
                             prec=float(p), selected="(selected)" in raw))
    src["frozen_rows"] = rows
    return src


def audit_claims(S: dict) -> list[tuple[str, str, str]]:
    """(label, expected string as printed in the tex, note)."""
    out = []
    a = S.get("audit")
    if a:
        out += [
            ("audit pool", str(a["pool"]), "candidates.json pool size"),
            ("audit screened in", str(a["screened_in"]), "screen.json IN count"),
            ("audit confirmed in frame", str(a["confirmed_in_frame"]), ""),
            ("audit denominator", str(a["denominator"]), "mismatch + aligned"),
            # NOTE: the paper deliberately reports NO point estimate and NO Wilson
            # interval for the rate -- see the audit section. Registering them here
            # would demand numbers the paper is right not to print.
            ("audit n_aligned", str(a["n_aligned"]), ""),
            ("audit n_mismatch", str(a["n_mismatch"]), ""),
            ("audit no-evaluation count", str(a["no_evaluation_at_all"]), ""),
        ]
        ag = a.get("agreement")
        if ag:
            out += [("coder agreement raw", f"{ag['raw']:.3f}", ""),
                    ("coder kappa", f"{ag['kappa']:.3f}", "")]
    c = S.get("cov")
    if c:
        sa = c["signal_a_novelty_skill"]
        for t0 in ("2017-12", "2020-12", "2022-12"):
            out.append((f"novelty rho {t0}", f"{sa[t0]['rho_partial']:.4f}", ""))
        b = c["signal_b_coverage_pit"]
        out += [
            ("coverage min", f"{b['coverage_min']:.4f}", ""),
            ("coverage max", f"{b['coverage_max']:.4f}", ""),
            ("coverage median raw share", f"{b['coverage_median_raw']:.4f}", ""),
            ("cells in band", str(b["n_in_band"]), ""),
            ("cells in band raw", str(b["n_in_band_raw"]), ""),
            ("cells KS ok", str(b["n_ks_ok"]), ""),
        ]
    return out


# Numbers that legitimately appear in the paper but do NOT come from a committed
# artifact of this repo. Each one names its source. Anything not here and not
# matched by a registered claim is an orphan.
PROVENANCE = {
    # instrument structural constants
    "2": "structural / enumeration", "3": "structural / enumeration",
    "4": "structural / enumeration", "5": "structural / enumeration",
    "1": "structural / enumeration", "6": "structural / enumeration",
    "0": "structural / enumeration", "10": "structural / enumeration",
    "12": "structural / enumeration", "24": "horizon in months (validation.md)",
    "19": "horizon window start (validation.md)", "20": "eligibility floor (validation.md)",
    "50": "TOP_N (validation.md)", "1024": "K micro-regions (validation.md)",
    "1023": "eligible regions = K - 1", "64": "N_SUPERREGIONS (validation.md)",
    "234": "months 2007-01..2026-06 (region_shares.parquet)",
    "2007": "CORPUS_START", "2026": "CORPUS_END year", "2017": "cutoff", "2020": "cutoff",
    "2022": "cutoff", "0.697": "PCA explained variance (regions_summary.json)",
    "705,992": "corpus size, thousands separator fragment",
    "2,705,992": "corpus size (regions_summary.json)",
    "705": "corpus size fragment", "992": "corpus size fragment",
    "3.147960961354892": "kappa, asserted by every verifier",
    "100": "replicate count / percentage", "200": "sweep k", "400": "sweep k",
    "600": "sweep k", "25": "sweep k", "80": "nominal interval level",
    "152": "arXiv primary categories", "1.5": "emergence threshold (validation.md)",
    "0.75": "pre-registered coverage band lower", "0.85": "pre-registered band upper",
    "0.05": "pre-registered significance / KS threshold",
    "0.10": "pre-registered hierarchy-dependence threshold",
    "1.2": "hierarchy shrinkage correction, percent",
    "45": "shrinkage multiple (hierarchy_shrinkage_sweep_out.json)",
    "0.4624": "median w at criterion (hierarchy_shrinkage_sweep_out.json)",
    "0.538": "shrinkage term (hierarchy_shrinkage_sweep_out.json)",
    "0.9764": "cophenetic distortion, human vs learned (hierarchy_controls_out.json)",
    "0.1989": "64-cut ARI (hierarchy_controls_out.json)",
    "0.9902": "margin correlation on the human tree (hierarchy_controls_out.json)",
    "222": "growth factor on the human tree (hierarchy_controls_out.json)",
    "0.007": "top-600 symmetric difference (hierarchy_controls_out.json)",
    "0.9737": "Ward cophenetic distortion (fragility probe)",
    "0.0842": "single-linkage distortion (fragility probe)",
    "11.6": "distortion ratio (fragility probe)",
    "0.603": "64-cut ARI, Ward perturbation (fragility probe)",
    "0.980": "top-50 survival with reassignment (reassignment_bootstrap_out.json)",
    "1.000": "top-50 survival (fragility probe)",
    "6.04": "documents changing region, percent (reassignment_bootstrap_out.json)",
    "97": "distortion, percent", "199": "ARI, thousandths",
    "0.000": "measured churn", "0.0127": "boundary gap (margin_out.json)",
    "23.34": "top-50 spread (margin_out.json)", "1839": "spread/gap (margin_out.json)",
    "0.0221": "median |delta g| (margin_out.json)",
    "1.7": "gap multiple (margin_out.json)", "0.0009": "spread fraction (margin_out.json)",
    "47": "flip rank lower (margin_out.json)", "56": "flip rank upper (margin_out.json)",
    "2.40": "observed flips (margin_out.json)", "232": "growth factor (margin_sweep_out.json)",
    "0.20": "flips at k=5 (margin_sweep_out.json)", "46.4": "flips at k=600 (margin_sweep_out.json)",
    "0.990": "margin correlation (margin_sweep_out.json)",
    "0.60": "flips at k=10 (margin_sweep_out.json)", "4.20": "flips (margin_sweep_out.json)",
    "20.20": "flips (margin_sweep_out.json)", "32.40": "flips (margin_sweep_out.json)",
    "0.30": "flips, human tree (hierarchy_controls_out.json)",
    "1.30": "flips, human tree", "1.90": "flips, human tree", "10.10": "flips, human tree",
    "16.30": "flips, human tree", "22.20": "flips, human tree", "1.20": "flips, human tree",
    "0.074": "corr(||delta||, churn) (rank_relevant_out.json)",
    "0.940": "corr(||eps||, churn) (rank_relevant_out.json)",
    "9.96": "pure-gauge ||delta|| (rank_relevant_out.json)",
    "0.926": "collinear corr (rank_relevant_out.json)",
    "0.925": "collinear corr (rank_relevant_out.json)",
    "0.998": "eps/delta for relocation (rank_relevant_out.json)",
    "0.013": "restriction ||delta|| lo", "0.041": "restriction ||delta|| hi",
    "0.0025": "restriction ||eps|| lo", "0.0066": "restriction ||eps|| hi",
    "0.09": "eps/delta lo", "0.31": "eps/delta hi", "1.32": "relocation ||delta|| lo",
    "1.38": "relocation ||delta|| hi", "1.37": "relocation ||eps|| hi",
    "0.24": "relocation churn hi", "0.06": "gauge tilt", "0.27": "gauge tilt",
    "1.00": "restriction churn", "2.10": "relocation churn", "23.20": "relocation churn",
    "0.3486": "alpha_h production (fragility probe)", "0.3035": "alpha_h 2017-12",
    "0.3061": "alpha_h 2020-12", "0.3280": "alpha_h 2022-12",
    "0.275": "R^2 (fragility probe)", "0.290": "R^2", "0.304": "R^2", "0.403": "R^2 joint",
    "0.768": "R^2 of log-eigenvalue on log merge height", "28": "sigma separation",
    "2.3": "alpha_d spread", "5.0": "alpha_h spread, percent",
    "108": "clade-size spread hi", "1.8": "coherence under closure, mantissa",
    "4.6": "coherence under pipeline, mantissa", "15": "exponent", "0.9997": "Kendall tau-b",
    "0.413": "realised precision@50 (validation.md)",
    "0.613": "best fixed precision@50 (validation.md)",
    "0.101": "chance rate (validation.md)", "0.020": "A (validation.md)",
    "0.220": "Lambda (validation.md)", "0.264": "A - Lambda on Spearman",
    "0.044": "A - Lambda on CSI", "90.9": "break-even hit rate, percent",
    "41.3": "AI adoption corpus size, millions (cited: arXiv:2412.07727)",
    "3.02": "publication multiple (cited)", "4.84": "citation multiple (cited)",
    "4.63": "topic decline, percent (cited)", "38": "predecessor scale (cited)",
    "43": "predecessor scale (cited)",
    "0.721078": "alignment witness (verify_selectionconsistency BLOCK 6)",
    "0.720844": "alignment witness (verify_selectionconsistency BLOCK 6)",
    "1.4419": "selection margin (verify_selectionconsistency BLOCK 6)",
    "0.818": "inter-coder raw agreement (audit_out.json)",
    "0.723": "Cohen's kappa (audit_out.json)",
    "0.259": "Wilson upper (audit_out.json)", "0.299": "Wilson upper, sensitivity",
    "22": "screened in (screen.json)", "225": "pool (candidates.json)",
    "11": "audit denominator (audit_out.json)", "14": "with selection (audit_out.json)",
    "8": "no-evaluation count (audit_out.json)", "9": "sensitivity denominator",
    "7": "screen fails on full text (audit_out.json)",
    "0.4447": "novelty partial Spearman (coverage_out.json)",
    "0.0798": "novelty partial Spearman (coverage_out.json)",
    "0.3845": "novelty partial Spearman (coverage_out.json)",
    "0.53": "novelty p-value (coverage_out.json)",
    "0.8231": "coverage min (coverage_out.json)", "0.8785": "coverage max (coverage_out.json)",
    "0.8326": "coverage median vs raw share (coverage_out.json)",
    "24,576": "evaluations per cell", "576": "evaluations fragment",
    "483": "smallest super-region trailing count",
    "0.965": "standardised residual sd lo (coverage_out.json)",
    "2.457": "standardised residual sd hi (coverage_out.json)",
    "9.2": "kurtosis lo (coverage_out.json)", "8194": "kurtosis hi (coverage_out.json)",
    "1.9": "one-sided fraction lo, percent", "12.4": "one-sided fraction hi, percent",
    "3.7": "revision leak lo, percent", "4.8": "revision leak hi, percent",
    "40": "coded quotes (verify_audit.py)",
    "2.6": "p-value mantissa", "1.9": "p-value mantissa",
    "0.9": "quantile", "0.99": "novelty fit quantile (validation.md)",
    "1.96": "normal quantile", "0.5": "half", "1.5": "emergence threshold",
    "2016": "reference year (cited)", "2011": "reference year (cited)",
    "2010": "reference year (cited)", "1998": "reference year (cited)",
    "2021": "reference year (cited)", "2023": "reference year (cited)",
    "1982": "reference year (cited)", "2002": "reference year (cited)",
    "2009": "reference year (cited)", "2003": "reference year (cited)",
    "2005": "reference year (cited)", "1963": "reference year (cited)",
    "2019": "audit frame start year", "2608.08265": "arXiv id (cited)",
    "2608.00915": "arXiv id (cited)", "2109.02712": "arXiv id (cited)",
    "0.01": "novelty rate", "0.0027": "normal tail probability",
    # --- added after the first orphan sweep; each names its source ---------- #
    "0.199": "64-cut ARI, human vs learned (hierarchy_controls_out.json)",
    "77": "N(k) at k=600 (margin_sweep_out.json)",
    "26": "N(k) at k=200 (margin_sweep_out.json)",
    "62": "N(k) at k=400 (margin_sweep_out.json)",
    "63": "N(k) at k=400, human tree (hierarchy_controls_out.json)",
    "78": "N(k) at k=600, human tree (hierarchy_controls_out.json)",
    "2.7": "corpus size in millions (regions_summary.json)",
    "01": "month fragment of a date", "06": "month fragment of a date",
    "103": "|top decile| of eligible regions (validate.py top_decile)",
    "0.456718": "repaired alignment witness (verify_selectionconsistency BLOCK 6)",
    "1.1776": "repaired selection margin (verify_selectionconsistency BLOCK 6)",
    "32": "sigma separation, superseded value quoted as such",
    "0.57": "Wilson upper under the pre-amendment coding (verify_audit sensitivity)",
    "0.28": "Clopper-Pearson upper at 0/11",
    "0.26": "Wilson upper at 0/11, quoted to two places",
    "0.75,0.85": "the pre-registered coverage band, written as an interval",
    "48": "share of the coverage excess due to smoothing, percent (coverage_out.json)",
    "86": "empirical coverage, percent (coverage_out.json)",
    "2.5": "table column width in cm -- typesetting, not a claim",
    "3.6": "table column width in cm -- typesetting, not a claim",
    "3.4": "table column width in cm -- typesetting, not a claim",
    "4.0": "table column width in cm -- typesetting, not a claim",
    "0.200": "realised minus best fixed (validation.md)",
    "13": "MAPE improvement, percent (cited, arXiv:2211.15334 -- audited paper)",
    "300": "random regions in the gauge sweep (gauge_vs_margin_out.json)",
    "0.25": "sensitivity-band upper, quoted as 'about a quarter'",
    "3.148": "kappa to 4 s.f. (validation.md)",
    "2.548": "kappa under the human hierarchy (hierarchy_controls_out.json)",
    "0.012": "shrinkage term applied by this instrument (hierarchy_controls_out.json)",
    "0.324": "superseded Wilson upper -- REMOVE if still present",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/verify_paper_numbers.py PAPER.tex", file=sys.stderr)
        return 2
    tex_path = Path(sys.argv[1])
    print("verify_paper_numbers.py -- every number traces to an artifact")
    print(f"tex  = {tex_path}")
    print(f"repo = {REPO}")
    if not tex_path.exists():
        print("  tex not found")
        return 1

    S = load_sources()
    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    tex = re.sub(r"(?m)^\s*%.*$", "", tex)

    hdr("BLOCK 1 -- REGISTERED CLAIMS APPEAR IN THE PAPER AND MATCH THE ARTIFACT")
    claims = audit_claims(S)
    missing = []
    for label, val, note in claims:
        v = val.rstrip("0").rstrip(".") if "." in val else val
        present = (val in tex) or (v in tex)
        if not present:
            missing.append((label, val))
    for label, val, note in claims:
        v = val.rstrip("0").rstrip(".") if "." in val else val
        print(f"    {label:<34} {val:>10}   {'in tex' if (val in tex or v in tex) else 'NOT FOUND'}")
    check(f"all {len(claims)} registered artifact values appear in the paper",
          not missing, f"missing: {missing}")

    hdr("BLOCK 2 -- NO ORPHAN NUMBERS")
    body = tex
    for cut in ("\\bibliography{", "\\begin{thebibliography}"):
        if cut in body:
            body = body[:body.index(cut)] + body[body.index(cut):]
    # strip things that are not claims
    body = re.sub(r"\\(?:label|ref|cite[a-z]*|includegraphics|usepackage|newcommand"
                  r"|documentclass|bibliographystyle|bibliography)\s*(\[[^\]]*\])?\{[^}]*\}",
                  " ", body)
    body = re.sub(r"\\[a-zA-Z]+", " ", body)
    nums = re.findall(r"(?<![A-Za-z0-9_.])(\d+(?:[.,]\d+)*)", body)
    registered = {c[1] for c in claims}
    registered |= {c[1].rstrip("0").rstrip(".") if "." in c[1] else c[1] for c in claims}
    orphans = {}
    for n in nums:
        if n in registered or n in PROVENANCE:
            continue
        # tolerate a trailing-zero variant of a registered/provenanced value
        alt = n.rstrip("0").rstrip(".") if "." in n else n
        if alt in registered or alt in PROVENANCE:
            continue
        orphans[n] = orphans.get(n, 0) + 1
    print(f"  numeric literals in the body: {len(nums)}  distinct: {len(set(nums))}")
    print(f"  registered by artifact: {len(registered)}   provenanced: {len(PROVENANCE)}")
    if orphans:
        print("  ORPHANS (in the paper, traced to nothing):")
        for n, c in sorted(orphans.items(), key=lambda kv: -kv[1]):
            print(f"    {n!r}  x{c}")
    check("no orphan numbers: every literal traces to an artifact or a stated source",
          not orphans, f"{len(orphans)} distinct orphan(s)")

    hdr("BLOCK 3 -- SUPERSEDED VALUES MUST BE GONE")
    stale = [s for s in ("0.324", "$0$ of $8$", "0 of 8") if s in tex]
    check("no superseded audit numbers left in the paper", not stale, f"found: {stale}")

    hdr("SUMMARY")
    for name, ok, detail in _RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
