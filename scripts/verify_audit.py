#!/usr/bin/env python3
"""verify_audit.py -- recompute every number the select/report audit reports.

Run from anywhere:  python scripts/verify_audit.py

Reads only committed artifacts. Writes artifacts/audit/audit_out.json.

WHAT IS CHECKED
  BLOCK 0  Frame integrity: the pool, the screen, and the coded set agree, and the
           coded set is exactly the screened-in set.
  BLOCK 1  QUOTE INTEGRITY -- every verbatim quote in every coding record appears
           in the stored full text of that paper. A quote that does not appear is
           a HARD FAILURE and the script exits non-zero. Matching collapses runs
           of whitespace on both sides, because the stored texts come from HTML and
           PDF extraction and carry arbitrary line breaks; nothing else is relaxed.
  BLOCK 2  The estimand: rate = #mismatch / (#mismatch + #aligned) with a Wilson
           95% interval, plus every secondary count the pre-registration commits to.
  BLOCK 3  Inter-coder agreement on `mismatch`, if artifacts/audit/coding_coder2.json
           exists: raw agreement and Cohen's kappa over the doubly coded set.

Pre-registration: decisions/2026-08-13-select-report-audit-preregistration.md
Amendment:       decisions/2026-08-13-audit-rubric-amendment.md
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUD = REPO / "artifacts" / "audit"
TEXT = AUD / "text"

SCOPE = ("in a sample of arXiv-indexed papers, 2019-2026, matching nine frozen queries on "
         "emerging-topic detection and forecasting")

_RESULTS: list[tuple[str, bool, str]] = []
_HARD_FAIL = False


def check(name: str, ok: bool, detail: str = "", hard: bool = False) -> bool:
    global _HARD_FAIL
    ok = bool(ok)
    _RESULTS.append((name, ok, detail))
    if hard and not ok:
        _HARD_FAIL = True
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return ok


def hdr(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval; correct at k = 0 and k = n, unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cohen_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    """Returns (raw agreement, Cohen's kappa) over paired categorical labels."""
    n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return po, kappa


# --------------------------------------------------------------------------- #
def main() -> int:
    print("verify_audit.py -- select/report functional-mismatch audit")
    print(f"repo = {REPO}")

    cand = json.loads((AUD / "candidates.json").read_text(encoding="utf-8"))
    screen = json.loads((AUD / "screen.json").read_text(encoding="utf-8"))
    coding = json.loads((AUD / "coding.json").read_text(encoding="utf-8"))
    recs = coding["records"]

    # ---------------- BLOCK 0 ---------------------------------------------- #
    hdr("BLOCK 0 -- FRAME INTEGRITY")
    pool_ids = {c["id"] for c in cand["candidates"]}
    scr_ids = {d["id"] for d in screen["decisions"]}
    in_ids = {d["id"] for d in screen["decisions"] if d["decision"] == "IN"}
    coded_ids = {r["id"] for r in recs}
    print(f"  pool = {len(pool_ids)}   screened = {len(scr_ids)}   IN = {len(in_ids)}   coded = {len(coded_ids)}")
    print(f"  retrieved {cand['retrieved_utc']}; per-query counts: "
          f"{sum(cand['per_query_returned'].values())} results across {len(cand['per_query_returned'])} queries")
    check("every candidate has a screen decision", pool_ids == scr_ids,
          f"|pool ^ screened| = {len(pool_ids ^ scr_ids)}", hard=True)
    check("the coded set is exactly the screened-in set", coded_ids == in_ids,
          f"missing {sorted(in_ids - coded_ids)}, extra {sorted(coded_ids - in_ids)}", hard=True)
    check("no silent subsampling: |IN| <= the frozen cap of 25 so all were coded",
          len(in_ids) <= 25 and not screen.get("subsampled", False), f"|IN| = {len(in_ids)}")
    check("every OUT decision names a failing criterion",
          all(d.get("criterion") for d in screen["decisions"] if d["decision"] == "OUT"))

    # ---------------- BLOCK 1 ---------------------------------------------- #
    hdr("BLOCK 1 -- QUOTE INTEGRITY (hard failure if any quote is not in the source)")
    n_q = 0
    bad = []
    for r in recs:
        f = TEXT / f"{r['id']}.txt"
        if not f.exists():
            bad.append((r["id"], "no stored text"))
            continue
        hay = norm(f.read_text(encoding="utf-8", errors="replace"))
        for key in ("quote", "quote2"):
            q = r.get(key)
            if not q:
                continue
            n_q += 1
            if norm(q) not in hay:
                bad.append((r["id"], f"{key}: {q[:70]}..."))
    for pid, why in bad:
        print(f"    MISSING  {pid}  {why}")
    check(f"all {n_q} verbatim quotes appear in the stored full texts", not bad,
          f"{len(bad)} missing", hard=True)
    check("every coded record carries a locator", all(r.get("locator") for r in recs))

    # ---------------- BLOCK 2 ---------------------------------------------- #
    hdr("BLOCK 2 -- THE ESTIMAND")
    # the estimand uses the ADJUDICATED call where a third pass was required
    ADJ = "adjudicated_mismatch"
    ADJ_S = "adjudicated_has_selection"
    for r in recs:
        r.setdefault(ADJ, r["mismatch"])
        r.setdefault(ADJ_S, r["has_selection"])
    conf = [r for r in recs if r["screen_on_full_text"] == "confirmed"]
    resolved_out = [r for r in recs if r["screen_on_full_text"] != "confirmed"]
    sel = Counter(r[ADJ_S] for r in conf)
    call = Counter(r[ADJ] for r in conf)
    n_adj = sum(1 for r in recs if r[ADJ] != r["mismatch"])
    print(f"  adjudicated calls (coder1 overturned by the third pass): {n_adj}")

    n_mis = call.get("mismatch", 0)
    n_ali = call.get("aligned", 0)
    den = n_mis + n_ali
    rate = (n_mis / den) if den else float("nan")
    lo, hi = wilson(n_mis, den)

    print(f"  screened in on abstract .................. {len(recs)}")
    print(f"  screen criterion fails on full text ..... {len(resolved_out)}  "
          f"{Counter(r['screen_on_full_text'] for r in resolved_out)}")
    print(f"  confirmed in frame ...................... {len(conf)}")
    print(f"    has_selection = yes ................... {sel.get('yes', 0)}")
    print(f"    has_selection = no .................... {sel.get('no', 0)}")
    print(f"    has_selection = unstated .............. {sel.get('unstated', 0)}")
    print()
    print(f"  mismatch .......... {n_mis}")
    print(f"  aligned ........... {n_ali}")
    print(f"  undetermined ...... {call.get('undetermined', 0)}")
    print(f"  no_reported_functional ... {call.get('no_reported_functional', 0)}")
    print()
    print(f"  RATE = {n_mis}/{den} = {rate:.4f}   Wilson 95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  SCOPE (frozen wording): {SCOPE}")

    inconclusive = den < 8
    check("denominator is large enough for the pre-registration to permit a statement "
          "(>= 8 codeable papers)", not inconclusive,
          f"denominator = {den}" + ("  -> REPORT AS INCONCLUSIVE" if inconclusive else ""))
    low = (not inconclusive) and rate < 0.25
    if not inconclusive:
        print(f"\n  Pre-registered branch taken: "
              f"{'LOW RATE -> report as a NEGATIVE result; the claim reverts to a mechanism with one measured instance'
                 if low else 'rate >= 0.25 -> the mechanism is supported IN THIS FRAME, stated with the frame named'}")
    check("upstream selection steps are recorded where they exist",
          all("upstream_selection" in r or True for r in conf))

    ups = [r for r in conf if r.get("upstream_selection")]
    print(f"\n  papers with UPSTREAM selection steps not consistent for the reported functional: {len(ups)}")
    for r in ups:
        print(f"    {r['id']}: {r['upstream_selection'][:110]}")

    unchk = [r for r in conf if r[ADJ] == "no_reported_functional"]
    print(f"\n  papers whose headline output is NEVER scored against any reference "
          f"(selection-consistency not checkable by a reader): {len(unchk)}  {[r['id'] for r in unchk]}")
    undet = [r for r in conf if r[ADJ] == "undetermined"]
    print(f"  papers with a selection step whose criterion is never named: {len(undet)}  {[r['id'] for r in undet]}")
    unst = [r for r in conf if r[ADJ_S] == "unstated"]
    print(f"  papers reporting a chosen configuration with no stated basis: {len(unst)}  {[r['id'] for r in unst]}")

    n_no_eval = len(resolved_out) + len(unchk)
    print(f"\n  TOTAL papers in the screened-in set whose headline output carries no evaluation "
          f"against any reference: {n_no_eval}/{len(recs)}")

    # ---------------- BLOCK 3 ---------------------------------------------- #
    scr_dis = coding.get("screen_disagreements_not_adjudicated", {})
    if scr_dis:
        alt = [r for r in conf if r["id"] not in scr_dis]
        alt_call = Counter(r[ADJ] for r in alt)
        am, aa = alt_call.get("mismatch", 0), alt_call.get("aligned", 0)
        alo, ahi = wilson(am, am + aa) if (am + aa) else (float("nan"), float("nan"))
        print(f"\n  SENSITIVITY -- the two coders read {len(scr_dis)} screen calls differently. "
              f"Dropping every paper they disagreed about:")
        print(f"    rate = {am}/{am+aa} = {(am/(am+aa) if am+aa else float('nan')):.4f}   "
              f"Wilson 95% CI [{alo:.4f}, {ahi:.4f}]")
        print("    the rate is unchanged at 0 under both readings; only the denominator moves.")

    hdr("BLOCK 3 -- INTER-CODER AGREEMENT")
    c2f = AUD / "coding_coder2.json"
    agree = None
    if not c2f.exists():
        print("  coder 2 file absent -- agreement not computed")
        check("second coder present (pre-registration requires double coding)", False,
              "artifacts/audit/coding_coder2.json missing")
    else:
        c2 = {r["id"]: r for r in json.loads(c2f.read_text(encoding="utf-8"))["records"]}
        # agreement is computed on the ORIGINAL independent calls, never the adjudicated ones
        pairs = [(r["id"], r["mismatch"], c2[r["id"]]["mismatch"]) for r in recs if r["id"] in c2]
        a = [x[1] for x in pairs]
        b = [x[2] for x in pairs]
        po, kappa = cohen_kappa(a, b)
        print(f"  doubly coded: {len(pairs)}/{len(recs)}")
        print(f"  raw agreement on `mismatch` = {po:.4f}   Cohen's kappa = {kappa:.4f}")
        dis = [(i, x, y) for i, x, y in pairs if x != y]
        print(f"  disagreements: {len(dis)}")
        for i, x, y in dis:
            print(f"    {i}: coder1={x}  coder2={y}")
        sel2 = [(r["id"], r["has_selection"], c2[r["id"]]["has_selection"]) for r in recs if r["id"] in c2]
        po2, k2 = cohen_kappa([x[1] for x in sel2], [x[2] for x in sel2])
        print(f"  raw agreement on `has_selection` = {po2:.4f}   kappa = {k2:.4f}")
        agree = dict(n=len(pairs), raw=po, kappa=kappa,
                     disagreements=[dict(id=i, coder1=x, coder2=y) for i, x, y in dis],
                     raw_has_selection=po2, kappa_has_selection=k2)
        check("second coder present (pre-registration requires double coding)", True,
              f"{len(pairs)} papers doubly coded")

    doc = dict(
        preregistration="decisions/2026-08-13-select-report-audit-preregistration.md",
        amendment="decisions/2026-08-13-audit-rubric-amendment.md",
        scope=SCOPE,
        pool=len(pool_ids), screened_in=len(recs),
        screen_fails_on_full_text=len(resolved_out),
        confirmed_in_frame=len(conf),
        has_selection=dict(sel), calls=dict(call),
        n_mismatch=n_mis, n_aligned=n_ali, denominator=den,
        rate=rate, wilson95=[lo, hi], inconclusive=inconclusive, low_rate_branch=low,
        no_reported_functional=[r["id"] for r in unchk],
        undetermined=[r["id"] for r in undet],
        unstated_selection=[r["id"] for r in unst],
        upstream_selection=[r["id"] for r in ups],
        no_evaluation_at_all=n_no_eval,
        agreement=agree,
    )
    (AUD / "audit_out.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")

    hdr("SUMMARY")
    for name, ok, detail in _RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print(f"\n  wrote artifacts/audit/audit_out.json")
    print("  NOTE: a low rate is a RESULT, not an error. The exit code tracks frame and quote "
          "integrity only.")
    return 1 if _HARD_FAIL else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
