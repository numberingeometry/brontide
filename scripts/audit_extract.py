#!/usr/bin/env python3
"""audit_extract.py -- pull the passages a coder needs out of a stored full text.

Not a coder: it only surfaces windows around selection- and evaluation-related
language so the human coder reads the right part of the methods section rather
than the whole paper. The coding decision, and the verbatim quote, come from the
coder reading these windows (and, when they are not enough, the file itself).

  python scripts/audit_extract.py <arxiv-id> [--width N]
  python scripts/audit_extract.py --all
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXT = REPO / "artifacts" / "audit" / "text"

SELECT = [
    r"\bwe (?:select|choose|chose|selected|pick|picked|adopt|adopted)\b",
    r"\bbest[- ](?:performing|model|method|configuration|setting|score)\b",
    r"\bmodel selection\b", r"\bhyper-?parameter", r"\bgrid search\b", r"\brandom search\b",
    r"\bcross[- ]?validat", r"\bvalidation (?:set|split|data|score)\b", r"\bheld[- ]out\b",
    r"\bfine[- ]tun", r"\boptimal (?:number|value|parameter|configuration|model)\b",
    r"\bwe (?:set|fix|use)\b[^.]{0,60}\b(?:k|K|alpha|threshold|number of (?:topics|clusters))\b",
    r"\bearly stopping\b", r"\bcoherence score\b", r"\bsilhouette\b", r"\belbow\b",
    r"\bAIC\b", r"\bBIC\b", r"\bperplexity\b", r"\bgrid of\b", r"\btuned?\b",
]
REPORT = [
    r"\bprecision@?\s?\d*\b", r"\brecall@?\s?\d*\b", r"\bF1\b", r"\bF-?measure\b",
    r"\baccuracy\b", r"\bAUC\b", r"\bROC\b", r"\baverage precision\b",
    r"\bMAE\b", r"\bRMSE\b", r"\bMAPE\b", r"\bMASE\b", r"\bMSE\b", r"\bSMAPE\b",
    r"\bR\^?2\b", r"\bR2\b", r"\bSpearman\b", r"\bKendall\b", r"\bNDCG\b", r"\bMRR\b",
    r"\bhit rate\b", r"\blog[- ]?loss\b", r"\bCRPS\b", r"\bcoverage\b",
    r"\bevaluation metric", r"\bwe (?:report|evaluate|measure|assess)\b",
    r"\bground truth\b", r"\bgold standard\b", r"\bbaselines?\b",
]
HEADING = re.compile(
    r"(?m)^\s*(?:\d+(?:\.\d+)*\.?\s+)?((?:[A-Z][A-Za-z\- ]{2,60})"
    r"(?:Method|Methods|Methodology|Experiment|Experiments|Experimental|Evaluation|"
    r"Setup|Set-up|Results|Data|Dataset|Model|Approach|Analysis|Metric|Metrics|Validation))\s*$"
)


def windows(txt: str, pats: list[str], width: int) -> list[tuple[int, str]]:
    hits = []
    for p in pats:
        for m in re.finditer(p, txt, flags=re.I):
            hits.append(m.start())
    hits.sort()
    merged: list[tuple[int, int]] = []
    for h in hits:
        lo, hi = max(0, h - width), min(len(txt), h + width)
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return [(lo, txt[lo:hi]) for lo, hi in merged]


def show(pid: str, width: int = 420) -> None:
    f = TEXT / f"{pid}.txt"
    txt = f.read_text(encoding="utf-8", errors="replace")
    print("#" * 100)
    print(f"# {pid}   ({len(txt)} chars)")
    print("#" * 100)
    heads = [m.group(1).strip() for m in HEADING.finditer(txt)]
    print("HEADINGS:", " | ".join(heads[:40]) if heads else "(none matched)")
    print()
    for tag, pats in (("SELECTION", SELECT), ("REPORTING", REPORT)):
        ws = windows(txt, pats, width)
        print(f"----- {tag}: {len(ws)} window(s) -----")
        for lo, w in ws[:40]:
            print(f"[@{lo}] ...{' '.join(w.split())}...")
            print()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if "--all" in sys.argv:
        for f in sorted(TEXT.glob("*.txt")):
            show(f.stem)
    else:
        w = 420
        if "--width" in sys.argv:
            w = int(sys.argv[sys.argv.index("--width") + 1])
        show(sys.argv[1], w)
