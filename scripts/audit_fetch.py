#!/usr/bin/env python3
"""audit_fetch.py -- build the frozen sample frame for the select/report audit.

Frame is fixed by decisions/2026-08-13-select-report-audit-preregistration.md.
This script only *retrieves*; it makes no coding decision.

  stage 1:  python scripts/audit_fetch.py meta
            -> artifacts/audit/candidates.json   (deduped, sorted pool + per-query counts)

  stage 2:  python scripts/audit_fetch.py text
            -> artifacts/audit/text/<id>.txt     (full text for every SCREENED-IN id)
               reads the include list from artifacts/audit/screen.json

Deterministic: the query list, the date window, the dedup rule and the sort are
frozen in the ADR and hard-coded here. Re-running reproduces the same pool up to
arXiv adding new matching papers after the retrieval date, which is recorded.
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "audit"
TEXT = OUT / "text"

# --- FROZEN: the nine queries of the pre-registration ----------------------- #
QUERIES = [
    'abs:"emerging topic detection"',
    'abs:"emerging topics" AND abs:"prediction"',
    'abs:"emerging research topics"',
    'abs:"scientific trends" AND abs:"predict"',
    'abs:"emerging technology" AND abs:"forecasting"',
    'abs:"science of science" AND abs:"forecast"',
    'abs:"topic popularity" AND abs:"predict"',
    'abs:"research topics" AND abs:"forecasting"',
    'abs:"emerging trends" AND abs:"scientific"',
]
DATE_WINDOW = "submittedDate:[201901010000 TO 202608130000]"
MAX_PER_QUERY = 200
API = "https://export.arxiv.org/api/query"
UA = {"User-Agent": "brontide-audit/1.0 (metascience audit; contact via repo)"}


def _get(url: str, timeout: int = 60, tries: int = 4) -> bytes:
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception as e:                                    # noqa: BLE001
            last = e
            time.sleep(4 + 6 * k)
    raise RuntimeError(f"GET failed after {tries}: {url}: {last}")


def _q(s: str) -> str:
    return urllib.parse.quote(s, safe=":[]")


# --------------------------------------------------------------------------- #
def stage_meta() -> None:
    import urllib.parse  # noqa: F401  (kept local; _q uses the module-level import)

    OUT.mkdir(parents=True, exist_ok=True)
    per_query: dict[str, int] = {}
    recs: dict[str, dict] = {}

    for q in QUERIES:
        # arXiv's API wants '+' for spaces inside search_query; %20 is rejected (400).
        sq = urllib.parse.quote(q + " AND " + DATE_WINDOW, safe=':[]').replace("%20", "+")
        url = f"{API}?search_query={sq}&start=0&max_results={MAX_PER_QUERY}"
        raw = _get(url).decode("utf-8", "replace")
        entries = re.findall(r"<entry>(.*?)</entry>", raw, flags=re.S)
        per_query[q] = len(entries)
        for e in entries:
            m = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", e)
            if not m:
                continue
            full = m.group(1)
            base = full.split("v")[0]
            title = re.search(r"<title>(.*?)</title>", e, flags=re.S)
            summ = re.search(r"<summary>(.*?)</summary>", e, flags=re.S)
            pub = re.search(r"<published>([^<]+)</published>", e)
            cats = re.findall(r'<category term="([^"]+)"', e)
            rec = {
                "id": base,
                "versioned_id": full,
                "title": html.unescape(" ".join((title.group(1) if title else "").split())),
                "abstract": html.unescape(" ".join((summ.group(1) if summ else "").split())),
                "published": pub.group(1) if pub else "",
                "categories": sorted(set(cats)),
                "queries": [],
            }
            if base in recs:
                recs[base]["queries"].append(q)
            else:
                rec["queries"] = [q]
                recs[base] = rec
        print(f"  {per_query[q]:>4}  {q}", flush=True)
        time.sleep(3.2)                      # arXiv asks for >=3s between calls

    pool = [recs[k] for k in sorted(recs)]
    doc = {
        "frame": "arXiv API, 9 frozen queries x submittedDate:[2019-01-01, 2026-08-13]",
        "preregistration": "decisions/2026-08-13-select-report-audit-preregistration.md",
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_results_per_query": MAX_PER_QUERY,
        "per_query_returned": per_query,
        "pool_size": len(pool),
        "candidates": pool,
    }
    (OUT / "candidates.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"\npool = {len(pool)} unique arXiv ids -> {OUT/'candidates.json'}")


# --------------------------------------------------------------------------- #
_TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>", flags=re.S | re.I)
_ANY = re.compile(r"<[^>]+>")


def _html_to_text(raw: bytes) -> str:
    s = raw.decode("utf-8", "replace")
    s = _TAG.sub(" ", s)
    s = re.sub(r"</(p|div|h1|h2|h3|h4|li|tr|section)>", "\n", s, flags=re.I)
    s = _ANY.sub(" ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\xa0]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def _pdf_to_text(raw: bytes) -> str:
    import io
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader           # type: ignore
        except ImportError:
            return ""
    try:
        r = PdfReader(io.BytesIO(raw))
        return "\n\n".join((p.extract_text() or "") for p in r.pages)
    except Exception:                              # noqa: BLE001
        return ""


def fetch_text(base_id: str, versioned: str) -> tuple[str, str]:
    """Return (source_tag, text). Empty text means NO-TEXT."""
    for tag, url in (
        ("arxiv-html", f"https://arxiv.org/html/{versioned}"),
        ("arxiv-html-v1", f"https://arxiv.org/html/{base_id}v1"),
        ("ar5iv", f"https://ar5iv.labs.arxiv.org/html/{base_id}"),
    ):
        try:
            raw = _get(url, timeout=90, tries=2)
        except Exception:                          # noqa: BLE001
            continue
        txt = _html_to_text(raw)
        # arXiv returns a stub page when no HTML build exists
        if len(txt) > 12000 and "no html for" not in txt[:400].lower():
            return tag, txt
    try:
        raw = _get(f"https://arxiv.org/pdf/{versioned}", timeout=120, tries=2)
        txt = _pdf_to_text(raw)
        if len(txt) > 8000:
            return "pdf", txt
    except Exception:                              # noqa: BLE001
        pass
    return "none", ""


def stage_text() -> None:
    TEXT.mkdir(parents=True, exist_ok=True)
    cands = {c["id"]: c for c in json.loads((OUT / "candidates.json").read_text(encoding="utf-8"))["candidates"]}
    screen = json.loads((OUT / "screen.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in screen["decisions"] if r["decision"] == "IN"]
    print(f"fetching full text for {len(ids)} screened-in papers")
    prov = {}
    for i, pid in enumerate(ids, 1):
        dest = TEXT / f"{pid}.txt"
        if dest.exists() and dest.stat().st_size > 8000:
            prov[pid] = {"source": "cached", "chars": dest.stat().st_size}
            print(f"  [{i}/{len(ids)}] {pid}  cached", flush=True)
            continue
        tag, txt = fetch_text(pid, cands[pid]["versioned_id"])
        if txt:
            dest.write_text(txt, encoding="utf-8")
        prov[pid] = {"source": tag, "chars": len(txt)}
        print(f"  [{i}/{len(ids)}] {pid}  {tag}  {len(txt)} chars", flush=True)
        time.sleep(3.2)
    (OUT / "text_provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    n_none = sum(1 for v in prov.values() if v["source"] == "none")
    print(f"\ndone. NO-TEXT attrition: {n_none}/{len(ids)}")


if __name__ == "__main__":
    import urllib.parse
    stage = sys.argv[1] if len(sys.argv) > 1 else "meta"
    if stage == "meta":
        stage_meta()
    elif stage == "text":
        stage_text()
    else:
        raise SystemExit("usage: audit_fetch.py [meta|text]")
