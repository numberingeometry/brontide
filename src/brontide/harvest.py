"""harvest.py -- fetch metadata for papers that arrived AFTER the corpus was built.

    py -3 -m brontide.harvest --from 2026-07-01 --to 2026-09-04
    py -3 -m brontide.harvest --from 2026-08-01 --focus          # focus categories only
    py -3 -m brontide.harvest --from 2026-08-01 --openalex       # non-arXiv too

Writes newline-delimited JSON to ``data/harvest/<source>-<from>-<to>.jsonl`` and is
resumable: a day already written is skipped, so an interrupted run costs only the day it
was on.

## Why the arXiv API and not the Kaggle snapshot

``ingest.py`` reads a ~4 GB full-corpus Kaggle snapshot. That is the right tool for a
bootstrap and the wrong one for "what appeared last month": it re-downloads 2.7M papers to
obtain 26,000.

## Why the API and not OAI-PMH

OAI-PMH is the obvious choice for incremental harvesting and it is the wrong one here,
because its ``from``/``until`` range is on **datestamp** -- last modified -- not submission.
Measured on this endpoint: a one-day window returned a paper created in **2008**, because
someone revised it. Fetching three months that way means pulling hundreds of thousands of
revisions to find the new submissions among them.

The API's ``submittedDate`` range is exactly the question we are asking. Measured: one day
of arXiv is ~580-1300 papers, comfortably inside the API's result cap, so **slicing by day**
never needs paging tricks and keeps each request small. arXiv asks for ~3 s between
requests, which is the real cost: about 15 minutes of wall clock per month harvested, none
of it CPU.

## Sources and the flag that matters

Every row carries ``source``. arXiv rows are ``"arxiv"``; anything else is not, and that
distinction is load-bearing downstream -- brontide's scored statistic is a *share over
arXiv months*, so a non-arXiv row must never enter the series. ``update.py`` enforces it;
this module only makes sure the field is always there to enforce on.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

from . import config

HARVEST = config.DATA_DIR / "harvest"
UA = {"User-Agent": "brontide-harvest/1.0 (research; contact via repo)"}

ARXIV_API = "http://export.arxiv.org/api/query"
OPENALEX = "https://api.openalex.org/works"
ATOM = {"a": "http://www.w3.org/2005/Atom",
        "os": "http://a9.com/-/spec/opensearch/1.1/"}

# Focus categories, for the case where only part of arXiv needs a fast cadence. The
# narrow set is ~8% of arXiv and the broad set ~29%, so a weekly narrow run costs under
# a minute of GPU against 8.4 minutes for a monthly pass over everything. Edit to suit
# the areas being tracked.
FOCUS_NARROW = ("quant-ph", "stat.ML", "math.ST", "stat.TH")
FOCUS_BROAD = FOCUS_NARROW + ("cs.LG", "cs.IT", "math.PR", "math.NT", "math.AG",
                              "cs.CC", "math.OC")

# OpenAlex concept ids for the focus areas (stable ids, not names -- names drift).
OA_CONCEPTS = (
    "C121332964",   # Physics
    "C33923547",    # Mathematics
    "C119857082",   # Machine learning
    "C105795698",   # Statistics
)

# Concept filtering alone is unusable, and this was measured rather than assumed: a TWO-DAY
# window on Physics+Mathematics returned 7,091 works -- about 106,000/month, four times
# arXiv's entire output -- and the top of the list was a Cell paper, two Zenodo supplements
# and a DOAJ aggregator stub. Volume without precision is worse than nothing here: every one
# of those rows would get embedded, placed on the map, and offered as a suggestion.
#
# So the non-arXiv path is VENUE-driven. The point of it is the specific gap arXiv leaves --
# journal-only statistics and mathematics -- not "all of science", and a named venue is a far
# better relevance signal than a concept tree.
OA_EXCLUDE_VENUE = ("zenodo", "doaj", "ssrn", "researchgate", "figshare", "preprints.org",
                    "authorea", "techrxiv", "biorxiv", "medrxiv", "hal", "osf",
                    "semantic scholar", "crossref", "openalex")

# Journal-only venues that actually matter for these areas. Matched case-insensitively as a
# substring of OpenAlex's display_name, so "Annals of Statistics" catches
# "The Annals of Statistics".
OA_FOCUS_VENUES = (
    "annals of statistics", "annals of applied statistics", "annals of probability",
    "journal of the american statistical association", "biometrika", "bernoulli",
    "journal of the royal statistical society", "statistical science",
    "journal of machine learning research", "machine learning",
    "ieee transactions on information theory", "ieee transactions on pattern analysis",
    "journal of computational and graphical statistics", "electronic journal of statistics",
    "probability theory and related fields", "stochastic processes and their applications",
    "quantum", "prx quantum", "physical review a", "physical review letters",
    "physical review x", "npj quantum information", "quantum information processing",
    "communications in mathematical physics", "journal of mathematical physics",
    "inventiones mathematicae", "annals of mathematics", "duke mathematical journal",
    "compositio mathematica", "mathematische annalen", "advances in mathematics",
)

_PAGE = 200            # API page size; a day rarely needs more than a handful of pages
_SLEEP = 3.1           # arXiv's requested courtesy interval
_RETRIES = 5


def _get(url: str, timeout: int = 120) -> bytes:
    """One request, with the backoff arXiv's rate limiter expects."""
    last = None
    for attempt in range(_RETRIES):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last = e
            wait = _SLEEP * (2 ** attempt)
            code = getattr(e, "code", None)
            if code == 503:
                wait = max(wait, int(getattr(e, "headers", {}).get("Retry-After") or 20))
            print(f"    retry {attempt + 1}/{_RETRIES} after {wait:.0f}s ({e})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"gave up after {_RETRIES} attempts: {last}")


def _arxiv_day(day: date, categories: tuple[str, ...] | None) -> list[dict]:
    """Every paper SUBMITTED on `day`. Category filter is applied server-side when given."""
    lo = day.strftime("%Y%m%d") + "0000"
    hi = (day + timedelta(days=1)).strftime("%Y%m%d") + "0000"
    q = f"submittedDate:[{lo}+TO+{hi}]"
    if categories:
        q = "(" + "+OR+".join(f"cat:{c}" for c in categories) + ")+AND+" + q
    out: list[dict] = []
    start = 0
    while True:
        url = (f"{ARXIV_API}?search_query={q}&start={start}&max_results={_PAGE}"
               f"&sortBy=submittedDate&sortOrder=ascending")
        root = ET.fromstring(_get(url))
        entries = root.findall("a:entry", ATOM)
        for e in entries:
            raw = (e.findtext("a:id", "", ATOM) or "")
            aid = raw.rsplit("/abs/", 1)[-1]
            aid = aid.rsplit("v", 1)[0] if "v" in aid.rsplit("/", 1)[-1] else aid
            pub = e.findtext("a:published", "", ATOM) or ""
            title = " ".join((e.findtext("a:title", "", ATOM) or "").split())
            summary = " ".join((e.findtext("a:summary", "", ATOM) or "").split())
            cats = [c.get("term") for c in e.findall("a:category", ATOM) if c.get("term")]
            if not (aid and title and pub):
                continue
            out.append({"id": aid, "month": pub[:7], "categories": cats,
                        "title": title, "abstract": summary, "source": "arxiv",
                        "submitted": pub[:10]})
        if len(entries) < _PAGE:
            break
        start += _PAGE
        time.sleep(_SLEEP)
    return out


def _openalex_window(lo: date, hi: date, concepts: tuple[str, ...],
                     venues: tuple[str, ...] | None = None) -> list[dict]:
    """Non-arXiv works in a date window, for the focus areas only.

    Filtered to works that HAVE an abstract and are NOT already on arXiv -- an arXiv
    preprint reached this way would duplicate a row the arXiv path already owns, under a
    different id, and land twice on the map. Deduping on title downstream catches the rest.
    """
    out: list[dict] = []
    cursor = "*"
    flt = (f"from_publication_date:{lo.isoformat()},to_publication_date:{hi.isoformat()},"
           f"has_abstract:true,type:article,"
           f"concepts.id:" + "|".join(concepts))
    while cursor:
        url = (f"{OPENALEX}?filter={urllib.parse.quote(flt, safe=':,|')}"
               f"&per-page=200&cursor={urllib.parse.quote(cursor)}"
               f"&select=id,doi,title,publication_date,abstract_inverted_index,"
               f"primary_location,concepts")
        blob = json.loads(_get(url))
        for w in blob.get("results") or []:
            inv = w.get("abstract_inverted_index") or {}
            if not inv or not w.get("title"):
                continue
            # OpenAlex ships abstracts as an inverted index; rebuild the word order.
            pos: dict[int, str] = {}
            for word, idxs in inv.items():
                for i in idxs:
                    pos[i] = word
            abstract = " ".join(pos[i] for i in sorted(pos))
            loc = (w.get("primary_location") or {}).get("source") or {}
            venue = str(loc.get("display_name") or "").lower()
            if "arxiv" in venue:
                continue                     # the arXiv path owns these
            if any(x in venue for x in OA_EXCLUDE_VENUE):
                continue                     # repositories and aggregators, not journals
            if venues is not None and not any(v in venue for v in venues):
                continue
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            oid = str(w.get("id", "")).rsplit("/", 1)[-1]
            out.append({"id": f"doi:{doi}" if doi else f"openalex:{oid}",
                        "month": str(w.get("publication_date") or "")[:7],
                        "categories": [c["display_name"] for c in (w.get("concepts") or [])[:4]],
                        "title": " ".join(str(w["title"]).split()),
                        "abstract": abstract, "source": "openalex",
                        "venue": loc.get("display_name"),
                        "submitted": str(w.get("publication_date") or "")[:10]})
        cursor = (blob.get("meta") or {}).get("next_cursor")
        time.sleep(1.0)
    return out


def arxiv(lo: date, hi: date, categories: tuple[str, ...] | None = None) -> Path:
    """Day-by-day harvest into a resumable jsonl. Returns the path."""
    HARVEST.mkdir(parents=True, exist_ok=True)
    tag = "arxiv" + ("-focus" if categories else "")
    out = HARVEST / f"{tag}-{lo.isoformat()}-{hi.isoformat()}.jsonl"
    done: set[str] = set()
    if out.exists():
        # Heal a torn tail before appending. A killed run flushes a partial buffer, so the
        # file can end mid-line -- and appending after that permanently embeds an unreadable
        # record in the middle of the file. Truncate back to the last complete, parseable,
        # newline-terminated line so the resume writes onto a clean boundary.
        raw = out.read_bytes()
        keep, nl = 0, -1
        for line in raw.split(b"\n"):
            seg = len(line) + 1
            try:
                rec = json.loads(line.decode("utf-8"))
                done.add(rec["submitted"])
                nl = keep + len(line)          # end of a known-good line
            except (ValueError, KeyError, UnicodeDecodeError):
                pass
            keep += seg
        good_end = nl + 1 if nl >= 0 else 0
        if good_end < len(raw):
            out.write_bytes(raw[:good_end])
            print(f"  healed a torn tail: dropped {len(raw) - good_end} trailing byte(s)")
        print(f"  resuming: {len(done)} day(s) already harvested")

    day, total = lo, 0
    with open(out, "a", encoding="utf-8") as fh:
        while day < hi:
            if day.isoformat() in done:
                day += timedelta(days=1)
                continue
            rows = _arxiv_day(day, categories)
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            total += len(rows)
            print(f"  {day.isoformat()}  {len(rows):>5} papers   (running {total:,})",
                  flush=True)
            day += timedelta(days=1)
            time.sleep(_SLEEP)
    return out


def openalex(lo: date, hi: date, concepts: tuple[str, ...],
             venues: tuple[str, ...] | None = OA_FOCUS_VENUES) -> Path:
    HARVEST.mkdir(parents=True, exist_ok=True)
    out = HARVEST / f"openalex-{lo.isoformat()}-{hi.isoformat()}.jsonl"
    rows = _openalex_window(lo, hi, concepts, venues)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  openalex: {len(rows):,} non-arXiv works with abstracts")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="lo", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--to", dest="hi", default=None, help="YYYY-MM-DD (exclusive); default today")
    ap.add_argument("--focus", action="store_true", help="focus categories only (29% of arXiv)")
    ap.add_argument("--narrow", action="store_true", help="narrow focus only (8%)")
    ap.add_argument("--openalex", action="store_true", help="also harvest non-arXiv works")
    a = ap.parse_args()

    lo = date.fromisoformat(a.lo)
    hi = date.fromisoformat(a.hi) if a.hi else date.today()
    cats = FOCUS_NARROW if a.narrow else (FOCUS_BROAD if a.focus else None)
    print(f"harvest arXiv {lo} -> {hi}" + (f"  categories={len(cats)}" if cats else "  (all)"))
    p = arxiv(lo, hi, cats)
    print(f"  -> {p}")
    if a.openalex:
        print(f"harvest OpenAlex {lo} -> {hi} (non-arXiv only)")
        print(f"  -> {openalex(lo, hi, OA_CONCEPTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
