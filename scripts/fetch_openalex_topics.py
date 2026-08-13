"""Fetch OpenAlex topic assignments for the corpus, by arXiv DOI. Resumable.

Two uses, and the second matters more than the first:

  1. A finer, cross-field hierarchy (4,516 topics / 252 subfields / 26 fields / 4 domains,
     CC0) over the same documents, as a second human-built alternative to the Ward tree.

  2. A REAL corpus boundary. A measured 64% of a random sample matches by
     `doi:10.48550/arxiv.*`; `ids.arxiv` and landing-page filters are not supported, so
     that is the join. The unmatched 36% is not a nuisance -- restricting to the covered
     subset CUTS super-regions instead of removing whole ones, which is precisely the case
     where the sigma-saturation hypothesis of the gauge theorem fails. Synthetic
     whole-super drops cannot test that; a database coverage boundary can.

Writes data/openalex_topics.parquet (gitignored, local-only, like everything in data/).
Checkpoints every CHUNK batches so it can be killed and resumed.

Note for whoever runs this: the API response carries a `cost_usd` field
(~1e-4 per request at 50 ids/request, so ~5 USD for the full corpus at the time of
writing). It is not clear whether that is billed or informational. Run with --limit first.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
import pyarrow.parquet as pq

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from brontide import config                                              # noqa: E402

OUT = REPO / "data" / "openalex_topics.parquet"
MAILTO = "contact@example.invalid"          # polite pool
BATCH, SLEEP, CKPT = 50, 0.60, 40


def fetch(batch, retries=6):
    """Returns (results, ok). ok=False means the REQUEST failed.

    This distinction is load-bearing. An earlier version returned [] on failure, so the
    caller wrote every id in a rate-limited batch as matched=False -- making a transport
    error indistinguishable from a genuine absence from OpenAlex. That silently drove the
    apparent match rate from 39.9% to 16.3%. A failed batch must be recorded as UNKNOWN
    and retried, never as a negative.
    """
    q = "|".join(f"10.48550/arxiv.{i}" for i in batch)
    url = ("https://api.openalex.org/works?filter=doi:" + urllib.parse.quote(q, safe="|:.")
           + "&per-page=100&select=doi,primary_topic&mailto=" + MAILTO)
    for a in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)["results"], True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if a == retries - 1:
                print(f"    batch UNRESOLVED after {retries} tries: {type(e).__name__}", flush=True)
                return [], False
            time.sleep(min(2 ** a, 30))
    return [], False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stop after N ids (smoke test)")
    a = ap.parse_args()

    ids = pq.read_table(config.RAW_PARQUET, columns=["id"]).column("id").to_pylist()
    if a.limit:
        ids = ids[:a.limit]

    done, rows = set(), []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        if "matched" in prev.columns:
            # Only positives are trustworthy from any earlier run: a False may be a genuine
            # absence OR a swallowed transport error. Re-fetch everything not confirmed.
            good = prev[prev["matched"] == True]                        # noqa: E712
            dropped = len(prev) - len(good)
            if dropped:
                print(f"discarding {dropped:,} unconfirmed rows from a previous run "
                      f"(matched=False is ambiguous); keeping {len(good):,} confirmed")
            prev = good
        rows = prev.to_dict("records")
        done = set(prev["arxiv_id"])
        print(f"resuming: {len(done):,} confirmed already")

    todo = [i for i in ids if i not in done]
    print(f"{len(todo):,} to fetch of {len(ids):,}  ({BATCH}/request, ~{len(todo)//BATCH:,} requests)",
          flush=True)

    t0, n_hit, n_res, n_unres = time.time(), 0, 0, 0
    for bi in range(0, len(todo), BATCH):
        batch = todo[bi:bi + BATCH]
        results, ok = fetch(batch)
        if not ok:
            n_unres += len(batch)          # left out of the file entirely; retried next run
            time.sleep(5.0)
            continue
        got = {}
        for w in results:
            d = (w.get("doi") or "").lower().split("arxiv.")[-1]
            pt = w.get("primary_topic") or {}
            got[d] = {
                "topic": pt.get("display_name"),
                "subfield": (pt.get("subfield") or {}).get("display_name"),
                "field": (pt.get("field") or {}).get("display_name"),
                "domain": (pt.get("domain") or {}).get("display_name"),
            }
        for i in batch:
            g = got.get(i.lower())
            rows.append({"arxiv_id": i, "matched": g is not None,
                         **(g or {"topic": None, "subfield": None, "field": None, "domain": None})})
            n_hit += g is not None
        n_res += len(batch)
        time.sleep(SLEEP)

        if (bi // BATCH) % CKPT == 0 and bi:
            pd.DataFrame(rows).to_parquet(OUT, index=False)
            n = bi + len(batch)
            el = time.time() - t0
            print(f"  {n:,}/{len(todo):,}  resolved {n_res:,} unresolved {n_unres:,}  "
                  f"match {100*n_hit/max(n_res,1):.1f}% (of RESOLVED)  "
                  f"{el/60:.1f} min  eta {(el/n)*(len(todo)-n)/3600:.1f} h", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    m = df["matched"].mean()
    print(f"\nwrote {OUT}  rows={len(df):,}  match rate={100*m:.2f}%")
    print(f"unmatched = {int((~df['matched']).sum()):,} -- this is the real coverage boundary, "
          f"and a restriction to the matched subset CUTS super-regions rather than removing them.")
    if m > 0:
        sub = df[df["matched"]]
        print(f"distinct: topics={sub['topic'].nunique()} subfields={sub['subfield'].nunique()} "
              f"fields={sub['field'].nunique()} domains={sub['domain'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
