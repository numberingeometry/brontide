"""update.py -- append newly harvested papers to the frozen corpus, without moving anything.

    py -3 -m brontide.update --harvest data/harvest/arxiv-2026-07-01-2026-09-04.jsonl
    py -3 -m brontide.update --harvest <f> --dry-run       # say what would change
    py -3 -m brontide.update --harvest <f> --limit 500     # a small real append

## The one invariant

**Appends never move an existing row.** New papers go on the end of every artifact, so
every row index, region id and map coordinate already in circulation keeps meaning exactly
what it meant. That is what makes this safe to run monthly against a corpus other things
depend on, and it is why ``generation`` exists: ``build_id`` changes on every append (the
files did change), but ``generation`` changes only on a **refit**, so a consumer holding
cached row indices can tell "there is more" from "everything you cached is void".

## What is NOT recomputed, deliberately

The scored artifacts. brontide's published result is pre-registered against a frozen corpus
-- *2,705,992 papers, 2007-01 to 2026-06* -- and recomputing regions, series or forecasts
here would quietly invalidate that claim. So:

- **k-means is not refit.** New papers are assigned to the nearest EXISTING centroid, which
  is exact and is what the geometry was frozen for.
- **PCA is not refit.** New papers are projected through the saved basis (see geometry.py).
- **UMAP is not refit.** With a saved reducer, ``transform``; without one, corpus k-NN --
  ``geometry.to_umap`` reports which, and this prints it.
- **series / emergence / forecasts are not touched at all.**

``BUILD.json`` records ``frozen_result`` (what the publication was scored on) separately
from ``live`` (what is on disk now), so both statements stay true at once.

## Non-arXiv rows

Carried, embedded and placed like any other -- and marked ``source`` != ``"arxiv"``. The
scored statistic is a share over arXiv months, so a non-arXiv row entering the series would
change every region's denominator. ``series_mask()`` here is the single place that
distinction is enforced, and ``--allow-nonarxiv-in-series`` does not exist.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from . import config, geometry

SOURCE_COL = "source"
LABELS = config.ARTIFACTS_DIR / "region_labels.i16.npy"


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _set_rows(path: Path, n: int) -> None:
    p = _sidecar(path)
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta["n"] = int(n)
    if "dim" in meta:
        per = int(meta["dim"]) * (2 if meta.get("dtype") == "float16" else 4)
        meta["expected_bytes"] = int(n) * per
    p.write_text(json.dumps(meta, indent=2))


def existing_ids() -> tuple[set[str], set[str], int]:
    """(ids, normalised titles, row count). One pass over the parquet."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(config.RAW_PARQUET)
    ids: set[str] = set()
    titles: set[str] = set()
    n = 0
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["id", "title"])
        ids.update(str(x) for x in t.column("id").to_pylist())
        titles.update(_norm_title(str(x)) for x in t.column("title").to_pylist())
        n += t.num_rows
    return ids, titles, n


def load_harvest(paths: list[Path]) -> list[dict]:
    """Read harvest jsonl tolerantly. One bad byte must not cost 68,719 good rows.

    Opened as BYTES and decoded per line, because a text-mode iterator raises
    UnicodeDecodeError from the `for` statement itself -- one malformed byte anywhere in a
    100 MB file aborts the whole read, and the traceback points at a buffer offset rather
    than a file offset, which sends you looking in the wrong place.

    That is not hypothetical: a killed harvest run left a torn line (a partially flushed
    buffer, cut mid-UTF-8-character), and the resumed run appended after it. Exactly one
    line was unreadable and the other 68,719 were perfect. Losing the batch to that would be
    the wrong failure mode for a job that reads from the open internet.
    """
    rows: dict[str, dict] = {}
    skipped = 0
    for p in paths:
        with open(p, "rb") as fh:
            for raw in fh:
                try:
                    r = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    if raw.strip():
                        skipped += 1
                    continue
                if isinstance(r, dict) and r.get("id") and r.get("title") and r.get("abstract"):
                    rows[r["id"]] = r        # last wins; harvest files may overlap
    if skipped:
        print(f"  note: skipped {skipped} unreadable line(s) (torn write or bad bytes)")
    return list(rows.values())


def append_parquet(new: list[dict], n_before: int) -> int:
    """Rewrite papers.parquet with `new` appended. Order preserved, so indices are stable.

    A rewrite rather than a sidecar file: pyarrow cannot append in place, and a
    multi-file dataset would make "row 812,443" ambiguous, which is the one thing every
    consumer of this corpus depends on being unambiguous. 1.2 GB costs about a minute.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(config.RAW_PARQUET)
    old_schema = pf.schema_arrow
    has_source = SOURCE_COL in old_schema.names
    fields = list(old_schema)
    if not has_source:
        fields.append(pa.field(SOURCE_COL, pa.string()))
    schema = pa.schema(fields)

    tmp = config.RAW_PARQUET.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
    try:
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i)
            if not has_source:
                # Every pre-existing row came from the arXiv snapshot by construction.
                t = t.append_column(SOURCE_COL,
                                    pa.array(["arxiv"] * t.num_rows, pa.string()))
            writer.write_table(t.select(schema.names))
        if new:  # noqa: SIM102 -- kept flat for the finally below
            cols = {
                "id": pa.array([r["id"] for r in new], pa.string()),
                "month": pa.array([r["month"] for r in new], pa.string()),
                "categories": pa.array([list(r.get("categories") or []) for r in new],
                                       pa.list_(pa.string())),
                "title": pa.array([r["title"] for r in new], pa.string()),
                "abstract": pa.array([r["abstract"] for r in new], pa.string()),
                SOURCE_COL: pa.array([r.get("source", "arxiv") for r in new], pa.string()),
            }
            writer.write_table(pa.Table.from_pydict(
                {k: cols[k] for k in schema.names}, schema=schema))
    finally:
        writer.close()
        # Windows refuses os.replace while ANY handle on the target is open, and
        # ParquetFile holds one for the whole read. Closing it is not tidiness -- without
        # it the swap dies with WinError 5 after doing all the work.
        try:
            pf.close()
        except Exception:
            pass
        del pf
    os.replace(tmp, config.RAW_PARQUET)
    total = n_before + len(new)
    _sidecar(config.RAW_PARQUET).write_text(json.dumps({
        "n": total, "corpus_start": config.CORPUS_START,
        "appended_rows": len(new), "appended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "columns": list(schema.names),
        "row_order": ("snapshot order then append order; a row index is stable across "
                      "appends and is invalidated only by a refit (see generation)"),
    }, indent=2))
    return total


def series_mask() -> np.ndarray:
    """True for rows the scored series may use: arXiv only.

    The single place this is decided. brontide's forecast is a per-region SHARE of arXiv
    papers per month; a journal-only paper appearing in the numerator changes every
    region's denominator and would move a pre-registered number. Non-arXiv rows are for
    navigation.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(config.RAW_PARQUET)
    if SOURCE_COL not in pf.schema_arrow.names:
        return np.ones(pf.metadata.num_rows, dtype=bool)
    out = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=[SOURCE_COL])
        out.append(np.array([str(x) == "arxiv" for x in t.column(SOURCE_COL).to_pylist()]))
    return np.concatenate(out) if out else np.zeros(0, dtype=bool)


def _bump_build(appended: int, live_end: str, mode: str) -> dict:
    from . import manifest
    m = manifest.build()
    p = manifest.MANIFEST
    prev = {}
    if p.exists():
        try:
            prev = json.loads(p.read_text())
        except ValueError:
            pass
    # generation survives an append and increments only on a refit.
    m["generation"] = int(prev.get("generation", 1))
    m["frozen_result"] = prev.get("frozen_result") or {
        "corpus_end": "2026-06", "n_papers": 2705992,
        "note": ("the corpus the pre-registered validation was scored on; the scored "
                 "artifacts under artifacts/ are never recomputed by an append"),
    }
    m["live"] = {"corpus_end": live_end, "n_papers": m.get("n_papers"),
                 "last_append": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "appended_rows": appended, "umap_mode": mode}
    p.write_text(json.dumps(m, indent=2) + "\n")
    return m


def project_tail(log=print) -> int:
    """Bring pca50 / umap2 up to whatever the EMBEDDINGS already cover.

    An append is five stages across four artifacts, so it can die half-done -- and on
    2026-09-04 it did, in the embed stage, leaving the parquet and the embeddings at
    2,706,067 while pca50 and umap2 sat at 2,705,992. Re-running the append could not repair
    that: the new ids were already in the parquet, so it saw nothing new and returned
    "nothing to append" while the corpus stayed inconsistent.

    So the tail projection is its OWN operation, keyed on what is actually on disk rather
    than on what a harvest file says. Idempotent: with nothing to do it does nothing.
    """
    import pyarrow.parquet as pq
    n_papers = pq.ParquetFile(config.RAW_PARQUET).metadata.num_rows
    side = _sidecar(config.EMBED_MEMMAP).with_name(
        config.EMBED_MEMMAP.name + ".progress.json")
    n_emb = int(json.loads(side.read_text()).get("n_done", 0)) if side.exists() else 0
    n_pca = int(json.loads(_sidecar(config.PCA_MEMMAP).read_text())["n"])
    n_umap = int(json.loads(_sidecar(config.UMAP_MEMMAP).read_text())["n"])
    log(f"  parquet {n_papers:,}   embeddings {n_emb:,}   pca50 {n_pca:,}   umap2 {n_umap:,}")

    target = min(n_papers, n_emb)
    lo = min(n_pca, n_umap)
    if lo >= target:
        log("  nothing to project -- all four artifacts agree")
        return 0
    log(f"  projecting rows {lo:,} .. {target:,} ({target - lo:,} rows)")

    basis = geometry.pca_basis()
    emb = np.memmap(config.EMBED_MEMMAP, dtype=np.float16, mode="r",
                    shape=(max(target, n_emb), config.EMBED_DIM))
    vec = np.asarray(emb[lo:target], dtype=np.float64)
    if not len(vec):
        return 0
    bad = int((np.linalg.norm(vec, axis=1) == 0).sum())
    if bad:
        raise RuntimeError(f"{bad} of {len(vec)} rows in the tail are all-zero -- the embed "
                           f"did not actually cover them; do not project these")

    pca_new = geometry.to_pca(vec, basis)
    pmm = np.memmap(config.PCA_MEMMAP, dtype=np.float32, mode="r+",
                    shape=(target, config.PCA_DIM))
    pmm[lo:target] = pca_new
    pmm.flush(); del pmm
    _set_rows(config.PCA_MEMMAP, target)

    micro, sup = geometry.to_region(pca_new)
    if LABELS.exists():
        old_labels = np.load(LABELS)
        if len(old_labels) == lo:
            np.save(LABELS, np.concatenate([old_labels, micro.astype(np.int16)]))

    xy, mode = geometry.to_umap(pca_new)
    umm = np.memmap(config.UMAP_MEMMAP, dtype=np.float32, mode="r+", shape=(target, 2))
    umm[lo:target] = xy
    umm.flush(); del umm
    _set_rows(config.UMAP_MEMMAP, target)

    from collections import Counter
    names = json.loads((config.ARTIFACTS_DIR / "region_names.json").read_text())["names"]
    for nm, c in Counter(names.get(str(int(x)), str(x)) for x in sup).most_common(5):
        log(f"      {c:>5}  {nm}")
    log(f"  placed {len(pca_new):,} rows   umap mode={mode}")
    return len(pca_new)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--harvest", nargs="*", default=[], type=Path)
    ap.add_argument("--finish", action="store_true",
                    help="repair only: project any rows the embeddings cover but pca50/umap2 "
                         "do not, then update BUILD.json. Idempotent.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="append at most N (for a trial)")
    ap.add_argument("--skip-embed", action="store_true",
                    help="append metadata only; run brontide.embed yourself afterwards")
    a = ap.parse_args()

    if a.finish:
        print("finishing a partial append (projecting the tail)")
        n = project_tail()
        if n:
            import pyarrow.parquet as pq
            t = pq.read_table(config.RAW_PARQUET, columns=["month"]).column("month").to_pylist()
            m = _bump_build(n, max(str(x) for x in t[-n:]), "knn")
            print(f"\nBUILD.json  build_id={m['build_id']}  generation={m['generation']}")
            print(f"  frozen_result: {m['frozen_result']['corpus_end']} / "
                  f"{m['frozen_result']['n_papers']:,}")
            print(f"  live         : {m['live']['corpus_end']} / {m['live']['n_papers']:,}")
        return 0

    if not a.harvest:
        ap.error("--harvest is required unless --finish is given")
    rows = load_harvest(a.harvest)
    print(f"harvested rows: {len(rows):,}")
    ids, titles, n_before = existing_ids()
    print(f"corpus on disk: {n_before:,} rows")

    new, dup_id, dup_title = [], 0, 0
    seen_t: set[str] = set()
    for r in sorted(rows, key=lambda x: (x.get("submitted") or "", x["id"])):
        if r["id"] in ids:
            dup_id += 1
            continue
        nt = _norm_title(r["title"])
        if nt in titles or nt in seen_t:
            dup_title += 1               # same work under a second id (arXiv vs journal)
            continue
        seen_t.add(nt)
        new.append(r)
    if a.limit:
        new = new[:a.limit]
    src = {}
    for r in new:
        src[r.get("source", "arxiv")] = src.get(r.get("source", "arxiv"), 0) + 1
    months = sorted({r["month"] for r in new})
    print(f"new: {len(new):,}   (skipped {dup_id:,} known ids, {dup_title:,} duplicate titles)")
    print(f"  by source: {src}")
    print(f"  months   : {months[0] if months else '-'} .. {months[-1] if months else '-'}")

    if a.dry_run or not new:
        print("\ndry run -- nothing written" if a.dry_run else "\nnothing new to append")
        return 0

    print(f"\n[1/5] rewriting papers.parquet with {len(new):,} rows appended ...", flush=True)
    total = append_parquet(new, n_before)
    print(f"      {n_before:,} -> {total:,} rows")

    if a.skip_embed:
        print("\n--skip-embed: run `python -m brontide.embed` next, then re-run without it")
        return 0

    print(f"\n[2/5] embedding the new rows (resumes at {n_before:,}) ...", flush=True)
    from . import embed
    embed.run()

    print("\n[3/5] projecting through the saved PCA basis ...", flush=True)
    basis = geometry.pca_basis()
    emb = np.memmap(config.EMBED_MEMMAP, dtype=np.float16, mode="r",
                    shape=(total, config.EMBED_DIM))
    newvec = np.asarray(emb[n_before:total], dtype=np.float64)
    pca_new = geometry.to_pca(newvec, basis)
    pmm = np.memmap(config.PCA_MEMMAP, dtype=np.float32, mode="r+",
                    shape=(total, config.PCA_DIM))
    pmm[n_before:total] = pca_new
    pmm.flush(); del pmm
    _set_rows(config.PCA_MEMMAP, total)
    print(f"      basis={basis['source']}   appended {len(pca_new):,} rows")

    print("\n[4/5] regions (nearest existing centroid) ...", flush=True)
    micro, sup = geometry.to_region(pca_new)
    if LABELS.exists():
        old = np.load(LABELS)
        np.save(LABELS, np.concatenate([old, micro.astype(np.int16)]))
        print(f"      region_labels {len(old):,} -> {len(old) + len(micro):,}")
    from collections import Counter
    names = json.loads((config.ARTIFACTS_DIR / "region_names.json").read_text())["names"]
    top = Counter(names.get(str(int(s)), str(s)) for s in sup).most_common(5)
    for nm, c in top:
        print(f"        {c:>5}  {nm}")

    print("\n[5/5] map coordinates ...", flush=True)
    xy, mode = geometry.to_umap(pca_new)
    umm = np.memmap(config.UMAP_MEMMAP, dtype=np.float32, mode="r+", shape=(total, 2))
    umm[n_before:total] = xy
    umm.flush(); del umm
    _set_rows(config.UMAP_MEMMAP, total)
    print(f"      mode={mode}" + ("" if mode == "reducer" else
          "   (no saved reducer for this build -- corpus k-NN, median ~0.95% of the map)"))

    live_end = max(r["month"] for r in new)
    m = _bump_build(len(new), live_end, mode)
    print(f"\nBUILD.json  build_id={m['build_id']}  generation={m['generation']} "
          f"(unchanged by an append)")
    print(f"  frozen_result: {m['frozen_result']['corpus_end']} / "
          f"{m['frozen_result']['n_papers']:,}")
    print(f"  live         : {m['live']['corpus_end']} / {m['live']['n_papers']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
