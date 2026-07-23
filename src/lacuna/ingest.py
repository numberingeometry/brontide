"""Stage 1 — Ingest (host, ~1h).

Download the arXiv Kaggle snapshot (~2.6M papers, abstracts included) via
``kagglehub``, stream-parse the JSON, and write a tidy parquet:
``id, month, categories, title, abstract``.

Monthly binning makes arXiv-heatmap's 18:00/business-day logic unnecessary —
just take the calendar month of the paper's date (``CORPUS_START`` onward).
"""
from __future__ import annotations

import json
from email.utils import parsedate_to_datetime
from pathlib import Path

from . import config

_DATASET = "Cornell-University/arxiv"
_SNAPSHOT_NAME = "arxiv-metadata-oai-snapshot.json"

# Rows are flushed to the parquet writer in chunks of this size. The snapshot is
# ~2.6M records of ~1.2 KB each, so accumulating everything before writing would
# cost several GB of Python strings; batching keeps memory flat instead.
_BATCH_ROWS = 100_000


def _schema():
    import pyarrow as pa

    # `month` is a plain "YYYY-MM" string rather than a pandas Period: pyarrow
    # cannot serialise period dtype to parquet. It sorts lexicographically and
    # converts downstream with pd.PeriodIndex(df["month"], freq="M").
    return pa.schema([
        ("id", pa.string()),
        ("month", pa.string()),
        ("categories", pa.list_(pa.string())),
        ("title", pa.string()),
        ("abstract", pa.string()),
    ])


def _clean(text: str) -> str:
    """Collapse the snapshot's hard-wrapped whitespace into single spaces."""
    return " ".join(text.split())


def _month_of(record: dict) -> str | None:
    """Calendar month (YYYY-MM) of the paper's first version, else None.

    ``versions[0].created`` is an RFC 2822 date string, e.g.
    "Mon, 2 Apr 2007 19:18:42 GMT".
    """
    versions = record.get("versions")
    if not versions:
        return None
    created = versions[0].get("created")
    if not created:
        return None
    try:
        return parsedate_to_datetime(created).strftime("%Y-%m")
    except (TypeError, ValueError):
        return None


def _locate_snapshot() -> Path:
    """Download (or reuse the cached) Kaggle snapshot; return the JSON path.

    Needs Kaggle credentials -- [configured directory] or KAGGLE_USERNAME +
    KAGGLE_KEY. Without them kagglehub tries to prompt, which fails outright in
    an unattended run.
    """
    import kagglehub

    root = Path(kagglehub.dataset_download(_DATASET))
    matches = sorted(root.rglob(_SNAPSHOT_NAME))
    if not matches:  # fall back to the largest .json if Kaggle renames the file
        matches = sorted(root.rglob("*.json"), key=lambda p: p.stat().st_size, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No JSON snapshot found under {root}")
    return matches[0]


def run(sample: int | None = None, out: Path = config.RAW_PARQUET) -> Path:
    """Ingest the snapshot to parquet. ``sample`` = first-N rows for smoke tests.

    Returns the parquet path. Columns: id, month (YYYY-MM period), categories
    (list[str]), title, abstract.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = _locate_snapshot()
    schema = _schema()

    ids: list[str] = []
    months: list[str] = []
    cats: list[list[str]] = []
    titles: list[str] = []
    abstracts: list[str] = []

    n_kept = n_read = n_no_date = n_pre_corpus = n_bad_json = 0
    month_lo: str | None = None
    month_hi: str | None = None

    writer = pq.ParquetWriter(out, schema, compression="zstd")

    def flush() -> None:
        if not ids:
            return
        writer.write_table(pa.Table.from_pydict(
            {"id": ids, "month": months, "categories": cats,
             "title": titles, "abstract": abstracts},
            schema=schema,
        ))
        ids.clear(); months.clear(); cats.clear(); titles.clear(); abstracts.clear()

    try:
        with snapshot.open("r", encoding="utf-8") as fh:
            bar = tqdm(fh, desc=f"ingest[{config.resolve_sample(sample)}]", unit="rec")
            for line in bar:
                n_read += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_bad_json += 1
                    continue

                month = _month_of(rec)
                if month is None:
                    n_no_date += 1
                    continue
                # "YYYY-MM" strings order correctly, so no date parsing needed here.
                if month < config.CORPUS_START:
                    n_pre_corpus += 1
                    continue

                ids.append(rec.get("id", ""))
                months.append(month)
                cats.append((rec.get("categories") or "").split())
                titles.append(_clean(rec.get("title") or ""))
                abstracts.append(_clean(rec.get("abstract") or ""))
                n_kept += 1
                # Tracked here because `months` is cleared on every flush.
                if month_lo is None or month < month_lo:
                    month_lo = month
                if month_hi is None or month > month_hi:
                    month_hi = month

                if len(ids) >= _BATCH_ROWS:
                    flush()

                # `sample` counts rows KEPT, not lines read: the snapshot leads
                # with old-style ids from 1991, so the first N *lines* would be
                # almost entirely pre-CORPUS_START and a smoke test would land a
                # near-empty parquet.
                if sample is not None and n_kept >= sample:
                    break
            bar.close()
        flush()
    finally:
        writer.close()

    n_rows = pq.ParquetFile(out).metadata.num_rows
    print(
        f"\ningest[{config.resolve_sample(sample)}] -> {out}\n"
        f"  rows kept : {n_rows:,}  (read {n_read:,} records)\n"
        f"  months    : {month_lo} .. {month_hi}\n"
        f"  skipped   : {n_pre_corpus:,} pre-{config.CORPUS_START}, "
        f"{n_no_date:,} undated, {n_bad_json:,} unparseable\n"
        f"  file size : {out.stat().st_size / 1e6:,.1f} MB"
    )
    return out
