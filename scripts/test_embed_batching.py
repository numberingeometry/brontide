"""Prove the embed loop's index arithmetic covers every row exactly once. No GPU, no model.

Written after the row-group-skip change shipped a real crash: with the old reader every
chunk was exactly ``_READ_CHUNK`` rows (a multiple of ``EMBED_BATCH``), so the slice bounds
were correct by accident. Reading per row group makes the last batch of each group short,
and the mismatch only surfaced after the model had run, 20 minutes into a 14-hour job.

This replays the same arithmetic against the REAL row-group layout of papers.parquet and
asserts the two properties that matter:

  1. every row in [n_done, n_total) is written exactly once
  2. every write's slice length equals the number of texts available for it

    conda run -n brontide python scripts/test_embed_batching.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyarrow.parquet as pq  # noqa: E402

from brontide import config  # noqa: E402

_READ_CHUNK = 8192


def replay(group_rows: list[int], n_total: int, n_done: int) -> tuple[list[int], list[str]]:
    """The loop, with the model replaced by bookkeeping. Returns (writes_per_row, errors)."""
    written = [0] * n_total
    errors: list[str] = []

    start_group, cum = 0, 0
    if n_done:
        for gi, g in enumerate(group_rows):
            if cum + g > n_done:
                break
            cum += g
            start_group = gi + 1
    row = cum

    for gi in range(start_group, len(group_rows)):
        # one row group -> batches of at most _READ_CHUNK, last one short
        remaining = group_rows[gi]
        offs = []
        while remaining > 0:
            offs.append(min(_READ_CHUNK, remaining))
            remaining -= offs[-1]
        for chunk_len in offs:
            if row >= n_total:
                break
            n_titles = chunk_len
            for i in range(0, n_titles, config.EMBED_BATCH):
                lo = row + i
                hi = min(lo + config.EMBED_BATCH, n_total, row + n_titles)
                if hi <= lo:
                    break
                if hi <= n_done:
                    continue
                k = hi - lo
                n_texts = len(range(i, min(i + k, n_titles)))
                if n_texts != k:
                    errors.append(f"group {gi} i={i}: k={k} but only {n_texts} texts")
                for r in range(lo, hi):
                    if r < n_total:
                        written[r] += 1
            row += n_titles
    return written, errors


def main() -> int:
    pf = pq.ParquetFile(config.RAW_PARQUET)
    group_rows = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    n_total = sum(group_rows)
    print(f"papers.parquet: {pf.num_row_groups} row groups, {n_total:,} rows")
    print(f"  group sizes: min {min(group_rows):,} max {max(group_rows):,}")
    print(f"  EMBED_BATCH={config.EMBED_BATCH}  _READ_CHUNK={_READ_CHUNK}")
    print(f"  any group NOT a multiple of EMBED_BATCH: "
          f"{sum(1 for g in group_rows if g % config.EMBED_BATCH)} of {len(group_rows)}"
          "  <- why the old bounds were wrong\n")

    ok = True
    for n_done in (0, 50_048, 1_000_000, n_total - 75, n_total):
        written, errors = replay(group_rows, n_total, n_done)
        tail = written[n_done:]
        head = written[:n_done]
        twice = sum(1 for x in tail if x > 1)
        never = sum(1 for x in tail if x == 0)
        touched_done = sum(1 for x in head if x > 0)
        good = not errors and not twice and not never
        ok &= good
        print(f"  n_done={n_done:>9,}  to-embed={len(tail):>9,}  "
              f"written-twice={twice}  never-written={never}  "
              f"already-done-rows-touched={touched_done}  "
              f"slice/text mismatches={len(errors)}  ->  {'PASS' if good else 'FAIL'}")
        for e in errors[:3]:
            print(f"      {e}")
    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
