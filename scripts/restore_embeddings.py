"""Restore the embedding memmap and finish the pending append.

    conda run -n brontide python scripts/restore_embeddings.py

Why this exists: on 2026-09-04 an append made ``embed._load_progress`` return 0 (it treated
a grown corpus as a different run), which made ``embed.run`` open the memmap with mode
``"w+"``. That call truncates, so 4.16 GB of embeddings -- about 14 hours of GPU -- were
zeroed before a single paper was embedded. Both causes are now fixed in ``embed.py``: the
resume survives an append, and a ``"w+"`` over a non-empty file refuses without
``BRONTIDE_EMBED_TRUNCATE=1``.

The loss is fully recoverable and this script recovers it. Nothing was lost that is not a
pure function of ``papers.parquet`` and a fixed model: the embed is deterministic, which was
demonstrated by accident -- the ~50,000 rows the runaway did rewrite came back matching the
untouched ``pca50`` to **0.000e+00**. Everything derived from the embeddings (``pca50``,
``umap2``, centroids, regions) was never touched and is intact.

Resumable: interrupt it and run it again. It picks up at the progress checkpoint.

After it finishes, run the rest of the append:
    python -m brontide.update --harvest data/harvest/<file>.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brontide import config, embed  # noqa: E402


def main() -> int:
    prog = config.EMBED_MEMMAP.with_suffix(config.EMBED_MEMMAP.suffix + ".progress.json")
    state = json.loads(prog.read_text()) if prog.exists() else {}
    n_done = int(state.get("n_done", 0))
    n_total = int(state.get("n_total", 0))
    print(f"embedding memmap : {config.EMBED_MEMMAP}")
    print(f"  resume at      : {n_done:,} / {n_total:,}")
    print(f"  to embed       : {max(0, n_total - n_done):,} rows")
    print(f"  at 52.7/sec    : {max(0, n_total - n_done) / 52.7 / 3600:.1f} h\n", flush=True)

    t0 = time.perf_counter()
    embed.run()
    print(f"\ndone in {(time.perf_counter() - t0) / 3600:.2f} h", flush=True)

    after = json.loads(prog.read_text())
    print(f"  n_done now: {after.get('n_done'):,} / {after.get('n_total'):,}")
    print("\nnext:  python -m brontide.update --harvest data/harvest/"
          "arxiv-focus-2026-08-01-2026-08-03.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
