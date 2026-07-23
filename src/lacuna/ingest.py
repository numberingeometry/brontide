"""Stage 1 — Ingest (host, ~1h).

Download the arXiv Kaggle snapshot (~2.6M papers, abstracts included) via
``kagglehub``, stream-parse the JSON, and write a tidy parquet:
``id, month, categories, title, abstract``.

Monthly binning makes arXiv-heatmap's 18:00/business-day logic unnecessary —
just take the calendar month of the paper's date (``CORPUS_START`` onward).
"""
from __future__ import annotations

from pathlib import Path

from . import config


def run(sample: int | None = None, out: Path = config.RAW_PARQUET) -> Path:
    """Ingest the snapshot to parquet. ``sample`` = first-N rows for smoke tests.

    Returns the parquet path. Columns: id, month (YYYY-MM period), categories
    (list[str]), title, abstract.
    """
    # TODO:
    #   1. kagglehub.dataset_download("Cornell-University/arxiv")
    #   2. stream the ~4GB JSON lines (ijson / line-by-line json.loads); if sample,
    #      stop after `sample` rows.
    #   3. parse `versions[0].created` -> monthly period; keep >= CORPUS_START.
    #   4. write parquet (pyarrow) with the schema above.
    raise NotImplementedError("ingest.run")
