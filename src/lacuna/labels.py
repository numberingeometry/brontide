"""Stage 9 — Labels (host, minutes).

c-TF-IDF top terms per super-region (class-based TF-IDF over the 64 super-region
document-aggregates), plus nearest-centroid paper titles as a sanity check.
LLM label polish is a later phase — v1 ships the c-TF-IDF terms.
"""
from __future__ import annotations

from pathlib import Path

from . import config


def run(out: Path = config.ARTIFACTS_DIR / "labels.json") -> Path:
    """Write ``{super_region_id: {terms: [...], sample_titles: [...]}}``. Returns path."""
    # TODO:
    #   1. concat title+abstract per super-region -> one pseudo-doc each.
    #   2. c-TF-IDF (CountVectorizer over 64 docs, class-based idf) -> top terms.
    #   3. nearest-to-centroid titles for eyeball sanity.
    raise NotImplementedError("labels.run")
