#!/usr/bin/env python3
"""Stage 1 runner — ingest the arXiv snapshot to parquet."""
from brontide import cli, ingest

if __name__ == "__main__":
    sample = cli.parse_sample(__doc__)
    out = ingest.run(sample=sample)
    print(f"ingest -> {out}")
