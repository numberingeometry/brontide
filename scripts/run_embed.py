#!/usr/bin/env python3
"""Stage 2 runner — SPECTER2 embeddings (host GPU). Prints papers/sec (fallback trigger)."""
from brontide import cli, embed

if __name__ == "__main__":
    sample = cli.parse_sample(__doc__)
    out = embed.run(sample=sample)
    print(f"embed -> {out}")
