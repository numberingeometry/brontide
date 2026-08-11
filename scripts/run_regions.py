#!/usr/bin/env python3
"""Stage 4 runner — K=1024 microclusters + Ward tree -> 64 super-regions."""
from brontide import cli, regions

if __name__ == "__main__":
    sample = cli.parse_sample(__doc__)
    out = regions.run(sample=sample)
    print(f"regions -> {out}")
