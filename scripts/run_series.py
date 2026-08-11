#!/usr/bin/env python3
"""Stage 5 runner — EB-smoothed region shares + novelty stream."""
from brontide import cli, series

if __name__ == "__main__":
    sample = cli.parse_sample(__doc__)
    shares = series.run_shares(sample=sample)
    novelty = series.run_novelty(sample=sample)
    print(f"series -> shares={shares} novelty={novelty}")
