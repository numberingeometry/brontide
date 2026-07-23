#!/usr/bin/env python3
"""Stage 8 runner — out-of-time harness (3 cutoffs) vs smoothed-momentum baseline.

Pre-registration gate: artifacts/validation.md thresholds must already be
committed (dated) BEFORE this scoring run.
"""
from lacuna import cli, validate

if __name__ == "__main__":
    cli.parse_sample(__doc__)  # validate always runs full-history per cutoff; --sample is a no-op smoke flag
    out = validate.run()
    print(f"validate -> {out}")
