#!/usr/bin/env python3
"""Stage 9 runner — c-TF-IDF top terms per super-region."""
from lacuna import cli, labels

if __name__ == "__main__":
    cli.parse_sample(__doc__)
    out = labels.run()
    print(f"labels -> {out}")
