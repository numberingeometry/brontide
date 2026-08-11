#!/usr/bin/env python3
"""Stage 10b runner — rebuild every figure from committed artifacts.

Needs no raw data, no GPU and no network: everything is read out of ``artifacts/``.
Takes about a minute, most of it the 20-frame GIF.
"""
from brontide import cli, export_figs

if __name__ == "__main__":
    cli.parse_sample(__doc__)
    made = export_figs.run()
    print(f"figures -> {len(made)} written to figures/")
