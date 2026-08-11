#!/usr/bin/env python3
"""Stage 10c runner — build the self-contained dashboard.html from committed artifacts."""
from brontide import cli, export_dashboard

if __name__ == "__main__":
    cli.parse_sample(__doc__)
    out = export_dashboard.run()
    print(f"dashboard -> {out}")
