#!/usr/bin/env python3
"""Stage 10 runner — write the site's map binaries + per-region forecast JSON."""
from lacuna import cli, export_figs, export_map

if __name__ == "__main__":
    cli.parse_sample(__doc__)
    mp = export_map.run()
    figs = export_figs.run()
    print(f"export -> map={mp}, {len(figs)} figures")
