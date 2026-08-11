#!/usr/bin/env python3
"""Whole-pipeline smoke test on a --sample (CPU-only environment, before the host full run).

Runs every stage end-to-end on a small sample so interface/shape bugs surface on
the CPU-only environment, not 6 hours into a host embed. Default sample = 50k.
"""
from lacuna import (cli, embed, emergence, export_dashboard, export_figs, export_map,
                    forecast, ingest, labels, reduce, regions, series, validate)

STAGES = [
    ("ingest", lambda s: ingest.run(sample=s)),
    ("embed", lambda s: embed.run(sample=s)),
    ("reduce.pca", lambda s: reduce.run_pca(sample=s)),
    ("reduce.umap", lambda s: reduce.run_umap(sample=s)),
    ("regions", lambda s: regions.run(sample=s)),
    ("series.shares", lambda s: series.run_shares(sample=s)),
    ("series.novelty", lambda s: series.run_novelty(sample=s)),
    ("forecast", lambda s: forecast.run(sample=s)),
    ("emergence", lambda s: emergence.run()),
    ("labels", lambda s: labels.run()),
    ("export_map", lambda s: export_map.run()),
    ("export_figs", lambda s: export_figs.run()),
    ("export_dashboard", lambda s: export_dashboard.run()),
]

if __name__ == "__main__":
    p = cli.sample_parser(__doc__)
    p.set_defaults(sample=50_000)
    sample = p.parse_args().sample
    print(f"smoke test @ sample={sample}")
    for name, fn in STAGES:
        print(f"--- {name} ---")
        fn(sample)
    print("smoke test complete")
