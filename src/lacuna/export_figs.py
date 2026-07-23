"""Stage 10b — Export figures (CPU-only environment).

Write per-super-region history + forecast JSON for the site's draggable OJS figure
(site Phase 3a). One record per super-region: the smoothed-share history, the
24-month forecast with intervals, and the emergence label/ĝ.
"""
from __future__ import annotations

from pathlib import Path

from . import config


def run(shares: Path = config.ARTIFACTS_DIR / "region_shares.parquet",
        forecasts: Path = config.ARTIFACTS_DIR / "forecasts.parquet",
        emergence: Path = config.ARTIFACTS_DIR / "emergence.parquet",
        out: Path = config.ARTIFACTS_DIR / "region_figs.json") -> Path:
    """Emit ``[{id, label, history:[{month,share}], forecast:[{month,mean,lo,hi}],
    g_hat, class}]`` for the OJS figure. Returns path."""
    # TODO: aggregate micro-regions to the 64 super-regions; join labels + emergence;
    #       serialize compact JSON (site reads it directly).
    raise NotImplementedError("export_figs.run")
