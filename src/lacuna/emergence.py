"""Stage 7 — Emergence (CPU-only environment).

Pre-registered rules (ALL numbers in ``config.py`` + frozen in
``artifacts/validation.md`` before any scoring run):

At cutoff T0, rank regions by
    ĝ(r) = forecast smoothed-share (avg of horizon months 19–24)
           / smoothed-share(T0, trailing-12m avg).

- **Emerging** = top-``TOP_N`` by ĝ with ĝ ≥ ``G_EMERGING`` (1.5), eligibility
  floor ≥ ``ELIGIBILITY_FLOOR`` papers in trailing 12m at T0, ties by region size.
- **Declining** = bottom-``TOP_N`` with ĝ ≤ ``G_DECLINING`` (0.75) — the honest
  label: share decline, not saturation.
- **Saturating** (reported separately) = flattening after growth:
  trailing-24m growth ≥ ``SATURATING_GROWTH_24M`` AND ĝ in ``SATURATING_G_BAND``.
"""
from __future__ import annotations

from pathlib import Path

from . import config


def run(forecasts: Path = config.ARTIFACTS_DIR / "forecasts.parquet",
        shares: Path = config.ARTIFACTS_DIR / "region_shares.parquet",
        cutoff: str | None = None,
        out: Path = config.ARTIFACTS_DIR / "emergence.parquet") -> Path:
    """Compute ĝ and assign emerging/declining/saturating labels. Returns path."""
    # TODO:
    #   1. g_hat = mean(forecast[months 19..24]) / trailing12m_avg(share, T0).
    #   2. apply the pre-registered thresholds above (config constants only).
    raise NotImplementedError("emergence.run")
