"""Stage 6 — Forecast (host/CPU-only environment, ~30min CPU).

``statsforecast`` (Nixtla) roster from ``config.FORECAST_MODELS``; the serious
baseline is **SmoothedMomentum** (trailing-12m vs prior-12m share growth, carried
forward). Horizon = ``HORIZON_MONTHS`` (24); rolling-origin CV with 24-month test
windows (``CV_N_ORIGINS`` origins, ``CV_MIN_TRAIN_MONTHS`` min train), scored
overall AND on months 19–24 (the horizon the emergence call reads).

**Co-primary selection metrics:** per-series **MASE** (reported per-series, not
pooled) + **Spearman** of forecast-growth vs realized-growth ranking on CV folds
(the deliverable is a ranking — select for it).

v1 forecasts are **EB-shrunk toward the 64 super-regions** (cheap; uses the Ward
tree already computed). Full multi-level MinT reconciliation = v1.5. Disclosed
limitation: shares are compositional (sum to 1); 1024 univariate forecasts ignore
the constraint — tolerable for ranking, stated explicitly.
"""
from __future__ import annotations

from pathlib import Path

from . import config


def run(sample: int | None = None,
        shares: Path = config.ARTIFACTS_DIR / "region_shares.parquet",
        cutoff: str | None = None,
        out: Path = config.ARTIFACTS_DIR / "forecasts.parquet") -> Path:
    """Fit the roster and forecast ``HORIZON_MONTHS`` ahead from ``cutoff`` (or
    latest). Writes forecasts + the CV metrics table (MASE, Spearman). Returns path."""
    # TODO:
    #   1. build StatsForecast(models=[...], freq="MS"); custom SmoothedMomentum.
    #   2. cross_validation(h=24, test_size=24, n_windows=CV_N_ORIGINS, step_size=...).
    #   3. per-series MASE + Spearman(forecast-growth, realized-growth) per fold.
    #   4. EB-shrink point forecasts toward super-region forecast.
    raise NotImplementedError("forecast.run")


def smoothed_momentum(y, h: int = config.HORIZON_MONTHS):
    """The serious baseline: carry forward trailing-12m vs prior-12m share growth."""
    raise NotImplementedError("forecast.smoothed_momentum")
