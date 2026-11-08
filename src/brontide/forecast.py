"""Stage 6 — Forecast (CPU forecasting).

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

The pure baselines (SmoothedMomentum / Naive / SeasonalNaive) + the rolling-origin
CV + the EB-shrink run with numpy alone; the AutoETS/AutoARIMA/AutoTheta roster is
added via ``statsforecast`` when it is importable (guarded — the run degrades to
the baseline roster with a printed note otherwise).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config

# horizon months the emergence call reads (1-based, inclusive) -> 0-based slice
_H_LO, _H_HI = config.EMERGENCE_HORIZON_MONTHS          # (19, 24)


# --- baselines (pure) -------------------------------------------------------
def _annual_growth(y: np.ndarray) -> float:
    """trailing-12m mean / prior-12m mean; 1.0 if there is no prior year or it is 0."""
    if y.size < 24:
        return 1.0
    prior = y[-24:-12].mean()
    return float(y[-12:].mean() / prior) if prior > 0 else 1.0


def smoothed_momentum(y, h: int = config.HORIZON_MONTHS):
    """The serious baseline: carry forward trailing-12m vs prior-12m share growth.

    ``forecast[k] = trailing12m_mean * growth**(k/12)`` for k = 1..h — the observed
    annual share-growth extrapolated geometrically. Shares are add-α/EB smoothed
    (strictly positive), so no log guard is needed.
    """
    y = np.asarray(y, dtype=np.float64)
    level = y[-12:].mean() if y.size >= 12 else y.mean()
    g = _annual_growth(y)
    k = np.arange(1, h + 1)
    return level * g ** (k / 12.0)


def _naive(y, h):
    y = np.asarray(y, dtype=np.float64)
    return np.full(h, y[-1])


def _seasonal_naive(y, h, m: int = 12):
    y = np.asarray(y, dtype=np.float64)
    if y.size < m:
        return np.full(h, y[-1])
    return np.tile(y[-m:], int(np.ceil(h / m)))[:h]


_BASELINES = {
    "SmoothedMomentum": smoothed_momentum,
    "Naive": _naive,
    "SeasonalNaive": _seasonal_naive,
}


# --- CV metrics (pure) ------------------------------------------------------
def _mase(y_train: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, m: int = 12) -> float:
    """per-series MASE: mean(|test − pred|) / mean(|seasonal-naive error on train|)."""
    y_train = np.asarray(y_train, dtype=np.float64)
    if y_train.size > m:
        denom = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    else:
        denom = np.mean(np.abs(np.diff(y_train))) if y_train.size > 1 else 0.0
    denom = denom or 1.0
    return float(np.mean(np.abs(np.asarray(y_test, np.float64) - np.asarray(y_pred, np.float64))) / denom)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


def _horizon_growth(block: np.ndarray, base12: np.ndarray) -> np.ndarray:
    """mean share over horizon months 19–24 / trailing-12m base, per region."""
    hz = block[:, _H_LO - 1:_H_HI].mean(axis=1)
    return hz / np.where(base12 > 0, base12, 1.0)


def _rolling_cv(Y: np.ndarray, models: dict, n_windows: int = config.CV_N_ORIGINS,
                h: int = config.HORIZON_MONTHS, min_train: int = config.CV_MIN_TRAIN_MONTHS):
    """Rolling-origin CV over the (region × month) matrix ``Y``. Non-overlapping
    24-month test windows stepping back from the end; each model forecasts h ahead
    from the fold's train. Returns per-(model, window): median MASE + Spearman of
    forecast-growth vs realized-growth ranking across regions (horizon 19–24)."""
    R, T = Y.shape
    rows = []
    for w in range(n_windows):
        test_end = T - w * h
        test_start = test_end - h
        if test_start < min_train:
            break
        train, test = Y[:, :test_start], Y[:, test_start:test_end]
        base12 = train[:, -12:].mean(axis=1)
        realized_g = _horizon_growth(test, base12)
        for name, fn in models.items():
            pred = np.vstack([fn(train[r], h) for r in range(R)])
            mase = np.array([_mase(train[r], test[r], pred[r]) for r in range(R)])
            rows.append({
                "model": name, "window": w,
                "mase_median": float(np.median(mase)),
                "spearman": _spearman(_horizon_growth(pred, base12), realized_g),
            })
    return rows


# --- EB-shrink point forecasts toward the 64 super-regions ------------------
def _eb_shrink(fc: np.ndarray, super_of_region: np.ndarray, trailing_counts: np.ndarray) -> np.ndarray:
    """Shrink each region's forecast path toward its super's PER-REGION forecast
    level F_S/n_S (same P_S/n_S semantics as series), weight w_r = n_r/(n_r+κ) with
    n_r the trailing-12m count and κ the same DM-MoM estimate. (fc: R×h.)"""
    from .series import _estimate_kappa
    n_super = config.N_SUPERREGIONS
    F_S = np.zeros((n_super, fc.shape[1]), dtype=np.float64)
    np.add.at(F_S, super_of_region, fc)
    n_s = np.bincount(super_of_region, minlength=n_super).astype(np.float64)
    per_region_super = (F_S / np.where(n_s > 0, n_s, 1.0)[:, None])[super_of_region]  # R×h
    kappa = _estimate_kappa(trailing_counts, super_of_region)
    w = (trailing_counts / (trailing_counts + kappa))[:, None]
    return w * fc + (1.0 - w) * per_region_super


def run(sample: int | None = None,
        shares: Path = config.ARTIFACTS_DIR / "region_shares.parquet",
        cutoff: str | None = None,
        roster: str = "full",
        out: Path = config.ARTIFACTS_DIR / "forecasts.parquet") -> Path:
    """Fit the roster and forecast ``HORIZON_MONTHS`` ahead from ``cutoff`` (or
    latest). Writes forecasts (EB-shrunk) + the CV metrics table (MASE, Spearman).
    ``roster="baseline"`` runs only the numpy baselines (fast; skips the
    ~30-min statsforecast auto-models). Returns the forecasts path."""
    import json

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.read_parquet(shares)
    t0 = cutoff or config.CORPUS_END
    df = df[df["month"] <= t0]
    months = sorted(df["month"].unique())
    K = config.N_REGIONS
    h = config.HORIZON_MONTHS

    # (region × month) matrices of the smoothed share (the target) and raw counts.
    piv = df.pivot(index="region", columns="month", values="share_smoothed").reindex(range(K))
    Y = piv[months].to_numpy(np.float64)
    super_of_region = (df.groupby("region")["super_region"].first().reindex(range(K)).to_numpy(np.int64))
    cnt = df.pivot(index="region", columns="month", values="count").reindex(range(K))[months].to_numpy(np.float64)
    trailing_counts = cnt[:, -12:].sum(axis=1)                     # raw trailing-12m count, per region

    models = dict(_BASELINES)
    interval_fc: dict[str, tuple] = {}   # model -> (mean R×h, lo R×h, hi R×h) from statsforecast

    # --- statsforecast auto-models (guarded) --------------------------------
    sf_ok = False
    if roster == "full":
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
            sf_ok = True
        except Exception as e:                                    # noqa: BLE001
            print(f"forecast.run: statsforecast unavailable ({e.__class__.__name__}); "
                  f"baseline roster only (SmoothedMomentum/Naive/SeasonalNaive).")

    cv_rows = _rolling_cv(Y, models)

    if sf_ok:
        long = df[["region", "month", "share_smoothed"]].rename(
            columns={"region": "unique_id", "month": "ds", "share_smoothed": "y"})
        long["ds"] = pd.PeriodIndex(long["ds"], freq="M").to_timestamp()
        auto = [AutoETS(season_length=12), AutoARIMA(season_length=12), AutoTheta(season_length=12)]
        sf = StatsForecast(models=auto, freq="MS", n_jobs=-1)
        # CV for the auto-models (statsforecast batch); fold into the same metrics.
        cvdf = sf.cross_validation(df=long, h=h, step_size=h, n_windows=config.CV_N_ORIGINS)
        cv_rows += _sf_cv_metrics(cvdf, super_of_region.shape[0])
        fc = sf.forecast(df=long, h=h, level=[80])
        for m in ("AutoETS", "AutoARIMA", "AutoTheta"):
            mean = fc.pivot(index="unique_id", columns="ds", values=m).reindex(range(K)).to_numpy(np.float64)
            lo = fc.pivot(index="unique_id", columns="ds", values=f"{m}-lo-80").reindex(range(K)).to_numpy(np.float64)
            hi = fc.pivot(index="unique_id", columns="ds", values=f"{m}-hi-80").reindex(range(K)).to_numpy(np.float64)
            interval_fc[m] = (mean, lo, hi)

    # --- final forecast per model, EB-shrunk, written long ------------------
    fc_months = _future_months(months[-1], h)
    out_rows = []
    for name, fn in models.items():
        raw = np.vstack([fn(Y[r], h) for r in range(K)])
        shrunk = _eb_shrink(raw, super_of_region, trailing_counts)
        for r in range(K):
            for j in range(h):
                out_rows.append((r, fc_months[j], name, shrunk[r, j], np.nan, np.nan))
    for name, (mean, lo, hi) in interval_fc.items():
        shrunk = _eb_shrink(np.nan_to_num(mean), super_of_region, trailing_counts)
        for r in range(K):
            for j in range(h):
                out_rows.append((r, fc_months[j], name, shrunk[r, j], lo[r, j], hi[r, j]))

    fc_df = pd.DataFrame(out_rows, columns=["region", "month", "model", "mean", "lo", "hi"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(fc_df, preserve_index=False), out, compression="zstd")

    # CV summary + model selection (co-primary: max mean-Spearman, tie-break median MASE).
    cv = pd.DataFrame(cv_rows)
    summ = (cv.groupby("model").agg(spearman=("spearman", "mean"),
                                    mase=("mase_median", "median")).reset_index())
    summ = summ.sort_values(["spearman", "mase"], ascending=[False, True])
    selected = summ.iloc[0]["model"]
    cv.to_parquet(out.with_name("forecast_cv.parquet"), index=False)
    (out.with_suffix(out.suffix + ".meta.json")).write_text(json.dumps({
        "cutoff_T0": t0, "horizon": h, "models": list(models) + list(interval_fc),
        "selected_model": selected, "cv_summary": summ.to_dict("records"),
        "statsforecast": sf_ok,
    }, indent=2))

    print(f"\nforecast.run[{config.resolve_sample(sample)}] -> {out}\n"
          f"  T0={t0}  horizon={h}m  models={list(models) + list(interval_fc)}\n"
          f"  selected (co-primary MASE+Spearman) = {selected}\n"
          + summ.to_string(index=False))
    return out


def _future_months(last: str, h: int) -> list[str]:
    y, m = map(int, last.split("-"))
    o = y * 12 + (m - 1)
    return [f"{(o + k) // 12:04d}-{(o + k) % 12 + 1:02d}" for k in range(1, h + 1)]


def _sf_cv_metrics(cvdf, n_regions: int):
    """Per-(auto-model, window) median MASE + Spearman(growth ranks) from a
    statsforecast cross_validation frame. Mirrors ``_rolling_cv``'s metrics."""
    import numpy as np
    import pandas as pd

    rows = []
    models = [c for c in cvdf.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    for w, (_, g) in enumerate(cvdf.groupby("cutoff")):
        g = g.sort_values(["unique_id", "ds"])
        realized = g.groupby("unique_id")["y"].apply(
            lambda s: s.to_numpy()[_H_LO - 1:_H_HI].mean())
        base = g.groupby("unique_id")["y"].apply(lambda s: s.to_numpy()[:12].mean())
        for m in models:
            pred_g = g.groupby("unique_id")[m].apply(
                lambda s: s.to_numpy()[_H_LO - 1:_H_HI].mean()) / base.replace(0, np.nan)
            real_g = realized / base.replace(0, np.nan)
            mase = g.groupby("unique_id").apply(
                lambda s: np.mean(np.abs(s["y"].to_numpy() - s[m].to_numpy())) / (np.mean(np.abs(np.diff(s["y"].to_numpy()))) or 1.0))
            rows.append({"model": m, "window": w,
                         "mase_median": float(mase.median()),
                         "spearman": _spearman(pred_g.fillna(0).to_numpy(), real_g.fillna(0).to_numpy())})
    return rows
