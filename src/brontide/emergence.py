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

import numpy as np

from . import config


def g_hat(forecast_means: np.ndarray, trailing12m_share: np.ndarray) -> np.ndarray:
    """ĝ = mean(forecast over horizon months 19–24) / trailing-12m avg share at T0.

    ``forecast_means``: (R, HORIZON) point forecasts (month k=1..24 in column k−1);
    ``trailing12m_share``: (R,) trailing-12m avg of the smoothed share at T0.
    """
    lo, hi = config.EMERGENCE_HORIZON_MONTHS               # (19, 24) 1-based inclusive
    num = forecast_means[:, lo - 1:hi].mean(axis=1)
    return num / np.where(trailing12m_share > 0, trailing12m_share, np.nan)


def run(forecasts: Path = config.ARTIFACTS_DIR / "forecasts.parquet",
        shares: Path = config.ARTIFACTS_DIR / "region_shares.parquet",
        cutoff: str | None = None,
        model: str | None = None,
        out: Path = config.ARTIFACTS_DIR / "emergence.parquet") -> Path:
    """Compute ĝ and assign emerging/declining/saturating labels. ``model`` selects
    which roster forecast to rank by (default: the ``selected_model`` recorded in the
    forecasts meta). Returns the emergence path."""
    import json

    import pandas as pd

    fc = pd.read_parquet(forecasts)
    if model is None:
        meta = json.loads(Path(str(forecasts) + ".meta.json").read_text())
        model = meta.get("selected_model", "SmoothedMomentum")
    fc = fc[fc["model"] == model].copy()
    K = config.N_REGIONS
    h = config.HORIZON_MONTHS

    # (region × horizon-month) forecast means, columns ordered by month.
    fmonths = sorted(fc["month"].unique())
    fmeans = (fc.pivot(index="region", columns="month", values="mean")
                .reindex(range(K))[fmonths].to_numpy(np.float64))

    sh = pd.read_parquet(shares)
    t0 = cutoff or config.CORPUS_END
    sh = sh[sh["month"] <= t0]
    smonths = sorted(sh["month"].unique())
    S = sh.pivot(index="region", columns="month", values="share_smoothed").reindex(range(K))[smonths].to_numpy(np.float64)
    C = sh.pivot(index="region", columns="month", values="count").reindex(range(K))[smonths].to_numpy(np.float64)
    super_of_region = sh.groupby("region")["super_region"].first().reindex(range(K)).to_numpy(np.int64)

    trailing12_share = S[:, -12:].mean(axis=1)
    trailing12_count = C[:, -12:].sum(axis=1)                       # RAW count for the eligibility floor
    growth_24m = np.where(S[:, -24:-12].mean(axis=1) > 0,
                          S[:, -12:].mean(axis=1) / S[:, -24:-12].mean(axis=1), np.nan)
    gh = g_hat(fmeans, trailing12_share)

    eligible = trailing12_count >= config.ELIGIBILITY_FLOOR
    order_desc = np.lexsort((trailing12_count, np.nan_to_num(gh)))[::-1]   # ĝ desc, ties by size
    cls = np.array(["none"] * K, dtype=object)

    # Emerging: top-N eligible by ĝ with ĝ >= threshold.
    emerging = [r for r in order_desc if eligible[r] and gh[r] >= config.G_EMERGING][:config.TOP_N]
    cls[emerging] = "emerging"
    # Declining: bottom-N eligible with ĝ <= threshold.
    order_asc = np.lexsort((trailing12_count, np.nan_to_num(gh, nan=np.inf)))
    declining = [r for r in order_asc if eligible[r] and gh[r] <= config.G_DECLINING][:config.TOP_N]
    cls[declining] = "declining"
    # Saturating (separate, non-exclusive with the above only where unset): flattening after growth.
    sat_lo, sat_hi = config.SATURATING_G_BAND
    sat = eligible & (growth_24m >= config.SATURATING_GROWTH_24M) & (gh >= sat_lo) & (gh <= sat_hi)
    cls[np.where(sat & (cls == "none"))[0]] = "saturating"

    import pandas as pd  # noqa: F811  (local, keeps the top import optional)
    em = pd.DataFrame({
        "region": np.arange(K), "super_region": super_of_region,
        "g_hat": gh, "trailing12m_share": trailing12_share,
        "trailing12m_count": trailing12_count.astype(np.int64),
        "growth_24m": growth_24m, "eligible": eligible, "class": cls, "model": model,
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    em.to_parquet(out, index=False)

    n = {c: int((cls == c).sum()) for c in ("emerging", "declining", "saturating")}
    print(f"\nemergence.run -> {out}\n"
          f"  T0={t0}  model={model}  eligible={int(eligible.sum())}/{K}\n"
          f"  emerging={n['emerging']}  declining={n['declining']}  saturating={n['saturating']}\n"
          f"  g-hat (eligible): min {np.nanmin(gh[eligible]):.2f}  med {np.nanmedian(gh[eligible]):.2f}  max {np.nanmax(gh[eligible]):.2f}")
    return out
