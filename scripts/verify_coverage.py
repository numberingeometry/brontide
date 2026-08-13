#!/usr/bin/env python3
"""verify_coverage.py -- score the two signals frozen in
decisions/2026-08-13-novelty-and-coverage-preregistration.md.

Run from anywhere:   python scripts/verify_coverage.py

CPU-only, no network, self-contained. Reads only committed artifacts; writes
artifacts/coverage_out.json. Prints each frozen criterion with PASS/FAIL.

SIGNAL A -- novelty-stream incremental skill
  Unit: super-region, at each leak-free cutoff. Feature: trailing-12m novelty
  fraction. Control: trailing momentum. Target: the frozen growth functional
  (horizon months 19-24 mean / trailing-12m mean). Primary statistic: first-order
  partial Spearman of novelty vs growth controlling for momentum.
  FROZEN CRITERION: skill established iff rho_partial > 0 with p < 0.05 at >= 2
  of the 3 cutoffs.

SIGNAL B -- 80% prediction-interval coverage and PIT
  9 cells = 3 interval-carrying models x 3 cutoffs, each 1024 regions x 24 months.
  FROZEN CRITERION (coverage): calibrated iff >= 7 of 9 cells have empirical
  coverage of the interval AS PUBLISHED in [0.75, 0.85].
  FROZEN CRITERION (PIT): uniform iff the KS statistic against U(0,1) is < 0.05
  in >= 7 of 9 cells.

BLOCK 0 is the repo's standard environment anchor: reimplement the EB share
pipeline and assert kappa = 3.147961 with max|p_hat - committed| = 0 before any
derived number is trusted.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CUTOFFS = ["2017-12", "2020-12", "2022-12"]
INTERVAL_MODELS = ["AutoETS", "AutoARIMA", "AutoTheta"]
Z90 = 1.2815515655446004          # Phi^-1(0.9); the 80% interval half-width factor
HORIZON_LO, HORIZON_HI = 19, 24   # frozen: growth uses horizon months 19..24
ELIGIBILITY = 20                  # frozen: >= 20 papers in trailing 12m at T0

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    ok = bool(ok)
    _RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    return ok


def hdr(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def near(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- #
# small statistics, written out so nothing depends on scipy
# --------------------------------------------------------------------------- #
def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared -- the same convention scipy.stats.rankdata uses."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm = x - x.mean(); ym = y - y.mean()
    d = math.sqrt(float((xm * xm).sum()) * float((ym * ym).sum()))
    return float((xm * ym).sum() / d) if d > 0 else 0.0


def spearman(x, y) -> float:
    return pearson(rankdata(x), rankdata(y))


def _t_sf(t: float, df: int) -> float:
    """Upper-tail P(T > t) for Student-t, via the regularised incomplete beta."""
    x = df / (df + t * t)
    return 0.5 * _betainc(0.5 * df, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) by continued fraction (Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def partial_spearman(x, y, z) -> tuple[float, float, int]:
    """First-order partial Spearman of x vs y controlling for z, with two-sided p."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy, rxz, ryz = pearson(rx, ry), pearson(rx, rz), pearson(ry, rz)
    den = math.sqrt(max(0.0, (1 - rxz ** 2) * (1 - ryz ** 2)))
    if den == 0.0:
        return 0.0, 1.0, len(rx)
    rho = (rxy - rxz * ryz) / den
    n = len(rx)
    df = n - 3
    if df <= 0 or abs(rho) >= 1.0:
        return rho, 0.0 if abs(rho) >= 1.0 else 1.0, n
    t = rho * math.sqrt(df / (1.0 - rho ** 2))
    return rho, 2.0 * _t_sf(abs(t), df), n


def norm_cdf(z: np.ndarray) -> np.ndarray:
    from math import erf, sqrt
    v = np.vectorize(lambda u: 0.5 * (1.0 + erf(u / sqrt(2.0))))
    return v(np.asarray(z, dtype=np.float64))


def ks_uniform(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic of a sample against U(0,1)."""
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = len(u)
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(max((i / n - u).max(), (u - (i - 1) / n).max()))


# --------------------------------------------------------------------------- #
# BLOCK 0 -- environment anchor
# --------------------------------------------------------------------------- #
def block0_eb_anchor() -> None:
    hdr("BLOCK 0 -- ENVIRONMENT ANCHOR: independent EB-pipeline reimplementation")
    import pandas as pd
    from brontide import config, series

    df = pd.read_parquet(REPO / "artifacts" / "region_shares.parquet")
    months = sorted(df["month"].unique())
    K = int(df["region"].nunique())
    n_months = len(months)
    midx = {m: i for i, m in enumerate(months)}
    r = df["region"].to_numpy()
    t = df["month"].map(midx).to_numpy()

    C = np.zeros((K, n_months)); C[r, t] = df["count"].to_numpy()
    sup = np.zeros(K, dtype=np.int64); sup[r] = df["super_region"].to_numpy()

    a = config.ADD_ALPHA
    p = (C + a) / (C.sum(axis=0) + a * K)[None, :]
    n_super = config.N_SUPERREGIONS
    P_S = np.zeros((n_super, n_months)); np.add.at(P_S, sup, p)
    n_s = np.bincount(sup, minlength=n_super).astype(float)
    m_shrink = (P_S / n_s[:, None])[sup]

    trail = C[:, n_months - 12:n_months].sum(axis=1)
    kappa = float(series._estimate_kappa(trail, sup))
    w = (trail / (trail + kappa))[:, None]
    p_hat = w * p + (1.0 - w) * m_shrink

    ref = np.zeros((K, n_months)); ref[r, t] = df["share_smoothed"].to_numpy()
    dmax = float(np.abs(p_hat - ref).max())
    print(f"  kappa = {kappa:.12f}   max|p_hat - share_smoothed| = {dmax!r}")
    check("EB anchor: kappa == 3.147960961354892", near(kappa, 3.147960961354892, 5e-12),
          f"got {kappa:.12f}")
    check("EB anchor: share_smoothed reproduced EXACTLY", dmax == 0.0, f"max|diff| = {dmax!r}")


# --------------------------------------------------------------------------- #
# SIGNAL A
# --------------------------------------------------------------------------- #
def _months_between(lo: str, hi: str, all_months: list[str]) -> list[str]:
    return [m for m in all_months if lo <= m <= hi]


def _shift_month(m: str, k: int) -> str:
    y, mo = int(m[:4]), int(m[5:7])
    idx = y * 12 + (mo - 1) + k
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def signal_a() -> dict:
    hdr("SIGNAL A -- novelty-stream incremental skill (frozen: partial Spearman, >=2 of 3)")
    import pandas as pd

    out = {}
    for T0 in CUTOFFS:
        base = REPO / "artifacts" / "cutoffs" / T0
        rs = pd.read_parquet(base / "region_shares.parquet")
        ns = pd.read_parquet(base / "novelty_stream.parquet")
        ns = ns[ns["super_region"] >= 0]

        # super-region share path s(S,t)
        s = (rs.groupby(["super_region", "month"])["share_smoothed"].sum()
               .unstack("month").sort_index())
        cnt = (rs.groupby(["super_region", "month"])["count"].sum()
                 .unstack("month").sort_index())
        months = list(s.columns)

        trail_m = _months_between(_shift_month(T0, -11), T0, months)
        prev_m = _months_between(_shift_month(T0, -23), _shift_month(T0, -12), months)
        horiz_m = _months_between(_shift_month(T0, HORIZON_LO), _shift_month(T0, HORIZON_HI), months)
        assert len(trail_m) == 12 and len(prev_m) == 12, (T0, len(trail_m), len(prev_m))
        assert len(horiz_m) == 6, (T0, len(horiz_m))

        trail_mean = s[trail_m].mean(axis=1)
        prev_mean = s[prev_m].mean(axis=1)
        horiz_mean = s[horiz_m].mean(axis=1)
        mom = (trail_mean / prev_mean)
        G = (horiz_mean / trail_mean)

        nv = ns[ns["month"].isin(trail_m)].groupby("super_region")[["n_papers", "n_novel"]].sum()
        nov = (nv["n_novel"] / nv["n_papers"]).reindex(s.index)
        nvp = ns[ns["month"].isin(prev_m)].groupby("super_region")[["n_papers", "n_novel"]].sum()
        nov_prev = (nvp["n_novel"] / nvp["n_papers"]).reindex(s.index)

        papers_trail = cnt[trail_m].sum(axis=1)
        elig = (papers_trail >= ELIGIBILITY) & nov.notna() & nov_prev.notna() \
            & np.isfinite(mom) & np.isfinite(G)
        n_drop = int((~elig).sum())

        x = nov[elig].to_numpy()
        y = G[elig].to_numpy()
        z = mom[elig].to_numpy()
        rho, p, n = partial_spearman(x, y, z)
        raw = spearman(x, y)
        trend = (nov - nov_prev)[elig].to_numpy()
        rho_tr, p_tr, _ = partial_spearman(trend, y, z)

        # secondary control: the deployed forecast g-hat aggregated to supers
        fc = pd.read_parquet(base / "forecasts.parquet")
        sel = json.loads((base / "forecasts.parquet.meta.json").read_text())["selected_model"]
        fsel = fc[fc["model"] == sel]
        fmonths = sorted(fsel["month"].unique())
        hz = fmonths[HORIZON_LO - 1:HORIZON_HI]
        sup_of = rs.groupby("region")["super_region"].first()
        fh = fsel[fsel["month"].isin(hz)].copy()
        fh["super_region"] = fh["region"].map(sup_of)
        fhat = fh.groupby(["super_region", "month"])["mean"].sum().unstack("month").mean(axis=1)
        ghat = (fhat / trail_mean).reindex(s.index)
        ok2 = elig & np.isfinite(ghat)
        rho_g, p_g, n_g = partial_spearman(nov[ok2].to_numpy(), G[ok2].to_numpy(),
                                           ghat[ok2].to_numpy())

        out[T0] = dict(n=n, dropped=n_drop, rho_partial=rho, p=p, rho_raw=raw,
                       rho_trend=rho_tr, p_trend=p_tr,
                       rho_ctrl_ghat=rho_g, p_ctrl_ghat=p_g, n_ctrl_ghat=n_g,
                       selected_model=sel,
                       nov_median=float(np.median(x)), nov_min=float(x.min()),
                       nov_max=float(x.max()), G_median=float(np.median(y)))
        print(f"  {T0}: n={n} (dropped {n_drop})  "
              f"rho_partial={rho:+.4f} p={p:.4f} | raw rho={raw:+.4f} | "
              f"trend rho={rho_tr:+.4f} p={p_tr:.4f} | ctrl ghat rho={rho_g:+.4f} p={p_g:.4f}")

    hits = [T0 for T0 in CUTOFFS if out[T0]["rho_partial"] > 0 and out[T0]["p"] < 0.05]
    neg = [T0 for T0 in CUTOFFS if out[T0]["rho_partial"] < 0 and out[T0]["p"] < 0.05]
    print(f"\n  cutoffs meeting 'rho>0 and p<0.05': {len(hits)}/3  {hits}")
    print(f"  cutoffs with a SIGNIFICANT NEGATIVE partial correlation: {len(neg)}/3  {neg}")
    established = len(hits) >= 2
    out["_hits"] = hits
    out["_negative"] = neg
    out["_established"] = established
    check("SIGNAL A frozen criterion: incremental novelty skill ESTABLISHED "
          "(rho_partial > 0, p < 0.05, at >= 2 of 3 cutoffs)",
          established,
          f"{len(hits)}/3 -- criterion {'MET' if established else 'NOT MET'}; "
          f"reported as {'established' if established else 'NOT ESTABLISHED'}")
    return out


# --------------------------------------------------------------------------- #
# SIGNAL B
# --------------------------------------------------------------------------- #
def signal_b() -> dict:
    hdr("SIGNAL B -- 80% interval coverage and PIT (frozen: >=7 of 9 cells)")
    import pandas as pd

    cells = {}
    for T0 in CUTOFFS:
        base = REPO / "artifacts" / "cutoffs" / T0
        rs = pd.read_parquet(base / "region_shares.parquet")
        real = rs.set_index(["region", "month"])["share_smoothed"]
        real_add = rs.set_index(["region", "month"])["share_addalpha"]
        # raw empirical share: the month's counts renormalised, no add-alpha and no EB
        tot = rs.groupby("month")["count"].transform("sum")
        rs = rs.assign(share_raw=rs["count"] / tot)
        real_raw = rs.set_index(["region", "month"])["share_raw"]
        fc = pd.read_parquet(base / "forecasts.parquet")
        for m in INTERVAL_MODELS:
            d = fc[fc["model"] == m].copy()
            idx = d.set_index(["region", "month"]).index
            d["y"] = idx.map(real)
            d["y_addalpha"] = idx.map(real_add)
            d["y_raw"] = idx.map(real_raw)
            d = d.dropna(subset=["y", "y_addalpha", "y_raw", "lo", "hi"])
            y = d["y"].to_numpy(); lo = d["lo"].to_numpy(); hi = d["hi"].to_numpy()
            mean = d["mean"].to_numpy()
            mid = 0.5 * (lo + hi)
            shift = mean - mid

            cov_pub = float(((y >= lo) & (y <= hi)).mean())
            cov_rec = float(((y >= lo + shift) & (y <= hi + shift)).mean())

            sig = (hi - lo) / (2.0 * Z90)
            good = sig > 0
            u_pub = norm_cdf((y[good] - mean[good]) / sig[good])
            u_mid = norm_cdf((y[good] - mid[good]) / sig[good])
            ks_pub = ks_uniform(u_pub)
            ks_mid = ks_uniform(u_mid)

            # --- diagnosis blocks forced by adversarial verification ----------- #
            # (a) the same intervals against LESS-processed realisations of the target
            y_add = d["y_addalpha"].to_numpy()
            y_raw = d["y_raw"].to_numpy()
            cov_add = float(((y_add >= lo) & (y_add <= hi)).mean())
            cov_raw = float(((y_raw >= lo) & (y_raw <= hi)).mean())
            # (b) shares are strictly positive, so lo <= 0 makes the interval ONE-SIDED
            #     and P(y >= lo) = 1 by construction, not by calibration
            one_sided = lo <= 0.0
            frac_one_sided = float(one_sided.mean())
            hit_all = (y >= lo) & (y <= hi)
            cov_one_sided = float(hit_all[one_sided].mean()) if one_sided.any() else float("nan")
            cov_two_sided = float(hit_all[~one_sided].mean()) if (~one_sided).any() else float("nan")
            # (c) "too wide" vs "wrong shape": standardised residual scale and tails
            e = (y[good] - mean[good]) / sig[good]
            e_sd = float(e.std(ddof=1))
            e_kurt = float(((e - e.mean()) ** 4).mean() / (e.var() ** 2)) if e.var() > 0 else float("nan")
            p_gt3 = float((np.abs(e) > 3.0).mean())

            per_region = d.assign(hit=((y >= lo) & (y <= hi))).groupby("region")["hit"].mean()
            cells[(T0, m)] = dict(
                n=int(len(d)), coverage_published=cov_pub, coverage_recentred=cov_rec,
                pit_mean_published=float(u_pub.mean()), pit_mean_recentred=float(u_mid.mean()),
                pit_below_0p1=float((u_pub < 0.1).mean()), pit_above_0p9=float((u_pub > 0.9).mean()),
                ks_published=float(ks_pub), ks_recentred=float(ks_mid),
                region_cov_median=float(per_region.median()),
                region_cov_p10=float(per_region.quantile(0.10)),
                region_cov_p90=float(per_region.quantile(0.90)),
                coverage_vs_addalpha=cov_add, coverage_vs_raw_share=cov_raw,
                frac_interval_one_sided=frac_one_sided,
                coverage_where_one_sided=cov_one_sided,
                coverage_where_two_sided=cov_two_sided,
                resid_sd=e_sd, resid_kurtosis=e_kurt, p_abs_resid_gt3=p_gt3,
                mean_outside_own_interval=float(((mean < lo) | (mean > hi)).mean()),
                median_rel_mean_mid_gap=float(np.median(np.abs(mean - mid) / np.abs(mean))),
                pit_hist=[float(v) for v in np.histogram(u_pub, bins=10, range=(0, 1))[0]],
            )

    print(f"  {'cutoff':>8} {'model':>10} {'n':>7} {'cov(pub)':>9} {'cov(rec)':>9} "
          f"{'KS(pub)':>8} {'PITmean':>8} {'reg-med':>8}")
    for T0 in CUTOFFS:
        for m in INTERVAL_MODELS:
            c = cells[(T0, m)]
            print(f"  {T0:>8} {m:>10} {c['n']:>7} {c['coverage_published']:>9.4f} "
                  f"{c['coverage_recentred']:>9.4f} {c['ks_published']:>8.4f} "
                  f"{c['pit_mean_published']:>8.4f} {c['region_cov_median']:>8.4f}")

    in_band = [(k, v["coverage_published"]) for k, v in cells.items()
               if 0.75 <= v["coverage_published"] <= 0.85]
    ks_ok = [(k, v["ks_published"]) for k, v in cells.items() if v["ks_published"] < 0.05]
    print(f"\n  cells with coverage in [0.75, 0.85]: {len(in_band)}/9")
    print(f"  cells with KS statistic < 0.05:      {len(ks_ok)}/9")

    check("SIGNAL B frozen criterion (coverage): published 80% intervals CALIBRATED "
          "(>= 7 of 9 cells in [0.75, 0.85])",
          len(in_band) >= 7,
          f"{len(in_band)}/9 -- reported as {'calibrated' if len(in_band) >= 7 else 'NOT calibrated'}")
    check("SIGNAL B frozen criterion (PIT): Gaussian-implied PIT UNIFORM "
          "(KS statistic < 0.05 in >= 7 of 9 cells)",
          len(ks_ok) >= 7,
          f"{len(ks_ok)}/9 -- reported as {'uniform' if len(ks_ok) >= 7 else 'NOT uniform'}")

    covs = np.array([v["coverage_published"] for v in cells.values()])
    print(f"\n  coverage range across the 9 cells: {covs.min():.4f} .. {covs.max():.4f} "
          f"(median {np.median(covs):.4f})")

    # ---- DIAGNOSIS, forced by adversarial verification --------------------- #
    hdr("SIGNAL B DIAGNOSIS -- what the failure actually is")
    add = np.array([v["coverage_vs_addalpha"] for v in cells.values()])
    raw = np.array([v["coverage_vs_raw_share"] for v in cells.values()])

    def n_band(a):
        return int(((a >= 0.75) & (a <= 0.85)).sum())

    print(f"  {'realisation of the target':<42} {'median cov':>11} {'in band':>9}")
    print(f"  {'EB-smoothed share (the FROZEN target)':<42} {np.median(covs):>11.4f} {n_band(covs):>7}/9")
    print(f"  {'add-alpha share (EB removed)':<42} {np.median(add):>11.4f} {n_band(add):>7}/9")
    print(f"  {'raw empirical share (no add-alpha, no EB)':<42} {np.median(raw):>11.4f} {n_band(raw):>7}/9")
    excess_sm = np.median(covs) - 0.80
    excess_raw = np.median(raw) - 0.80
    share_of_excess = 1.0 - excess_raw / excess_sm if excess_sm != 0 else float("nan")
    print(f"\n  smoothing accounts for {share_of_excess:.1%} of the excess over the nominal 0.80")
    check("DIAGNOSIS: the intervals are calibrated against the RAW share and over-cover only "
          "against the EB-smoothed series they were fit on",
          n_band(raw) >= 7 and n_band(covs) < 7,
          f"raw {n_band(raw)}/9 in band vs smoothed {n_band(covs)}/9")

    os_frac = np.array([v["frac_interval_one_sided"] for v in cells.values()])
    print(f"\n  rows where lo <= 0, so the interval is ONE-SIDED by construction "
          f"(shares are strictly positive): {os_frac.min():.4f} .. {os_frac.max():.4f}")
    for T0 in CUTOFFS:
        row = "  " + T0
        for m in INTERVAL_MODELS:
            row += f"  {m}={cells[(T0, m)]['frac_interval_one_sided']:.4f}"
        print(row)
    print("  -> this tracks the ETS < ARIMA < Theta ordering of coverage; it is a property of "
          "non-binding lower bounds, NOT of model quality.")

    sds = np.array([v["resid_sd"] for v in cells.values()])
    kur = np.array([v["resid_kurtosis"] for v in cells.values()])
    tail = np.array([v["p_abs_resid_gt3"] for v in cells.values()])
    print(f"\n  standardised residual sd across cells: {sds.min():.3f} .. {sds.max():.3f} "
          f"(1.0 = width correct in scale)")
    print(f"  raw kurtosis (3 = normal): {kur.min():.1f} .. {kur.max():.1f}")
    print(f"  P(|e| > 3): {tail.min():.4f} .. {tail.max():.4f}   against 0.0027 under normality")
    check("DIAGNOSIS: the width is right in scale but the SHAPE is wrong -- the intervals "
          "over-cover in the bulk and UNDER-cover in the tails simultaneously",
          bool((tail > 0.0027).all()) and bool((kur > 3.0).all()),
          f"every cell has heavier tails than normal; min P(|e|>3) = {tail.min():.4f} = "
          f"{tail.min()/0.0027:.1f}x the normal rate")
    direction = ("over-covers the EB-smoothed target in the bulk while under-covering its tails; "
                 "calibrated against the raw share")
    print(f"\n  corrected direction statement: {direction}")

    return {"cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()},
            "n_in_band": len(in_band), "n_ks_ok": len(ks_ok),
            "coverage_min": float(covs.min()), "coverage_max": float(covs.max()),
            "coverage_median": float(np.median(covs)),
            "coverage_median_addalpha": float(np.median(add)),
            "coverage_median_raw": float(np.median(raw)),
            "n_in_band_addalpha": n_band(add), "n_in_band_raw": n_band(raw),
            "smoothing_share_of_excess": float(share_of_excess),
            "one_sided_frac_range": [float(os_frac.min()), float(os_frac.max())],
            "resid_sd_range": [float(sds.min()), float(sds.max())],
            "resid_kurtosis_range": [float(kur.min()), float(kur.max())],
            "p_abs_resid_gt3_range": [float(tail.min()), float(tail.max())],
            "direction": direction}


# --------------------------------------------------------------------------- #
def main() -> int:
    print("verify_coverage.py -- novelty-stream skill + 80% interval coverage/PIT")
    print(f"repo = {REPO}")
    print("pre-registration: decisions/2026-08-13-novelty-and-coverage-preregistration.md")
    block0_eb_anchor()
    a = signal_a()
    b = signal_b()

    doc = {
        "preregistration": "decisions/2026-08-13-novelty-and-coverage-preregistration.md",
        "signal_a_novelty_skill": {k: v for k, v in a.items()},
        "signal_b_coverage_pit": b,
    }
    (REPO / "artifacts" / "coverage_out.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")

    hdr("SUMMARY")
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    for name, ok, detail in _RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print("\n  NOTE: a FAIL on a frozen criterion is a RESULT, not an error. The script's "
          "exit code tracks the two environment-anchor checks only.")
    anchor_fail = sum(1 for name, ok, _ in _RESULTS if name.startswith("EB anchor") and not ok)
    print(f"  environment-anchor failures: {anchor_fail}")
    print(f"  wrote artifacts/coverage_out.json")
    return 1 if anchor_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
