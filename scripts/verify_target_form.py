"""Does the FUNCTIONAL FORM of the reported target decide whether geometric fragility
reaches the verdict?

Standing observation (measured 2026-08-12): the emergence verdict of this instrument is
essentially immune to two large perturbations --- corpus restriction (shares move 4.6e-3,
top-50 symmetric difference 0.000) and centroid perturbation with full reassignment (6% of
papers change region, top-50 survival 0.980).

The proposed mechanism is that the reported statistic is a RATIO of two time-averages,

    g_ratio(r) = mean_{t in H} p_hat_r(t)  /  mean_{t in T} p_hat_r(t),

and both perturbations act approximately as a time-CONSTANT factor on p_hat_r (the EB weight
w_r is fixed per origin), which a ratio annihilates exactly.

PRE-STATED PREDICTION, written before running:
    If the mechanism is real, then replacing the ratio by a DIFFERENCE on the same shares,

        g_diff(r)  = mean_{t in H} p_hat_r(t)  -  mean_{t in T} p_hat_r(t),

    -- the shape of the Delta s = s_test - s_train target used elsewhere in the
    science-of-science forecasting literature -- should churn substantially under the SAME
    perturbations, because a difference does not cancel a multiplicative factor.
    Concretely: g_diff top-50 symmetric-difference fraction should exceed g_ratio's by a
    wide margin under both perturbations. If instead g_diff is also immune, the mechanism
    is wrong and the immunity is a property of the data, not of the statistic.

Reported either way. EXPLORATORY: not one of the frozen quantities in
decisions/2026-08-12-scale-spectrum-preregistration.md; it is a test of an explanation.

Runtime ~2-3 min. CPU only.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from brontide import config, series                                      # noqa: E402

T0, N_REP, CHUNK = "2024-06", 10, 100_000


def eb_pipeline(C, sup, n_s, n_anchor, nsup):
    a, K = config.ADD_ALPHA, C.shape[0]
    p = (C + a) / (C.sum(axis=0) + a * K)[None, :]
    P_S = np.zeros((nsup, C.shape[1]), np.float64)
    np.add.at(P_S, sup, p)
    m = (P_S / np.maximum(n_s, 1)[:, None])[sup]
    trail = C[:, max(0, n_anchor - 12):n_anchor].sum(axis=1)
    kappa = series._estimate_kappa(trail, sup)
    w = (trail / (trail + kappa))[:, None]
    return w * p + (1.0 - w) * m, float(kappa)


def windows(P, i_t0):
    lo, hi = config.EMERGENCE_HORIZON_MONTHS
    return P[:, i_t0 + lo - 1:i_t0 + hi].mean(axis=1), P[:, i_t0 - 12:i_t0].mean(axis=1)


def g_ratio(P, i_t0):
    num, den = windows(P, i_t0)
    return num / np.where(den > 0, den, np.nan)


def g_diff(P, i_t0):
    num, den = windows(P, i_t0)
    return num - den                                    # the Delta s shape


def top_n(g, elig_idx, n=None):
    n = n or config.TOP_N
    order = elig_idx[np.argsort(-np.nan_to_num(g[elig_idx], nan=-np.inf))]
    return set(int(r) for r in order[:n])


def symdiff(a, b):
    return len(a ^ b) / (2 * config.TOP_N)


def assign_nearest(pca, cent, n):
    c2 = (cent ** 2).sum(axis=1)
    out = np.empty(n, dtype=np.int16)
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        blk = np.asarray(pca[lo:hi], dtype=np.float32)
        out[lo:hi] = np.argmin(c2[None, :] - 2.0 * (blk @ cent.T), axis=1).astype(np.int16)
    return out


def counts_from_labels(lab, midx, K, n_months):
    C = np.zeros((K, n_months), dtype=np.float64)
    ok = midx >= 0
    np.add.at(C, (lab[ok].astype(np.int64), midx[ok]), 1.0)
    return C


def main() -> int:
    art, data = REPO / "artifacts", REPO / "data"
    K, dim, nsup = config.N_REGIONS, config.PCA_DIM, config.N_SUPERREGIONS
    sh = pd.read_parquet(art / "region_shares.parquet")
    months = sorted(sh["month"].unique())
    n_months, i_t0, i_anchor = len(months), months.index(T0) + 1, len(months)
    Xs = (sh.pivot(index="region", columns="month", values="share_smoothed")
            .reindex(range(K))[months].to_numpy(np.float64))
    Cnt = (sh.pivot(index="region", columns="month", values="count")
             .reindex(range(K))[months].to_numpy(np.float64))
    sup0 = sh.groupby("region")["super_region"].first().reindex(range(K)).to_numpy(np.int64)

    elig = Cnt[:, i_t0 - 12:i_t0].sum(axis=1) >= config.ELIGIBILITY_FLOOR
    TARGETS = {"ratio (deployed)": g_ratio, "difference (Delta s shape)": g_diff}

    # ---------- perturbation A: corpus restriction, 64 super-region drops ----------
    print("A. corpus restriction (drop each of the 64 super-regions)\n")
    resA = {k: [] for k in TARGETS}
    for c in range(nsup):
        S = np.where(sup0 != c)[0]
        if S.size < 2 or (sup0 == c).sum() == 0:
            continue
        sup_S = sup0[S]
        n_s_S = np.bincount(sup_S, minlength=nsup)
        Xr, _ = eb_pipeline(Cnt[S], sup_S, n_s_S, i_anchor, nsup)
        el_local = np.where(elig[S])[0]
        for name, fn in TARGETS.items():
            full = fn(Xs, i_t0)[S]
            resA[name].append(symdiff(top_n(full, el_local), top_n(fn(Xr, i_t0), el_local)))

    # ---------- perturbation B: centroid perturbation + full reassignment ----------
    print("B. centroid perturbation at the frozen scale + reassignment of all 2.7M papers\n")
    meta = json.loads((data / "pca50.f32.memmap.meta.json").read_text())
    n = int(meta["n"])
    pca = np.memmap(data / "pca50.f32.memmap", dtype=np.float32, mode="r", shape=(n, dim))
    labels0 = np.load(data / "labels" / "region_labels.i16.npy")
    cent0 = np.load(art / "centroids.f32.npy").astype(np.float64)
    m_all = pd.read_parquet(config.RAW_PARQUET, columns=["month"])["month"].to_numpy()
    pos = {m: i for i, m in enumerate(months)}
    midx = np.fromiter((pos.get(m, -1) for m in m_all), dtype=np.int32, count=len(m_all))

    cnt_r = np.zeros(K); s1 = np.zeros((K, dim)); s2 = np.zeros((K, dim))
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        lab = labels0[lo:hi].astype(np.int64)
        blk = np.asarray(pca[lo:hi], dtype=np.float64)
        np.add.at(cnt_r, lab, 1.0); np.add.at(s1, lab, blk); np.add.at(s2, lab, blk ** 2)
    mean = s1 / cnt_r[:, None]
    sigma = np.sqrt(np.maximum(s2 / cnt_r[:, None] - mean ** 2, 0.0))
    scale = sigma / np.sqrt(Cnt[:, i_anchor - 12:i_anchor].sum(axis=1))[:, None]

    base = {name: top_n(fn(Xs, i_t0), np.where(elig)[0]) for name, fn in TARGETS.items()}
    resB = {k: [] for k in TARGETS}
    for seed in range(N_REP):
        rng = np.random.default_rng(seed)
        cp = cent0 + rng.standard_normal(cent0.shape) * scale
        lab = assign_nearest(pca, cp.astype(np.float32), n)
        C = counts_from_labels(lab, midx, K, n_months)
        sup = fcluster(linkage(cp, method="ward"), nsup, criterion="maxclust") - 1
        X, _ = eb_pipeline(C, sup, np.bincount(sup, minlength=nsup), i_anchor, nsup)
        el = np.where(C[:, i_t0 - 12:i_t0].sum(axis=1) >= config.ELIGIBILITY_FLOOR)[0]
        for name, fn in TARGETS.items():
            resB[name].append(symdiff(base[name], top_n(fn(X, i_t0), el)))
        print(f"  seed {seed:2d}  " + "  ".join(
            f"{name}: {resB[name][-1]:.3f}" for name in TARGETS))

    # ---------------------------------- report ----------------------------------
    print("\n" + "=" * 74)
    print(f"{'target':<28} {'A: restriction':>18} {'B: reassignment':>18}")
    print("-" * 74)
    out = {}
    for name in TARGETS:
        a, b = np.array(resA[name]), np.array(resB[name])
        out[name] = {"A_median": float(np.median(a)), "A_max": float(a.max()),
                     "B_median": float(np.median(b)), "B_max": float(b.max())}
        print(f"{name:<28} {np.median(a):>10.3f} (max {a.max():.2f}) "
              f"{np.median(b):>10.3f} (max {b.max():.2f})")
    print("=" * 74)

    r, d = out["ratio (deployed)"], out["difference (Delta s shape)"]
    wider = (d["A_median"] > r["A_median"] + 0.05) or (d["B_median"] > r["B_median"] + 0.05)
    if wider:
        print("PASS: the difference-shaped target churns substantially more than the ratio under")
        print("      the same perturbations. The functional form of the reported target -- not the")
        print("      robustness of the pipeline -- is what decides whether geometric fragility")
        print("      reaches the verdict. A Delta s target does NOT inherit this immunity.")
    else:
        print("FAIL: the difference-shaped target is about as immune as the ratio. The proposed")
        print("      cancellation mechanism does NOT explain the observed immunity -- the survey's")
        print("      remark must be withdrawn or rewritten.")
    print("EXPLORATORY: a test of an explanation, not one of the frozen quantities.")

    (REPO / "scripts" / "target_form_out.json").write_text(json.dumps(
        {"T0": T0, "n_replicates_B": N_REP, "results": out,
         "prediction_supported": bool(wider),
         "status": "EXPLORATORY - tests the cancellation mechanism"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
