"""One theorem instead of two mechanisms: the rank-relevant part of a perturbation.

The paper currently explains its two null results separately -- corpus restriction is an
exact gauge (structural), document relocation survives by margin (statistical). That is
two mechanisms where there may be one.

THE PROPOSED UNIFICATION. Write any perturbation in log coordinates as

    delta_{r,t} = log p_hat'_{r,t} - log p_hat_{r,t},
    u_t = (1/K) sum_r delta_{r,t},          eps_{r,t} = delta_{r,t} - u_t     (sum_r eps = 0)

For a single-period log-ratio statistic log g(r) = log p_hat_{r,t2} - log p_hat_{r,t1},

    log g'(r) - log g(r) = (u_{t2} - u_{t1}) + (eps_{r,t2} - eps_{r,t1}),

and the first term is region-independent, so it CANNOT reorder anything however large it
is. The ranking depends on eps alone. Combined with the margin proposition, the top-k set
is unchanged whenever 2 max_r |eps-induced change| < g_(k) - g_(k+1).

This subsumes both mechanisms: a gauge is the special case eps == 0, where invariance is
structural and needs no margin; everything else is governed by the margin condition applied
to eps rather than to delta.

PRE-STATED PREDICTIONS, written before running:
  R1. For whole-super corpus restriction, eps is negligible relative to u -- essentially the
      entire perturbation is the gauge component.
  R2. For centroid perturbation with reassignment, eps is a substantial fraction of delta.
  R3. Across perturbations, churn tracks ||eps||, NOT ||delta||. If two perturbations have
      similar ||delta|| but very different ||eps||, the one with larger ||eps|| must churn
      more. A perturbation with large ||delta|| and near-zero ||eps|| must not churn at all.

R3 is the decisive one. If churn tracks ||delta|| just as well as ||eps||, the
decomposition explains nothing and the two-mechanism account should stand.

EXPLORATORY: a test of an explanation, not one of the frozen quantities.
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
sys.path.insert(0, str(REPO / "scripts"))
from brontide import config, series                                      # noqa: E402
from verify_margin import assign_nearest, counts_from_labels             # noqa: E402
from verify_reassignment_bootstrap import top_n                          # noqa: E402

T0, CHUNK = "2024-06", 100_000
KS = [5, 10, 25, 50, 100, 200, 400, 600]


def eb(C, sup, nsup, n_anchor, kappa=None):
    a, K = config.ADD_ALPHA, C.shape[0]
    p = (C + a) / (C.sum(axis=0) + a * K)[None, :]
    P_S = np.zeros((nsup, C.shape[1]), np.float64)
    np.add.at(P_S, sup, p)
    n_s = np.bincount(sup, minlength=nsup)
    m = (P_S / np.maximum(n_s, 1)[:, None])[sup]
    trail = C[:, max(0, n_anchor - 12):n_anchor].sum(axis=1)
    k = series._estimate_kappa(trail, sup) if kappa is None else kappa
    w = (trail / (trail + k))[:, None]
    return w * p + (1.0 - w) * m


def decompose(Xp, X0, idx):
    """delta = log Xp - log X0 on the common regions; split into gauge u_t and residual eps."""
    d = np.log(Xp[idx]) - np.log(X0[idx])
    u = d.mean(axis=0, keepdims=True)                 # per-period, common to all regions
    e = d - u
    return d, u, e


def single_period_g(X, i_t0):
    lo, hi = config.EMERGENCE_HORIZON_MONTHS
    return X[:, i_t0 + hi - 1] / X[:, i_t0 - 1]


def main() -> int:
    art, data = REPO / "artifacts", REPO / "data"
    K, dim, nsup = config.N_REGIONS, config.PCA_DIM, config.N_SUPERREGIONS
    sh = pd.read_parquet(art / "region_shares.parquet")
    months = sorted(sh["month"].unique())
    n_months, i_t0, i_anchor = len(months), months.index(T0) + 1, len(months)
    Cnt = (sh.pivot(index="region", columns="month", values="count")
             .reindex(range(K))[months].to_numpy(np.float64))
    sup0 = sh.groupby("region")["super_region"].first().reindex(range(K)).to_numpy(np.int64)
    elig = Cnt[:, i_t0 - 12:i_t0].sum(axis=1) >= config.ELIGIBILITY_FLOOR

    X0 = eb(Cnt, sup0, nsup, i_anchor)
    g0 = single_period_g(X0, i_t0)
    print(f"{'perturbation':<34} {'||delta||':>10} {'||u||':>10} {'||eps||':>10} "
          f"{'eps/delta':>10} {'churn k=50':>11}")
    print("-" * 92)

    rows = []

    # --- A. whole-super restrictions (the gauge case) ------------------------
    for c in [0, 5, 11, 23]:
        S = np.where(sup0 != c)[0]
        supS = sup0[S]
        Xr = eb(Cnt[S], supS, nsup, i_anchor)
        Xr_full = np.full_like(X0, np.nan); Xr_full[S] = Xr
        idx = S[elig[S]]
        d, u, e = decompose(Xr_full, X0, idx)
        gr = np.full(K, np.nan); gr[S] = single_period_g(Xr, i_t0)
        el = idx
        churn = len(top_n(g0, el, 50) ^ top_n(gr, el, 50)) / 100.0
        rows.append(("restriction: drop super %d" % c, np.abs(d).max(), np.abs(u).max(),
                     np.abs(e).max(), churn))

    # --- B. centroid perturbation + reassignment (the non-gauge case) --------
    meta = json.loads((data / "pca50.f32.memmap.meta.json").read_text())
    n = int(meta["n"])
    pca = np.memmap(data / "pca50.f32.memmap", dtype=np.float32, mode="r", shape=(n, dim))
    labels = np.load(data / "labels" / "region_labels.i16.npy")
    cent0 = np.load(art / "centroids.f32.npy").astype(np.float64)
    m_all = pd.read_parquet(config.RAW_PARQUET, columns=["month"])["month"].to_numpy()
    pos = {m: i for i, m in enumerate(months)}
    midx = np.fromiter((pos.get(m, -1) for m in m_all), dtype=np.int32, count=len(m_all))
    cnt_r = np.zeros(K); s1 = np.zeros((K, dim)); s2 = np.zeros((K, dim))
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        lab = labels[lo:hi].astype(np.int64)
        blk = np.asarray(pca[lo:hi], dtype=np.float64)
        np.add.at(cnt_r, lab, 1.0); np.add.at(s1, lab, blk); np.add.at(s2, lab, blk ** 2)
    mean = s1 / cnt_r[:, None]
    sigma = np.sqrt(np.maximum(s2 / cnt_r[:, None] - mean ** 2, 0.0))
    scale = sigma / np.sqrt(Cnt[:, i_anchor - 12:i_anchor].sum(axis=1))[:, None]

    for seed in range(4):
        rng = np.random.default_rng(seed)
        cp = cent0 + rng.standard_normal(cent0.shape) * scale
        lab = assign_nearest(pca, cp.astype(np.float32), n)
        C = counts_from_labels(lab, midx, K, n_months)
        sup = fcluster(linkage(cp, method="ward"), nsup, criterion="maxclust") - 1
        Xp = eb(C, sup, nsup, i_anchor)
        el = np.where((C[:, i_t0 - 12:i_t0].sum(axis=1) >= config.ELIGIBILITY_FLOOR) & elig)[0]
        d, u, e = decompose(Xp, X0, el)
        churn = len(top_n(g0, el, 50) ^ top_n(single_period_g(Xp, i_t0), el, 50)) / 100.0
        rows.append((f"relocation: seed {seed}", np.abs(d).max(), np.abs(u).max(),
                     np.abs(e).max(), churn))

    # --- C. THE DISCRIMINATING CONTROL: a synthetic PURE gauge ---------------
    # A and B are collinear: both families differ in ||delta|| and ||eps|| together, so
    # neither predictor can win on them. The decisive case is large ||delta|| with eps == 0
    # exactly. Multiplying every share in period t by an arbitrary lambda_t is exactly that.
    # If ||delta|| drove churn, these would churn enormously; if eps drives it, not at all.
    rng = np.random.default_rng(12345)
    el_all = np.where(elig)[0]
    for mag in (0.5, 2.0, 10.0):
        lam = np.exp(rng.uniform(-mag, mag, size=X0.shape[1]))   # one factor per period
        Xg = X0 * lam[None, :]
        d, u, e = decompose(Xg, X0, el_all)
        churn = len(top_n(g0, el_all, 50) ^ top_n(single_period_g(Xg, i_t0), el_all, 50)) / 100.0
        rows.append((f"PURE GAUGE (mag {mag})", np.abs(d).max(), np.abs(u).max(),
                     np.abs(e).max(), churn))

    for name, dd, uu, ee, ch in rows:
        print(f"{name:<34} {dd:>10.4f} {uu:>10.4f} {ee:>10.4f} {ee/max(dd,1e-12):>10.4f} {ch:>11.3f}")

    R = np.array([[r[1], r[3], r[4]] for r in rows])       # delta, eps, churn
    c_delta = float(np.corrcoef(R[:, 0], R[:, 2])[0, 1])
    c_eps = float(np.corrcoef(R[:, 1], R[:, 2])[0, 1])
    rest = R[:4]; relo = R[4:8]; gauge = R[8:]
    print("\n" + "=" * 92)
    print(f"R1 restriction: eps/delta = {rest[:,1].max()/max(rest[:,0].max(),1e-12):.2e}   "
          f"(prediction: negligible)")
    print(f"R2 relocation : eps/delta = {relo[:,1].mean()/max(relo[:,0].mean(),1e-12):.3f}   "
          f"(prediction: substantial)")
    print(f"   pure gauge    : max ||delta|| = {gauge[:,0].max():.4f} with ||eps|| = "
          f"{gauge[:,1].max():.2e} and churn = {gauge[:,2].max():.3f}")
    print(f"R3 corr(||delta||, churn) = {c_delta:+.4f}")
    print(f"   corr(||eps||,   churn) = {c_eps:+.4f}")
    ok = c_eps > c_delta + 0.05
    print("\n" + ("PASS: churn tracks the RESIDUAL, not the raw perturbation. The gauge and the\n"
                  "  margin accounts are one theorem -- only the region-varying part of a\n"
                  "  perturbation can reorder, and it is governed by the boundary margin."
                  if ok else
                  "FAIL: the residual is no better a predictor than the raw perturbation;\n"
                  "  keep the two-mechanism account."))
    print("EXPLORATORY: a test of an explanation, not one of the frozen quantities.")

    (REPO / "scripts" / "rank_relevant_out.json").write_text(json.dumps({
        "rows": [{"name": r[0], "delta_inf": r[1], "u_inf": r[2], "eps_inf": r[3],
                  "churn_k50": r[4]} for r in rows],
        "corr_delta_churn": c_delta, "corr_eps_churn": c_eps, "unification_supported": bool(ok),
        "status": "EXPLORATORY",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
