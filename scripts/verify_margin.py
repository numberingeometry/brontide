"""Why is the top-k verdict immune to large geometric perturbations?

`verify_target_form.py` REFUTED the first explanation. The proposed mechanism was that
the reported statistic is a ratio of two time-averages and therefore annihilates a
time-constant multiplicative perturbation. If that were the mechanism, a difference-shaped
target (the Delta s = s_test - s_train shape) would churn where the ratio does not. It does
not: both targets sit at 0.000 median symmetric difference under corpus restriction and
0.020 under centroid perturbation with full reassignment. The mechanism is not cancellation.

SECOND HYPOTHESIS, pre-stated here before running:

    The immunity is a MARGIN property of the top-k decision, not a property of the
    statistic's functional form. A top-k verdict is stable exactly when the k-th and
    (k+1)-th candidates are separated by more than the perturbation moves them. Under this
    hypothesis:

      (H1) the perturbation-induced change |Delta g| in the statistic is SMALL compared to
           the spread of g across the top of the ranking, but NOT small compared to the
           local gap between adjacent regions near rank k;
      (H2) the regions that actually enter or leave the top-50 are concentrated at ranks
           NEAR 50 in the unperturbed ordering -- not scattered through the ranking.

    (H2) is the decisive one and is nearly free to measure. If flips occur at ranks far
    from the boundary, the margin story is wrong too.

If the hypothesis holds, the transferable prescription is that a top-k verdict should be
reported with its boundary margin, because the margin -- not the geometry -- is what
determines whether the verdict is reproducible.

EXPLORATORY: a test of an explanation, not one of the frozen quantities.
Runtime ~2 min. CPU only.
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
K_TOP = config.TOP_N


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


def g_ratio(P, i_t0):
    lo, hi = config.EMERGENCE_HORIZON_MONTHS
    num = P[:, i_t0 + lo - 1:i_t0 + hi].mean(axis=1)
    den = P[:, i_t0 - 12:i_t0].mean(axis=1)
    return num / np.where(den > 0, den, np.nan)


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

    elig = Cnt[:, i_t0 - 12:i_t0].sum(axis=1) >= config.ELIGIBILITY_FLOOR
    eidx = np.where(elig)[0]
    g0 = g_ratio(Xs, i_t0)
    order = eidx[np.argsort(-np.nan_to_num(g0[eidx], nan=-np.inf))]
    rank_of = {int(r): i for i, r in enumerate(order)}                  # 0-based rank
    gs = g0[order]

    print(f"{len(eidx)} eligible regions; top-{K_TOP} boundary at rank {K_TOP}\n")
    print("unperturbed ordering, g near the decision boundary:")
    for i in range(K_TOP - 3, K_TOP + 3):
        print(f"   rank {i+1:3d}  g = {gs[i]:.5f}" + ("   <-- boundary" if i == K_TOP - 1 else ""))
    boundary_gap = float(gs[K_TOP - 1] - gs[K_TOP])
    spread_top = float(gs[0] - gs[K_TOP - 1])
    print(f"\nboundary gap  g_(50) - g_(51) = {boundary_gap:.6f}")
    print(f"spread of top-50   g_(1) - g_(50) = {spread_top:.6f}")
    print(f"ratio spread/gap = {spread_top / max(boundary_gap, 1e-12):.1f}\n")

    # ---- perturbation: centroid noise + full reassignment ----------------------
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

    T0set = set(int(r) for r in order[:K_TOP])
    flip_ranks, dg_rel = [], []
    for seed in range(N_REP):
        rng = np.random.default_rng(seed)
        cp = cent0 + rng.standard_normal(cent0.shape) * scale
        lab = assign_nearest(pca, cp.astype(np.float32), n)
        C = counts_from_labels(lab, midx, K, n_months)
        sup = fcluster(linkage(cp, method="ward"), nsup, criterion="maxclust") - 1
        X, _ = eb_pipeline(C, sup, np.bincount(sup, minlength=nsup), i_anchor, nsup)
        el = np.where(C[:, i_t0 - 12:i_t0].sum(axis=1) >= config.ELIGIBILITY_FLOOR)[0]
        g1 = g_ratio(X, i_t0)
        new = set(int(r) for r in el[np.argsort(-np.nan_to_num(g1[el], nan=-np.inf))][:K_TOP])
        for r in (T0set ^ new):
            if r in rank_of:
                flip_ranks.append(rank_of[r] + 1)
        common = np.array([r for r in order if np.isfinite(g1[r]) and np.isfinite(g0[r])])
        dg_rel.append(float(np.median(np.abs(g1[common] - g0[common]))))

    print(f"median |Delta g| per region over {N_REP} replicates: {np.median(dg_rel):.6f}")
    print(f"   as a multiple of the boundary gap : "
          f"{np.median(dg_rel)/max(boundary_gap,1e-12):.1f}x")
    print(f"   as a fraction of the top-50 spread: "
          f"{np.median(dg_rel)/max(spread_top,1e-12):.4f}\n")

    if flip_ranks:
        fr = np.array(flip_ranks)
        print(f"{len(fr)} membership flips over {N_REP} replicates; their ranks in the "
              f"UNPERTURBED ordering:")
        print(f"   min {fr.min()}   median {int(np.median(fr))}   max {fr.max()}")
        near = int((np.abs(fr - K_TOP) <= 10).sum())
        print(f"   within +/-10 of the rank-{K_TOP} boundary: {near}/{len(fr)} "
              f"({100*near/len(fr):.0f}%)")
    else:
        fr, near = np.array([]), 0
        print("no membership flips at all over the replicates.")

    # ---- (H3) the quantitative version: predict the flip COUNT from the band ----
    # A candidate can only cross if it lies within roughly one perturbation width of the
    # boundary value. The number of eligible regions in that band is therefore a parameter-
    # free prediction of the per-replicate symmetric-difference size.
    delta = float(np.median(dg_rel))
    gb = float(gs[K_TOP - 1])
    band = np.abs(np.nan_to_num(g0[eidx], nan=np.inf) - gb) < delta
    predicted = int(band.sum())
    observed = len(flip_ranks) / N_REP
    print(f"\n(H3) parameter-free flip-count prediction")
    print(f"   regions with |g - g_(50)| < median|delta g| = {delta:.5f} : {predicted}")
    print(f"   observed membership changes per replicate            : {observed:.2f}")
    ratio = observed / max(predicted, 1e-9)
    print(f"   observed / predicted                                 : {ratio:.2f}")
    h3 = 0.5 <= ratio <= 2.0

    h2 = len(flip_ranks) == 0 or (near / len(flip_ranks)) >= 0.8
    print("\n" + "=" * 74)
    if h2:
        print("PASS (H2): flips are confined to the neighbourhood of the decision boundary.")
        print("   The immunity is a MARGIN property of the top-k decision, not a property of")
        print("   the statistic's functional form and not robustness of the geometry. A top-k")
        print("   verdict is reproducible exactly when its boundary margin exceeds what the")
        print("   perturbation moves the statistic -- so the margin is the quantity that should")
        print("   be reported alongside the list.")
        print(f"   (H3) {'CONFIRMED' if h3 else 'NOT CONFIRMED'}: the boundary-band count "
              f"predicts the flip rate to within a factor {max(ratio,1/max(ratio,1e-9)):.2f}.")
    else:
        print("FAIL (H2): flips occur far from the boundary, so the margin story does not")
        print("   explain the immunity either. Both proposed mechanisms are now refuted; the")
        print("   survey must report the immunity as unexplained.")
    print("EXPLORATORY: a test of an explanation, not one of the frozen quantities.")

    (REPO / "scripts" / "margin_out.json").write_text(json.dumps({
        "T0": T0, "n_replicates": N_REP, "k": K_TOP,
        "boundary_gap": boundary_gap, "top_spread": spread_top,
        "median_abs_delta_g": float(np.median(dg_rel)),
        "delta_g_over_boundary_gap": float(np.median(dg_rel) / max(boundary_gap, 1e-12)),
        "n_flips": int(len(flip_ranks)),
        "flip_rank_min": int(fr.min()) if len(fr) else None,
        "flip_rank_max": int(fr.max()) if len(fr) else None,
        "frac_flips_within_10_of_boundary": float(near / len(fr)) if len(fr) else None,
        "H2_supported": bool(h2),
        "H3_predicted_band_count": predicted,
        "H3_observed_flips_per_replicate": observed,
        "H3_observed_over_predicted": ratio,
        "H3_supported": bool(h3),
        "status": "EXPLORATORY - second hypothesis after cancellation was refuted",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
