"""The margin account predicts a CURVE, not a number. This tests the curve.

`verify_margin.py` established that top-k verdict instability is controlled by the
boundary margin, and that the count of candidates within one perturbation width of the
boundary predicts the flip rate (3 predicted vs 2.40 observed at k=50).

That account makes a much stronger, falsifiable prediction, and it is the one that matters
for whether the result generalises beyond this instrument:

    PRE-STATED PREDICTION
    Immunity is not a property of the pipeline, so it must FAIL where the ranking is
    crowded. Sweeping the report size k moves the decision boundary into regions of
    different density. The observed symmetric difference should therefore
      (a) track the boundary-band count N(k) = #{r : |g(r) - g_(k)| < median|delta g|}
          across the whole sweep, not just at k=50; and
      (b) be substantially LARGER at k where the local density is high.
    If instead churn stays flat in k, the margin account is wrong and the immunity is a
    property of the data or the pipeline after all.

Same ten perturbation replicates as verify_margin.py (centroid noise at the frozen scale,
all 2.7M papers reassigned, tree and 64-cut refit, EB re-run). The perturbed statistic is
cached once per replicate, then every k is evaluated against it, so the sweep is free.

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
from brontide import config, series                                      # noqa: E402
sys.path.insert(0, str(REPO / "scripts"))
from verify_margin import (assign_nearest, counts_from_labels, eb_pipeline,  # noqa: E402
                           g_ratio)

T0, N_REP, CHUNK = "2024-06", 10, 100_000
KS = [5, 10, 25, 50, 100, 200, 400, 600]


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

    print(f"caching {N_REP} perturbed replicates (reassigning all {n:,} papers each) ...")
    reps = []
    for seed in range(N_REP):
        rng = np.random.default_rng(seed)
        cp = cent0 + rng.standard_normal(cent0.shape) * scale
        lab = assign_nearest(pca, cp.astype(np.float32), n)
        C = counts_from_labels(lab, midx, K, n_months)
        sup = fcluster(linkage(cp, method="ward"), nsup, criterion="maxclust") - 1
        X, _ = eb_pipeline(C, sup, np.bincount(sup, minlength=nsup), i_anchor, nsup)
        el = np.where(C[:, i_t0 - 12:i_t0].sum(axis=1) >= config.ELIGIBILITY_FLOOR)[0]
        reps.append((g_ratio(X, i_t0), el))
        print(f"  seed {seed} cached")

    common = np.array([r for r in eidx if np.isfinite(g0[r])])
    delta = float(np.median([np.median(np.abs(g1[common] - g0[common])) for g1, _ in reps]))
    print(f"\nmedian |delta g| = {delta:.6f}\n")

    order = eidx[np.argsort(-np.nan_to_num(g0[eidx], nan=-np.inf))]
    gs = g0[order]

    print(f"{'k':>5} {'gap g_(k)-g_(k+1)':>19} {'N(k) predicted':>15} "
          f"{'observed flips':>15} {'obs/pred':>9}")
    print("-" * 70)
    rows = []
    for k in KS:
        if k >= len(order):
            continue
        gk = float(gs[k - 1])
        gap = float(gs[k - 1] - gs[k])
        pred = int((np.abs(np.nan_to_num(g0[eidx], nan=np.inf) - gk) < delta).sum())
        base = set(int(r) for r in order[:k])
        flips = []
        for g1, el in reps:
            new = set(int(r) for r in el[np.argsort(-np.nan_to_num(g1[el], nan=-np.inf))][:k])
            flips.append(len(base ^ new))
        obs = float(np.mean(flips))
        ratio = obs / max(pred, 1e-9)
        rows.append({"k": k, "gap": gap, "predicted": pred, "observed": obs, "ratio": ratio})
        print(f"{k:>5} {gap:>19.6f} {pred:>15d} {obs:>15.2f} {ratio:>9.2f}")

    P = np.array([r["predicted"] for r in rows], float)
    O = np.array([r["observed"] for r in rows], float)
    corr = float(np.corrcoef(P, O)[0, 1])
    grows = O[-1] > O[0] + 1.0
    tracks = corr >= 0.8

    print("\n" + "=" * 70)
    print(f"corr(predicted N(k), observed flips) over the sweep = {corr:+.4f}")
    print(f"observed flips at k={rows[0]['k']}: {O[0]:.2f}   at k={rows[-1]['k']}: {O[-1]:.2f}")
    if tracks and grows:
        print("PASS: churn tracks the boundary-band count across the whole sweep AND grows")
        print("  sharply where the ranking is crowded. The immunity at k=50 is a margin")
        print("  property of that particular boundary, NOT robustness of the pipeline --")
        print("  the same pipeline, same perturbation, and a differently sized report is")
        print("  materially less reproducible.")
    elif grows and not tracks:
        print("PARTIAL: churn does grow with k, so immunity is boundary-specific, but the")
        print("  band count is not a good quantitative predictor across the sweep.")
    else:
        print("FAIL: churn does not grow where the ranking is crowded. The margin account")
        print("  does not generalise across k and must be restated or withdrawn.")
    print("EXPLORATORY: a test of an explanation, not one of the frozen quantities.")

    (REPO / "scripts" / "margin_sweep_out.json").write_text(json.dumps({
        "T0": T0, "n_replicates": N_REP, "median_abs_delta_g": delta,
        "sweep": rows, "corr_pred_obs": corr,
        "tracks": bool(tracks), "grows_with_k": bool(grows),
        "status": "EXPLORATORY - generalisation test of the margin account",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
