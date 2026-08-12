"""H1-H3: does the verdict survive replacing the learned hierarchy with a human one?

Pre-registered in decisions/2026-08-12-hierarchy-control-preregistration.md (commit f01f69d).

The Carlsson-Memoli and transport results say hierarchy-indexed statistics inherit
their hierarchy's instability, but both were only ever demonstrated against trees
derived from the SAME embedding. The arXiv taxonomy already in data/papers.parquet
-- 152 primary categories in 20 groups, 100% coverage -- gives a hierarchy people
built, over exactly the same documents.

Frozen design: hold the leaf set at the existing 1,024 micro-regions so both trees
are comparable objects on the same leaves; only the tree changes.

  T_W  the committed Ward linkage
  T_C  average linkage on the Aitchison distance between regions' category profiles

Both cut at 64; the full EB pipeline re-runs under each, so the shrinkage target,
kappa and the smoothed shares all inherit the hierarchy under test.

  H1 descriptive  cophenetic distortion between the trees; 64-cut ARI
  H2 scored       top-k symmetric difference between the two verdicts;
                  hierarchy-dependence established iff median over k >= 0.10
  H3 scored       margin account replicates on T_C iff corr(N(k), churn) >= 0.70
                  and churn(k=600) >= 5x churn(k=5)

Reported either way. Runtime ~4 min, CPU only.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from brontide import config, series                                      # noqa: E402
from verify_margin import assign_nearest, counts_from_labels, g_ratio      # noqa: E402
from verify_reassignment_bootstrap import top_n, within_region_sigma       # noqa: E402

T0, N_REP, CHUNK = "2024-06", 10, 100_000
KS = [5, 10, 25, 50, 100, 200, 400, 600]


def eb(C, sup, n_s, n_anchor, nsup):
    a, K = config.ADD_ALPHA, C.shape[0]
    p = (C + a) / (C.sum(axis=0) + a * K)[None, :]
    P_S = np.zeros((nsup, C.shape[1]), np.float64)
    np.add.at(P_S, sup, p)
    m = (P_S / np.maximum(n_s, 1)[:, None])[sup]
    trail = C[:, max(0, n_anchor - 12):n_anchor].sum(axis=1)
    k = series._estimate_kappa(trail, sup)
    w = (trail / (trail + k))[:, None]
    return w * p + (1.0 - w) * m, float(k)


def category_profiles(labels, K):
    """(K, n_cat) counts of each region's papers by FIRST-listed arXiv category."""
    cats = pq.read_table(config.RAW_PARQUET, columns=["categories"]).column("categories").to_pylist()
    prim = [c[0] if c else "" for c in cats]
    vocab = sorted({c for c in prim if c})
    idx = {c: i for i, c in enumerate(vocab)}
    P = np.zeros((K, len(vocab)), np.float64)
    ci = np.fromiter((idx.get(c, -1) for c in prim), dtype=np.int32, count=len(prim))
    ok = ci >= 0
    np.add.at(P, (labels[ok].astype(np.int64), ci[ok]), 1.0)
    groups = sorted({c.split(".")[0] for c in vocab})
    return P, vocab, groups


def aitchison_tree(P):
    """Average linkage on the Aitchison distance = Euclidean distance between clr vectors."""
    Q = P + config.ADD_ALPHA
    Q = Q / Q.sum(axis=1, keepdims=True)
    L = np.log(Q)
    clr = L - L.mean(axis=1, keepdims=True)
    return linkage(pdist(clr, metric="euclidean"), method="average")


def verdict(C, sup, nsup, i_t0, i_anchor):
    n_s = np.bincount(sup, minlength=nsup)
    X, kap = eb(C, sup, n_s, i_anchor, nsup)
    el = np.where(C[:, i_t0 - 12:i_t0].sum(axis=1) >= config.ELIGIBILITY_FLOOR)[0]
    return g_ratio(X, i_t0), el, kap


def main() -> int:
    art, data = REPO / "artifacts", REPO / "data"
    K, dim, nsup = config.N_REGIONS, config.PCA_DIM, config.N_SUPERREGIONS

    sh = pd.read_parquet(art / "region_shares.parquet")
    months = sorted(sh["month"].unique())
    n_months, i_t0, i_anchor = len(months), months.index(T0) + 1, len(months)
    Cnt = (sh.pivot(index="region", columns="month", values="count")
             .reindex(range(K))[months].to_numpy(np.float64))
    Xs = (sh.pivot(index="region", columns="month", values="share_smoothed")
            .reindex(range(K))[months].to_numpy(np.float64))
    sup_W0 = sh.groupby("region")["super_region"].first().reindex(range(K)).to_numpy(np.int64)

    # ---- GATE ---------------------------------------------------------------
    Xchk, kap = eb(Cnt, sup_W0, np.bincount(sup_W0, minlength=nsup), i_anchor, nsup)
    d = float(np.abs(Xchk - Xs).max())
    print(f"GATE kappa = {kap:.15f}   max|reimplemented - committed| = {d:.1e}")
    assert abs(kap - 3.147960961354892) < 1e-9 and d == 0.0, "gate failed; nothing below is trustworthy"

    labels = np.load(data / "labels" / "region_labels.i16.npy")
    P, vocab, groups = category_profiles(labels, K)
    print(f"category profiles: {len(vocab)} primary categories, {len(groups)} top-level groups; "
          f"regions with no labelled paper: {int((P.sum(axis=1) == 0).sum())}")

    Z_W = np.load(art / "linkage.f64.npy")
    Z_C = aitchison_tree(P)

    # ---- H1 (descriptive) ---------------------------------------------------
    uW, uC = cophenet(Z_W), cophenet(Z_C)
    dist = float(np.abs(uC / uC.max() - uW / uW.max()).max())
    cutW = fcluster(Z_W, nsup, criterion="maxclust")
    cutC = fcluster(Z_C, nsup, criterion="maxclust")
    ari = float(adjusted_rand_score(cutW, cutC))
    print(f"\nH1  normalised cophenetic distortion T_C vs T_W = {dist:.4f}"
          f"   (perturbation of T_W alone gives 0.9737)")
    print(f"    64-cut ARI = {ari:.4f}   max depth: Ward {len(np.unique(cutW))} cl, "
          f"category {len(np.unique(cutC))} cl")

    # ---- H2 (scored) --------------------------------------------------------
    gW, elW, kW = verdict(Cnt, cutW - 1, nsup, i_t0, i_anchor)
    gC, elC, kC = verdict(Cnt, cutC - 1, nsup, i_t0, i_anchor)
    el = np.intersect1d(elW, elC)
    print(f"\nH2  kappa under T_W = {kW:.4f}, under T_C = {kC:.4f}; "
          f"{len(el)} regions eligible on both sides")
    h2 = {}
    for k in KS:
        a, b = top_n(gW, el, k), top_n(gC, el, k)
        h2[k] = len(a ^ b) / (2 * k)
    med = float(np.median(list(h2.values())))
    print("    k:      " + "  ".join(f"{k:>6}" for k in KS))
    print("    symdiff:" + "  ".join(f"{h2[k]:>6.3f}" for k in KS))
    print(f"    median over k = {med:.3f}   -> {'PASS' if med >= 0.10 else 'FAIL'} (>= 0.10)")

    # ---- H3 (scored): margin account on the human tree ----------------------
    meta = json.loads((data / "pca50.f32.memmap.meta.json").read_text())
    n = int(meta["n"])
    pca = np.memmap(data / "pca50.f32.memmap", dtype=np.float32, mode="r", shape=(n, dim))
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

    print(f"\nH3  {N_REP} perturbation replicates on the CATEGORY hierarchy "
          f"(centroids jittered, all {n:,} papers reassigned, T_C rebuilt each time)")
    reps = []
    for seed in range(N_REP):
        rng = np.random.default_rng(seed)
        cp = cent0 + rng.standard_normal(cent0.shape) * scale
        lab = assign_nearest(pca, cp.astype(np.float32), n)
        C = counts_from_labels(lab, midx, K, n_months)
        Pp, _, _ = category_profiles(lab, K)
        cut = fcluster(aitchison_tree(Pp), nsup, criterion="maxclust")
        g1, el1, _ = verdict(C, cut - 1, nsup, i_t0, i_anchor)
        reps.append((g1, el1))
        print(f"    seed {seed} done")

    common = np.array([r for r in el if np.isfinite(gC[r])])
    delta = float(np.median([np.median(np.abs(g1[common] - gC[common])) for g1, _ in reps]))
    rows = []
    for k in KS:
        gk = float(np.sort(gC[el])[::-1][k - 1])
        pred = int((np.abs(np.nan_to_num(gC[el], nan=np.inf) - gk) < delta).sum())
        base = top_n(gC, el, k)
        obs = float(np.mean([len(base ^ top_n(g1, el1, k)) / 2.0 for g1, el1 in reps]))
        rows.append({"k": k, "predicted": pred, "observed": obs})
    corr = float(np.corrcoef([r["predicted"] for r in rows], [r["observed"] for r in rows])[0, 1])
    growth = rows[-1]["observed"] / max(rows[0]["observed"], 1e-9)
    print(f"\n    median |delta g| = {delta:.6f}")
    print("    k:        " + "  ".join(f"{r['k']:>6}" for r in rows))
    print("    N(k):     " + "  ".join(f"{r['predicted']:>6}" for r in rows))
    print("    churn:    " + "  ".join(f"{r['observed']:>6.2f}" for r in rows))
    print(f"    corr(N, churn) = {corr:+.4f}   churn(600)/churn(5) = {growth:.1f}x")
    h3 = corr >= 0.70 and growth >= 5.0
    print(f"    -> {'PASS' if h3 else 'FAIL'} (corr >= 0.70 and growth >= 5x)")

    print("\n" + "=" * 78)
    print(f"H2 hierarchy-dependence established : {'YES' if med >= 0.10 else 'NO'}  (median {med:.3f})")
    print(f"H3 margin account replicates        : {'YES' if h3 else 'NO'}  "
          f"(corr {corr:+.3f}, growth {growth:.1f}x)")
    print("=" * 78)

    (REPO / "scripts" / "hierarchy_controls_out.json").write_text(json.dumps({
        "T0": T0, "n_categories": len(vocab), "n_groups": len(groups),
        "H1_cophenetic_distortion": dist, "H1_cut_ari": ari,
        "kappa_ward": kW, "kappa_category": kC,
        "H2_symdiff_by_k": {str(k): v for k, v in h2.items()}, "H2_median": med,
        "H2_hierarchy_dependence": bool(med >= 0.10),
        "H3_rows": rows, "H3_corr": corr, "H3_growth": growth, "H3_replicates": bool(h3),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
