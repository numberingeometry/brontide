"""Close the stated limitation on the S3 stability result.

`decisions/2026-08-12-scale-spectrum-preregistration.md` scored S3 by perturbing the
1,024 centroids at the data's own sampling scale and recomputing the hierarchy. That
measured Ward's cophenetic ultrametric as ~97% distorted while the reported top-50
emerging set survived in 100/100 replicates. The freeze recorded the caveat honestly:

    "the perturbation moves centroids with paper->region assignment held FIXED, so the
     survival figure understates."

This script removes that caveat. After perturbing the centroids it **reassigns all
2,705,992 papers to their nearest perturbed centroid**, rebuilds the monthly counts
from the new assignment, re-runs the EB pipeline, refits the Ward tree and the 64-cut,
and only then recomputes the realized-growth top-50. Every downstream quantity moves.

EXPLORATORY: this is a follow-up to a frozen criterion, not one of the frozen
quantities. It can only weaken or confirm the "no effect" reading; the instability
reading (distortion 0.9737 >> 0.10) does not depend on it.

Runtime ~1-2 min for the default 10 replicates. CPU only.
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

N_REP = 10
T0 = "2024-06"
CHUNK = 100_000


def eb_pipeline(C, sup, n_s, n_anchor, n_supers):
    """series.run_shares' EB math. Verified to reproduce kappa=3.147961 exactly."""
    a, K = config.ADD_ALPHA, C.shape[0]
    p = (C + a) / (C.sum(axis=0) + a * K)[None, :]
    P_S = np.zeros((n_supers, C.shape[1]), np.float64)
    np.add.at(P_S, sup, p)
    m = (P_S / np.maximum(n_s, 1)[:, None])[sup]
    trail = C[:, max(0, n_anchor - 12):n_anchor].sum(axis=1)
    kappa = series._estimate_kappa(trail, sup)
    w = (trail / (trail + kappa))[:, None]
    return w * p + (1.0 - w) * m, float(kappa)


def g_real(P, i_t0):
    lo, hi = config.EMERGENCE_HORIZON_MONTHS
    num = P[:, i_t0 + lo - 1:i_t0 + hi].mean(axis=1)
    den = P[:, i_t0 - 12:i_t0].mean(axis=1)
    return num / np.where(den > 0, den, np.nan)


def top_n(g, elig_idx, n=None):
    n = n or config.TOP_N
    order = elig_idx[np.argsort(-np.nan_to_num(g[elig_idx], nan=-np.inf))]
    return set(int(r) for r in order[:n])


def assign_nearest(pca, cent, n, chunk=CHUNK):
    """argmin_j ||x_i - c_j||^2 == argmin_j (||c_j||^2 - 2 x_i . c_j). One matmul per chunk."""
    c2 = (cent ** 2).sum(axis=1)
    out = np.empty(n, dtype=np.int16)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        blk = np.asarray(pca[lo:hi], dtype=np.float32)
        out[lo:hi] = np.argmin(c2[None, :] - 2.0 * (blk @ cent.T), axis=1).astype(np.int16)
    return out


def counts_from_labels(lab, midx, K, n_months):
    C = np.zeros((K, n_months), dtype=np.float64)
    ok = midx >= 0
    np.add.at(C, (lab[ok].astype(np.int64), midx[ok]), 1.0)
    return C


def within_region_sigma(labels, pca, K, dim, n, chunk=CHUNK):
    cnt = np.zeros(K)
    s1 = np.zeros((K, dim))
    s2 = np.zeros((K, dim))
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        lab = labels[lo:hi].astype(np.int64)
        blk = np.asarray(pca[lo:hi], dtype=np.float64)
        np.add.at(cnt, lab, 1.0)
        np.add.at(s1, lab, blk)
        np.add.at(s2, lab, blk ** 2)
    mean = s1 / cnt[:, None]
    return np.sqrt(np.maximum(s2 / cnt[:, None] - mean ** 2, 0.0)), cnt


def main() -> int:
    art, data = REPO / "artifacts", REPO / "data"
    K, dim, nsup = config.N_REGIONS, config.PCA_DIM, config.N_SUPERREGIONS

    meta = json.loads((data / "pca50.f32.memmap.meta.json").read_text())
    n = int(meta["n"])
    pca = np.memmap(data / "pca50.f32.memmap", dtype=np.float32, mode="r", shape=(n, dim))
    labels0 = np.load(data / "labels" / "region_labels.i16.npy")
    cent0 = np.load(art / "centroids.f32.npy").astype(np.float64)

    sh = pd.read_parquet(art / "region_shares.parquet")
    months = sorted(sh["month"].unique())
    n_months, i_t0, i_anchor = len(months), months.index(T0) + 1, len(months)
    Xs = (sh.pivot(index="region", columns="month", values="share_smoothed")
            .reindex(range(K))[months].to_numpy(np.float64))
    Cnt0 = (sh.pivot(index="region", columns="month", values="count")
              .reindex(range(K))[months].to_numpy(np.float64))

    # paper -> month index, aligned to the committed month grid
    m_all = pd.read_parquet(config.RAW_PARQUET, columns=["month"])["month"].to_numpy()
    pos = {m: i for i, m in enumerate(months)}
    midx = np.fromiter((pos.get(m, -1) for m in m_all), dtype=np.int32, count=len(m_all))

    # sanity: rebuilding counts from the committed labels must reproduce the committed counts
    Cchk = counts_from_labels(labels0, midx, K, n_months)
    drift = float(np.abs(Cchk - Cnt0).max())
    print(f"label->count reconstruction: max |rebuilt - committed| = {drift:.0f}")
    if drift != 0:
        print("  WARNING: assignment/month alignment does not reproduce the committed counts;")
        print("  downstream numbers are not comparable to the frozen run.")

    elig = (Cnt0[:, i_t0 - 12:i_t0].sum(axis=1) >= config.ELIGIBILITY_FLOOR)
    elig_idx = np.where(elig)[0]
    T50_0 = top_n(g_real(Xs, i_t0), elig_idx)
    Z0 = linkage(cent0, method="ward")
    cut0 = fcluster(Z0, nsup, criterion="maxclust")

    print("computing within-region sigma over the 2.7M x 50 cloud ...")
    sigma, n_tot = within_region_sigma(labels0, pca, K, dim, n)
    trail12 = Cnt0[:, i_anchor - 12:i_anchor].sum(axis=1)
    scale = sigma / np.sqrt(trail12)[:, None]        # the frozen perturbation scale

    surv, reassigned, kappas = [], [], []
    for seed in range(N_REP):
        rng = np.random.default_rng(seed)
        cp = cent0 + rng.standard_normal(cent0.shape) * scale

        lab = assign_nearest(pca, cp.astype(np.float32), n)          # THE NEW STEP
        frac_moved = float((lab != labels0).mean())

        C = counts_from_labels(lab, midx, K, n_months)
        Zp = linkage(cp, method="ward")
        sup = fcluster(Zp, nsup, criterion="maxclust") - 1
        n_s = np.bincount(sup, minlength=nsup)
        X, kap = eb_pipeline(C, sup, n_s, i_anchor, nsup)

        el = np.where(C[:, i_t0 - 12:i_t0].sum(axis=1) >= config.ELIGIBILITY_FLOOR)[0]
        s = len(T50_0 & top_n(g_real(X, i_t0), el)) / config.TOP_N
        surv.append(s); reassigned.append(frac_moved); kappas.append(kap)
        print(f"  seed {seed:2d}: {frac_moved*100:5.2f}% papers reassigned   "
              f"kappa {kap:.3f}   top-50 survival {s:.3f}")

    surv, reassigned = np.array(surv), np.array(reassigned)
    print("\n" + "=" * 72)
    print(f"papers reassigned : median {np.median(reassigned)*100:.2f}%  "
          f"[{reassigned.min()*100:.2f}%, {reassigned.max()*100:.2f}%]")
    print(f"kappa             : {min(kappas):.3f} .. {max(kappas):.3f}  (frozen run: 3.148)")
    print(f"top-50 survival   : median {np.median(surv):.3f}  "
          f"[{surv.min():.3f}, {surv.max():.3f}]   over {N_REP} replicates")
    print("=" * 72)
    if np.median(surv) >= 0.90:
        print("PASS: the 'no effect on the verdict' reading SURVIVES full reassignment.")
        print("      The caveat recorded in the freeze is now closed, not merely stated.")
    else:
        print("FAIL: survival drops once documents are reassigned. The frozen S3 'no effect'")
        print("      reading was an artifact of holding the assignment fixed -- report this.")
    print("Note: EXPLORATORY follow-up; the instability finding (0.9737 >> 0.10) is unaffected.")

    (art / "..").resolve()
    out = REPO / "scripts" / "reassignment_bootstrap_out.json"
    out.write_text(json.dumps({
        "n_replicates": N_REP, "T0": T0,
        "count_reconstruction_max_abs_err": drift,
        "frac_reassigned_median": float(np.median(reassigned)),
        "frac_reassigned_min": float(reassigned.min()),
        "frac_reassigned_max": float(reassigned.max()),
        "kappa_min": float(min(kappas)), "kappa_max": float(max(kappas)),
        "top50_survival_median": float(np.median(surv)),
        "top50_survival_min": float(surv.min()), "top50_survival_max": float(surv.max()),
        "status": "EXPLORATORY follow-up to frozen criterion S3",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
