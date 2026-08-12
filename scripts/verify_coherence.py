#!/usr/bin/env python
"""verify_coherence.py -- numerical companion for O1 (subcompositional coherence
of emergence statistics) on the brontide production instrument.

Adversarial check of the claimed EXACT FACTORIZATION

    p_hat'_{r,t} = [ w'_r p_{r,t} + (1-w'_r) m_{r,t} ] / lambda_t
                 = p_hat^{(kappa_A)}_{r,t} / lambda_t

for every whole-super-region restriction A, plus the nine supporting predictions
(balance movement = kappa refit; closed-form oracle; contraction identity; exact
rank-coherence at fixed kappa; kappa-sweep margin; differential certificate;
incoherence of the share difference; the cut-super failure mode; the rolling-w
negative control), and the C1/C2/C3 classification of Part II.

Self-contained, CPU-only, no network. Reads ONLY
    artifacts/region_shares.parquet
    artifacts/super_region.i16.npy
    artifacts/linkage.f64.npy
and writes nothing.  Run from the repo root:  python scripts/verify_coherence.py

Convention: "symdiff" = |Top50' triangle Top50| / 50 (so one swapped element
reads 0.0400), matching the numbers quoted in the claim; the /(|A|+|B|) form is
printed alongside where it matters.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from brontide import config, series  # noqa: E402

ART = REPO / "artifacts"
NS = config.N_SUPERREGIONS          # 64
KFULL = config.N_REGIONS            # 1024
AA = config.ADD_ALPHA               # 1.0
TOP_N = config.TOP_N                # 50
FLOOR = config.ELIGIBILITY_FLOOR    # 20

TOL_ID = 1e-13                      # machine-precision tolerance for the identities

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> bool:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")
    return bool(ok)


def report(name: str, detail: str) -> None:
    print(f"  [ -- ] {name}\n        {detail}")


def hdr(s: str) -> None:
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# EB pipeline, reimplemented exactly as series.run_shares
# ---------------------------------------------------------------------------
def add_alpha_shares(C: np.ndarray, k_dim: int | None = None) -> np.ndarray:
    """p_{r,t} = (C+a)/(M_t + a*K).  k_dim overrides K (used for the frozen-K test)."""
    kk = C.shape[0] if k_dim is None else k_dim
    return (C + AA) / (C.sum(axis=0) + AA * kk)[None, :]


def super_mean(p: np.ndarray, sup: np.ndarray) -> np.ndarray:
    """m_{r,t} = P_{S(r),t} / n_{S(r)}, n_S counted over the rows PRESENT in `sup`."""
    P_S = np.zeros((NS, p.shape[1]), dtype=np.float64)
    np.add.at(P_S, sup, p)
    n_s = np.bincount(sup, minlength=NS).astype(np.float64)
    safe = np.where(n_s > 0, n_s, 1.0)
    return (P_S / safe[:, None])[sup]


def rolling_trail(C: np.ndarray) -> np.ndarray:
    """trail_{r,t} = sum of C over months [t-12, t)  (region-local, time-varying)."""
    cum = np.concatenate([np.zeros((C.shape[0], 1)), np.cumsum(C, axis=1)], axis=1)
    t = np.arange(C.shape[1])
    return cum[:, t] - cum[:, np.maximum(0, t - 12)]


def eb_run(C: np.ndarray, sup: np.ndarray, n_anchor: int,
           kappa: float | None = None, k_dim: int | None = None,
           rolling: bool = False) -> dict:
    """One full run of the EB stage on the given rows.  Mirrors series.run_shares."""
    p = add_alpha_shares(C, k_dim=k_dim)
    m = super_mean(p, sup)
    trail = C[:, max(0, n_anchor - 12):n_anchor].sum(axis=1)
    kap = series._estimate_kappa(trail, sup) if kappa is None else float(kappa)
    if rolling:
        w = rolling_trail(C) / (rolling_trail(C) + kap)
        p_hat = w * p + (1.0 - w) * m
    else:
        w = trail / (trail + kap)
        p_hat = w[:, None] * p + (1.0 - w[:, None]) * m
    return {"p": p, "m": m, "trail": trail, "kappa": kap, "w": w, "p_hat": p_hat}


# ---------------------------------------------------------------------------
# growth / ranking helpers
# ---------------------------------------------------------------------------
def growth(p_hat: np.ndarray, tr: slice, hz: slice) -> np.ndarray:
    return p_hat[:, hz].mean(axis=1) / p_hat[:, tr].mean(axis=1)


def topn(stat: np.ndarray, ids: np.ndarray, n: int = TOP_N) -> set:
    order = np.argsort(-stat, kind="stable")
    return set(int(i) for i in ids[order[:n]])


def sd50(a: set, b: set) -> float:
    """|a triangle b| / TOP_N  -- the convention the claim quotes (one swap = 0.04)."""
    return len(a ^ b) / TOP_N


# ---------------------------------------------------------------------------
# tree structure from the Ward linkage
# ---------------------------------------------------------------------------
def build_tree(Z: np.ndarray, k: int):
    """Psi (k-1,k) ilr contrasts; clade (k-1,k) leaf membership; parent + edge weights."""
    leaves: list[np.ndarray] = [np.array([i]) for i in range(k)]
    Psi = np.zeros((k - 1, k), dtype=np.float64)
    clade = np.zeros((k - 1, k), dtype=bool)
    parent = np.full(2 * k - 1, -1, dtype=np.int64)
    height = np.zeros(2 * k - 1, dtype=np.float64)
    for v in range(k - 1):
        i, j = int(Z[v, 0]), int(Z[v, 1])
        L, R = leaves[i], leaves[j]
        r, s = L.size, R.size
        Psi[v, L] = np.sqrt(s / (r * (r + s)))
        Psi[v, R] = -np.sqrt(r / (s * (r + s)))
        clade[v, L] = True
        clade[v, R] = True
        parent[i] = parent[j] = k + v
        height[k + v] = Z[v, 2]
        leaves.append(np.concatenate([L, R]))
    edge_w = np.array([height[parent[n]] - height[n] if parent[n] >= 0 else 0.0
                       for n in range(2 * k - 1)])
    return Psi, clade, parent, height, edge_w


def twd(mu: np.ndarray, nu: np.ndarray, clade_sub: np.ndarray,
        w_leaf: np.ndarray, w_node: np.ndarray) -> float:
    """Tree-Wasserstein W_1^T = sum_e w_e |mu(Gamma_e) - nu(Gamma_e)| (root edge = 0).

    ``clade_sub`` is the (n_internal, |A|) leaf-membership matrix restricted to the
    surviving leaves A; the tree topology and edge weights stay those of the full tree.
    """
    d = mu - nu
    return float((np.abs(d) * w_leaf).sum() + (np.abs(clade_sub @ d) * w_node).sum())


def main() -> int:
    t_start = time.time()

    # =====================================================================
    hdr("STEP 0 -- reproduction anchors (refuse to proceed if these fail)")
    # =====================================================================
    sh = pd.read_parquet(ART / "region_shares.parquet")
    months = sorted(sh["month"].unique())
    T = len(months)
    C = sh.pivot(index="region", columns="month", values="count") \
          .reindex(range(KFULL))[months].to_numpy(np.float64)
    SM = sh.pivot(index="region", columns="month", values="share_smoothed") \
           .reindex(range(KFULL))[months].to_numpy(np.float64)
    SA = sh.pivot(index="region", columns="month", values="share_addalpha") \
           .reindex(range(KFULL))[months].to_numpy(np.float64)
    sup = np.load(ART / "super_region.i16.npy").astype(np.int32)
    n_anchor = T

    base = eb_run(C, sup, n_anchor)
    p, m, trail, kappa, w, p_hat = (base["p"], base["m"], base["trail"],
                                    base["kappa"], base["w"], base["p_hat"])

    print(f"  months           : {T}  ({months[0]}..{months[-1]})")
    print(f"  kappa            : {kappa!r}")
    ok = check("A0 kappa reproduces 3.147960961354892",
               abs(kappa - 3.147960961354892) < 5e-13,
               f"|kappa - 3.147960961354892| = {abs(kappa - 3.147960961354892):.3e}")
    d_sm = float(np.abs(p_hat - SM).max())
    ok &= check("A1 share_smoothed reproduced to 0.0",
                d_sm == 0.0, f"max|p_hat - share_smoothed| = {d_sm!r}")
    d_sa = float(np.abs(p - SA).max())
    ok &= check("A2 share_addalpha reproduced to 0.0",
                d_sa == 0.0, f"max|p - share_addalpha| = {d_sa!r}")
    d_cl = float(np.abs(p.sum(axis=0) - 1.0).max())
    ok &= check("A3 D1 closure: sum_r p_{r,t} = 1",
                d_cl < 1e-14, f"max|sum_r p - 1| = {d_cl:.3e}")
    if not ok:
        print("\nABORT: the reimplementation does not reproduce the instrument.")
        return 1

    s_ph = p_hat.sum(axis=0)
    cov_term = np.zeros(T)
    for S in range(NS):
        idx = np.where(sup == S)[0]
        ww, pp = w[idx], p[idx]
        cov_term += idx.size * ((ww[:, None] * pp).mean(axis=0) - ww.mean() * pp.mean(axis=0))
    check("A4 D3 non-closure identity  sum_r p_hat = 1 + sum_S n_S cov_S(w,p)",
          np.abs(s_ph - (1.0 + cov_term)).max() < 1e-13,
          f"residual {np.abs(s_ph - (1.0 + cov_term)).max():.3e};  sum_r p_hat in "
          f"[{s_ph.min():.6f}, {s_ph.max():.6f}]  (claim 0.997809..1.003775)")
    report("A5 weights", f"w in [{w.min():.4f}, {w.max():.4f}], median {np.median(w):.4f} "
                         f"(claim [0.8579,0.9986], median 0.9879)")

    T0 = "2024-06"
    n_le = sum(mn <= T0 for mn in months)
    lo_h, hi_h = config.EMERGENCE_HORIZON_MONTHS
    TR = slice(n_le - 12, n_le)
    HZ = slice(n_le + lo_h - 1, n_le + hi_h)
    print(f"  T0={T0} n_le={n_le}  Tr={TR.start}..{TR.stop-1} ({months[TR.start]}..{months[TR.stop-1]})"
          f"  H={HZ.start}..{HZ.stop-1} ({months[HZ.start]}..{months[HZ.stop-1]})")
    eligible = C[:, TR].sum(axis=1) >= FLOOR
    check("A6 D4 eligibility 1023/1024 on the raw trailing count",
          int(eligible.sum()) == 1023, f"eligible = {int(eligible.sum())}/1024")

    g_full = growth(p_hat, TR, HZ)
    L_full = np.log(g_full)

    Z = np.load(ART / "linkage.f64.npy")
    Psi, clade, parent, height, edge_w = build_tree(Z, KFULL)
    check("A7 ilr basis orthonormal (Psi Psi^T = I, Psi 1 = 0)",
          np.abs(Psi @ Psi.T - np.eye(KFULL - 1)).max() < 1e-12
          and np.abs(Psi.sum(axis=1)).max() < 1e-12,
          f"max|PsiPsi^T - I| = {np.abs(Psi @ Psi.T - np.eye(KFULL-1)).max():.3e}, "
          f"max|Psi 1| = {np.abs(Psi.sum(axis=1)).max():.3e}")
    B_full = Psi @ np.log(p_hat)
    D_full = np.diff(B_full, axis=1).var(axis=1)
    hh, yy = np.log2(Z[:, 2]), np.log2(D_full)
    slope_h = np.polyfit(hh, yy, 1)[0]
    r2_h = np.corrcoef(hh, yy)[0, 1] ** 2
    depth = np.zeros(2 * KFULL - 1, dtype=int)
    for n in range(2 * KFULL - 3, -1, -1):
        if parent[n] >= 0:
            depth[n] = depth[parent[n]] + 1
    dep = depth[KFULL:2 * KFULL - 1]
    slope_d = np.polyfit(dep, yy, 1)[0]
    r2_d = np.corrcoef(dep, yy)[0, 1] ** 2
    csz = np.log2(clade.sum(axis=1))
    check("A8 tree reconstruction validated against the established S1 results",
          abs(np.corrcoef(dep, csz)[0, 1] - (-0.699)) < 1e-3
          and dep.max() == 14 and abs(r2_h - 0.275) < 1e-3,
          f"corr(depth, log2 clade size) = {np.corrcoef(dep, csz)[0,1]:.4f} (established -0.699); "
          f"depths 0..{dep.max()} (established 0..14); R^2 of the height fit = {r2_h:.4f} "
          f"(established 0.275)")
    report("A8b SIGN CONVENTION of the scale exponent -- flagged",
           f"RAW SLOPE of log2 D_v on log2(merge height) = {slope_h:+.4f} (R^2 {r2_h:.4f}); raw "
           f"slope on depth = {slope_d:+.4f} (R^2 {r2_d:.4f}). The established block reports "
           f"alpha_h = -0.3486 and alpha_d = 0.0358..0.0834, i.e. alpha = -slope in BOTH cases "
           f"(D_v proportional to scale^-alpha). Self-consistent, but the wording 'regressing "
           f"log2 D_v on log2(merge height): alpha_h = -0.3486' reads as if alpha_h were the "
           f"slope, and it is not: D_v INCREASES with merge height.")

    # =====================================================================
    hdr("STEP 1 -- PRIMARY: exact factorization over all 64 whole-super drops")
    # =====================================================================
    rows, store = [], {}
    for S in range(NS):
        A = np.where(sup != S)[0]
        run = eb_run(C[A], sup[A], n_anchor)                     # verbatim restricted re-run
        lam = p[A].sum(axis=0)
        kap_A = run["kappa"]
        wA = trail[A] / (trail[A] + kap_A)

        e_p = np.abs(run["p"] - p[A] / lam[None, :]).max() / np.abs(run["p"]).max()
        e_m = np.abs(run["m"] - m[A] / lam[None, :]).max() / np.abs(run["m"]).max()
        e_t = np.abs(run["trail"] - trail[A]).max()
        lhs = run["p_hat"] * lam[None, :]
        rhs = wA[:, None] * p[A] + (1.0 - wA[:, None]) * m[A]
        e_A = np.abs(lhs - rhs).max() / np.abs(run["p_hat"]).max()
        e_A2 = np.abs(run["p_hat"] - rhs / lam[None, :]).max() / np.abs(run["p_hat"]).max()

        store[S] = {"A": A, "lam": lam, "kappa_A": kap_A, "wA": wA,
                    "p_hat_r": run["p_hat"], "ph_kA_full": rhs}
        P_C = p[sup == S].sum(axis=0)
        rows.append((kap_A, e_p, e_m, e_t, e_A, e_A2, P_C.min(), P_C.max()))

    R = np.array(rows)
    kap_A_all = R[:, 0]
    kap_dlog = np.log(kap_A_all / kappa)
    check("P1 PRIMARY  max|p_hat'*lambda - (w' p + (1-w') m)| / max|p_hat'| <= 1e-13",
          R[:, 4].max() <= TOL_ID,
          f"max over 64 drops = {R[:, 4].max():.3e}  (claim 6.196e-15);  "
          f"p_hat' = p_hat^(kappa_A)/lambda to {R[:, 5].max():.3e}")
    check("P1a Lemma 1  p' = p/lambda", R[:, 1].max() <= TOL_ID,
          f"max rel err = {R[:, 1].max():.3e}  (claim 6.237e-15)")
    check("P1b Lemma 2  m' = m/lambda", R[:, 2].max() <= TOL_ID,
          f"max rel err = {R[:, 2].max():.3e}  (claim 1.192e-14)")
    check("P1c Lemma 3  trail' = trail exactly", R[:, 3].max() == 0.0,
          f"max|trail' - trail| = {R[:, 3].max()!r}  (claim 0.000e+00)")
    check("P1d kappa_A range == claimed [3.059732, 3.337904]",
          abs(kap_A_all.min() - 3.059732) < 2e-5 and abs(kap_A_all.max() - 3.337904) < 2e-5,
          f"kappa_A in [{kap_A_all.min():.6f}, {kap_A_all.max():.6f}];  "
          f"max|theta_A| = {np.abs(kap_dlog).max():.4f} (claim <= 0.0586)")
    report("P1e dropped-super share P_C,t",
           f"range [{R[:, 6].min():.4f}, {R[:, 7].max():.4f}] (claim 0.0006..0.0804)")

    # =====================================================================
    hdr("STEP 2 -- headline: growth ranking under the 64 restrictions")
    # =====================================================================
    sd, taus, Delta_store = [], [], {}
    for S in range(NS):
        st = store[S]
        A = st["A"]
        elig_A = A[eligible[A]]
        gp = np.full(KFULL, np.nan)
        gp[A] = growth(st["p_hat_r"], TR, HZ)
        sd.append(sd50(topn(gp[elig_A], elig_A), topn(g_full[elig_A], elig_A)))
        taus.append(kendalltau(gp[elig_A], g_full[elig_A]).statistic)
        Delta_store[S] = (elig_A, np.log(gp[elig_A]) - L_full[elig_A])
    sd = np.array(sd); taus = np.array(taus)
    check("P2 top-50 symmetric difference is 0 for all 64 whole-super drops",
          sd.max() == 0.0,
          f"symdiff |dAB|/50: median {np.median(sd):.4f} max {sd.max():.4f} (claim 0.0000/0.0000)")
    check("P2b Kendall tau-b matches the claim (median 0.999690, min 0.998452)",
          abs(np.median(taus) - 0.999690) < 5e-6 and abs(taus.min() - 0.998452) < 5e-6,
          f"tau-b median {np.median(taus):.6f} min {taus.min():.6f}")

    # =====================================================================
    hdr("STEP 3 -- prediction 1: balance movement is entirely the kappa refit")
    # =====================================================================
    e_kap, e_fix, n_disj, e_Dv = [], [], [], []
    for S in range(NS):
        st = store[S]
        A, kap_A = st["A"], st["kappa_A"]
        dropped = np.where(sup == S)[0]
        disj = ~clade[:, dropped].any(axis=1)
        n_disj.append(int(disj.sum()))
        if disj.sum() == 0:
            continue
        X = np.ones((KFULL, T)); X[A] = st["p_hat_r"]
        B_res = Psi[disj] @ np.log(X)
        w_kA = trail / (trail + kap_A)
        ph_kA = w_kA[:, None] * p + (1.0 - w_kA[:, None]) * m
        e_kap.append(np.abs(B_res - Psi[disj] @ np.log(ph_kA)).max())
        run_fx = eb_run(C[A], sup[A], n_anchor, kappa=kappa)
        Y = np.ones((KFULL, T)); Y[A] = run_fx["p_hat"]
        B_fx = Psi[disj] @ np.log(Y)
        e_fix.append(np.abs(B_fx - B_full[disj]).max())
        Dv = np.diff(B_fx, axis=1).var(axis=1)
        e_Dv.append(np.abs(Dv / D_full[disj] - 1.0).max())
    check("P3 disjoint-support balances: full restricted re-run == kappa_A refit alone",
          max(e_kap) <= TOL_ID, f"max |full - kappa-only| = {max(e_kap):.4e}  (claim 1.5099e-14)")
    check("P3b S2a: at FIXED kappa, restriction moves disjoint balances by 0",
          max(e_fix) <= TOL_ID,
          f"max |restricted - unrestricted| = {max(e_fix):.4e}  (claim 1.6875e-14; "
          f"pre-registered 1.777e-15)")
    check("P3c S1 scale spectrum is exactly corpus-coherent (D_v invariant at fixed kappa)",
          max(e_Dv) <= 1e-11,
          f"max relative change in D_v over disjoint nodes = {max(e_Dv):.3e}")
    report("P3d disjoint-support node count", f"min {min(n_disj)} max {max(n_disj)} of {KFULL-1}")

    # =====================================================================
    hdr("STEP 4 -- prediction 2: closed-form restriction oracle for g'_A")
    # =====================================================================
    e_or = []
    for S in range(NS):
        st = store[S]
        g_true = growth(st["p_hat_r"], TR, HZ)
        g_or = growth(st["ph_kA_full"] / st["lam"][None, :], TR, HZ)
        e_or.append(np.abs(g_or - g_true).max() / np.abs(g_true).max())
    check("P4 oracle g'_A from (p, m, trail, kappa_A) matches the full re-run",
          max(e_or) <= TOL_ID, f"max rel err = {max(e_or):.3e}  (claim 2.085e-15)")

    # =====================================================================
    hdr("STEP 5 -- prediction 3: the contraction identity (Theorem C)")
    # =====================================================================
    def g_at(kap: float) -> np.ndarray:
        wk = trail / (trail + kap)
        return growth(wk[:, None] * p + (1.0 - wk[:, None]) * m, TR, HZ)

    pbar_tr, pbar_h = p[:, TR].mean(axis=1), p[:, HZ].mean(axis=1)
    mbar_tr, mbar_h = m[:, TR].mean(axis=1), m[:, HZ].mean(axis=1)
    phbar_tr, phbar_h = p_hat[:, TR].mean(axis=1), p_hat[:, HZ].mean(axis=1)
    r1 = np.abs(w * pbar_tr + (1 - w) * mbar_tr - phbar_tr).max() / phbar_tr.min()
    r2 = np.abs(w * pbar_h + (1 - w) * mbar_h - phbar_h).max() / phbar_h.min()
    check("P5a w constant in t => EB commutes with time-averaging",
          max(r1, r2) < 1e-13, f"max relative residual = {max(r1, r2):.3e}")

    pi_tr = (1 - w) * mbar_tr / phbar_tr
    pi_h = (1 - w) * mbar_h / phbar_h
    G_S = mbar_h / mbar_tr
    Phi_1 = pi_h - pi_tr
    Phi_2 = pi_tr * (G_S / g_full - 1.0)
    check("P5b closed forms agree: pi^H - pi^Tr == pi^Tr (G_S/g - 1)",
          np.abs(Phi_1 - Phi_2).max() < 1e-14,
          f"max|form1 - form2| = {np.abs(Phi_1 - Phi_2).max():.3e}")
    h = 1e-6
    num = (np.log(g_at(kappa * np.exp(h))) - np.log(g_at(kappa * np.exp(-h)))) / (2 * h)
    check("P5c dlog g / dlog kappa == pi^H - pi^Tr (central differences)",
          np.abs(num - Phi_1).max() <= 1e-6,
          f"max|analytic - numeric| = {np.abs(num - Phi_1).max():.3e} (claim 6.520e-08); "
          f"max|Phi| = {np.abs(Phi_1).max():.4f} (claim 4.587e-01)")
    check("P5d |Phi_r| <= 1 universally", np.abs(Phi_1).max() <= 1.0,
          f"max|Phi| = {np.abs(Phi_1).max():.6f}")
    report("P5e prior-mass fraction",
           f"pi^H in [{pi_h.min():.3e}, {pi_h.max():.3e}] median {np.median(pi_h):.3e} "
           f"(claim 3.76e-4..0.652, median 1.32e-2)")
    check("P5f grain trap: kappa -> infinity collapses g onto the super growth G_S",
          np.abs(g_at(1e9) - G_S).max() < 1e-4,
          f"max|g(kappa=1e9) - G_S| = {np.abs(g_at(1e9) - G_S).max():.3e}")

    # mean-value-theorem form of the kappa displacement
    mvt_ok = True
    for S in range(NS):
        st = store[S]
        A, th = st["A"], kap_dlog[S]
        actual = np.log(growth(st["ph_kA_full"], TR, HZ)) - L_full[A]
        lo_k, hi_k = sorted((kappa, st["kappa_A"]))
        grid = np.exp(np.linspace(np.log(lo_k), np.log(hi_k), 25))
        Phis = []
        for kk in grid:
            wk = trail / (trail + kk)
            ph = wk[:, None] * p + (1.0 - wk[:, None]) * m
            a_tr, a_h = ph[:, TR].mean(axis=1), ph[:, HZ].mean(axis=1)
            Phis.append((1 - wk) * mbar_h / a_h - (1 - wk) * mbar_tr / a_tr)
        Phis = np.array(Phis)[:, A]
        lo_b, hi_b = th * Phis.min(axis=0), th * Phis.max(axis=0)
        lo_b, hi_b = np.minimum(lo_b, hi_b), np.maximum(lo_b, hi_b)
        mvt_ok &= bool(np.all(actual >= lo_b - 1e-12) and np.all(actual <= hi_b + 1e-12))
    check("P5g MVT form  log g^(kappa_A) - log g = theta_A Phi_r(kappa~) is attainable",
          mvt_ok, "actual displacement lies inside theta_A * [min Phi, max Phi] for all 64 drops")

    # =====================================================================
    hdr("STEP 6 -- prediction 4: EXACT rank-coherence at fixed kappa (C2)")
    # =====================================================================
    t1, t2 = TR.start, HZ.stop - 1
    rel_sd, n_moves, pair_err, lvl_sd, cagr_same, tau_fx, sd_fx = [], [], [], [], [], [], []
    for S in range(NS):
        A = np.where(sup != S)[0]
        run_fx = eb_run(C[A], sup[A], n_anchor, kappa=kappa)
        ratio_r = run_fx["p_hat"][:, t2] / run_fx["p_hat"][:, t1]
        ratio_u = p_hat[A, t2] / p_hat[A, t1]
        gauge = ratio_r / ratio_u
        rel_sd.append(gauge.std() / np.abs(gauge).mean())
        n_moves.append(int((np.argsort(-ratio_r, kind="stable")
                            != np.argsort(-ratio_u, kind="stable")).sum()))
        d_r = np.log(ratio_r)[:-1] - np.log(ratio_r)[1:]
        d_u = np.log(ratio_u)[:-1] - np.log(ratio_u)[1:]
        pair_err.append(np.abs(d_r - d_u).max())
        lv = run_fx["p_hat"][:, t2] / p_hat[A, t2]                # share level at fixed t (C2)
        lvl_sd.append(lv.std() / lv.mean())
        gg = growth(run_fx["p_hat"], TR, HZ)
        cagr = gg ** (1.0 / 24.0)                                 # strictly monotone transform
        cagr_same.append(int((np.argsort(-cagr, kind="stable")
                              != np.argsort(-gg, kind="stable")).sum()))
        gp = np.full(KFULL, np.nan); gp[A] = gg
        elig_A = A[eligible[A]]
        sd_fx.append(sd50(topn(gp[elig_A], elig_A), topn(g_full[elig_A], elig_A)))
        tau_fx.append(kendalltau(gp[elig_A], g_full[elig_A]).statistic)
    check("P6 single-month growth ratio: gauge factor identical across all regions",
          max(rel_sd) <= 1e-14,
          f"max relative sd of (restricted/unrestricted) = {max(rel_sd):.3e} (claim 4.216e-16)")
    check("P6b ZERO rank positions change (fixed kappa, single-month growth ratio)",
          max(n_moves) == 0, f"max positions changed = {max(n_moves)}  (claim 0)")
    check("P6c pairwise log-growth differences exactly invariant (C1)",
          max(pair_err) <= TOL_ID, f"max|difference| = {max(pair_err):.3e} (claim 7.772e-16)")
    check("P6d share level at fixed t is C2 (gauge 1/lambda_t, region-independent)",
          max(lvl_sd) <= 1e-14, f"max relative sd of the level gauge = {max(lvl_sd):.3e}")
    check("P6e CAGR is a monotone transform of g -> identical ranking",
          max(cagr_same) == 0, f"max rank positions differing = {max(cagr_same)}")
    report("P6f window-averaged g at FIXED kappa (tilt only)",
           f"top-50 symdiff max {max(sd_fx):.4f}; tau-b median {np.median(tau_fx):.6f} "
           f"min {min(tau_fx):.6f}  -> the residual tau-b shortfall is the tilt, not kappa")

    # block-constant gauge: g must become EXACTLY C2
    lam_blk = np.ones(T)
    lam_blk[TR] = 0.93
    lam_blk[HZ] = 0.88
    g_blk = growth(p_hat / lam_blk[None, :], TR, HZ)
    ratio = g_blk / g_full
    check("P6g block-constant closure subgroup: g is EXACTLY gauge-equivariant",
          ratio.std() / ratio.mean() <= 1e-14
          and int((np.argsort(-g_blk, kind="stable") != np.argsort(-g_full, kind="stable")).sum()) == 0,
          f"relative sd of g'/g = {ratio.std()/ratio.mean():.3e}; rank positions changed = "
          f"{int((np.argsort(-g_blk, kind='stable') != np.argsort(-g_full, kind='stable')).sum())}")

    # =====================================================================
    hdr("STEP 7 -- prediction 5: how far the kappa channel stretches")
    # =====================================================================
    elig_ids = np.where(eligible)[0]
    base_top = topn(g_full[elig_ids], elig_ids)

    def top_at(kap: float) -> set:
        return topn(g_at(kap)[elig_ids], elig_ids)

    dec_ok = all(top_at(kappa * np.exp(-d)) == base_top
                 for d in (0.0586, 0.2, 0.5, 1.0, 2.0, 5.0, float(np.log(kappa / 1e-3))))
    lo_d, hi_d = 0.0, 8.0
    for _ in range(60):
        mid = 0.5 * (lo_d + hi_d)
        if top_at(kappa * np.exp(mid)) == base_top:
            lo_d = mid
        else:
            hi_d = mid
    brk = lo_d
    max_theta = float(np.abs(kap_dlog).max())
    check("P7 top-50 survives every DECREASE of kappa (down to 1e-3)",
          dec_ok, f"top-50 unchanged at kappa*exp(-d) for d up to {np.log(kappa/1e-3):.3f} (w -> 1)")
    check("P7b upward break at Delta log kappa = 1.2231 +/- 0.01",
          abs(brk - 1.2231) <= 0.01,
          f"break at Delta log kappa = {brk:.4f} (kappa = {kappa*np.exp(brk):.4f}); restrictions "
          f"produce at most {max_theta:.4f} -> margin {brk/max_theta:.1f}x (claim 20.9x)")

    # =====================================================================
    hdr("STEP 8 -- prediction 6: the differential certificate at the boundary")
    # =====================================================================
    ord_full = elig_ids[np.argsort(-g_full[elig_ids], kind="stable")]
    Gamma = L_full[ord_full[TOP_N - 1]] - L_full[ord_full[TOP_N]]
    report("P8a unrestricted boundary gap",
           f"Gamma = L_(50) - L_(51) = {Gamma:.4e} (claim 6.8782e-03); "
           f"g_(50) = {g_full[ord_full[TOP_N-1]]:.4f}, g_(51) = {g_full[ord_full[TOP_N]]:.4f}")

    geff, diffs, margins, exact_ok = [], [], [], []
    ind_raw, ind_ctr, kap_part, tilt_part = [], [], [], []
    dk_max = 0.0
    for S in range(NS):
        elig_A, Delta = Delta_store[S]
        st = store[S]
        A, lam = st["A"], st["lam"]
        o = np.argsort(-L_full[elig_A], kind="stable")
        r_star, s_star = o[TOP_N - 1], o[TOP_N]
        Ge = L_full[elig_A][r_star] - L_full[elig_A][s_star]
        dif = Delta[s_star] - Delta[r_star]
        geff.append(Ge); diffs.append(dif); margins.append(Ge - dif)
        Lp = L_full[elig_A] + Delta
        exact_ok.append(bool(Lp[o[:TOP_N]].min() > Lp[o[TOP_N:]].max()))
        shift = np.log((1 / lam[HZ]).mean() / (1 / lam[TR]).mean())
        ind_raw.append(max(abs(Delta[r_star]), abs(Delta[s_star])))
        ind_ctr.append(max(abs(Delta[r_star] - shift), abs(Delta[s_star] - shift)))
        pos = {int(r): i for i, r in enumerate(A)}
        idxA = np.array([pos[int(r)] for r in elig_A])
        dk = np.log(growth(st["ph_kA_full"], TR, HZ))[idxA] - L_full[elig_A]
        dt = Delta - dk
        kap_part.append(abs(dk[s_star] - dk[r_star]))
        tilt_part.append(abs(dt[s_star] - dt[r_star]))
        dk_max = max(dk_max, float(np.abs(dk).max()))
    geff = np.array(geff); diffs = np.array(diffs); margins = np.array(margins)
    kap_part = np.array(kap_part); tilt_part = np.array(tilt_part)
    ind_raw = np.array(ind_raw); ind_ctr = np.array(ind_ctr)
    check("P8 exact top-50 criterion min_Top(L+D) > max_notTop(L+D) holds 64/64",
          all(exact_ok), f"{sum(exact_ok)}/64 restrictions satisfy the exact criterion")
    check("P8b differential certificate Gamma_eff - (D_s* - D_r*) > 0 for all 64",
          margins.min() > 0,
          f"min margin = {margins.min():.4e} (claim 2.412e-05); min Gamma_eff = {geff.min():.4e} "
          f"(claim 4.818e-05); worst clearance = {geff[np.argmin(margins)]/diffs[np.argmin(margins)]:.2f}x "
          f"(claim 2.0x)")
    report("P8c boundary differential",
           f"in [{diffs.min():.4e}, {diffs.max():.4e}] (claim [-1.115e-03, +4.906e-04])")
    report("P8d individual near-boundary perturbation (claim 3.240e-03 = 67x min Gamma_eff)",
           f"raw |Delta| max = {ind_raw.max():.4e} ({ind_raw.max()/geff.min():.0f}x); "
           f"centred by the rank-neutral shift max = {ind_ctr.max():.4e} "
           f"({ind_ctr.max()/geff.min():.0f}x); max_r|Delta_r| kappa term alone = {dk_max:.4e} "
           f"(claim 2.687e-02)")
    nz = np.abs(diffs) > 0
    report("P8e comonotone cancellation factor",
           f"raw median {np.median(ind_raw[nz]/np.abs(diffs)[nz]):.1f}x max "
           f"{(ind_raw[nz]/np.abs(diffs)[nz]).max():.0f}x; centred median "
           f"{np.median(ind_ctr[nz]/np.abs(diffs)[nz]):.2f}x max "
           f"{(ind_ctr[nz]/np.abs(diffs)[nz]).max():.0f}x  (claim median 3.2x, max 317x)")
    check("P8f at the boundary the TILT dominates the kappa term",
          int((tilt_part > kap_part).sum()) == 63,
          f"tilt > kappa in {(tilt_part > kap_part).sum()}/64 (claim 63/64); |kappa part| median "
          f"{np.median(kap_part):.3e} max {kap_part.max():.3e} (claim 2.47e-06 / 1.60e-04); "
          f"|tilt part| median {np.median(tilt_part):.3e} max {tilt_part.max():.3e} "
          f"(claim 1.26e-04 / 1.12e-03)")

    dec_err, tau_c, kap_c, rho_slack = [], [], [], []
    for S in range(NS):
        st = store[S]
        A, lam, ph_kA = st["A"], st["lam"], st["ph_kA_full"]
        out = {}
        for nm, W in (("Tr", TR), ("H", HZ)):
            inv = 1.0 / lam[W]
            Lam = inv.mean()
            dl = inv / Lam - 1.0
            phb = ph_kA[:, W].mean(axis=1)
            rho = (dl[None, :] * ph_kA[:, W]).mean(axis=1) / phb
            bound = dl.std() * (ph_kA[:, W].std(axis=1) / phb)
            out[nm] = (Lam, rho, bound)
        lhs = np.log(growth(st["p_hat_r"], TR, HZ))
        rhs = (np.log(out["H"][0] / out["Tr"][0]) + np.log(growth(ph_kA, TR, HZ))
               + np.log1p(out["H"][1]) - np.log1p(out["Tr"][1]))
        dec_err.append(np.abs(lhs - rhs).max())
        tt = np.log1p(out["H"][1]) - np.log1p(out["Tr"][1])
        tau_c.append(np.abs(tt - tt.mean()).max())
        kk = np.log(growth(ph_kA, TR, HZ)) - L_full[A]
        kap_c.append(np.abs(kk - kk.mean()).max())
        rho_slack.append(min((out["H"][2] - np.abs(out["H"][1])).min(),
                             (out["Tr"][2] - np.abs(out["Tr"][1])).min()))
    check("P8g Theorem D decomposition is an exact identity",
          max(dec_err) <= TOL_ID, f"max residual = {max(dec_err):.3e}")
    check("P8h Cauchy-Schwarz bound |rho^W_r| <= sd_W(delta)*CV_W(p_hat_r) respected",
          min(rho_slack) >= 0.0,
          f"min slack (bound - |rho|) over all regions/windows/drops = {min(rho_slack):.3e}")
    report("P8i centred terms",
           f"max centred tilt |tau_r| = {max(tau_c):.4e} (claim 4.800e-03); max centred kappa "
           f"term = {max(kap_c):.4e} (claim 2.035e-02)")

    cvh = [(1 / store[S]["lam"][HZ]).std() / (1 / store[S]["lam"][HZ]).mean() for S in range(NS)]
    cvt = [(1 / store[S]["lam"][TR]).std() / (1 / store[S]["lam"][TR]).mean() for S in range(NS)]
    report("P8j real within-window CV of 1/lambda",
           f"horizon max {max(cvh):.4e} (claim 6.471e-03), trailing max {max(cvt):.4e} "
           f"(claim 9.100e-03)")

    def max_shock(t_idx: int) -> float:
        def surv(f):
            lam = np.ones(T); lam[t_idx] = 1.0 - f
            return topn(growth(p_hat / lam[None, :], TR, HZ)[elig_ids], elig_ids) == base_top
        lo, hi = 0.0, 0.999
        if surv(hi):
            return hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if surv(mid) else (lo, mid)
        return lo
    sh_h = [max_shock(t) for t in range(HZ.start, HZ.stop)]
    sh_t = [max_shock(t) for t in range(TR.start, TR.stop)]
    check("P8k synthetic one-month closure shock (first horizon month / the T0 month)",
          abs(sh_h[0] - 0.197) < 0.002 and abs(sh_t[-1] - 0.555) < 0.002,
          f"{months[HZ.start]} survives {sh_h[0]*100:.1f}% removal (claim 19.7%); "
          f"{months[TR.stop-1]} survives {sh_t[-1]*100:.1f}% (claim 55.5%). WORST month: "
          f"horizon {min(sh_h)*100:.1f}%, trailing {min(sh_t)*100:.1f}%")

    # =====================================================================
    hdr("STEP 9 -- prediction 7 + Part II placement: what is C3")
    # =====================================================================
    flips, sd_diff = [], []
    d_full = phbar_h - phbar_tr
    for S in range(NS):
        st = store[S]
        A = st["A"]
        elig_A = A[eligible[A]]
        dp = np.full(KFULL, np.nan)
        ph_r = st["p_hat_r"]
        dp[A] = ph_r[:, HZ].mean(axis=1) - ph_r[:, TR].mean(axis=1)
        flips.append(int((np.sign(dp[elig_A]) != np.sign(d_full[elig_A])).sum()))
        sd_diff.append(sd50(topn(dp[elig_A], elig_A), topn(d_full[elig_A], elig_A)))
    check("P9 share difference flips SIGN under a single super drop (max 31, total 309)",
          max(flips) == 31 and sum(flips) == 309,
          f"max flips in one drop = {max(flips)} (claim 31); total over 64 drops = "
          f"{sum(flips)} (claim 309)")
    check("P9b share-difference top-50 is NOT invariant (unlike the ratio)",
          max(sd_diff) > 0 and abs(max(sd_diff) - 0.04) < 1e-12,
          f"symdiff |dAB|/50 max = {max(sd_diff):.4f} (claim 0.0400); growth-ratio "
          f"counterpart = {sd.max():.4f}")

    # scalar C3 statistics: Herfindahl, entropy, total variation, tree-Wasserstein
    def month_order(v):
        return np.argsort(-v, kind="stable")
    S_probe = int(np.argmax([store[S]["lam"].std() for S in range(NS)]))
    A = store[S_probe]["A"]
    run_fx = eb_run(C[A], sup[A], n_anchor, kappa=kappa)         # = p_hat[A]/lambda exactly
    x_u, x_r = p_hat[A], run_fx["p_hat"]                         # the raw (unnormalised) paths
    q_u = x_u / x_u.sum(axis=0)[None, :]                         # re-closed each month
    q_r = x_r / x_r.sum(axis=0)[None, :]
    cl_A = clade[:, A].astype(np.float64)
    w_leaf, w_node = edge_w[:KFULL][A], edge_w[KFULL:2 * KFULL - 1]

    stats = {}
    stats["Herfindahl (raw path)"] = ((x_u ** 2).sum(axis=0), (x_r ** 2).sum(axis=0))
    stats["Shannon entropy (raw path)"] = (-(x_u * np.log(x_u)).sum(axis=0),
                                           -(x_r * np.log(x_r)).sum(axis=0))
    stats["total variation (raw path)"] = (0.5 * np.abs(np.diff(x_u, axis=1)).sum(axis=0),
                                           0.5 * np.abs(np.diff(x_r, axis=1)).sum(axis=0))
    stats["tree-Wasserstein (raw path)"] = (
        np.array([twd(x_u[:, t], x_u[:, t + 1], cl_A, w_leaf, w_node) for t in range(T - 1)]),
        np.array([twd(x_r[:, t], x_r[:, t + 1], cl_A, w_leaf, w_node) for t in range(T - 1)]))
    reord = {k: int((month_order(u) != month_order(v)).sum()) for k, (u, v) in stats.items()}
    relmv = {k: float(np.abs(v / u - 1).max()) for k, (u, v) in stats.items()}
    check("P9c Herfindahl / entropy / TV / tree-Wasserstein of the RAW path are C3",
          min(reord.values()) > 0,
          f"at fixed kappa, dropping super {S_probe} reorders the months by "
          + ", ".join(f"{k.split(' (')[0]} in {n}" for k, n in reord.items())
          + " positions (a coherent or gauge-equivariant statistic gives 0); max relative "
          + f"value move: TWD {relmv['tree-Wasserstein (raw path)']:.3e}, "
          + f"TV {relmv['total variation (raw path)']:.3e}")
    e_q = float(np.abs(q_r - q_u).max())
    en_c = (-(q_u * np.log(q_u)).sum(axis=0), -(q_r * np.log(q_r)).sum(axis=0))
    hh_c = ((q_u ** 2).sum(axis=0), (q_r ** 2).sum(axis=0))
    check("P9d CORRECTION to Part II: after RE-CLOSING each month, entropy and Herfindahl "
          "are exactly C1, not C3",
          e_q < 1e-15 and np.abs(en_c[1] / en_c[0] - 1).max() < 1e-13
          and np.abs(hh_c[1] / hh_c[0] - 1).max() < 1e-13,
          f"max|q' - q| = {e_q:.3e}; entropy moves {np.abs(en_c[1]/en_c[0]-1).max():.3e} "
          f"relative, Herfindahl {np.abs(hh_c[1]/hh_c[0]-1).max():.3e}. Both are "
          f"scale-invariant, so by "
          f"Theorem B(a) they factor through the log-ratio coordinates. Their C3 placement "
          f"holds only because brontide's p_hat is NOT closed (D3) -- state the object, "
          f"not just the statistic.")

    # =====================================================================
    hdr("STEP 10 -- prediction 8: the cut-super failure mode (Lemma 2 breaks)")
    # =====================================================================
    def half_experiment(kap: float, mode: str, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        gb = growth(eb_run(C, sup, n_anchor, kappa=kap)["p_hat"], TR, HZ)
        out = []
        for S in range(NS):
            idx = np.where(sup == S)[0]
            drop = (rng.permutation(idx)[:idx.size // 2] if mode == "random"
                    else idx[np.argsort(-gb[idx], kind="stable")][:idx.size // 2])
            keep = np.ones(KFULL, bool); keep[drop] = False
            Ah = np.where(keep)[0]
            run = eb_run(C[Ah], sup[Ah], n_anchor, kappa=kap)
            # Lemma 2 diagnostic: does m' still equal m/lambda?
            gp = np.full(KFULL, np.nan); gp[Ah] = growth(run["p_hat"], TR, HZ)
            eA = Ah[eligible[Ah]]
            out.append(sd50(topn(gp[eA], eA), topn(gb[eA], eA)))
        return np.array(out)

    S0 = 0
    idx0 = np.where(sup == S0)[0]
    keep0 = np.ones(KFULL, bool); keep0[idx0[: idx0.size // 2]] = False
    A0 = np.where(keep0)[0]
    run0 = eb_run(C[A0], sup[A0], n_anchor, kappa=kappa)
    lam_h = p[A0].sum(axis=0)
    e_m_cut = np.abs(run0["m"] - m[A0] / lam_h[None, :]).max() / np.abs(run0["m"]).max()
    e_p_cut = np.abs(run0["p"] - p[A0] / lam_h[None, :]).max() / np.abs(run0["p"]).max()
    check("P10a cutting a super breaks Lemma 2 (m' != m/lambda) but not Lemma 1",
          e_m_cut > 1e-6 and e_p_cut <= TOL_ID,
          f"max rel |m' - m/lambda| = {e_m_cut:.3e} (whole-super drops: {R[:, 2].max():.3e}); "
          f"p' = p/lambda still exact at {e_p_cut:.3e}")
    v_lo = half_experiment(kappa, "random", 0)
    v_lo_f = half_experiment(kappa, "fastest", 0)
    v_hi = half_experiment(300.0, "random", 0)
    v_hi_f = half_experiment(300.0, "fastest", 0)
    check("P10b break magnitude scales with the prior-mass fraction (kappa)",
          abs(v_lo.max() - 0.04) < 1e-12 and abs(v_hi.max() - 0.12) < 1e-12
          and abs(v_hi_f.max() - 0.20) < 1e-12,
          f"kappa=3.148: random half median {np.median(v_lo):.4f} max {v_lo.max():.4f} "
          f"(claim median 0.0000 max 0.0400), fastest-growing half max {v_lo_f.max():.4f}; "
          f"kappa=300: random max {v_hi.max():.4f} (claim 0.1200), fastest-growing max "
          f"{v_hi_f.max():.4f} (claim 0.2000)")
    seeds = [half_experiment(kappa, "random", s).max() for s in range(6)]
    report("P10c seed sensitivity of the random-half probe at production kappa",
           f"max symdiff over seeds 0..5: {[round(float(x), 4) for x in seeds]}")

    # =====================================================================
    hdr("STEP 11 -- prediction 9: negative control, w made TIME-VARYING (rolling)")
    # =====================================================================
    base_roll = eb_run(C, sup, n_anchor, rolling=True)
    g_roll = growth(base_roll["p_hat"], TR, HZ)
    e_fac, sd_roll, tau_roll = [], [], []
    for S in range(NS):
        A = np.where(sup != S)[0]
        run = eb_run(C[A], sup[A], n_anchor, rolling=True)
        lam = p[A].sum(axis=0)
        tr_t = rolling_trail(C[A])
        wt = tr_t / (tr_t + run["kappa"])
        rhs = wt * p[A] + (1.0 - wt) * m[A]
        e_fac.append(np.abs(run["p_hat"] * lam[None, :] - rhs).max() / np.abs(run["p_hat"]).max())
        gp = np.full(KFULL, np.nan); gp[A] = growth(run["p_hat"], TR, HZ)
        elig_A = A[eligible[A]]
        sd_roll.append(sd50(topn(gp[elig_A], elig_A), topn(g_roll[elig_A], elig_A)))
        tau_roll.append(kendalltau(gp[elig_A], g_roll[elig_A]).statistic)
    check("P11 factorization survives TIME-VARYING w (time-constancy is not load-bearing)",
          max(e_fac) <= TOL_ID, f"max rel err = {max(e_fac):.3e}  (claim 6.276e-15)")
    check("P11b rolling-w top-50 symmetric difference is still 0",
          max(sd_roll) == 0.0,
          f"max symdiff = {max(sd_roll):.4f}; tau-b median {np.median(tau_roll):.6f} min "
          f"{min(tau_roll):.6f} (claim median 0.999637, min 0.998164)")

    # =====================================================================
    hdr("STEP 12 -- F4 stress: freeze K at 1024 in the restricted denominator")
    # =====================================================================
    sd_fk, tau_fk, gauge_fk = [], [], []
    for S in range(NS):
        A = np.where(sup != S)[0]
        run = eb_run(C[A], sup[A], n_anchor, k_dim=KFULL)         # closure broken, gauge intact
        rat = run["p"] / p[A]
        gauge_fk.append(float((rat.std(axis=0) / rat.mean(axis=0)).max()))
        gp = np.full(KFULL, np.nan); gp[A] = growth(run["p_hat"], TR, HZ)
        elig_A = A[eligible[A]]
        sd_fk.append(sd50(topn(gp[elig_A], elig_A), topn(g_full[elig_A], elig_A)))
        tau_fk.append(kendalltau(gp[elig_A], g_full[elig_A]).statistic)
    check("P12 frozen-K restriction keeps the per-month gauge and the top-50",
          max(sd_fk) == 0.0 and max(gauge_fk) < 1e-13,
          f"max symdiff = {max(sd_fk):.4f} (claim 0.0000); tau-b median {np.median(tau_fk):.6f} "
          f"(claim 0.999696); max across-region relative sd of p''/p = {max(gauge_fk):.3e} "
          f"-> a per-month scalar, exactly as Lemma 1's remark predicts")

    # =====================================================================
    hdr("STEP 13 -- ADVERSARIAL: unions of MANY supers, and the a-priori certificate")
    # =====================================================================
    e_uni, sd_uni, kap_uni, brk = [], [], [], {}
    for nd in (2, 3, 4, 6, 8, 12, 16, 24, 32):
        rng = np.random.default_rng(1234)
        loc = []
        for _ in range(40):
            drop = rng.permutation(NS)[:nd]
            A = np.where(~np.isin(sup, drop))[0]
            run = eb_run(C[A], sup[A], n_anchor)
            lam = p[A].sum(axis=0)
            wA = trail[A] / (trail[A] + run["kappa"])
            rhs = wA[:, None] * p[A] + (1.0 - wA[:, None]) * m[A]
            e_uni.append(np.abs(run["p_hat"] * lam[None, :] - rhs).max() / np.abs(run["p_hat"]).max())
            kap_uni.append(run["kappa"])
            gp = np.full(KFULL, np.nan); gp[A] = growth(run["p_hat"], TR, HZ)
            elig_A = A[eligible[A]]
            loc.append(sd50(topn(gp[elig_A], elig_A), topn(g_full[elig_A], elig_A)))
        loc = np.array(loc)
        sd_uni.extend(loc)
        brk[nd] = (int((loc > 0).sum()), float(loc.max()))
    check("P13 Theorem A holds for unions of MANY whole supers, not just singletons",
          max(e_uni) <= TOL_ID,
          f"360 random unions of 2..32 supers: max factorization rel err = {max(e_uni):.3e}; "
          f"kappa_A spans [{min(kap_uni):.4f}, {max(kap_uni):.4f}] "
          f"(vs [3.0597, 3.3379] for singletons)")
    check("P13b the top-50 invariance is NOT structural: it BREAKS outside the 64 "
          "single-super drops",
          max(sd_uni) > 0,
          "broken draws / 40 by #supers dropped: "
          + ", ".join(f"{k}->{v[0]}" for k, v in brk.items())
          + f"; max symdiff always exactly {max(sd_uni):.4f} (one swap at the near-tied "
            f"boundary pair). Even 2 dropped supers already break it.")

    n_cert = 0
    for S in range(NS):
        elig_A, Delta = Delta_store[S]
        lam = store[S]["lam"]
        shift = np.log((1 / lam[HZ]).mean() / (1 / lam[TR]).mean())
        B = np.abs(Delta - shift)
        LA = L_full[elig_A]
        o = np.argsort(-LA, kind="stable")
        top, rest = o[:TOP_N], o[TOP_N:]
        if bool((LA[top][:, None] - LA[rest][None, :]
                 > B[top][:, None] + B[rest][None, :]).all()):
            n_cert += 1
    check("P13c Corollary D1's a-priori certificate discharges only 61 of 64 (claimed)",
          n_cert == 61,
          f"{n_cert}/64 restrictions certified by the uniform bound, with the tilt computed "
          f"exactly rather than bounded -- the claim's own honest negative reproduces")

    # =====================================================================
    hdr("STEP 14 -- ADVERSARIAL: the other three origins (masked cutoffs)")
    # =====================================================================
    for T0m in config.VALIDATION_CUTOFFS:
        d = ART / "cutoffs" / T0m
        if not (d / "region_shares.parquet").exists():
            report(f"P14[{T0m}]", "cutoff artifacts absent -- skipped")
            continue
        import json
        meta = json.loads((d / "region_shares.parquet.meta.json").read_text())
        shm = pd.read_parquet(d / "region_shares.parquet")
        mm = sorted(shm["month"].unique())
        Cm = shm.pivot(index="region", columns="month", values="count") \
                .reindex(range(KFULL))[mm].to_numpy(np.float64)
        SMm = shm.pivot(index="region", columns="month", values="share_smoothed") \
                 .reindex(range(KFULL))[mm].to_numpy(np.float64)
        supm = np.load(d / "super_region.i16.npy").astype(np.int32)
        nle = int(meta["n_le_t0"])
        bm = eb_run(Cm, supm, nle)
        TRm, HZm = slice(nle - 12, nle), slice(nle + lo_h - 1, nle + hi_h)
        eligm = Cm[:, TRm].sum(axis=1) >= FLOOR
        gm = growth(bm["p_hat"], TRm, HZm)
        pm, mmm, trm, kpm = bm["p"], bm["m"], bm["trail"], bm["kappa"]
        e_m2, sd_m, ka_m = [], [], []
        for S in range(NS):
            A = np.where(supm != S)[0]
            if A.size == KFULL:
                continue
            run = eb_run(Cm[A], supm[A], nle)
            lam = pm[A].sum(axis=0)
            wA = trm[A] / (trm[A] + run["kappa"])
            rhs = wA[:, None] * pm[A] + (1.0 - wA[:, None]) * mmm[A]
            e_m2.append(np.abs(run["p_hat"] * lam[None, :] - rhs).max() / np.abs(run["p_hat"]).max())
            ka_m.append(run["kappa"])
            gp = np.full(KFULL, np.nan); gp[A] = growth(run["p_hat"], TRm, HZm)
            eA = A[eligm[A]]
            sd_m.append(sd50(topn(gp[eA], eA), topn(gm[eA], eA)))
        sd_m = np.array(sd_m)
        check(f"P14[{T0m}] instrument reproduced (kappa + share_smoothed) and Theorem A holds",
              abs(kpm - meta["kappa"]) < 1e-12 and np.abs(bm["p_hat"] - SMm).max() == 0.0
              and max(e_m2) <= TOL_ID,
              f"kappa = {kpm:.12f} (meta {meta['kappa']:.12f}); max|p_hat - share_smoothed| = "
              f"{np.abs(bm['p_hat'] - SMm).max()!r}; factorization max rel err = {max(e_m2):.3e}")
        report(f"P14[{T0m}] top-50 invariance at this origin",
               f"BROKEN in {int((sd_m > 0).sum())}/{len(sd_m)} single-super drops "
               f"(median symdiff {np.median(sd_m):.4f}, max {sd_m.max():.4f}); "
               f"kappa = {kpm:.4f}, kappa_A in [{min(ka_m):.4f}, {max(ka_m):.4f}]")

    # =====================================================================
    hdr("SUMMARY")
    # =====================================================================
    npass = sum(1 for _, o, _ in _RESULTS if o)
    for nm, o, dt in _RESULTS:
        if not o:
            print(f"  FAILED: {nm} -- {dt}")
    print(f"\n  {npass}/{len(_RESULTS)} checks PASSED   (elapsed {time.time() - t_start:.1f}s)")
    verdict = "PASS" if npass == len(_RESULTS) else "FAIL"
    print(f"\n  OVERALL: {verdict} -- the exact factorization "
          f"p_hat' = p_hat^(kappa_A)/lambda and the C1/C2/C3 rank-coherence "
          f"classification are {'CONFIRMED' if verdict == 'PASS' else 'NOT fully confirmed'} "
          f"on the production instrument.")
    print("\n  NOTE: check P9d PASSING is a CORRECTION to the claim, not a confirmation of "
          "it -- Part II lists Shannon entropy and Herfindahl as C3, which is true only of "
          "the unnormalised path; on a re-closed composition both are exactly C1.")
    print("  NOTE: P13b/P14 sharpen the claim's own Part IV caveat into a measurement. The "
          "factorization (Theorem A) is structural and holds on all four origins and for "
          "unions of up to 32 supers; the top-50 invariance is NOT -- it breaks for unions "
          "of as few as 2 supers at the production origin, and for single-super drops at "
          "two of the three masked origins, where kappa is ~7 instead of ~3.15.")
    print("  NOTE: P8d/P8e print two centrings because the claim's 'individual near-boundary "
          "perturbation = 3.240e-03 (67x)' and 'comonotone factor median 3.2x / max 317x' do "
          "not state which constant Delta is centred by; neither centring reproduces them. "
          "The load-bearing certificate numbers (P8b) reproduce exactly.")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
