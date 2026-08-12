"""Numerical companion for O2 -- "Ultrametric diffusion as a model of knowledge dynamics".

Checks, against the committed artifacts, every number the O2 write-up asserts:

  THEOREM 1  exact spectral theorem for the finite ultrametric (ABKO/Vladimirov)
             generator: the ilr/SBP balance vectors ARE the eigenvectors, with
             lambda_v given in closed form by the ancestral chain.
             -> identity, checked to machine precision.
  COR. 2     Langevin closure: D_v = (sigma^2/lambda)(1-e^{-lambda tau}) and
             rho_1(v) = -(1-e^{-lambda tau})/2 in (-1/2, 0).
             -> a RANGE prediction, checked node by node (both ends).
  PROP. 3 /  slope-free kernel invariant b_n/b_h, and its claimed refutation of
  THEOREM 4  every power kernel u^{-theta}.
             -> inequality, checked with slack + a full scan of the model ratio.
  THEOREM 5  replacement two-component law D_v = c_1 D^null_v + c_2/lambda_v(theta).

Everything is recomputed from `artifacts/` (+ the per-cutoff copies); nothing is
read from the write-up.  The EB pipeline is reimplemented from scratch and gated
on reproducing kappa = 3.147961 and share_smoothed to EXACTLY 0.0 difference on
all four origins before any derived number is trusted.

EXPLORATORY except where it re-derives quantities frozen in
decisions/2026-08-12-scale-spectrum-preregistration.md (alpha_h, R^2, the
matched-N multinomial null).  Theorems 1/4/5 are not pre-registered.

CPU only, no network.  Runtime ~3 min.  Run from the repo root:

    python scripts/verify_ultradiffusion.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import cophenet, is_monotonic
from scipy.optimize import least_squares
from scipy.spatial.distance import squareform
from scipy.stats import f as f_dist

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from brontide import config, series                                    # noqa: E402

K = config.N_REGIONS
NSUP = config.N_SUPERREGIONS
NV = K - 1                       # internal nodes
NULL_SEEDS = 30                  # seeds 0..29 (published null used 200)
TAU = 1.0                        # months

ORIGINS = [
    ("production", REPO / "artifacts"),
    ("2017-12", REPO / "artifacts" / "cutoffs" / "2017-12"),
    ("2020-12", REPO / "artifacts" / "cutoffs" / "2020-12"),
    ("2022-12", REPO / "artifacts" / "cutoffs" / "2022-12"),
]

CHECKS: list[tuple[str, bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "", tag: str = "CORE") -> bool:
    """tag CORE  = a load-bearing claim of the write-up.
       tag DETAIL = an as-stated wording/secondary number; a FAIL here is a defect
                    in the write-up's statement, not necessarily in its verdict."""
    CHECKS.append((name, bool(ok), detail, tag))
    print(f"  [{'PASS' if ok else 'FAIL'}] ({tag}) {name}" + (f"   {detail}" if detail else ""))
    return bool(ok)


def close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def hdr(s: str) -> None:
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ---------------------------------------------------------------- primitives
def build_psi(Z: np.ndarray) -> np.ndarray:
    """SBP/ilr balance matrix (K-1, K) transported to the dendrogram (PhILR)."""
    mem: list = [None] * (2 * K - 1)
    for i in range(K):
        mem[i] = np.array([i], dtype=np.int64)
    Psi = np.zeros((NV, K))
    for v in range(NV):
        a, b = int(Z[v, 0]), int(Z[v, 1])
        L, R = mem[a], mem[b]
        r, s = L.size, R.size
        Psi[v, L] = np.sqrt(s / (r * (r + s)))
        Psi[v, R] = -np.sqrt(r / (s * (r + s)))
        mem[K + v] = np.concatenate([L, R])
    return Psi


def shell_matrix(Z: np.ndarray) -> np.ndarray:
    """M[v,k] = mu-mass of the k-th ancestral shell of v (counting measure).

    lambda_v(w) = sum_k M[v,k] * w(h_k) is exactly the Theorem-1 closed form:
    M[v,v] = n_v and M[v,u_k] = n_{u_k} - n_{u_{k-1}} along the ancestral chain.
    """
    par = np.full(2 * K - 1, -1, dtype=np.int64)
    for v in range(NV):
        par[int(Z[v, 0])] = K + v
        par[int(Z[v, 1])] = K + v
    n = Z[:, 3]
    M = np.zeros((NV, NV))
    for v in range(NV):
        M[v, v] = n[v]
        prev, p = n[v], par[K + v]
        while p != -1:
            k = int(p) - K
            M[v, k] = n[k] - prev
            prev = n[k]
            p = par[p]
    return M


def eb_pipeline(C: np.ndarray, sup: np.ndarray, n_anchor: int):
    """EXACT reimplementation of series.run_shares' share_smoothed."""
    a = config.ADD_ALPHA
    p = (C + a) / (C.sum(axis=0) + a * K)[None, :]
    n_s = np.bincount(sup, minlength=NSUP)
    P_S = np.zeros((NSUP, C.shape[1]))
    np.add.at(P_S, sup, p)
    m = (P_S / n_s[:, None])[sup]
    trail = C[:, max(0, n_anchor - 12):n_anchor].sum(axis=1)
    kappa = series._estimate_kappa(trail, sup)
    w = (trail / (trail + kappa))[:, None]
    return w * p + (1.0 - w) * m, float(kappa), p


def ols(X: np.ndarray, y: np.ndarray):
    """OLS with intercept -> (beta, se, R^2, cov)."""
    X = np.atleast_2d(X)
    if X.shape[0] != y.size:
        X = X.T
    X1 = np.column_stack([np.ones(y.size), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ beta
    n, k = X1.shape
    cov = (r @ r / (n - k)) * np.linalg.inv(X1.T @ X1)
    r2 = 1.0 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return beta, np.sqrt(np.diag(cov)), float(r2), cov


def ols_slopes_many(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Slopes of many regressions sharing design X (n,k). Y is (n,m) -> (k+1,m)."""
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    return np.linalg.pinv(X1) @ Y


def r2_of(y: np.ndarray, yhat: np.ndarray) -> float:
    r = y - yhat
    return float(1.0 - (r @ r) / ((y - y.mean()) @ (y - y.mean())))


def load_origin(name: str, d: pathlib.Path) -> dict:
    df = pq.read_table(d / "region_shares.parquet").to_pandas()
    meta = json.loads((d / "region_shares.parquet.meta.json").read_text())
    months = sorted(df["month"].unique().tolist())
    mi = {m: i for i, m in enumerate(months)}
    T = len(months)
    ri = df["region"].to_numpy()
    ti = df["month"].map(mi).to_numpy()
    C = np.zeros((K, T)); C[ri, ti] = df["count"].to_numpy()
    SS = np.zeros((K, T)); SS[ri, ti] = df["share_smoothed"].to_numpy()
    PA = np.zeros((K, T)); PA[ri, ti] = df["share_addalpha"].to_numpy()
    Z = np.load(d / "linkage.f64.npy")
    return dict(name=name, dir=d, C=C, SS=SS, PA=PA, months=months, meta=meta,
                n_anchor=int(meta.get("n_le_t0", meta["n_months"])),
                sup=np.load(d / "super_region.i16.npy").astype(np.int32),
                Z=Z, h=Z[:, 2].copy(), n=Z[:, 3].copy(), Psi=build_psi(Z))


def dispersion(Psi: np.ndarray, X: np.ndarray):
    """b_v(t), Delta b, D_v = Var_t[Delta b] (ddof=1)."""
    b = Psi @ np.log(X)
    db = np.diff(b, axis=1)
    return b, db, db.var(axis=1, ddof=1)


# =========================================================== 0. pipeline gate
def section0():
    hdr("SECTION 0 -- pipeline fidelity gate (nothing downstream is trusted "
        "until this passes)")
    O = {}
    for name, d in ORIGINS:
        o = load_origin(name, d)
        phat, kappa, p = eb_pipeline(o["C"], o["sup"], o["n_anchor"])
        dss = float(np.abs(phat - o["SS"]).max())
        dpa = float(np.abs(p - o["PA"]).max())
        ok = (dss == 0.0) and (dpa == 0.0) and close(kappa, o["meta"]["kappa"], 1e-9)
        check(f"EB reimplementation exact [{name}]", ok,
              f"kappa={kappa:.6f} (meta {o['meta']['kappa']:.6f})  "
              f"max|d share_smoothed|={dss:.1e}  max|d share_addalpha|={dpa:.1e}  "
              f"n_le_t0={o['n_anchor']} T={len(o['months'])}")
        o["kappa"] = kappa
        O[name] = o
    check("kappa(production) == 3.147961", close(O["production"]["kappa"], 3.147961, 5e-7),
          f"{O['production']['kappa']:.6f}")
    # the documented failure mode: anchoring the trailing window at the window end
    o = O["2017-12"]
    _, kbad, _ = eb_pipeline(o["C"], o["sup"], o["C"].shape[1])
    check("window-end anchoring IS a real corruption (control)",
          abs(kbad - o["kappa"]) > 1.0,
          f"2017-12 kappa would be {kbad:.4f} not {o['kappa']:.4f} -- so the "
          f"n_le_t0 anchor is load-bearing, as the claim warns")
    return O


# ================================================== 1. Theorem 1 (the identity)
def section1(O):
    hdr("SECTION 1 -- THEOREM 1: L psi_v = -lambda_v psi_v with the closed-form "
        "lambda_v  (identity -> machine precision)")
    thetas = (0.5, 1.0, 1.75, 3.0)
    bound_res, bound_rel = 1.2e-13, 1e-13
    for name, _ in ORIGINS:
        o = O[name]
        Z, Psi = o["Z"], o["Psi"]
        check(f"is_monotonic(Z) [{name}]", bool(is_monotonic(Z)),
              f"h in [{o['h'].min():.4f}, {o['h'].max():.4f}]  "
              f"= {o['h'].max()/o['h'].min():.2f}x = "
              f"{np.log2(o['h'].max()/o['h'].min()):.2f} octaves")
        check(f"Psi Psi^T = I, Psi 1 = 0 [{name}]",
              np.abs(Psi @ Psi.T - np.eye(NV)).max() < 1e-14
              and np.abs(Psi.sum(axis=1)).max() < 1e-14,
              f"|PsiPsi^T-I|={np.abs(Psi @ Psi.T - np.eye(NV)).max():.2e}  "
              f"|Psi1|={np.abs(Psi.sum(axis=1)).max():.2e}  "
              f"min share={o['SS'].min():.3e} (>0: {bool(o['SS'].min() > 0)})")
        U = squareform(cophenet(Z))
        M = shell_matrix(Z)
        o["M"] = M
        iu = np.triu_indices(K, 1)
        res_all, rel_all = [], []
        for th in thetas:
            W = np.zeros((K, K))
            W[iu] = U[iu] ** (-th)
            W += W.T
            L = W.copy()
            np.fill_diagonal(L, -W.sum(axis=1))
            lam = M @ (o["h"] ** (-th))                       # Theorem-1 closed form
            LP = Psi @ L                                       # rows = L psi_v (L sym)
            res = float(np.abs(LP + lam[:, None] * Psi).max())
            ray = -np.einsum("vi,vi->v", Psi, LP)              # Rayleigh quotient
            rel = float(np.abs(lam / ray - 1.0).max())
            res_all.append(res); rel_all.append(rel)
            del W, L, LP
        check(f"eigen-identity + closed form, theta={thetas} [{name}]",
              max(res_all) <= bound_res and max(rel_all) <= bound_rel,
              "max||L psi+lam psi||_inf = " + " / ".join(f"{x:.2e}" for x in res_all)
              + f"  (bound {bound_res:.1e}, slack x{bound_res/max(res_all):.0f})\n"
              + "            max|lam_formula/lam_Rayleigh-1| = "
              + " / ".join(f"{x:.2e}" for x in rel_all)
              + f"  (bound {bound_rel:.1e}, slack x{bound_rel/max(rel_all):.0f})")
        del U


# ============================== 2. pre-registered S1 anchors + shape diagnostics
def section2(O):
    hdr("SECTION 2 -- pre-registered S1 anchors, tree mass dimension, curvature")
    # NOTE the sign convention: the pre-registration writes log2 D = c - alpha_h log2 h,
    # and the measured slope of log2 D on log2 h is POSITIVE, so alpha_h is reported
    # NEGATIVE (-0.3486). D_v grows with merge height.
    want_a = {"production": -0.3486, "2017-12": -0.3035, "2020-12": -0.3061, "2022-12": -0.3280}
    want_r = {"production": 0.2752, "2017-12": 0.2903, "2020-12": 0.2746, "2022-12": 0.3044}
    for name, _ in ORIGINS:
        o = O[name]
        _, db, D = dispersion(o["Psi"], o["SS"])
        o["db"], o["D"] = db, D
        beta, se, r2, _ = ols(np.log2(o["h"]), np.log2(D))
        o["alpha_h"], o["se_alpha"], o["r2_pl"] = -beta[1], se[1], r2
        check(f"alpha_h / R^2 reproduce [{name}]",
              close(-beta[1], want_a[name], 5e-4) and close(r2, want_r[name], 5e-4),
              f"alpha_h={-beta[1]:+.4f} (claim {want_a[name]:+})  SE={se[1]:.4f}  "
              f"R^2={r2:.4f} (claim {want_r[name]})")
    msk = [O[n]["alpha_h"] for n in ("2017-12", "2020-12", "2022-12")]
    spread = max(msk) - min(msk)
    check("SE(alpha_h) = 0.0177 on production, and the masked spread is ~1.4 x SE "
          "(so the '5% cross-cutoff stability' carries no information beyond one fit's precision)",
          close(O["production"]["se_alpha"], 0.0177, 5e-5) and 1.0 < spread / 0.0177 < 2.0,
          f"SE={O['production']['se_alpha']:.4f}; masked alpha_h spread = {spread:.4f} = "
          f"{spread/O['production']['se_alpha']:.2f} x SE "
          f"({100*spread/abs(np.mean(msk)):.1f}% of the mean)")

    o = O["production"]
    b, se, r2, _ = ols(np.log2(o["h"]), np.log2(o["n"]))
    check("tree mass dimension d = 1.6463 +/- 0.0171, R^2 = 0.9003",
          close(b[1], 1.6463, 5e-4) and close(se[1], 0.0171, 5e-4) and close(r2, 0.9003, 5e-4),
          f"d={b[1]:.4f} +/- {se[1]:.4f}  R^2={r2:.4f}")

    lh = np.log2(o["h"])
    b, se, r2, _ = ols(np.column_stack([lh, lh ** 2]), np.log2(o["D"]))
    t2 = b[2] / se[2]
    check("curvature b_2 = +0.0154 +/- 0.0128 (t = +1.20), sign is WRONG for a crossover",
          close(b[2], 0.0154, 5e-4) and close(se[2], 0.0128, 5e-4) and b[2] > 0,
          f"b_2={b[2]:+.4f} +/- {se[2]:.4f}  t={t2:+.2f}  (crossover needs b_2 < 0)")

    # 3-parameter crossover fit D = S*phi(Lambda h^-a), phi(x)=(1-e^-x)/x
    y = np.log2(o["D"])
    def resid(q):
        logS, logL, a = q
        x = np.exp(logL) * o["h"] ** (-a)
        phi = np.where(x < 1e-8, 1.0 - x / 2, (1.0 - np.exp(-x)) / np.maximum(x, 1e-300))
        return y - (logS / np.log(2) + np.log2(np.maximum(phi, 1e-300)))
    best = None
    for l0 in (0.0, 2.0, 4.0):
        for a0 in (0.1, 0.35, 1.0):
            r = least_squares(resid, [0.0, l0, a0], max_nfev=20000)
            if best is None or r.cost < best.cost:
                best = r
    r2x = r2_of(y, y - best.fun)
    xfit = np.exp(best.x[1]) * o["h"] ** (-best.x[2])
    check("3-param crossover fit collapses onto the pure power law (R^2 identical to 4 dp)",
          close(r2x, o["r2_pl"], 5e-5),
          f"R^2_crossover={r2x:.4f} vs R^2_powerlaw={o['r2_pl']:.4f}; fitted lambda*tau in "
          f"[{xfit.min():.0f}, {xfit.max():.0f}] -- deep in the phi(x)=1/x regime, where S and "
          f"Lambda are not separately identified (the write-up quotes 25..105, a different "
          f"point on the same flat ridge), so NO saturation is resolvable either way")


# ============================================ 3. Corollary 2 -- the rho_1 range
def section3(O):
    hdr("SECTION 3 -- COROLLARY 2: rho_1(v) = -(1-e^{-lambda tau})/2 must lie in "
        "(-1/2, 0)  (a two-sided range prediction)")
    def lag1(db):
        c = db - db.mean(axis=1, keepdims=True)
        return (c[:, 1:] * c[:, :-1]).sum(axis=1) / (c * c).sum(axis=1)

    o = O["production"]
    rho = lag1(o["db"])
    o["rho1"] = rho
    n_pos = int((rho > 0).sum())
    n_below = int((rho < -0.5).sum())
    check("upper end of the range holds: 0 of 1023 nodes have rho_1 > 0",
          n_pos == 0,
          f"rho_1 in [{rho.min():.4f}, {rho.max():.4f}], median {np.median(rho):.4f}; "
          f"n(rho>0) = {n_pos}")
    check("LOWER end of the range holds: rho_1 > -1/2 for all nodes  "
          "(the write-up asserts rho_1 in (-1/2,0) and calls the prediction universal)",
          n_below == 0,
          f"n(rho_1 < -1/2) = {n_below} / {NV} = {100*n_below/NV:.1f} % -- these nodes admit "
          f"NO real lambda_v under the Langevin closure (1 + 2 rho <= 0). They are also outside "
          f"OU + i.i.d. observation noise, whose rho_1 -> -1/2 strictly from above. "
          f"The write-up quotes the range [-0.645, -0.259] and the interval (-1/2, 0) on the "
          f"same page without noting that the former is not contained in the latter. "
          f"(Cuts the SAME way as the write-up's verdict: one more failure of the closure.)",
          tag="DETAIL")
    # is the whole rho_1 signature just the sampling floor?  the matched-N null has
    # i.i.d.-across-months noise, for which E[rho_1(Delta b)] = -1/2 exactly.
    C, sup, na = o["C"], o["sup"], o["n_anchor"]
    Nt = np.rint(C.sum(axis=0)).astype(np.int64)
    q = C.sum(axis=1) / C.sum()
    rn = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        Cn = np.empty_like(C)
        for t in range(C.shape[1]):
            Cn[:, t] = rng.multinomial(Nt[t], q)
        Xn, _, _ = eb_pipeline(Cn, sup, na)
        rn.append(lag1(np.diff(o["Psi"] @ np.log(Xn), axis=1)))
    rn = np.concatenate(rn)
    check("rho_1 discriminates the model from pure sampling noise "
          "(the write-up offers rho_1 < 0 as confirmation of the Langevin closure)",
          abs(np.median(rho) - np.median(rn)) > 0.05,
          f"observed median rho_1 = {np.median(rho):.4f}, frac < -1/2 = {n_below/NV:.3f};  "
          f"matched-N NULL median rho_1 = {np.median(rn):.4f}, frac < -1/2 = "
          f"{(rn < -0.5).mean():.3f}  -> the null reproduces the observed rho_1 signature "
          f"almost exactly, so 'rho_1 < 0 on all 1023 nodes' is NOT evidence for the "
          f"diffusion; i.i.d. sampling noise alone predicts rho_1 -> -1/2. "
          f"(Cuts the SAME way as the write-up's verdict: the sign test has no power.)",
          tag="DETAIL")
    ok = 1.0 + 2.0 * rho > 0
    lt = -np.log(1.0 + 2.0 * rho[ok])
    check("median implied lambda*tau = 2.43 and 5th pct = 1.32, as published",
          close(np.median(lt), 2.43, 6e-3) and close(np.quantile(lt, 0.05), 1.32, 6e-3),
          f"median={np.median(lt):.4f}, q05={np.quantile(lt, 0.05):.4f} -- reproduced, BUT only "
          f"after silently dropping the {NV - int(ok.sum())} nodes with rho_1 <= -1/2; the "
          f"monotone map of the FULL median rho_1={np.median(rho):.4f} gives lambda*tau="
          f"{-np.log(1+2*np.median(rho)):.4f}, so '2.43' is a conditional median the write-up "
          f"does not flag as conditional")
    check("conclusion 'fully relaxed at monthly resolution' survives either way",
          np.quantile(lt, 0.05) > 1.0,
          f"5th pct of the implied relaxation time = {1/np.quantile(lt, 0.05):.2f} months "
          f"< tau = 1 month, so D_v = sigma_v^2/lambda_v and (sigma_v, lambda_v) are "
          f"NOT separately identified -- the identifiability collapse is confirmed")


# ================= 4. Proposition 3 / Theorem 4 -- the slope-free kernel invariant
def section4(O):
    hdr("SECTION 4 -- PROP. 3 / THEOREM 4: the slope-free invariant b_n/b_h and "
        "the claimed refutation of every power kernel")
    o = O["production"]
    lh, ln_ = np.log2(o["h"]), np.log2(o["n"])
    X = np.column_stack([lh, ln_])
    y = np.log2(o["D"])
    beta, se, r2, cov = ols(X, y)
    bh, bn = beta[1], beta[2]
    g = np.array([-bn / bh ** 2, 1.0 / bh])
    se_ratio = float(np.sqrt(g @ cov[1:, 1:] @ g))
    ratio = bn / bh
    check("two-exponent fit: b_h=+1.0625+/-0.0509, b_n=-0.4336+/-0.0293, R^2=0.4030",
          close(bh, 1.0625, 5e-4) and close(bn, -0.4336, 5e-4) and close(r2, 0.4030, 5e-4),
          f"b_h={bh:+.4f}+/-{se[1]:.4f}  b_n={bn:+.4f}+/-{se[2]:.4f}  R^2={r2:.4f}")
    check("measured invariant b_n/b_h = -0.4081 +/- 0.0110 (delta method)",
          close(ratio, -0.4081, 5e-4) and close(se_ratio, 0.0110, 5e-5),
          f"{ratio:.4f} +/- {se_ratio:.4f}")

    # --- model ratio over the power-kernel family --------------------------
    M, h = o["M"], o["h"]
    def model_ratio_and_r2(ths):
        LAM = M @ np.exp(-np.log(h)[:, None] * np.asarray(ths)[None, :])
        Y = -np.log2(LAM)
        B = ols_slopes_many(X, Y)
        rat = B[2] / B[1]
        # constrained fit: log2 D = a + s * (-log2 lambda), model needs s >= 0
        r2c = np.empty(len(ths))
        for j in range(len(ths)):
            bb, _, rr, _ = ols(Y[:, j], y)
            r2c[j] = rr if bb[1] >= 0 else 0.0
        return rat, r2c, B

    def beta_h_of(th: float) -> float:
        return float(ols_slopes_many(X, (-np.log2(M @ (h ** (-th))))[:, None])[1, 0])

    def ratio_of(th: float) -> float:
        b_ = ols_slopes_many(X, (-np.log2(M @ (h ** (-th))))[:, None])[:, 0]
        return float(b_[2] / b_[1])

    ths = np.round(np.arange(0.02, 6.0001, 0.02), 4)
    rat, r2c, B = model_ratio_and_r2(ths)
    # locate the pole of the ratio (beta_h = 0) by bisection, then split the branches
    sgn = np.sign(B[1])
    j = int(np.where(np.diff(sgn) != 0)[0][0])
    lo, hi = float(ths[j]), float(ths[j + 1])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.sign(beta_h_of(mid)) == np.sign(beta_h_of(lo)):
            lo = mid
        else:
            hi = mid
    pole = 0.5 * (lo + hi)
    branchB = ths > pole
    # refine the minimizer of the ratio on branch B
    i = int(np.argmin(np.where(branchB, rat, np.inf)))
    fine = np.linspace(ths[i] - 0.05, ths[i] + 0.05, 201)
    rf = np.array([ratio_of(t) for t in fine])
    th_min, r_min = float(fine[np.argmin(rf)]), float(rf.min())
    check("on the branch where the model has any log-h dependence, the model ratio "
          "bottoms out at -0.2857 at theta ~ 2.0",
          close(r_min, -0.2857, 5e-4) and close(th_min, 2.0, 0.05),
          f"min = {r_min:.6f} at theta = {th_min:.3f} (write-up says 2.0; ratio at exactly "
          f"theta=2.00 is {ratio_of(2.0):.6f}, so the minimum is flat to 1e-5 over "
          f"theta in [1.95, 2.05])")

    # the write-up's literal universal claim: ratio >= -0.286 for ALL theta in (0, inf)
    worst = float(rat.min())
    check("write-up's literal claim 'the ratio attains its minimum -0.2857 at theta = 2.0 "
          "and satisfies beta_n/beta_h >= -0.286 THROUGHOUT theta in (0, inf)'",
          worst >= -0.286 - 1e-6,
          f"FALSIFIED as stated: min over the scan = {worst:.4f} at theta = "
          f"{ths[int(np.argmin(rat))]:.2f}.  beta_h(theta) crosses zero at theta = "
          f"{pole:.6f}, so the ratio has a POLE there and is unbounded below on "
          f"(0, {pole:.4f}) -- it is neither a minimum nor a bound over (0, inf)",
          tag="DETAIL")

    # the honest version: the ATTAINABLE SET of the ratio and whether -0.4081 is in it
    S = M @ np.log(h)                       # theta->0+ limit direction of -log lambda
    bl, _, _, _ = ols(X, S)
    limit0 = bl[2] / bl[1]                  # sup of the lower branch
    lo_branch = rat[~branchB]
    check("REPAIRED FORM: the measured ratio is still NOT attained by any power kernel "
          "-- it falls in the open gap between the two branches",
          limit0 < ratio < r_min and float(lo_branch.max()) <= limit0 + 1e-6,
          f"attainable set over theta in (0,inf) = (-inf, {limit0:.4f}] U [{r_min:.4f}, +inf) "
          f"(the lower branch's sup is the exact theta->0+ limit {limit0:.4f}); measured "
          f"{ratio:.4f} lies inside the open gap, so the refutation SURVIVES the repair")
    n_se = (r_min - ratio) / se_ratio
    check("distance from the measurement to the nearest attainable model ratio >= 11 SE",
          n_se >= 11.0,
          f"({r_min:.4f} - ({ratio:.4f})) / {se_ratio:.4f} = {n_se:.2f} SE "
          f"(write-up says 11.2; distance to the far branch is "
          f"{(ratio - limit0)/se_ratio:.0f} SE)")
    lowB = ~branchB
    check("the pole region is empty of explanatory power, so excluding it is legitimate",
          float(r2c[lowB].max()) < 0.05,
          f"max R^2_c on the lower branch (theta < {pole:.3f}) = {r2c[lowB].max():.4f}; "
          f"max_theta R^2_c overall = {r2c.max():.4f} at theta = "
          f"{ths[int(np.argmax(r2c))]:.2f} (ratio there {rat[int(np.argmax(r2c))]:.4f}); "
          f"R^2_c(1.75)={r2c[np.argmin(np.abs(ths-1.75))]:.4f} (write-up 0.3502), "
          f"R^2_c(1.85)={r2c[np.argmin(np.abs(ths-1.85))]:.4f} (write-up 0.3659)")

    # --- self-similar reduction: theta* = 1.85, R^2(log2 lambda ~ log2 h) = 0.7680
    lam185 = M @ (h ** -1.85)
    _, _, r2h, _ = ols(lh, np.log2(lam185))
    d_mass = ols(lh, ln_)[0][1]
    a_ss = 1.85 - d_mass
    check("Corollary 1's route fails: R^2(log2 lambda(1.85) ~ log2 h) = 0.7680, and the "
          "self-similar exponent alpha = theta - d = 0.204 misses the measured 0.3486 by 71%",
          close(r2h, 0.7680, 5e-4) and close(a_ss, 0.204, 1e-3)
          and close(100 * (abs(o["alpha_h"]) - a_ss) / a_ss, 71.0, 1.5),
          f"R^2={r2h:.4f} -> {100*(1-r2h):.0f}% of the log-eigenvalue variance is orthogonal "
          f"to log h; alpha_selfsim = {a_ss:.4f} vs |alpha_h| = {abs(o['alpha_h']):.4f} "
          f"= {100*(abs(o['alpha_h'])-a_ss)/a_ss:.0f}% error")

    # --- two-parameter kernel u^-theta e^{-u/xi} ---------------------------
    # NOTE the damping must act on the ANCESTRAL SHELL heights, not on lambda: the shell
    # decomposition M carries w(h_k) per ancestor k, so exp(-h_k/xi) multiplies column k.
    th_g = np.round(np.arange(0.25, 4.001, 0.25), 4)
    xi_g = list(np.geomspace(1.0, 1000.0, 25)) + [np.inf]
    best, viol, reach = None, [], []
    for xi in xi_g:
        damp = np.zeros(NV) if np.isinf(xi) else -h / xi
        LAM = M @ np.exp(-np.log(h)[:, None] * th_g[None, :] + damp[:, None])
        Y = -np.log2(LAM)
        Bm = ols_slopes_many(X, Y)
        rr = Bm[2] / Bm[1]
        for j, th in enumerate(th_g):
            bb, _, r2j, _ = ols(Y[:, j], y)
            r2j = r2j if bb[1] >= 0 else 0.0
            if best is None or r2j > best[0]:
                best = (r2j, th, xi, rr[j])
            if r2j > 0.05 and rr[j] < -0.29:
                viol.append((th, xi, rr[j], r2j))
            if abs(rr[j] - ratio) < 2 * se_ratio:
                reach.append((th, xi, rr[j], r2j))
    if viol:
        print("        grid points with R^2_c > 0.05 and ratio < -0.29:")
        for th, xi, rr_, r2j in sorted(viol, key=lambda z: z[2])[:8]:
            print(f"          theta={th:.2f} xi={'inf' if np.isinf(xi) else f'{xi:8.2f}'} "
                  f"ratio={rr_:+.4f} R^2_c={r2j:.4f}")
    check("two-parameter kernel u^-theta e^{-u/xi} does not reach the measured ratio "
          "anywhere it explains a meaningful share of the spectrum",
          not any(abs(z[2] - ratio) < 2 * se_ratio and z[3] > 0.05 for z in reach),
          f"R^2_c maximizer on the grid: theta={best[1]}, "
          f"xi={'inf' if np.isinf(best[2]) else f'{best[2]:.1f}'}, R^2_c={best[0]:.4f}, "
          f"ratio={best[3]:.4f} -> {(best[3]-ratio)/se_ratio:.1f} SE from the measurement. "
          f"Grid points within 2 SE of the measured ratio: {len(reach)}, of which "
          f"{sum(1 for z in reach if z[3] > 0.05)} have R^2_c > 0.05 "
          f"(max R^2_c among them = {max([z[3] for z in reach], default=0.0):.4f})")

    # --- the invariant across the masked origins and two controls ----------
    for name, _ in ORIGINS[1:]:
        oo = O[name]
        bb, _, rr, _ = ols(np.column_stack([np.log2(oo["h"]), np.log2(oo["n"])]),
                           np.log2(oo["D"]))
        oo["ratio"] = bb[2] / bb[1]
        print(f"        {name}: b_h={bb[1]:+.4f} b_n={bb[2]:+.4f} ratio={bb[2]/bb[1]:.4f} R^2={rr:.4f}")
    want = {"2017-12": -0.366, "2020-12": -0.385, "2022-12": -0.355}
    check("masked-origin ratios reproduce (-0.366 / -0.385 / -0.355)",
          all(close(O[n]["ratio"], want[n], 1e-3) for n in want),
          "all four origins are below the model's attainable -0.2857")

    # control: EB shrinkage removed (raw add-alpha shares)
    _, _, Draw = dispersion(o["Psi"], o["PA"])
    braw, _, r2raw, _ = ols(X, np.log2(Draw))
    araw = -ols(lh, np.log2(Draw))[0][1]
    check("control -- none of this is an EB-shrinkage artifact "
          "(alpha_h=-0.3397, b=(1.088,-0.455), ratio=-0.418, R^2=0.3960)",
          close(araw, -0.3397, 1e-3) and close(braw[1], 1.088, 2e-3)
          and close(braw[2], -0.455, 2e-3) and close(braw[2] / braw[1], -0.418, 2e-3)
          and close(r2raw, 0.3960, 1e-3),
          f"raw add-alpha shares (EB removed): alpha_h={araw:+.4f} b_h={braw[1]:+.4f} "
          f"b_n={braw[2]:+.4f} ratio={braw[2]/braw[1]:.4f} R^2={r2raw:.4f}")
    o["ratio"], o["se_ratio"], o["model_min_ratio"] = ratio, se_ratio, rat[i]


# ============================ 5. the matched-N null + Theorem 5 (the replacement)
def section5(O):
    hdr("SECTION 5 -- matched-N multinomial null (seeds 0..29) and THEOREM 5: "
        "D_v = c_1 D^null_v + c_2 / lambda_v(theta)")
    for name, _ in ORIGINS:
        o = O[name]
        C, sup, na = o["C"], o["sup"], o["n_anchor"]
        Nt = np.rint(C.sum(axis=0)).astype(np.int64)
        q = C.sum(axis=1) / C.sum()                    # time-averaged (pooled) shares
        acc = np.zeros(NV)
        alphas, ratios = [], []
        Xd = np.column_stack([np.log2(o["h"]), np.log2(o["n"])])
        for seed in range(NULL_SEEDS):
            rng = np.random.default_rng(seed)
            Cn = np.empty_like(C)
            for t in range(C.shape[1]):
                Cn[:, t] = rng.multinomial(Nt[t], q)
            Xn, _, _ = eb_pipeline(Cn, sup, na)
            _, _, Dn = dispersion(o["Psi"], Xn)
            acc += Dn
            alphas.append(-ols(np.log2(o["h"]), np.log2(Dn))[0][1])
            bb = ols(Xd, np.log2(Dn))[0]
            ratios.append(bb[2] / bb[1])
        o["Dnull"] = acc / NULL_SEEDS
        o["alpha_null"] = (float(np.mean(alphas)), float(np.std(alphas, ddof=1)))
        o["ratio_null"] = (float(np.mean(ratios)), float(np.std(ratios, ddof=1)))
        print(f"        {name}: alpha_null = {np.mean(alphas):.4f} +/- {np.std(alphas, ddof=1):.4f}"
              f"   ratio_null = {np.mean(ratios):.4f} +/- {np.std(ratios, ddof=1):.4f}")

    a, s = O["production"]["alpha_null"]
    sig = (abs(O["production"]["alpha_h"]) - 0.1151) / 0.0072
    check("null alpha_h = -0.114 +/- 0.006 (published -0.1151 +/- 0.0072 @200 seeds), "
          "reproducing the recorded ~32 sigma separation",
          close(a, -0.114, 4e-3) and 30.0 < sig < 35.0,
          f"alpha_null = {a:+.4f} +/- {s:.4f} over {NULL_SEEDS} seeds;  separation of the "
          f"observed {O['production']['alpha_h']:+.4f} = ({abs(O['production']['alpha_h']):.4f} "
          f"- 0.1151)/0.0072 = {sig:.1f} sigma")
    rn, rns = O["production"]["ratio_null"]
    check("the null's OWN invariant ratio = -0.5031 +/- 0.0059, i.e. the observation "
          "is BRACKETED by sampling and relaxation",
          close(rn, -0.5031, 4e-3),
          f"null ratio {rn:.4f} +/- {rns:.4f}; observed {O['production']['ratio']:.4f} is "
          f"{(O['production']['ratio']-rn)/rns:.1f} null-sigma from pure sampling and "
          f"{(O['production']['model_min_ratio']-O['production']['ratio'])/O['production']['se_ratio']:.1f} "
          f"SE from pure relaxation")

    want_c1 = {"production": 0.999, "2017-12": 1.032, "2020-12": 1.016, "2022-12": 0.936}
    want_th = {"production": 2.77, "2017-12": 2.48, "2020-12": 2.56, "2022-12": 2.52}
    want_r2 = {"production": 0.5561, "2017-12": 0.4626, "2020-12": 0.5268, "2022-12": 0.4856}
    want_nl = {"production": 0.4547, "2017-12": 0.3404, "2020-12": 0.4560, "2022-12": 0.3114}
    print("\n        origin      c_1     theta    R^2_thm5  R^2_null  R^2_pl   "
          "med Dnull/D  samp%  alpha_fit  alpha_meas")
    for name, _ in ORIGINS:
        o = O[name]
        y = np.log2(o["D"])
        M, h = o["M"], o["h"]
        lam_of = lambda th: M @ (h ** (-th))                            # noqa: E731

        def resid(p):
            c1, c2, th = p[0], np.exp(p[1]), p[2]
            pred = c1 * o["Dnull"] + c2 / lam_of(th)
            return y - np.log2(np.maximum(pred, 1e-300))

        best = None
        for th0 in (1.5, 2.5, 3.5):
            for lc2 in (-5.0, 0.0, 5.0):
                r = least_squares(resid, [1.0, lc2, th0],
                                  bounds=([0, -50, 0.05], [10, 50, 12]), max_nfev=8000)
                if best is None or r.cost < best.cost:
                    best = r
        c1, c2, th = best.x[0], np.exp(best.x[1]), best.x[2]
        r2t5 = r2_of(y, y - best.fun)
        _, _, r2null, _ = ols(np.log2(o["Dnull"]), y)
        fit = c1 * o["Dnull"] + c2 / lam_of(th)
        a_fit = -ols(np.log2(h), np.log2(fit))[0][1]
        samp = float((c1 * o["Dnull"]).sum() / fit.sum())
        med = float(np.median(o["Dnull"] / o["D"]))
        o["thm5"] = (c1, th, r2t5, r2null, samp, med, a_fit, best)
        print(f"        {name:<11} {c1:6.3f}  {th:6.2f}   {r2t5:7.4f}  {r2null:8.4f}  "
              f"{o['r2_pl']:6.4f}   {med:9.3f}  {100*samp:5.1f}   {a_fit:8.3f}  {o['alpha_h']:9.3f}")

    ok_c1 = all(close(O[n]["thm5"][0], want_c1[n], 8e-3) for n, _ in ORIGINS)
    ok_th = all(close(O[n]["thm5"][1], want_th[n], 0.03) for n, _ in ORIGINS)
    ok_r2 = all(close(O[n]["thm5"][2], want_r2[n], 3e-3) for n, _ in ORIGINS)
    ok_nl = all(close(O[n]["thm5"][3], want_nl[n], 3e-3) for n, _ in ORIGINS)
    check("Theorem 5 c_1 = 0.999/1.032/1.016/0.936 (theory: exactly 1, freely estimated)",
          ok_c1, "  ".join(f"{O[n]['thm5'][0]:.3f}" for n, _ in ORIGINS))
    check("Theorem 5 theta = 2.77/2.48/2.56/2.52, masked spread 3.2%", ok_th,
          "  ".join(f"{O[n]['thm5'][1]:.2f}" for n, _ in ORIGINS)
          + f"   masked spread = {100*(max(O[n]['thm5'][1] for n in want_th if n!='production') - min(O[n]['thm5'][1] for n in want_th if n!='production'))/np.mean([O[n]['thm5'][1] for n in want_th if n!='production']):.1f}%")
    check("Theorem 5 R^2 = 0.5561/0.4626/0.5268/0.4856", ok_r2,
          "  ".join(f"{O[n]['thm5'][2]:.4f}" for n, _ in ORIGINS))
    check("sampling floor alone R^2 = 0.4547/0.3404/0.4560/0.3114", ok_nl,
          "  ".join(f"{O[n]['thm5'][3]:.4f}" for n, _ in ORIGINS))
    check("median D^null/D = 0.875 and sampling carries 83.9% of sum_v D_v (production)",
          close(O["production"]["thm5"][5], 0.875, 5e-3)
          and close(100 * O["production"]["thm5"][4], 83.9, 0.6),
          f"median ratio {O['production']['thm5'][5]:.4f}, sampling share "
          f"{100*O['production']['thm5'][4]:.1f}%")
    check("the fitted surface regenerates the pre-registered exponent on all four origins",
          all(abs(O[n]["thm5"][6] - O[n]["alpha_h"]) <= 0.02 for n, _ in ORIGINS),
          "  ".join(f"{n}:{O[n]['thm5'][6]:.3f}v{O[n]['alpha_h']:.3f}" for n, _ in ORIGINS))
    check("Theorem 5 beats the pre-registered power law on every origin",
          all(O[n]["thm5"][2] > O[n]["r2_pl"] for n, _ in ORIGINS),
          "  ".join(f"{O[n]['thm5'][2]:.4f}>{O[n]['r2_pl']:.4f}" for n, _ in ORIGINS))

    # kernel-constrained dynamic term vs a free two-exponent dynamic term
    print("\n        kernel-constrained (c1, c2, theta) vs free two-exponent (c1, c2, p, q) "
          "dynamic term:")
    rejected = []
    for name, _ in ORIGINS:
        o = O[name]
        y = np.log2(o["D"]); h, n_ = o["h"], o["n"]
        c1c, thc = o["thm5"][0], o["thm5"][1]
        sse_c = float(2 * o["thm5"][7].cost)

        def residF(p):
            c1, c2, pw, qw = p[0], np.exp(p[1]), p[2], p[3]
            pred = c1 * o["Dnull"] + c2 * h ** pw * n_ ** qw
            return y - np.log2(np.maximum(pred, 1e-300))

        # seed one start from the constrained optimum: fit log(1/lambda(theta_hat)) with
        # a power law in (h, n) so the free model starts strictly no worse.
        bs = ols(np.column_stack([np.log(h), np.log(n_)]),
                 -np.log(o["M"] @ (h ** (-thc))))[0]
        c2c = np.exp(o["thm5"][7].x[1])
        starts = [[c1c, np.log(c2c) + bs[0], bs[1], bs[2]],
                  [1.0, 0.0, -1.0, 0.3], [1.0, -5.0, -2.0, 0.5], [1.0, 5.0, -0.5, 0.1],
                  [c1c, np.log(max(c2c, 1e-12)), 0.0, -1.0]]
        bf = None
        for st in starts:
            try:
                r = least_squares(residF, st, max_nfev=40000)
            except Exception:
                continue
            if bf is None or r.cost < bf.cost:
                bf = r
        sse_f = min(float(2 * bf.cost), sse_c)     # free model cannot be allowed to be worse
        F1 = ((sse_c - sse_f) / 1.0) / (sse_f / (NV - 4))
        F2 = F1 / 2.0                               # the write-up's numerator df = 2
        p1 = float(1 - f_dist.cdf(F1, 1, NV - 4)) if F1 > 0 else 1.0
        p2 = float(1 - f_dist.cdf(F2, 2, NV - 4)) if F2 > 0 else 1.0
        rejected.append((p1 < 0.05, p2 < 0.05))
        print(f"        {name:<11} SSE_c={sse_c:7.3f} SSE_f={sse_f:7.3f} | correct df: "
              f"F(1,{NV-4})={F1:6.2f} p={p1:.4f} | write-up's df: F(2,{NV-4})={F2:5.2f} "
              f"p={p2:.4f} | free (p,q)=({bf.x[2]:+.3f},{bf.x[3]:+.3f})")
    n1 = sum(r[0] for r in rejected)
    n2 = sum(r[1] for r in rejected)
    check("write-up's closing claim 'the kernel-constrained dynamic term is NOT "
          "significantly rejected against a free two-exponent alternative on three of "
          "four origins'",
          n1 <= 1,
          f"REFUTED under the correct df. The write-up's F values (2.7 / 26.6 / 5.3 / 0.0) "
          f"reproduce EXACTLY as the F(2,1019) column above, so its numerator df is 2 -- "
          f"which independently confirms this script's SSEs match the write-up's. "
          f"But the constrained model has 3 free parameters "
          f"(c1, c2, theta) and the alternative 4 (c1, c2, p, q), so the numerator df is 1 "
          f"and every F doubles. Rejected at p<0.05 on {n1}/4 origins with df=1, versus "
          f"{n2}/4 with the write-up's df=2. So 'the ABKO kernel survives on the residual' "
          f"does NOT hold as stated; the honest statement is that the kernel-constrained "
          f"residual term is rejected on 3 of 4 origins, though by a small SSE margin "
          f"(<0.7% of SSE on production).",
          tag="DETAIL")


def main():
    t0 = time.time()
    print(__doc__.split("\n\n")[0])
    print(f"repo = {REPO}")
    O = section0()
    if not all(c[1] for c in CHECKS):
        print("\nPIPELINE GATE FAILED -- refusing to report any derived number.")
        return 1
    section1(O)
    section2(O)
    section3(O)
    section4(O)
    section5(O)

    hdr("SUMMARY")
    bad_core = [c for c in CHECKS if not c[1] and c[3] == "CORE"]
    bad_det = [c for c in CHECKS if not c[1] and c[3] == "DETAIL"]
    print(f"  {sum(1 for c in CHECKS if c[1])} / {len(CHECKS)} checks passed  "
          f"({time.time() - t0:.0f}s)")
    if bad_core:
        print("\n  CORE FAILURES -- a load-bearing claim does not hold:")
        for name, _, det, _ in bad_core:
            print(f"    * {name}\n        {det}")
    if bad_det:
        print("\n  AS-STATED DEFECTS -- the verdict survives but the wording/number is wrong:")
        for name, _, det, _ in bad_det:
            print(f"    * {name}\n        {det}")
    print()
    if not bad_core and not bad_det:
        print("  OVERALL: PASS -- every asserted number reproduces.")
    elif not bad_core:
        print("  OVERALL: PASS WITH CORRECTIONS -- every load-bearing claim reproduces on "
              "the real instrument;\n           the flagged statements above are wrong as "
              "written and must be repaired in the paper.")
    else:
        print("  OVERALL: FAIL -- at least one load-bearing claim does not hold.")
    return 1 if bad_core else 0


if __name__ == "__main__":
    sys.exit(main())
