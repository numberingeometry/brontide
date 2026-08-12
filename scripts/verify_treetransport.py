"""verify_treetransport.py — numerical companion for O3 (slug: treetransport).

Checks the level-decomposed tree-Wasserstein claims against the committed brontide
instrument.  CPU-only, no network, runnable as ``python scripts/verify_treetransport.py``
from the repo root.  Writes nothing, commits nothing, modifies no existing file.

GATE  — reimplement the EB pipeline; kappa == 3.147961 and share_smoothed diff == 0.0.
CHECK 1 — the exact identities (co-area, factorisation, min-flow, corr(TWD,TV) + Thm C).
CHECK 2 — Theorem B: the depth profile has no modulus of continuity (counterexample
          + the realised Ward-vs-single version on this corpus).
CHECK 3 — Lemma 3 partition sandwich on real data (+ a negative control).
CHECK 4 — Theorem A / A' certificate: vacuous on Ward, non-vacuous on single (100 seeds).
CHECK 5 — the look-ahead dose-response across the four origins (the decisive one).

Every bound is checked in the direction it is asserted and the slack is reported.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brontide import config, series  # noqa: E402

K = 1024
N_SUPER = 64
T_START = time.time()
FAILURES: list[str] = []
NOTES: list[str] = []


def hdr(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label + (f" :: {detail}" if detail else ""))
    return ok


def note(s: str) -> None:
    NOTES.append(s)
    print(f"  [note] {s}")


def rel(a: float, b: float) -> float:
    """|a-b|/|b| guarded."""
    return abs(a - b) / max(abs(b), 1e-300)


# ---------------------------------------------------------------------------
# tree machinery
# ---------------------------------------------------------------------------
class Tree:
    """Dendrogram from a scipy linkage matrix Z (n-1, 4)."""

    def __init__(self, Z: np.ndarray):
        Z = np.asarray(Z, dtype=np.float64)
        self.Z = Z
        self.n = Z.shape[0] + 1
        self.N = 2 * self.n - 1
        self.root = self.N - 1
        self.h = np.zeros(self.N)
        self.h[self.n:] = Z[:, 2]
        self.parent = np.full(self.N, -1, dtype=np.int64)
        self.left = Z[:, 0].astype(np.int64)
        self.right = Z[:, 1].astype(np.int64)
        self.parent[self.left] = np.arange(self.n, self.N)
        self.parent[self.right] = np.arange(self.n, self.N)
        # h(parent); root gets its own height so its edge weight is 0
        self.hp = np.where(self.parent >= 0, self.h[np.maximum(self.parent, 0)], self.h)
        self.w = self.hp - self.h            # edge weight, 0 at the root
        self.depth = np.zeros(self.N, dtype=np.int64)
        for i in range(self.n - 2, -1, -1):
            v = self.n + i
            self.depth[self.left[i]] = self.depth[v] + 1
            self.depth[self.right[i]] = self.depth[v] + 1
        self.H = float(self.h[self.root])
        self.maxdepth = int(self.depth.max())

    def node_mass(self, X: np.ndarray) -> np.ndarray:
        """X (T, n) leaf measures -> (T, N) node masses mu(Gamma(v))."""
        X = np.atleast_2d(X)
        M = np.zeros((X.shape[0], self.N))
        M[:, : self.n] = X
        for i in range(self.n - 1):
            M[:, self.n + i] = M[:, self.left[i]] + M[:, self.right[i]]
        return M

    def w1(self, absd: np.ndarray) -> np.ndarray:
        """absd (T, N) = |mu(Gamma(v)) - nu(Gamma(v))| -> W1^T per row."""
        return absd @ self.w

    def alive(self, t: float) -> np.ndarray:
        """boolean (N,): v is a block root of P_t, i.e. h(v) <= t < h(p(v))."""
        return (self.h <= t) & ((self.parent < 0) | (self.hp > t))

    def g(self, absd: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """discrepancy profile g(t) -> (T, len(ts))."""
        A = np.stack([self.alive(float(t)).astype(np.float64) for t in ts], axis=1)
        return absd @ A

    def Gint(self, absd: np.ndarray, taus: np.ndarray) -> np.ndarray:
        """int_tau^inf g(t) dt, exact closed form -> (T, len(taus)).

        int_tau^inf g = sum_v (h(p(v)) - max(h(v), tau))^+ * |dmu(Gamma(v))|,
        and for tau < 0 the convention g(t)=g(0) for t<0 adds |tau| * g(0).
        """
        taus = np.asarray(taus, dtype=np.float64)
        tp = np.maximum(taus, 0.0)
        WT = np.maximum(0.0, self.hp[None, :] - np.maximum(self.h[None, :], tp[:, None]))
        out = absd @ WT.T
        neg = np.maximum(0.0, -taus)
        if np.any(neg > 0):
            g0 = absd[:, : self.n].sum(axis=1)      # g(0) = sum over leaves |dmu|
            out = out + g0[:, None] * neg[None, :]
        return out

    def coarse_share(self, absd: np.ndarray, lo: int | None = None,
                     hi: int | None = None) -> np.ndarray:
        """fraction of W1^T carried by edges whose CHILD depth is in [lo, hi]."""
        sel = np.ones(self.N, dtype=bool)
        if lo is not None:
            sel &= self.depth >= lo
        if hi is not None:
            sel &= self.depth <= hi
        num = absd[:, sel] @ self.w[sel]
        den = absd @ self.w
        return num / den

    def cophenetic_square(self) -> np.ndarray:
        """(n, n) cophenetic ultrametric u_T."""
        from scipy.cluster.hierarchy import cophenet
        from scipy.spatial.distance import squareform
        return squareform(cophenet(self.Z))


def tree_plan(tree: Tree, d: np.ndarray) -> list[tuple[int, int, float, float]]:
    """Tree-greedy optimal plan for the signed leaf discrepancy d = mu - nu.

    Returns matches (x, y, mass, h(LCA)).  x carries surplus, y carries deficit.
    """
    n = tree.n
    pos: list = [None] * tree.N
    neg: list = [None] * tree.N
    for x in range(n):
        dx = float(d[x])
        if dx > 0:
            pos[x], neg[x] = [(x, dx)], []
        elif dx < 0:
            pos[x], neg[x] = [], [(x, -dx)]
        else:
            pos[x], neg[x] = [], []
    matches: list[tuple[int, int, float, float]] = []
    L, R, Zh = tree.left, tree.right, tree.Z[:, 2]
    for i in range(n - 1):
        a, b, v = int(L[i]), int(R[i]), n + i
        hv = float(Zh[i])
        P = pos[a] + pos[b]
        Ng = neg[a] + neg[b]
        pos[a] = neg[a] = pos[b] = neg[b] = None
        while P and Ng:
            x, mp = P[-1]
            y, mn = Ng[-1]
            m = mp if mp < mn else mn
            matches.append((x, y, m, hv))
            if mp - m <= 0.0:
                P.pop()
            else:
                P[-1] = (x, mp - m)
            if mn - m <= 0.0:
                Ng.pop()
            else:
                Ng[-1] = (y, mn - m)
        pos[v], neg[v] = P, Ng
    return matches


def ols_nw(y: np.ndarray, x: np.ndarray, lags: int = 12):
    """OLS y ~ 1 + x with Newey-West (Bartlett) HAC standard errors."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = y.size
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        if l >= n:
            break
        wgt = 1.0 - l / (lags + 1.0)
        G = u[l:].T @ u[:-l]
        S += wgt * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return float(beta[1]), float(se[1]), float(beta[1] / se[1])


def spread_pm(vals) -> float:
    """(max-min)/(2*mean) -- the '+/- x%' convention."""
    v = np.asarray(vals, float)
    return float((v.max() - v.min()) / (2.0 * abs(v.mean())))


def spread_maxdev(vals) -> float:
    """max|v-mean|/|mean|."""
    v = np.asarray(vals, float)
    return float(np.abs(v - v.mean()).max() / abs(v.mean()))


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def artifacts_dir(origin: str) -> Path:
    return ROOT / "artifacts" if origin == "production" else ROOT / "artifacts" / "cutoffs" / origin


def load_origin(origin: str):
    """-> (Tree, months list, shares (K, n_months))."""
    import pyarrow.parquet as pq
    adir = artifacts_dir(origin)
    Z = np.load(adir / "linkage.f64.npy")
    tab = pq.read_table(adir / "region_shares.parquet")
    months_col = tab.column("month").to_pylist()
    sh = np.asarray(tab.column("share_smoothed").to_numpy(), dtype=np.float64)
    n_months = len(set(months_col))
    months = months_col[:n_months]
    assert len(sh) == K * n_months, (len(sh), K, n_months)
    X = sh.reshape(K, n_months)
    return Tree(Z), months, X


def drift_measures(X: np.ndarray):
    """columns renormalised to sum 1 -> (mu (T,K) at t-1, nu (T,K) at t) for consecutive t."""
    col = X.sum(axis=0)
    P = (X / col[None, :]).T          # (n_months, K)
    return P[:-1], P[1:], col


# ---------------------------------------------------------------------------
# GATE
# ---------------------------------------------------------------------------
def gate() -> bool:
    hdr("GATE — reimplement the EB pipeline from raw labels + papers.parquet")
    import pyarrow.parquet as pq

    labels = np.load(ROOT / "data" / "labels" / "region_labels.i16.npy").astype(np.int32)
    sup = np.load(ROOT / "artifacts" / "super_region.i16.npy").astype(np.int32)
    n_s = np.bincount(sup, minlength=N_SUPER)
    months_all = pq.read_table(ROOT / "data" / "papers.parquet", columns=["month"]).column("month").to_pylist()

    def ordinal(m: str) -> int:
        y, mo = m.split("-")
        return int(y) * 12 + int(mo) - 1

    o0, o1 = ordinal(config.CORPUS_START), ordinal(config.CORPUS_END)
    month_labels = [f"{o // 12:04d}-{o % 12 + 1:02d}" for o in range(o0, o1 + 1)]
    midx = np.fromiter((ordinal(m) - o0 if o0 <= ordinal(m) <= o1 else -1 for m in months_all),
                       dtype=np.int32, count=len(months_all))
    keep = midx >= 0
    n_months = len(month_labels)
    C = np.zeros((K, n_months))
    np.add.at(C, (labels[keep], midx[keep]), 1.0)
    M_t = C.sum(axis=0)
    a = config.ADD_ALPHA
    p = (C + a) / (M_t + a * K)[None, :]
    P_S = np.zeros((N_SUPER, n_months))
    np.add.at(P_S, sup, p)
    m = (P_S / n_s[:, None])[sup]
    n_anchor = n_months
    trail = C[:, n_anchor - 12:n_anchor].sum(axis=1)
    kappa = series._estimate_kappa(trail, sup)
    w = (trail / (trail + kappa))[:, None]
    p_hat = w * p + (1.0 - w) * m

    tab = pq.read_table(ROOT / "artifacts" / "region_shares.parquet")
    committed = np.asarray(tab.column("share_smoothed").to_numpy(), float).reshape(K, n_months)
    ccount = np.asarray(tab.column("count").to_numpy(), float).reshape(K, n_months)

    dk = abs(kappa - 3.147961)
    dsh = float(np.abs(committed - p_hat).max())
    dct = float(np.abs(ccount - C).max())
    print(f"  n_months={n_months}  months={month_labels[0]}..{month_labels[-1]}  papers={int(keep.sum()):,}")
    print(f"  kappa (reimplemented) = {kappa!r}")
    print(f"  max |count diff|          = {dct:.17g}")
    print(f"  max |share_smoothed diff| = {dsh:.17g}")
    ok = check("kappa == 3.147961 (tol 1e-6)", dk < 1e-6, f"|diff| = {dk:.3e}")
    ok &= check("max |share_smoothed reimpl - committed| == 0.0 exactly", dsh == 0.0, f"= {dsh:.3e}")
    ok &= check("counts match exactly", dct == 0.0)
    # column-sum range quoted in the claim
    csum = committed.sum(axis=0)
    print(f"  share_smoothed column sums: min={csum.min():.6f} max={csum.max():.6f} "
          f"(claim: 0.9978-1.0038)")
    check("column sums in [0.9978, 1.0038] (renormalisation needed, <0.4%)",
          csum.min() >= 0.9977 and csum.max() <= 1.0039,
          f"[{csum.min():.6f}, {csum.max():.6f}]")
    # save trailing counts for the S3 perturbation
    globals()["_TRAIL"] = trail
    return ok


def _lemma2_lp_check(m: int = 24, seed: int = 7) -> None:
    """Independent optimality check of Lemma 2 on a small random dendrogram.

    Solves min_pi <pi, d_T> subject to the marginal constraints with a general LP
    (scipy.optimize.linprog), and compares with (a) the edge-sum formula for W1^T
    and (b) the tree-greedy plan's cost.  d_T = 2 u_T on the leaves.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.optimize import linprog
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((m, 4))
    tr = Tree(linkage(pts, method="average"))
    a = rng.random(m) + 0.05
    b = rng.random(m) + 0.05
    a /= a.sum()
    b /= b.sum()
    U = _coph_from_tree(tr)
    cost = (2.0 * U).reshape(-1)
    A_eq = np.zeros((2 * m, m * m))
    for i in range(m):
        A_eq[i, i * m:(i + 1) * m] = 1.0
        A_eq[m + i, i::m] = 1.0
    r = linprog(cost, A_eq=A_eq, b_eq=np.concatenate([a, b]), bounds=(0, None), method="highs")
    absd = np.abs(tr.node_mass(a[None, :]) - tr.node_mass(b[None, :]))
    w1_edge = float(tr.w1(absd)[0])
    greedy = tree_plan(tr, a - b)
    w1_greedy = 2.0 * sum(mm * hv for _, _, mm, hv in greedy)
    print(f"  (Lemma 2, independent LP on a random {m}-leaf tree)")
    print(f"       LP optimum      = {r.fun:.15f}")
    print(f"       edge-sum W1^T   = {w1_edge:.15f}   |diff| = {abs(r.fun - w1_edge):.3e}")
    print(f"       tree-greedy plan= {w1_greedy:.15f}   |diff| = {abs(r.fun - w1_greedy):.3e}")
    check("(L2) tree-greedy plan is LP-optimal and equals the edge-sum W1^T (tol 1e-9)",
          r.status == 0 and abs(r.fun - w1_edge) < 1e-9 and abs(r.fun - w1_greedy) < 1e-9,
          f"LP {r.fun:.12f} vs edge {w1_edge:.12f} vs greedy {w1_greedy:.12f}")


# ---------------------------------------------------------------------------
# CHECK 1
# ---------------------------------------------------------------------------
def check1(tree: Tree, mu: np.ndarray, nu: np.ndarray, months: list[str]):
    hdr("CHECK 1 — the exact identities (production, 233 monthly drift pairs)")
    Mmu = tree.node_mass(mu)
    Mnu = tree.node_mass(nu)
    absd = np.abs(Mmu - Mnu)
    W1 = tree.w1(absd)
    print(f"  T = {mu.shape[0]} drift pairs; H = {tree.H:.5f}; max depth = {tree.maxdepth}")

    # ---- (1a) co-area: integrate g over the breakpoint grid, independently ----
    bp = np.unique(np.concatenate([[0.0], tree.h[tree.n:]]))
    gvals = tree.g(absd, bp[:-1])                      # g constant on [bp_k, bp_{k+1})
    integral = gvals @ np.diff(bp)
    d1a = float(np.abs(W1 - integral).max())
    print(f"  (1a) breakpoints = {bp.size};  max |sum_e w_e|dmu| - int_0^inf g dt| = {d1a:.3e}")
    check("(1a) co-area identity W1^T = int_0^inf g(t) dt  (tol 1e-12)", d1a <= 1e-12,
          f"measured {d1a:.3e} (claim 1.42e-14)")

    # ---- (1b) factorisation ----
    g0 = absd[:, :tree.n].sum(axis=1)
    TV = 0.5 * np.abs(mu - nu).sum(axis=1)
    d_g0 = float(np.abs(g0 - 2 * TV).max())
    hbar = integral / g0
    d1b = float(np.abs(W1 - 2 * TV * hbar).max())
    print(f"  (1b) max |g(0) - 2 TV| = {d_g0:.3e};  max |W1^T - 2*TV*hbar| = {d1b:.3e}")
    print(f"       hbar: median {np.median(hbar):.4f}  range [{hbar.min():.4f}, {hbar.max():.4f}]"
          f"  hbar/H median = {np.median(hbar) / tree.H:.4f}")
    check("(1b) factorisation W1^T = 2*TV*hbar  (tol 1e-12)", d1b <= 1e-12,
          f"measured {d1b:.3e} (claim 1.78e-15)")
    check("(1b) hbar median ~ 22.539", rel(float(np.median(hbar)), 22.539) < 2e-3,
          f"median {np.median(hbar):.4f}  range [{hbar.min():.3f},{hbar.max():.3f}] (claim 22.539, 11.77-56.38)")
    check("(1b) hbar/H ~ 0.194", rel(float(np.median(hbar) / tree.H), 0.194) < 5e-3,
          f"{np.median(hbar) / tree.H:.5f}")

    # ---- (1c) min-flow / tree-greedy plan ----
    U = tree.cophenetic_square()
    max_mass_err = 0.0
    max_u_err = 0.0
    max_w1_err = 0.0
    n_pairs_tot = 0
    for t in range(mu.shape[0]):
        d = mu[t] - nu[t]
        M = tree_plan(tree, d)
        n_pairs_tot += len(M)
        mass = sum(mm for _, _, mm, _ in M)
        max_mass_err = max(max_mass_err, abs(mass - TV[t]))
        ue = max(abs(U[x, y] - hv) for x, y, _, hv in M)
        max_u_err = max(max_u_err, ue)
        cost = sum(mm * hv for _, _, mm, hv in M)
        max_w1_err = max(max_w1_err, abs(2.0 * cost - W1[t]))
    print(f"  (1c) plan built for all months, {n_pairs_tot:,} matched pairs total")
    print(f"       max |plan mass - TV|                 = {max_mass_err:.3e}")
    print(f"       max |u_T(x,y) - h(LCA at match)|     = {max_u_err:.3e}")
    print(f"       max |2 * sum(mass*u) - W1^T|         = {max_w1_err:.3e}")
    check("(1c) plan mass == TV (tol 1e-12)", max_mass_err <= 1e-12,
          f"= {max_mass_err:.3e} (claim says 0.0 exactly; the residue is |sum mu - sum nu|/2 "
          f"after renormalisation, i.e. pure float rounding)")
    check("(1c) u on every matched pair == LCA merge height exactly", max_u_err == 0.0,
          f"= {max_u_err:.3e}")
    check("(1c) W1^T = 2 * sum over plan of mass*u  (tol 1e-12)", max_w1_err <= 1e-12,
          f"= {max_w1_err:.3e}")

    # g(t) == 2 * sum_A (mu(A)-nu(A))^+ at t = 30
    t0 = 30.0
    al = tree.alive(t0)
    pos_sum = np.maximum(Mmu[:, al] - Mnu[:, al], 0.0).sum(axis=1)
    g30 = absd[:, al].sum(axis=1)
    d1c = float(np.abs(g30 - 2 * pos_sum).max())
    print(f"       at t=30: |P_t| = {int(al.sum())} blocks;  max |g(t) - 2 sum (mu-nu)^+| = {d1c:.3e}")
    check("(1c) g(t) = 2 sum_A (mu(A)-nu(A))^+ at t=30  (tol 1e-12)", d1c <= 1e-12,
          f"= {d1c:.3e}, {int(al.sum())} blocks (claim 3.33e-16, 16 blocks)")
    check("(1c) block count at t=30 == 16", int(al.sum()) == 16, f"= {int(al.sum())}")

    # min over couplings: plan mass beyond t equals g(t)/2, for several t
    worst = 0.0
    for t in (5.0, 30.0, 60.0):
        for tm in (0, 100, 232):
            M = tree_plan(tree, mu[tm] - nu[tm])
            beyond = sum(mm for _, _, mm, hv in M if hv > t)
            gt = float(absd[tm, tree.alive(t)].sum())
            worst = max(worst, abs(2 * beyond - gt))
    check("(1c) min-flow reading: 2*pi(u>t) == g(t) for the tree-greedy plan (tol 1e-12)",
          worst <= 1e-12, f"max dev {worst:.3e}")

    # INDEPENDENT optimality check of Lemma 2: brute-force LP on a small random tree.
    _lemma2_lp_check()

    # ---- (1d) corr(TWD, TV) and Theorem C ----
    c_raw = float(np.corrcoef(W1, TV)[0, 1])
    lw, lt, lh = np.log(W1), np.log(TV), np.log(hbar)
    c_log = float(np.corrcoef(lw, lt)[0, 1])
    s1 = float(lt.std(ddof=0))          # population sd -- reproduces the claim's 0.1006/0.3016
    s2 = float(lh.std(ddof=0))
    rho = float(np.corrcoef(lt, lh)[0, 1])
    pred = (s1 + rho * s2) / np.sqrt(s1**2 + s2**2 + 2 * rho * s1 * s2)
    cap = s1 / np.sqrt(s1**2 + s2**2)
    decor = -s1 / s2
    d_ident = float(np.abs(lw - (np.log(2 * TV) + lh)).max())
    print(f"  (1d) corr(TWD, TV) raw = {c_raw:.6f}   (claim -0.121425)")
    print(f"       corr(log TWD, log TV)  = {c_log:.6f}  (claim -0.09857)")
    print(f"       sd(log TV)={s1:.4f}  sd(log hbar)={s2:.4f}  ratio={s2/s1:.3f}  rho={rho:.4f}")
    print(f"       Theorem C formula      = {pred:.6f}  (claim -0.0985)")
    print(f"       independence cap = {cap:.4f}   exact-decorrelation rho = {decor:.4f}")
    print(f"       max |log W1 - log(2TV) - log hbar| = {d_ident:.3e}")
    check("(1d) log identity log W1 = log(2TV) + log hbar (tol 1e-14)", d_ident <= 1e-14,
          f"= {d_ident:.3e}")
    check("(1d) corr(TWD,TV) reproduces -0.121425", abs(c_raw + 0.121425) < 5e-5,
          f"{c_raw:.6f}")
    check("(1d) Theorem C formula == measured corr(log,log) (tol 1e-10)",
          abs(pred - c_log) < 1e-10, f"formula {pred:.8f} vs measured {c_log:.8f}")
    check("(1d) sd(log TV) ~ 0.1006, sd(log hbar) ~ 0.3016, rho ~ -0.4232",
          abs(s1 - 0.1006) < 5e-4 and abs(s2 - 0.3016) < 5e-4 and abs(rho + 0.4232) < 5e-4,
          f"({s1:.4f}, {s2:.4f}, {rho:.4f})")
    return dict(absd=absd, W1=W1, TV=TV, hbar=hbar, integral=integral, g0=g0, U=U,
                Mmu=Mmu, Mnu=Mnu)


# ---------------------------------------------------------------------------
# CHECK 2 — Theorem B
# ---------------------------------------------------------------------------
def _bal_linkage(m: int, eps: float) -> np.ndarray:
    k = int(round(np.log2(m)))
    assert 2 ** k == m
    nodes = list(range(m))
    nid = m
    rows = []
    for j in range(k - 1, -1, -1):
        hgt = 1.0 + eps * (k - j) / k
        new = []
        for i in range(0, len(nodes), 2):
            rows.append([nodes[i], nodes[i + 1], hgt, 0])
            new.append(nid)
            nid += 1
        nodes = new
    Z = np.array(rows, dtype=np.float64)
    return _fix_counts(Z, m)


def _cat_linkage(m: int, eps: float) -> np.ndarray:
    cur = 0
    nid = m
    rows = []
    for i in range(1, m):
        rows.append([cur, i, 1.0 + eps * i / (m - 1), 0])
        cur = nid
        nid += 1
    return _fix_counts(np.array(rows, dtype=np.float64), m)


def _fix_counts(Z: np.ndarray, m: int) -> np.ndarray:
    cnt = np.ones(2 * m - 1)
    for i in range(m - 1):
        cnt[m + i] = cnt[int(Z[i, 0])] + cnt[int(Z[i, 1])]
        Z[i, 3] = cnt[m + i]
    return Z


def _coph_from_tree(tree: Tree) -> np.ndarray:
    """u_T as a dense (n,n) matrix, built directly from the tree (no scipy)."""
    n = tree.n
    U = np.zeros((n, n))
    leaves: list = [None] * tree.N
    for x in range(n):
        leaves[x] = np.array([x])
    for i in range(n - 1):
        a, b, v = int(tree.left[i]), int(tree.right[i]), n + i
        La, Lb = leaves[a], leaves[b]
        U[np.ix_(La, Lb)] = tree.Z[i, 2]
        U[np.ix_(Lb, La)] = tree.Z[i, 2]
        leaves[v] = np.concatenate([La, Lb])
        leaves[a] = leaves[b] = None
    return U


def check2(prod_tree: Tree, mu: np.ndarray, nu: np.ndarray):
    hdr("CHECK 2 — Theorem B: the depth profile has no modulus of continuity")
    for m, eps in ((64, 1e-3), (256, 1e-6)):
        Tb = Tree(_bal_linkage(m, eps))
        Tc = Tree(_cat_linkage(m, eps))
        Ub, Uc = _coph_from_tree(Tb), _coph_from_tree(Tc)
        sup = float(np.abs(Ub - Uc).max())
        a, b = m - 1, m - 2      # the two leaves attached to the caterpillar spine at depths 1, 2
        assert Tc.depth[a] == 1 and Tc.depth[b] == 2, (Tc.depth[a], Tc.depth[b])
        x = np.zeros((1, m)); x[0, a] = 1.0
        y = np.zeros((1, m)); y[0, b] = 1.0
        res = {}
        for name, tr in (("bal", Tb), ("cat", Tc)):
            ad = np.abs(tr.node_mass(x) - tr.node_mass(y))
            w1 = float(tr.w1(ad)[0])
            D = tr.maxdepth
            cs = float(tr.coarse_share(ad, hi=D // 3)[0])
            res[name] = (w1, D, cs, ad)
        w1b, Db, csb, adb = res["bal"]
        w1c, Dc, csc, adc = res["cat"]
        print(f"  m={m}, eps={eps:g}")
        print(f"    ||u_bal - u_cat||_inf = {sup:.6e}   (<= eps = {eps:g}?  {sup <= eps})")
        print(f"    W1_bal = {w1b:.9f}   W1_cat = {w1c:.9f}   |diff| = {abs(w1b-w1c):.6e} "
              f"(<= 2 eps = {2*eps:g}?  {abs(w1b-w1c) <= 2*eps})")
        print(f"    max depth: bal {Db}  cat {Dc}")
        print(f"    coarse-third depth share (d <= D/3): bal {csb:.4f}   cat {csc:.4f}")
        check(f"(B) m={m}: ||u-u'||_inf <= eps", sup <= eps + 1e-18, f"{sup:.4e}")
        check(f"(B) m={m}: |W1_bal - W1_cat| <= 2 eps", abs(w1b - w1c) <= 2 * eps + 1e-18,
              f"{abs(w1b-w1c):.4e}")
        check(f"(B) m={m}: coarse-third share 0.0000 (bal) vs 1.0000 (cat)",
              abs(csb) < 1e-12 and abs(csc - 1.0) < 1e-12, f"{csb:.6f} / {csc:.6f}")
        # profile total-variation distance between the two depth profiles
        L = max(Db, Dc)
        pb = np.zeros(L + 1); pc = np.zeros(L + 1)
        for l in range(L + 1):
            pb[l] = float(Tb.coarse_share(adb, lo=l, hi=l)[0]) if l <= Db else 0.0
            pc[l] = float(Tc.coarse_share(adc, lo=l, hi=l)[0]) if l <= Dc else 0.0
        tvprof = 0.5 * np.abs(pb - pc).sum()
        check(f"(B) m={m}: ||pi_bal - pi_cat||_TV == 1", abs(tvprof - 1.0) < 1e-12,
              f"{tvprof:.10f}")
        # sanity: the two counterexample trees really are legal dendrograms
        ok_mono = bool((Tb.w[np.arange(Tb.N) != Tb.root] > 0).all() and
                       (Tc.w[np.arange(Tc.N) != Tc.root] > 0).all())
        check(f"(B) m={m}: both are legal dendrograms (heights strictly increase root-ward)", ok_mono)

    # ---- the realised version on this corpus: Ward vs single on the SAME centroids ----
    from scipy.cluster.hierarchy import linkage
    C = np.load(ROOT / "artifacts" / "centroids.f32.npy").astype(np.float64)
    Zs = linkage(C, method="single")
    Ts = Tree(Zs)
    adw = np.abs(prod_tree.node_mass(mu) - prod_tree.node_mass(nu))
    ads = np.abs(Ts.node_mass(mu) - Ts.node_mass(nu))
    # the claim's 0.409 / 0.0015 pair is the 2007-2010 ERA average, not the full-window mean
    n_pairs = mu.shape[0]
    eraA = np.zeros(n_pairs, dtype=bool)
    eraA[: 4 * 12 - 1] = True                      # 2007-02 .. 2010-12
    cw_a = float(prod_tree.coarse_share(adw, hi=4)[eraA].mean())
    cs_a = float(Ts.coarse_share(ads, hi=4)[eraA].mean())
    cw_f = float(prod_tree.coarse_share(adw, hi=4).mean())
    cs_f = float(Ts.coarse_share(ads, hi=4).mean())
    print(f"  realised on this corpus (same 1024 centroids, only the linkage rule changes):")
    print(f"    Ward   max depth {prod_tree.maxdepth}, H = {prod_tree.H:.4f}, "
          f"coarse (d<=4) share: 2007-2010 {cw_a:.4f}, full window {cw_f:.4f}")
    print(f"    single max depth {Ts.maxdepth}, H = {Ts.H:.4f}, "
          f"coarse (d<=4) share: 2007-2010 {cs_a:.6f}, full window {cs_f:.6f}")
    print(f"    ratio on the matched 2007-2010 era = {cw_a/max(cs_a,1e-12):.0f}x   (claim ~270x)")
    check("(B) realised: Ward max depth 15, single max depth >= 300 (claim 366; base tree is 371)",
          prod_tree.maxdepth == 15 and Ts.maxdepth >= 300,
          f"{prod_tree.maxdepth} / {Ts.maxdepth}")
    check("(B) realised: coarse(d<=4) share 0.409 (Ward) vs 0.0015 (single), 2007-2010 era",
          abs(cw_a - 0.409) < 3e-3 and abs(cs_a - 0.0015) < 3e-4,
          f"{cw_a:.4f} / {cs_a:.6f}  ratio {cw_a/cs_a:.0f}x")
    check("(B) realised: changing ONLY the linkage rule moves the depth profile by >100x",
          cw_a / max(cs_a, 1e-12) > 100.0, f"{cw_a/max(cs_a,1e-12):.0f}x")
    return Ts


# ---------------------------------------------------------------------------
# perturbation machinery (pre-registration S3)
# ---------------------------------------------------------------------------
def within_region_sigma() -> np.ndarray:
    """per-coordinate within-region sd of the PCA-50 coordinates (K, 50)."""
    meta = json.loads((config.PCA_MEMMAP.with_suffix(config.PCA_MEMMAP.suffix + ".meta.json")).read_text())
    n = int(meta["n"])
    pca = np.memmap(config.PCA_MEMMAP, dtype=np.float32, mode="r", shape=(n, config.PCA_DIM))
    lab = np.load(ROOT / "data" / "labels" / "region_labels.i16.npy").astype(np.int32)
    s1 = np.zeros((K, config.PCA_DIM))
    s2 = np.zeros((K, config.PCA_DIM))
    cnt = np.zeros(K)
    CH = 250_000
    for lo in range(0, n, CH):
        hi = min(lo + CH, n)
        Xc = np.asarray(pca[lo:hi], dtype=np.float64)
        Lc = lab[lo:hi]
        np.add.at(s1, Lc, Xc)
        np.add.at(s2, Lc, Xc * Xc)
        np.add.at(cnt, Lc, 1.0)
    mean = s1 / cnt[:, None]
    var = np.maximum(s2 / cnt[:, None] - mean**2, 0.0)
    return np.sqrt(var)


def perturbed_centroids(C0: np.ndarray, sigma: np.ndarray, trail: np.ndarray,
                        seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return C0 + scale * rng.standard_normal(C0.shape) * (sigma / np.sqrt(trail)[:, None])


# ---------------------------------------------------------------------------
# CHECK 3 — Lemma 3 sandwich
# ---------------------------------------------------------------------------
def sandwich_violations(base: Tree, pert: Tree, absd_b: np.ndarray, absd_p: np.ndarray,
                        taus: np.ndarray, eps: float, tol: float = 1e-12):
    """count violations of g(t+eps) <= g'(t) <= g(t-eps)."""
    gp = pert.g(absd_p, taus)
    hi = base.g(absd_b, np.maximum(taus - eps, -1.0))       # t-eps < 0 -> g(0)
    # for t-eps<0 the convention is g(t)=g(0); alive() at t<0 gives nothing, so patch:
    negmask = (taus - eps) < 0
    if negmask.any():
        g0 = absd_b[:, :base.n].sum(axis=1)
        hi[:, negmask] = g0[:, None]
    lo = base.g(absd_b, taus + eps)
    v_lo = int((gp < lo - tol).sum())
    v_hi = int((gp > hi + tol).sum())
    slack_lo = float((gp - lo).min())
    slack_hi = float((hi - gp).min())
    return v_lo, v_hi, slack_lo, slack_hi, gp.size


def check3(prod: Tree, mu: np.ndarray, nu: np.ndarray, sigma: np.ndarray, trail: np.ndarray):
    hdr("CHECK 3 — Lemma 3 partition sandwich on real data")
    from scipy.cluster.hierarchy import linkage
    from scipy.cluster.hierarchy import cophenet
    C0 = np.load(ROOT / "artifacts" / "centroids.f32.npy").astype(np.float64)
    u0 = cophenet(prod.Z)
    absd_b = np.abs(prod.node_mass(mu) - prod.node_mass(nu))
    taus = np.linspace(0.0, prod.H, 60)

    for scale, tag in ((1.0, "pre-registered scale"), (1e-3, "1e-3x scale (non-trivial eps)")):
        Cp = perturbed_centroids(C0, sigma, trail, 0, scale)
        Zp = linkage(Cp, method="ward")
        Tp = Tree(Zp)
        up = cophenet(Zp)
        eps = float(np.abs(up - u0).max())
        absd_p = np.abs(Tp.node_mass(mu) - Tp.node_mass(nu))
        v_lo, v_hi, s_lo, s_hi, ntot = sandwich_violations(prod, Tp, absd_b, absd_p, taus, eps)
        print(f"  seed 0, {tag}: eps = {eps:.5f}  (eps/max u = {eps/prod.H:.4f})")
        print(f"    checks = {2*ntot:,};  violations low-side {v_lo}, high-side {v_hi}")
        print(f"    min slack: g'(t) - g(t+eps) = {s_lo:.6e};  g(t-eps) - g'(t) = {s_hi:.6e}")
        check(f"(L3) zero violations of g(t+eps) <= g'(t) <= g(t-eps) [{tag}]",
              v_lo == 0 and v_hi == 0, f"{v_lo}+{v_hi} of {2*ntot:,} (claim 0/27,960 at eps=112.16)")
        if scale == 1.0:
            check("(L3) eps at seed 0 == 112.16", abs(eps - 112.16094) < 1e-3, f"{eps:.5f}")
            check("(L3) total checks == 27,960", 2 * ntot == 27960, f"{2*ntot}")
            # NEGATIVE CONTROL: the same test at eps = 0 must fail, else the test is vacuous
            v0_lo, v0_hi, _, _, _ = sandwich_violations(prod, Tp, absd_b, absd_p, taus, 0.0)
            print(f"    negative control at eps = 0: violations {v0_lo} + {v0_hi} (must be > 0)")
            check("(L3) negative control at eps=0 DOES detect violations (test is not vacuous)",
                  (v0_lo + v0_hi) > 0, f"{v0_lo}+{v0_hi}")
        else:
            # is the sandwich non-trivial here?  count grid points where the band collapses
            # to the trivial upper bound g(0)
            g0 = absd_b[:, :prod.n].sum(axis=1)
            band_top = prod.g(absd_b, np.maximum(taus - eps, 0.0))
            frac_trivial = float(((band_top >= g0[:, None] - 1e-15)).mean())
            print(f"    fraction of grid points where the upper band is the trivial g(0): "
                  f"{frac_trivial:.3f}")
            check("(L3) the sandwich is checked in a NON-TRIVIAL regime too "
                  "(band collapses to g(0) at < 10% of grid points)", frac_trivial < 0.10,
                  f"{frac_trivial:.3f} at eps = {eps:.3e}")

    # --- how small must the perturbation be before Ward's certificate stops being vacuous? ---
    hb = float(np.median(prod.Gint(absd_b, np.array([0.0]))[:, 0]
                         / absd_b[:, :prod.n].sum(axis=1)))
    print("  dose-response of Ward's cophenetic distortion in the perturbation scale "
          "(9 seeds each):")
    print(f"    {'scale':>10s} {'eps median':>12s} {'eps min':>10s} {'eps max':>10s} {'eta median':>11s}")
    trans = None
    for sc in (1.0, 1e-1, 1e-2, 4e-3, 3e-3, 1e-3):
        es = [float(np.abs(cophenet(linkage(perturbed_centroids(C0, sigma, trail, s, sc),
                                            method="ward")) - u0).max()) for s in range(9)]
        em = float(np.median(es))
        print(f"    {sc:10.4g} {em:12.5f} {min(es):10.5f} {max(es):10.5f} {em/hb:11.4f}")
        if trans is None and em / hb < 1.0:
            trans = sc
    print(f"    -> Ward's eta crosses 1 only below scale ~{trans:g} of the actual sampling "
          f"noise; the distortion is a near-step function of the perturbation size "
          f"(a top-level merge either flips or it does not).")
    check("(L3) Ward's certificate is vacuous even at 1/100 of the sampling-noise scale",
          trans is not None and trans < 1e-2,
          f"eta crosses 1 only below scale {trans:g}")
    return


# ---------------------------------------------------------------------------
# CHECK 4 — the certificate
# ---------------------------------------------------------------------------
def check4(prod: Tree, single: Tree, mu: np.ndarray, nu: np.ndarray,
           sigma: np.ndarray, trail: np.ndarray, c1: dict):
    hdr("CHECK 4 — Theorem A / A' certificate: vacuous on Ward, non-vacuous on single")
    from scipy.cluster.hierarchy import linkage, cophenet, fcluster
    from sklearn.metrics import adjusted_rand_score

    C0 = np.load(ROOT / "artifacts" / "centroids.f32.npy").astype(np.float64)
    u_w = cophenet(prod.Z)
    u_s = cophenet(single.Z)
    maxu_w, maxu_s = float(u_w.max()), float(u_s.max())
    f_w = fcluster(prod.Z, 64, criterion="maxclust")
    f_s = fcluster(single.Z, 64, criterion="maxclust")

    absd_w = c1["absd"]
    absd_s = np.abs(single.node_mass(mu) - single.node_mass(nu))
    hbar_w = c1["hbar"]
    W1_w = c1["W1"]
    g0 = c1["g0"]
    Gs_int = single.Gint(absd_s, np.array([0.0]))[:, 0]
    hbar_s = Gs_int / absd_s[:, :single.n].sum(axis=1)
    W1_s = single.w1(absd_s)

    taus_w = np.linspace(0.0, prod.H, 60)
    taus_s = np.linspace(0.0, single.H, 60)
    G_w = prod.Gint(absd_w, taus_w) / W1_w[:, None]
    G_s = single.Gint(absd_s, taus_s) / W1_s[:, None]
    Gm_w = G_w.mean(axis=0)
    Gm_s = G_s.mean(axis=0)

    n_seeds = 100
    rec = {"ward": [], "single": []}
    t0 = time.time()
    for seed in range(n_seeds):
        Cp = perturbed_centroids(C0, sigma, trail, seed)
        for name, meth, base, u0, maxu, f0, taus, Gm, absd0, W10 in (
                ("ward", "ward", prod, u_w, maxu_w, f_w, taus_w, Gm_w, absd_w, W1_w),
                ("single", "single", single, u_s, maxu_s, f_s, taus_s, Gm_s, absd_s, W1_s)):
            Zp = linkage(Cp, method=meth)
            Tp = Tree(Zp)
            up = cophenet(Zp)
            eps = float(np.abs(up - u0).max())
            ari = float(adjusted_rand_score(f0, fcluster(Zp, 64, criterion="maxclust")))
            adp = np.abs(Tp.node_mass(mu) - Tp.node_mass(nu))
            W1p = Tp.w1(adp)
            Gp = Tp.Gint(adp, taus) / W1p[:, None]
            dsup = float(np.abs(Gp.mean(axis=0) - Gm).max())
            # era-averaged version
            rec[name].append(dict(seed=seed, eps=eps, epsrel=eps / maxu, ari=ari,
                                  maxdepth=Tp.maxdepth, dsup=dsup,
                                  w1rel=float(np.abs(W1p / W10 - 1.0).max()),
                                  Hp=Tp.H))
    print(f"  100 seeds x 2 linkages in {time.time()-t0:.1f}s")

    out = {}
    for name, hb, H, maxu in (("ward", hbar_w, prod.H, maxu_w), ("single", hbar_s, single.H, maxu_s)):
        R = rec[name]
        eps_med = float(np.median([r["eps"] for r in R]))
        epsrel_med = float(np.median([r["epsrel"] for r in R]))
        ari_med = float(np.median([r["ari"] for r in R]))
        hb_med = float(np.median(hb))
        eta = eps_med / hb_med
        dsups = np.array([r["dsup"] for r in R])
        mds = [r["maxdepth"] for r in R]
        out[name] = dict(eps=eps_med, epsrel=epsrel_med, ari=ari_med, hbar=hb_med, eta=eta,
                         dsup_med=float(np.median(dsups)), dsup_max=float(dsups.max()),
                         mind=min(mds), maxd=max(mds))
        print(f"  {name:7s}: eps median {eps_med:.4f} (epsrel {epsrel_med:.4f}), "
              f"hbar median {hb_med:.4f}, eta = {eta:.4f}, ARI median {ari_med:.4f}")
        print(f"           max depth across seeds {min(mds)}-{max(mds)};  "
              f"observed ||G'-G||_inf median {np.median(dsups):.4f} max {dsups.max():.4f}")

    check("(A) Ward epsrel median reproduces pre-registered 0.9737",
          abs(out["ward"]["epsrel"] - 0.9737) < 2e-3, f"{out['ward']['epsrel']:.4f}")
    check("(A) single epsrel median reproduces pre-registered 0.0842",
          abs(out["single"]["epsrel"] - 0.0842) < 5e-3, f"{out['single']['epsrel']:.4f}")
    check("(A) Ward 64-cut ARI median reproduces 0.6032",
          abs(out["ward"]["ari"] - 0.6032) < 1e-2, f"{out['ward']['ari']:.4f}")
    check("(A) single 64-cut ARI median reproduces 0.8870",
          abs(out["single"]["ari"] - 0.8870) < 3e-2, f"{out['single']['ari']:.4f}")
    check("(A) Ward certificate is VACUOUS: eta > 1", out["ward"]["eta"] > 1.0,
          f"eta = {out['ward']['eta']:.4f} (claim 5.01)")
    check("(A) single certificate is NON-VACUOUS: eta < 1", out["single"]["eta"] < 1.0,
          f"eta = {out['single']['eta']:.4f} (claim 0.1313)")
    check("(A) observed ||G'-G||_inf medians ~ 0.0083 (Ward) / 0.0063 (single)",
          abs(out["ward"]["dsup_med"] - 0.0083) < 3e-3 and abs(out["single"]["dsup_med"] - 0.0063) < 3e-3,
          f"{out['ward']['dsup_med']:.4f} / {out['single']['dsup_med']:.4f}")

    # ---- Theorem A on single linkage, where its hypothesis holds ----
    Cp = perturbed_centroids(C0, sigma, trail, 0)
    Zs_p = linkage(Cp, method="single")
    Tsp = Tree(Zs_p)
    eps_s = float(np.abs(cophenet(Zs_p) - u_s).max())
    ad_sp = np.abs(Tsp.node_mass(mu) - Tsp.node_mass(nu))
    W1sp = Tsp.w1(ad_sp)
    eta_t = eps_s / hbar_s                       # per-month eta (the theorem's own quantity)
    print(f"  Theorem A on single linkage, seed 0: eps = {eps_s:.5f}, "
          f"per-month eta in [{eta_t.min():.4f}, {eta_t.max():.4f}]")
    # (i)
    viol_i = int((np.abs(W1sp / W1_s - 1.0) > eta_t + 1e-12).sum())
    slack_i = float((eta_t - np.abs(W1sp / W1_s - 1.0)).min())
    check("(A-i) |W1'/W1 - 1| <= eta, per month", viol_i == 0,
          f"{viol_i} violations of {eta_t.size}; min slack {slack_i:.4e}")
    viol_i2 = int((np.abs(W1sp - W1_s) > 2 * eps_s * g0_single(absd_s, single) + 1e-12).sum())
    check("(A-i) |W1' - W1| <= 2 eps g(0), per month", viol_i2 == 0, f"{viol_i2} violations")
    # (ii)
    tau_fine = np.unique(np.concatenate([np.linspace(0.0, single.H, 200),
                                         [2 * single.H / 3]]))
    Gb = single.Gint(absd_s, tau_fine) / W1_s[:, None]
    Gb_m = single.Gint(absd_s, tau_fine - eps_s) / W1_s[:, None]
    Gb_p = single.Gint(absd_s, tau_fine + eps_s) / W1_s[:, None]
    Gp = Tsp.Gint(ad_sp, tau_fine) / W1sp[:, None]
    lo = Gb_p / (1.0 + eta_t)[:, None]
    hi = Gb_m / (1.0 - eta_t)[:, None]
    v2 = int((Gp < lo - 1e-12).sum()) + int((Gp > hi + 1e-12).sum())
    check("(A-ii) G(tau+eps)/(1+eta) <= G'(tau) <= G(tau-eps)/(1-eta), all tau x months",
          v2 == 0, f"{v2} violations of {2*Gp.size:,}; "
                   f"min slack {min(float((Gp-lo).min()), float((hi-Gp).min())):.4e}")
    # (iii) global and at tau = 2H/3.
    # The published 0.1081 / 0.2756 pair is the bound on the WINDOW-AVERAGED profile at the
    # MEDIAN eps: averaging Thm A(iii) over months gives |mean G' - mean G| <= mean_t omega_t
    # + eta over (1-eta).  Both that form and the strict per-month sup form are reported.
    eps_med = out["single"]["eps"]
    eta_med = eps_med / float(np.median(hbar_s))
    Gm_m = single.Gint(absd_s, tau_fine - eps_med) / W1_s[:, None]
    Gm_p = single.Gint(absd_s, tau_fine + eps_med) / W1_s[:, None]
    OM = Gm_m - Gm_p                                  # per-month, per-tau modulus
    i23 = int(np.argmin(np.abs(tau_fine - 2 * single.H / 3)))
    om_avg_23 = float(OM.mean(axis=0)[i23])
    om_avg_gl = float(OM.mean(axis=0).max())
    om_sup_23 = float(OM[:, i23].max())
    om_sup_gl = float(OM.max())
    bnd = lambda o: (o + eta_med) / (1.0 - eta_med)    # noqa: E731
    obs_sup = float(np.abs(Gp.mean(axis=0) - Gb.mean(axis=0)).max())
    obs_23 = float(abs(Gp.mean(axis=0)[i23] - Gb.mean(axis=0)[i23]))
    print(f"  single, eps = median {eps_med:.4f}, eta = {eta_med:.4f}:")
    print(f"     window-averaged omega: global {om_avg_gl:.4f} -> bound {bnd(om_avg_gl):.4f} "
          f"(claim 0.2612 -> 0.4519)")
    print(f"     window-averaged omega at tau=2H/3: {om_avg_23:.4f} -> bound {bnd(om_avg_23):.4f} "
          f"(claim 0.1081 -> 0.2756)")
    print(f"     strict per-month sup omega: global {om_sup_gl:.4f} -> {bnd(om_sup_gl):.4f}; "
          f"at 2H/3 {om_sup_23:.4f} -> {bnd(om_sup_23):.4f}")
    print(f"     observed |G'-G| (mean profile): sup {obs_sup:.4f}, at 2H/3 {obs_23:.4f}")
    check("(A-iii) observed sup |G'-G| respects the certificate on single linkage",
          obs_sup <= bnd(om_avg_gl), f"{obs_sup:.4f} <= {bnd(om_avg_gl):.4f}")
    check("(A-iii) single global bound reproduces 0.2612 -> 0.4519",
          abs(om_avg_gl - 0.2612) < 3e-3 and abs(bnd(om_avg_gl) - 0.4519) < 5e-3,
          f"{om_avg_gl:.4f} -> {bnd(om_avg_gl):.4f}")
    check("(A-iii) single bound at tau=2H/3 reproduces 0.1081 -> 0.2756",
          abs(om_avg_23 - 0.1081) < 3e-3 and abs(bnd(om_avg_23) - 0.2756) < 5e-3,
          f"{om_avg_23:.4f} -> {bnd(om_avg_23):.4f}")
    out["single"]["bound23"] = bnd(om_avg_23)
    out["single"]["obs23"] = obs_23

    # LIKE-FOR-LIKE: the claim compares a WARD era gap (0.085) against a SINGLE-linkage
    # certificate (0.276).  Do it properly: the single tree's OWN era gap vs its OWN bound.
    n_pairs = mu.shape[0]
    eA = np.zeros(n_pairs, bool); eA[: 4 * 12 - 1] = True          # 2007-02 .. 2010-12
    eB = np.zeros(n_pairs, bool); eB[-(3 * 12 + 6):] = True        # 2023-01 .. 2026-06
    G_s23 = (single.Gint(absd_s, np.array([2 * single.H / 3])) / W1_s[:, None])[:, 0]
    G_w23 = (prod.Gint(absd_w, np.array([2 * prod.H / 3])) / W1_w[:, None])[:, 0]
    gap_s = float(G_s23[eB].mean() - G_s23[eA].mean())
    gap_w = float(G_w23[eB].mean() - G_w23[eA].mean())
    print(f"  like-for-like era gap in G(2H/3), 2007-2010 -> 2023-2026:")
    print(f"     Ward   {G_w23[eA].mean():.4f} -> {G_w23[eB].mean():.4f}  gap {gap_w:.4f}   "
          f"(its own certificate: VACUOUS, eta = {out['ward']['eta']:.2f})")
    print(f"     single {G_s23[eA].mean():.4f} -> {G_s23[eB].mean():.4f}  gap {gap_s:.4f}   "
          f"(its own certificate: {bnd(om_avg_23):.4f}, ratio {bnd(om_avg_23)/max(abs(gap_s),1e-9):.1f}x)")
    check("(A-iii) like-for-like: the single tree's own era gap is inside its own certificate",
          abs(gap_s) <= bnd(om_avg_23), f"|{gap_s:.4f}| <= {bnd(om_avg_23):.4f}")
    note("the claim's '3.2x' compares a WARD era gap (0.0854) with a SINGLE-linkage "
         f"certificate (0.2756) -- different trees, not like-for-like. Like-for-like, the "
         f"single tree's own gap is {gap_s:.4f} vs its own bound {bnd(om_avg_23):.4f} "
         f"({bnd(om_avg_23)/max(abs(gap_s),1e-9):.1f}x), and the Ward gap has NO certificate at all.")

    # ---- Theorem A' : the transport-weighted robustification, on Ward ----
    Zw_p = linkage(Cp, method="ward")
    Twp = Tree(Zw_p)
    Uw = c1["U"]
    Uwp = Twp.cophenetic_square()
    dU = np.abs(Uw - Uwp)
    # pooled transport-weighted distribution of |u - u'| under BOTH optimal plans
    dev_a, wt_a, dev_b, wt_b = [], [], [], []
    for t in range(mu.shape[0]):
        d = mu[t] - nu[t]
        for M, store_d, store_w in ((tree_plan(prod, d), dev_a, wt_a),
                                    (tree_plan(Twp, d), dev_b, wt_b)):
            xs = np.fromiter((x for x, _, _, _ in M), dtype=np.int64, count=len(M))
            ys = np.fromiter((y for _, y, _, _ in M), dtype=np.int64, count=len(M))
            ms = np.fromiter((m for _, _, m, _ in M), dtype=np.float64, count=len(M))
            store_d.append(dU[xs, ys])
            store_w.append(ms)
    dev_a = np.concatenate(dev_a); wt_a = np.concatenate(wt_a)
    dev_b = np.concatenate(dev_b); wt_b = np.concatenate(wt_b)

    def wquant(dev, wt, q):
        o = np.argsort(dev)
        cw = np.cumsum(wt[o]) / wt.sum()
        return float(dev[o][np.searchsorted(cw, q)])

    qs = [wquant(dev_a, wt_a, q) / maxu_w for q in (0.5, 0.75, 0.90, 0.99)]
    print(f"  Ward transport-weighted |u-u'|/max u quantiles q50/q75/q90/q99 = "
          f"{qs[0]:.4f} / {qs[1]:.4f} / {qs[2]:.4f} / {qs[3]:.4f}   "
          f"(claim 0.0152 / 0.0758 / 0.3937 / 0.9002)")
    check("(A') Ward transport-weighted quantiles reproduce the claim (rel tol 20%)",
          all(rel(a, b) < 0.20 for a, b in zip(qs, (0.0152, 0.0758, 0.3937, 0.9002))),
          f"{qs}")

    Hmax = max(prod.H, Twp.H)
    hbar_med = float(np.median(hbar_w))
    grid = np.unique(np.concatenate([np.linspace(0, maxu_w, 4001), np.sort(dev_a)[::97]]))
    best = (np.inf, np.nan, np.nan)
    for e in grid:
        ba = float(wt_a[dev_a > e].sum() / wt_a.sum())
        bb = float(wt_b[dev_b > e].sum() / wt_b.sum())
        beta = max(ba, bb)
        er = (2.0 * e + beta * Hmax) / hbar_med
        if er < best[0]:
            best = (er, e, beta)
    eta_rob, e_star, beta_star = best
    print(f"  Theorem A' optimised on Ward: eta_rob = {eta_rob:.4f} at eps/max u = "
          f"{e_star/maxu_w:.4f}, betatilde = {beta_star:.4f}   (claim 1.843 / 0.051 / 0.288)")
    check("(A') Ward robust certificate still VACUOUS: eta_rob > 1", eta_rob > 1.0,
          f"eta_rob = {eta_rob:.4f}")
    e_claim = 0.051 * maxu_w
    b_claim = max(float(wt_a[dev_a > e_claim].sum() / wt_a.sum()),
                  float(wt_b[dev_b > e_claim].sum() / wt_b.sum()))
    er_claim = (2 * e_claim + b_claim * Hmax) / hbar_med
    print(f"     at the claim's own point (eps/max u = 0.051): betatilde = {b_claim:.4f} "
          f"(claim 0.288), eta_rob = {er_claim:.4f} (claim 1.843);  Hmax = {Hmax:.4f}")
    if rel(eta_rob, 1.843) > 0.03:
        note(f"eta_rob optimum is {eta_rob:.4f}, not the claimed 1.843; at the claim's own "
             f"stated (eps, betatilde) = (0.051*maxu, 0.288) the formula "
             f"(2*eps + beta*Hmax)/hbar gives {er_claim:.3f}, not 1.843 -- the claim's "
             f"arithmetic needs Hmax ~ 103 whereas Hmax = max(H,H') = {Hmax:.2f}. "
             f"The verdict is UNCHANGED and slightly stronger (more vacuous, not less).")

    # ---- Theorem A' is itself checked numerically (pointwise and in L1) ----
    e_t = 0.05 * maxu_w
    tgrid = np.linspace(0.0, max(prod.H, Twp.H), 400)
    adp = np.abs(Twp.node_mass(mu) - Twp.node_mass(nu))
    g_b = prod.g(absd_w, tgrid)
    g_p = Twp.g(adp, tgrid)
    g_bm = prod.g(absd_w, np.maximum(tgrid - e_t, 0.0))
    g_bp = prod.g(absd_w, tgrid + e_t)
    TVv = 0.5 * absd_w[:, :prod.n].sum(axis=1)
    # per-month betatilde at eps = e_t (max over the two optimal plans)
    off = 0
    beta_t = np.zeros(mu.shape[0])
    per_a = [len(tree_plan(prod, mu[t] - nu[t])) for t in range(mu.shape[0])]
    for t in range(mu.shape[0]):
        sl = slice(off, off + per_a[t])
        wa = wt_a[sl]
        beta_t[t] = float(wa[dev_a[sl] > e_t].sum() / max(wa.sum(), 1e-300))
        off += per_a[t]
    slack = 2.0 * beta_t[:, None] * TVv[:, None]
    v_lo = int((g_p < g_bp - slack - 1e-12).sum())
    v_hi = int((g_p > g_bm + slack + 1e-12).sum())
    l1 = np.trapezoid(np.abs(g_p - g_b), tgrid) / W1_w
    print(f"  Theorem A' checked directly at eps = 0.05*max u = {e_t:.3f}: "
          f"pointwise violations {v_lo}+{v_hi} of {2*g_p.size:,}")
    print(f"     measured ||g'-g||_L1 / W1^T: median {np.median(l1):.4f}, max {l1.max():.4f} "
          f"vs the bound eta_rob = {eta_rob:.4f}")
    check("(A') pointwise bound g(t+eps)-2*beta*TV <= g'(t) <= g(t-eps)+2*beta*TV holds",
          v_lo == 0 and v_hi == 0, f"{v_lo}+{v_hi} violations")
    check("(A') the L1 bound is respected (and enormously slack)", l1.max() <= eta_rob,
          f"max {l1.max():.4f} <= {eta_rob:.4f}")

    # ---- the "delete a few bad regions" escape route ----
    thr = 0.25 * maxu_w
    A = dU > thr
    np.fill_diagonal(A, False)
    alive = np.ones(K, dtype=bool)
    deg = A.sum(axis=1).astype(np.int64)
    removed = []
    while True:
        da = np.where(alive, deg, -1)
        v = int(da.argmax())
        if da[v] <= 0:
            break
        alive[v] = False
        removed.append(v)
        deg = deg - A[v].astype(np.int64)
        deg[v] = 0
    mean_mass = mu.mean(axis=0)
    mass_removed = float(mean_mass[~alive].sum() / mean_mass.sum())
    print(f"  forcing eps/max u <= 0.25 (greedy vertex cover): delete {len(removed)}/1024 regions "
          f"carrying {100*mass_removed:.1f}% of the mean mass   (claim 299 / 27.9%)")
    check("(A') 'delete a few bad leaves' escape is closed (>= 100 regions, >= 10% mass)",
          len(removed) >= 100 and mass_removed >= 0.10,
          f"{len(removed)} regions / {100*mass_removed:.1f}% mass")
    return out


def g0_single(absd: np.ndarray, tree: Tree) -> np.ndarray:
    return absd[:, :tree.n].sum(axis=1)


# ---------------------------------------------------------------------------
# CHECK 5 — the look-ahead dose-response
# ---------------------------------------------------------------------------
def check5():
    hdr("CHECK 5 — DECISIVE: the look-ahead dose-response across the four origins")
    origins = ["2017-12", "2020-12", "2022-12", "production"]
    W_LO, W_HI = "2007-02", "2019-12"
    res = {}
    for og in origins:
        tree, months, X = load_origin(og)
        m_lo, m_hi = drift_measures(X)[0], drift_measures(X)[1]
        pair_months = months[1:]                    # month t of the pair (mu_{t-1}, mu_t)
        sel = np.array([(W_LO <= mm <= W_HI) for mm in pair_months])
        absd = np.abs(tree.node_mass(m_lo) - tree.node_mass(m_hi))
        W1 = tree.w1(absd)
        g0 = absd[:, :tree.n].sum(axis=1)
        hbar = tree.Gint(absd, np.array([0.0]))[:, 0] / g0
        cs = [0.1, 0.2, 0.3, 0.4, 0.5, 2.0 / 3.0, 0.8]
        taus = np.array([c * tree.H for c in cs])
        G = tree.Gint(absd, taus) / W1[:, None]
        dep4 = tree.coarse_share(absd, hi=4)
        dep5 = tree.coarse_share(absd, hi=5)
        dep11 = tree.coarse_share(absd, lo=11)
        res[og] = dict(tree=tree, months=pair_months, sel=sel, G=G, hbar=hbar, W1=W1,
                       dep4=dep4, dep5=dep5, dep11=dep11, cs=cs,
                       hbarH=float(np.median(hbar)) / tree.H)
        print(f"  {og:11s}: {len(pair_months)} drift pairs, window n={int(sel.sum())}, "
              f"H={tree.H:.4f}, max depth={tree.maxdepth}, "
              f"hbar/H (window median) = {float(np.median(hbar[sel]))/tree.H:.4f}")

    check("(5) identical 155-month window on all four origins",
          all(int(res[o]['sel'].sum()) == 155 for o in origins),
          str({o: int(res[o]['sel'].sum()) for o in origins}))

    # ---- hbar / H across origins ----
    hbarH_full = {o: float(np.median(res[o]["hbar"])) / res[o]["tree"].H for o in origins}
    vals = [hbarH_full["production"], hbarH_full["2017-12"], hbarH_full["2020-12"], hbarH_full["2022-12"]]
    print(f"  hbar/H (each origin's OWN full window): "
          f"prod {vals[0]:.4f} / 2017-12 {vals[1]:.4f} / 2020-12 {vals[2]:.4f} / 2022-12 {vals[3]:.4f}")
    rngspread = (max(vals) - min(vals)) / float(np.mean(vals))
    print(f"       spread: (max-min)/mean = {100*rngspread:.1f}%, "
          f"max|dev|/mean = {100*spread_maxdev(vals):.1f}%, "
          f"(max-min)/(2*mean) = {100*spread_pm(vals):.1f}%   (claim 5.0%)")
    check("(5) hbar/H ~ 0.194 / 0.190 / 0.204 / 0.202 across origins",
          all(abs(a - b) < 0.006 for a, b in zip(vals, (0.194, 0.190, 0.204, 0.202))),
          f"{[round(v,4) for v in vals]}")
    check("(5) hbar/H is a cross-origin invariant (spread < 10% on every convention)",
          rngspread < 0.10, f"{100*rngspread:.1f}%")
    wvals = [float(np.median(res[o]["hbar"][res[o]["sel"]])) / res[o]["tree"].H for o in origins]
    wmask = wvals[:3]
    print(f"  hbar/H on the IDENTICAL 155-month window: "
          f"2017-12 {wvals[0]:.4f} / 2020-12 {wvals[1]:.4f} / 2022-12 {wvals[2]:.4f} "
          f"/ production {wvals[3]:.4f}")
    print(f"       the three LEAK-FREE origins agree to "
          f"{100*(max(wmask)-min(wmask))/np.mean(wmask):.1f}% while production sits "
          f"{100*(np.mean(wmask)-wvals[3])/np.mean(wmask):.1f}% below all three")
    check("(5) on a matched window the three leak-free hbar/H agree to < 2%",
          (max(wmask) - min(wmask)) / np.mean(wmask) < 0.02,
          f"{100*(max(wmask)-min(wmask))/np.mean(wmask):.2f}%")
    check("(5) production hbar/H is an outlier below all three leak-free origins on the "
          "matched window (another look-ahead signature)",
          wvals[3] < min(wmask), f"{wvals[3]:.4f} < {min(wmask):.4f}")
    note(f"hbar/H values reproduce exactly (0.1944/0.1898/0.2044/0.2022) but the quoted "
         f"'5.0% spread' does not: under the same (max-min)/mean convention the claim uses "
         f"for its LEVEL spreads it is {100*rngspread:.1f}%, and 5.0% is not produced by any "
         f"of the three conventions ({100*rngspread:.1f} / {100*spread_maxdev(vals):.1f} / "
         f"{100*spread_pm(vals):.1f}%). The 'coincidence' with alpha_h's 5.0% is not robust.")

    # ---- the dose-response trends on the identical window ----
    i23 = res["production"]["cs"].index(2.0 / 3.0)
    print("\n  OLS on the IDENTICAL window 2007-02..2019-12, Newey-West L=12:")
    print(f"  {'origin':12s} {'G(2H/3) trend/yr':>18s} {'t':>7s} | {'dep(d<=4) trend/yr':>20s} {'t':>7s}")
    trends_h, trends_d, ts_h, ts_d = {}, {}, {}, {}
    for og in origins:
        r = res[og]
        sel = r["sel"]
        x = np.arange(int(sel.sum())) / 12.0
        b_h, se_h, t_h = ols_nw(r["G"][sel, i23], x, 12)
        b_d, se_d, t_d = ols_nw(r["dep4"][sel], x, 12)
        trends_h[og], ts_h[og] = b_h, t_h
        trends_d[og], ts_d[og] = b_d, t_d
        print(f"  {og:12s} {b_h:+18.5f} {t_h:7.2f} | {b_d:+20.5f} {t_d:7.2f}")

    order = ["2017-12", "2020-12", "2022-12", "production"]
    mono_h = all(trends_h[order[i]] < trends_h[order[i + 1]] for i in range(3))
    mono_d = all(trends_d[order[i]] < trends_d[order[i + 1]] for i in range(3))
    check("(5) height trend is MONOTONE INCREASING in the dendrogram cutoff T0", mono_h,
          f"{[round(trends_h[o],5) for o in order]}  (1/24 = 4.2% under a random ordering)")
    if not mono_d:
        note("the DEPTH trend is NOT monotone in T0 (2017-12 +%.5f > 2020-12 +%.5f). "
             "The claim asserts 'the same monotone-in-T0 pattern' for the depth statistic, "
             "which its OWN reported numbers (+0.00063, +0.00032, +0.00536, +0.00789) "
             "already violate. What IS true: production exceeds every leak-free origin by "
             ">=1.5x and is the only one that rejects zero."
             % (trends_d['2017-12'], trends_d['2020-12']))
    check("(5) depth trend: production strictly exceeds all three leak-free origins",
          all(trends_d["production"] > trends_d[o] for o in order[:3]),
          f"{[round(trends_d[o],5) for o in order]}  (NOT monotone: "
          f"{trends_d['2017-12']:.5f} > {trends_d['2020-12']:.5f})")
    check("(5) height trend: production strictly exceeds all three leak-free origins",
          all(trends_h["production"] > trends_h[o] for o in order[:3]))
    # is the ordering just tracking the trees' height scale rather than T0?
    Hs = [res[o]["tree"].H for o in order]
    mono_H = all(Hs[i] < Hs[i + 1] for i in range(3))
    print(f"  confound probe: root heights H by origin = {[round(h,2) for h in Hs]} "
          f"-> monotone in T0? {mono_H}  (if False, the trend ordering is not merely tracking H)")
    check("(5) the trend ordering is NOT explained by a monotone ordering of H", not mono_H,
          f"H = {[round(h,2) for h in Hs]}")
    leak_free = order[:3]
    n_weak_h = sum(1 for o in leak_free if abs(ts_h[o]) < 2.0)
    n_weak_d = sum(1 for o in leak_free if abs(ts_d[o]) < 2.0)
    check("(5) >= 2 of 3 leak-free origins cannot reject zero height trend (|t| < 2)",
          n_weak_h >= 2, f"{n_weak_h}/3  t = {[round(ts_h[o],2) for o in leak_free]}")
    check("(5) >= 2 of 3 leak-free origins cannot reject zero depth trend (|t| < 2)",
          n_weak_d >= 2, f"{n_weak_d}/3  t = {[round(ts_d[o],2) for o in leak_free]}")
    check("(5) production (leak-prone) origin DOES reject zero (|t| > 2)",
          abs(ts_h["production"]) > 2.0, f"t = {ts_h['production']:.2f}")
    sh = [trends_h[o] for o in leak_free]
    sd_ = [trends_d[o] for o in leak_free]
    print(f"  masked spread: height (max-min)/(2*mean) = {100*spread_pm(sh):.0f}% "
          f"(claim 45%);  depth = {100*spread_pm(sd_):.0f}% (claim 120%)")
    check("(5) masked height-trend spread exceeds the pre-registration's +/-25% standard",
          spread_pm(sh) > 0.25, f"{100*spread_pm(sh):.0f}%")
    check("(5) all four origin trends are POSITIVE (the sign, weak evidence)",
          all(trends_h[o] > 0 for o in origins), f"{[round(trends_h[o],5) for o in origins]}")
    print(f"  magnitude ratio max/min across origins (height): "
          f"{max(trends_h.values())/min(trends_h.values()):.2f}x  (claim 6.5x)")

    # sign flip across tau on the leak-free 2017-12 origin
    r17 = res["2017-12"]
    taus17 = np.array([0.1, 0.9]) * r17["tree"].H
    G17 = r17["tree"].Gint(np.abs(r17["tree"].node_mass(drift_measures(load_origin("2017-12")[2])[0]) -
                                  r17["tree"].node_mass(drift_measures(load_origin("2017-12")[2])[1])),
                           taus17)
    W117 = r17["W1"]
    G17 = G17 / W117[:, None]
    x17 = np.arange(int(r17["sel"].sum())) / 12.0
    b01, _, t01 = ols_nw(G17[r17["sel"], 0], x17, 12)
    b09, _, t09 = ols_nw(G17[r17["sel"], 1], x17, 12)
    print(f"  2017-12 (leak-free): trend at tau=0.1H = {b01:+.5f} (t={t01:.2f}); "
          f"at tau=0.9H = {b09:+.5f} (t={t09:.2f})   (claim +0.00224 / -0.00063)")
    check("(5) the height trend CHANGES SIGN across the cut on the leak-free 2017-12 origin",
          b01 > 0 > b09, f"{b01:+.5f} / {b09:+.5f}")

    # ---- level reproducibility: height vs depth ----
    print("\n  window-mean LEVELS on the identical window (masked-origin spread):")
    print(f"  {'stat':12s} " + "  ".join(f"{o:>10s}" for o in order) +
          f" {'spread+-':>10s} {'spread rng':>11s}")
    lvl_spread = {}
    for j, c in enumerate(res["production"]["cs"]):
        row = {o: float(res[o]["G"][res[o]["sel"], j].mean()) for o in order}
        mk = [row[o] for o in leak_free]
        lvl_spread[c] = (spread_pm(mk), (max(mk) - min(mk)) / np.mean(mk))
        print(f"  G({c:.3f}H)   " + "  ".join(f"{row[o]:10.4f}" for o in order) +
              f" {100*lvl_spread[c][0]:9.1f}% {100*lvl_spread[c][1]:10.1f}%")
    dep_spread = {}
    for key, lab in (("dep4", "dep d<=4"), ("dep5", "dep d<=5"), ("dep11", "dep d>=11")):
        row = {o: float(res[o][key][res[o]["sel"]].mean()) for o in order}
        mk = [row[o] for o in leak_free]
        dep_spread[key] = (spread_pm(mk), (max(mk) - min(mk)) / np.mean(mk))
        print(f"  {lab:12s} " + "  ".join(f"{row[o]:10.4f}" for o in order) +
              f" {100*dep_spread[key][0]:9.1f}% {100*dep_spread[key][1]:10.1f}%")
    h_mean = np.mean([lvl_spread[c][0] for c in res["production"]["cs"][:6]])
    d_mean = np.mean([dep_spread[k][0] for k in ("dep4", "dep11")])
    h_rng = np.mean([lvl_spread[c][1] for c in res["production"]["cs"][:6]])
    d_rng = np.mean([dep_spread[k][1] for k in ("dep4", "dep11")])
    print(f"  mean masked spread: height (c<=2/3) {100*h_mean:.1f}% (+-) / {100*h_rng:.1f}% (range) "
          f"vs depth {100*d_mean:.1f}% / {100*d_rng:.1f}%  -> height {d_rng/h_rng:.1f}x more reproducible")
    check("(5) the HEIGHT level replicates cross-origin better than the DEPTH level",
          d_mean > h_mean, f"depth {100*d_mean:.1f}% vs height {100*h_mean:.1f}%")
    tgt_h = {0.1: 1.1, 0.2: 2.7, 0.3: 4.3, 0.4: 3.5, 0.5: 4.8, 2.0 / 3.0: 6.3, 0.8: 17.8}
    ok_h = all(abs(100 * lvl_spread[c][1] - tgt_h[c]) < 0.25 for c in res["production"]["cs"])
    check("(5) height-level spreads reproduce 1.1/2.7/4.3/3.5/4.8/6.3/17.8% "
          "[convention: (max-min)/mean over the 3 masked origins]", ok_h,
          str({round(c, 3): round(100 * lvl_spread[c][1], 1) for c in res["production"]["cs"]}))
    ok_d = (abs(100 * dep_spread["dep4"][1] - 12.3) < 0.25
            and abs(100 * dep_spread["dep5"][1] - 10.3) < 0.25
            and abs(100 * dep_spread["dep11"][1] - 32.1) < 0.25)
    check("(5) depth-level spreads reproduce 12.3 / 10.3 / 32.1%", ok_d,
          f"{100*dep_spread['dep4'][1]:.1f} / {100*dep_spread['dep5'][1]:.1f} / "
          f"{100*dep_spread['dep11'][1]:.1f}")
    note("the claim uses TWO different spread conventions without saying so: the LEVEL "
         "spreads (1.1..17.8%, 12.3/10.3/32.1%) are (max-min)/mean, while the TREND "
         "spreads (+/-45%, +/-120%) are (max-min)/(2*mean). Both reproduce exactly once "
         "the convention is matched, but they are not comparable to each other as printed.")

    # ---- crossing signature ----
    print("\n  crossing signature (window-mean G(2H/3) in 2009 vs 2017):")
    for yr in ("2009", "2017"):
        row = {}
        for og in order:
            r = res[og]
            s = np.array([mm.startswith(yr) for mm in r["months"]])
            row[og] = float(r["G"][s, i23].mean())
        print(f"    {yr}: " + "  ".join(f"{o} {row[o]:.4f}" for o in order))
        if yr == "2009":
            below = all(row["production"] < row[o] for o in leak_free)
        else:
            above = all(row["production"] > row[o] for o in leak_free)
    check("(5) production tree crosses the masked trees (below in 2009, above in 2017)",
          below and above, "crossing signature present")

    # ---- the recorded production headline ----
    print("\n  production headline on its own full window (233 pairs):")
    rp = res["production"]
    mn = rp["months"]
    eA = np.array([mm[:4] in ("2007", "2008", "2009", "2010") for mm in mn])
    eB = np.array([mm[:4] in ("2023", "2024", "2025", "2026") for mm in mn])
    c4A, c4B = float(rp["dep4"][eA].mean()), float(rp["dep4"][eB].mean())
    c5A, c5B = float(rp["dep5"][eA].mean()), float(rp["dep5"][eB].mean())
    f11A, f11B = float(rp["dep11"][eA].mean()), float(rp["dep11"][eB].mean())
    gA, gB = float(rp["G"][eA, i23].mean()), float(rp["G"][eB, i23].mean())
    xall = np.arange(len(mn)) / 12.0
    b4, _, t4 = ols_nw(rp["dep4"], xall, 12)
    bG, _, tG = ols_nw(rp["G"][:, i23], xall, 12)
    print(f"    coarse d<=4 : {c4A:.4f} -> {c4B:.4f}   (claim 0.4090 -> 0.5715)")
    print(f"    coarse d<=5 : {c5A:.4f} -> {c5B:.4f}   (claim 0.502 -> 0.654)  "
          f"-> boundary swing {100*abs(c5A-c4A)/c4A:.0f}% on the 2007-2010 level")
    print(f"    fine  d>=11 : {f11A:.4f} -> {f11B:.4f}  (claim 0.1349 -> 0.0736)")
    print(f"    G(2H/3)     : {gA:.4f} -> {gB:.4f}   (claim 0.0986 -> 0.1840), "
          f"era gap {gB-gA:.4f}")
    print(f"    OLS trends  : depth d<=4 {b4:+.5f}/yr (t={t4:.2f});  "
          f"G(2H/3) {bG:+.5f}/yr (t={tG:.2f})   (claim +0.01100 / +0.00562)")
    check("(5) production headline coarse d<=4: 0.4090 -> 0.5715",
          abs(c4A - 0.4090) < 2e-3 and abs(c4B - 0.5715) < 2e-3, f"{c4A:.4f} -> {c4B:.4f}")
    check("(5) production headline fine d>=11: 0.1349 -> 0.0736",
          abs(f11A - 0.1349) < 2e-3 and abs(f11B - 0.0736) < 2e-3, f"{f11A:.4f} -> {f11B:.4f}")
    check("(5) production depth trend +0.01100/yr", abs(b4 - 0.01100) < 5e-4, f"{b4:+.5f}")
    check("(5) production G(2H/3): 0.0986 -> 0.1840, trend +0.00562/yr",
          abs(gA - 0.0986) < 2e-3 and abs(gB - 0.1840) < 2e-3 and abs(bG - 0.00562) < 3e-4,
          f"{gA:.4f} -> {gB:.4f}, {bG:+.5f}")
    check("(5) the tercile boundary swings the headline by > 15%",
          abs(c5A - c4A) / c4A > 0.15, f"{100*abs(c5A-c4A)/c4A:.0f}%")
    return dict(trends_h=trends_h, ts_h=ts_h, trends_d=trends_d, ts_d=ts_d,
                hbarH=vals, headline=(c4A, c4B, f11A, f11B, gA, gB, b4, bG),
                lvl_spread=lvl_spread, dep_spread=dep_spread)


# ---------------------------------------------------------------------------
def main() -> int:
    print("verify_treetransport.py — level-decomposed tree-Wasserstein (slug: treetransport)")
    print(f"repo = {ROOT}")

    if not gate():
        print("\nGATE FAILED — every downstream number is untrustworthy. Stopping.")
        return 1

    prod, months, X = load_origin("production")
    mu, nu, colsum = drift_measures(X)
    print(f"\n  production: {len(months)} months, {mu.shape[0]} drift pairs "
          f"({months[1]}..{months[-1]})")

    c1 = check1(prod, mu, nu, months)
    single = check2(prod, mu, nu)

    sigma = within_region_sigma()
    trail = globals()["_TRAIL"]
    check3(prod, mu, nu, sigma, trail)
    check4(prod, single, mu, nu, sigma, trail, c1)
    check5()

    hdr("VERDICT")
    print(f"  elapsed {time.time() - T_START:.1f}s")
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED assertion(s):")
        for f in FAILURES:
            print(f"    - {f}")
        print("\nOVERALL: FAIL")
        return 1
    print("  All assertions passed.")
    print("  The claim is CONFIRMED in the direction it was made:")
    print("    * the two exact identities hold to machine precision;")
    print("    * the depth-indexed profile provably has no modulus of continuity;")
    print("    * Theorem A's certificate is vacuous on Ward and non-vacuous on single;")
    print("    * the 0.409 -> 0.571 coarsening trend is NOT licensed: it is monotone")
    print("      increasing in how far past the scored window the dendrogram could see,")
    print("      and 2 of 3 leak-free origins cannot reject a zero trend.")
    print("\nOVERALL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
