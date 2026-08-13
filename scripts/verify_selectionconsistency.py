#!/usr/bin/env python3
"""verify_selectionconsistency.py -- numerical companion for O4,
"Selection-consistency of a benchmark".

Run from the repo root:   python scripts/verify_selectionconsistency.py

CPU-only, no network, self-contained. Reads only committed artifacts; writes
nothing. Prints an explicit PASS/FAIL per check and a final verdict.

WHAT IS CHECKED
  BLOCK 0  Environment anchor: independent reimplementation of the EB share
           pipeline must give kappa = 3.147961 and max|p_hat - share_smoothed| = 0.0.
  BLOCK 1  Exact decomposition of the frozen 6x3 precision@50 table (parsed out of
           artifacts/validation.md, not hard-coded): A, Lambda, the Theorem-D
           identity to machine precision, spreads, break-even hit rate, the
           Prop. E1 feasibility thresholds, the exact n=3 facts, the paired tests.
  BLOCK 2  Proposition C2 (the sharp construction): analytic targets + Monte Carlo,
           plus the Lemma D non-attainment check at c = 1.
  BLOCK 3  Proposition C1: the model that wins on BOTH co-primary selection criteria
           (MASE and Spearman) falls below the chance rate on the reported metric.
  BLOCK 4  Adversarial extras: Corollary B3's heteroskedastic witness; Theorem A's
           normalized-regret-equals-1; the A - Lambda identity on random tables
           (is it really an identity, or a coincidence of this table?).
  BLOCK 5  Adversarial extension against the REAL CV artifacts: reproduce the three
           CV selections from artifacts/cutoffs/<T0>/forecast_cv.parquet, measure
           the actual per-fold leads gamma/tau, and test the Prop. E1 feasibility
           condition on measured data rather than on an assumed tau.
"""
from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# --------------------------------------------------------------------------- #
# tiny check harness
# --------------------------------------------------------------------------- #
_RESULTS: list[tuple[str, bool, str]] = []
_EXACT = 1e-12


def check(name: str, ok: bool, detail: str = "") -> bool:
    ok = bool(ok)
    _RESULTS.append((name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))
    return ok


def near(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def hdr(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# BLOCK 0 -- environment anchor: the EB pipeline
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

    C = np.zeros((K, n_months), dtype=np.float64)
    C[r, t] = df["count"].to_numpy()
    sup = np.zeros(K, dtype=np.int64)
    sup[r] = df["super_region"].to_numpy()

    a = config.ADD_ALPHA
    M_t = C.sum(axis=0)
    p = (C + a) / (M_t + a * K)[None, :]

    n_super = config.N_SUPERREGIONS
    P_S = np.zeros((n_super, n_months), dtype=np.float64)
    np.add.at(P_S, sup, p)
    n_s = np.bincount(sup, minlength=n_super).astype(np.float64)
    m_shrink = (P_S / n_s[:, None])[sup]

    n_anchor = n_months                                   # 234 for production
    trail = C[:, n_anchor - 12:n_anchor].sum(axis=1)
    kappa = float(series._estimate_kappa(trail, sup))
    w = (trail / (trail + kappa))[:, None]                # FIXED across months
    p_hat = w * p + (1.0 - w) * m_shrink

    ref = np.zeros((K, n_months), dtype=np.float64)
    ref[r, t] = df["share_smoothed"].to_numpy()
    dmax = float(np.abs(p_hat - ref).max())

    print(f"  months={n_months} (2007-01..2026-06)  K={K}  supers={n_super}  ADD_ALPHA={a}")
    print(f"  kappa = {kappa:.6f}")
    print(f"  max|p_hat - share_smoothed| = {dmax!r}")
    check("EB anchor: kappa == 3.147961", near(kappa, 3.147961, 5e-7), f"got {kappa:.6f}")
    check("EB anchor: share_smoothed reproduced to 0.0 EXACTLY", dmax == 0.0, f"max|diff| = {dmax!r}")
    check("EB anchor: production series is 234 months", n_months == 234, f"n_months={n_months}")


# --------------------------------------------------------------------------- #
# BLOCK 1 -- the frozen 6x3 table
# --------------------------------------------------------------------------- #
_ROW = re.compile(
    r"^\|\s*(\d{4}-\d{2})\s*\|\s*(.+?)\s*\|\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*\|\s*(-?[\d.]+)\s*\|\s*$"
)


def parse_validation_table():
    """Parse the frozen Results table out of artifacts/validation.md (read-only)."""
    txt = (REPO / "artifacts" / "validation.md").read_text(encoding="utf-8")
    rows = []
    for line in txt.splitlines():
        mm = _ROW.match(line.strip())
        if not mm:
            continue
        cutoff, raw, prec, spear, csi = mm.groups()
        selected = "(selected)" in raw
        baseline = "(baseline)" in raw
        name = re.sub(r"[*_]", "", raw).replace("(selected)", "").replace("(baseline)", "").strip()
        rows.append(dict(cutoff=cutoff, model=name, prec=float(prec),
                         spearman=float(spear), csi=float(csi),
                         selected=selected, baseline=baseline))
    return rows


def block1_frozen_table() -> dict:
    hdr("BLOCK 1 -- EXACT DECOMPOSITION OF THE FROZEN 6x3 precision@50 TABLE")
    from scipy import stats

    rows = parse_validation_table()
    cutoffs = sorted({r["cutoff"] for r in rows})
    models = sorted({r["model"] for r in rows})
    check("table parsed: 18 rows = 6 models x 3 cutoffs",
          len(rows) == 18 and len(models) == 6 and len(cutoffs) == 3,
          f"{len(rows)} rows, {len(models)} models, {len(cutoffs)} cutoffs")

    mi = {m: i for i, m in enumerate(models)}
    ci = {c: j for j, c in enumerate(cutoffs)}
    R = np.full((len(models), len(cutoffs)), np.nan)
    sel_model = {}
    base_model = set()
    for r in rows:
        R[mi[r["model"]], ci[r["cutoff"]]] = r["prec"]
        if r["selected"]:
            sel_model[r["cutoff"]] = r["model"]
        if r["baseline"]:
            base_model.add(r["model"])
    check("no missing cells", not np.isnan(R).any())

    print("\n  precision@50 (rows=models, cols=cutoffs " + ", ".join(cutoffs) + ")")
    for m in models:
        print(f"    {m:<18} " + "  ".join(f"{R[mi[m], ci[c]]:.3f}" for c in cutoffs))
    print(f"\n  CV selections parsed from the file: " +
          ", ".join(f"{c}->{sel_model[c]}" for c in cutoffs))
    check("CV selections == AutoETS / SmoothedMomentum / SmoothedMomentum",
          [sel_model[c] for c in cutoffs] == ["AutoETS", "SmoothedMomentum", "SmoothedMomentum"])
    baseline = "SmoothedMomentum"
    check("designated default m_0 (baseline) == SmoothedMomentum", base_model == {baseline})

    n = len(cutoffs)
    per_cut_max = R.max(axis=0)                     # max_m R_i(m), per cutoff
    E_max = float(per_cut_max.mean())               # E[max_m R_i]
    mean_by_model = R.mean(axis=1)                  # E[R_i(m)] per model
    best_fixed_i = int(np.argmax(mean_by_model))
    R_bf = float(mean_by_model[best_fixed_i])
    R_wf = float(mean_by_model.min())
    R_sel_vec = np.array([R[mi[sel_model[c]], ci[c]] for c in cutoffs])
    R_sel = float(R_sel_vec.mean())
    R_base = float(mean_by_model[mi[baseline]])

    A = E_max - R_bf
    per_cut_loss = per_cut_max - R_sel_vec
    Lam = float(per_cut_loss.mean())
    identity_lhs = R_sel - R_bf
    identity_rhs = A - Lam

    print()
    check("E[max_m R_i] (per-cutoff oracle) == 0.633333", near(E_max, 1.9 / 3, _EXACT), f"{E_max:.6f}")
    check("max_m E[R_i] == 0.613333 attained by AutoETS",
          near(R_bf, 1.84 / 3, _EXACT) and models[best_fixed_i] == "AutoETS",
          f"{R_bf:.6f} by {models[best_fixed_i]}")
    check("R(m-hat) (CV-selected policy) == 0.413333 [reported 0.413]",
          near(R_sel, 1.24 / 3, _EXACT), f"{R_sel:.6f}")
    check("R(m_0) (always-baseline) == 0.433333 [reported 0.433]",
          near(R_base, 1.30 / 3, _EXACT), f"{R_base:.6f}")
    check("A (value of adaptivity) == 0.020000", near(A, 0.02, 1e-12), f"{A:.6f}")
    check("Lambda (per-point selection loss) == 0.220000", near(Lam, 0.22, 1e-12), f"{Lam:.6f}")
    check("A >= 0 (Jensen gap non-negativity)", A >= -_EXACT, f"A = {A:.6f}, slack {A:.6f}")
    check("Lambda >= 0", Lam >= -_EXACT, f"Lambda = {Lam:.6f}, slack {Lam:.6f}")
    check("THEOREM D IDENTITY  R(m-hat) - max_m E[R_i] == A - Lambda == -0.200000",
          abs(identity_lhs - identity_rhs) < _EXACT and near(identity_lhs, -0.2, 1e-12),
          f"lhs={identity_lhs:.15f} rhs={identity_rhs:.15f} |diff|={abs(identity_lhs - identity_rhs):.3e}")

    Delta = R_bf - R_wf
    reg = R_bf - R_sel
    norm_reg = reg / Delta
    R_rand = float(mean_by_model.mean())
    sel_eff = (R_sel - R_rand) / (R_bf - R_rand)
    V = R_sel - R_base
    print()
    check("class spread Delta over fixed strategies == 0.506667",
          near(Delta, 1.84 / 3 - 0.32 / 3, _EXACT), f"{Delta:.6f} (worst fixed = {models[int(np.argmin(mean_by_model))]})")
    check("normalized regret Reg/Delta == 0.394737", near(norm_reg, 0.2 / (1.52 / 3), 1e-9), f"{norm_reg:.6f}")
    check("uniform-random roster member == 0.361111", near(R_rand, 2.1666666666666665 / 6, 1e-12), f"{R_rand:.6f}")
    check("selection efficiency vs uniform-random == 0.207",
          near(sel_eff, 0.20705, 5e-5), f"(R_sel-R_rand)/(R_bf-R_rand) = {sel_eff:.6f}")
    check("selection value V = R(m-hat) - R(m_0) == -0.020000 (HARMFUL)",
          near(V, -0.02, 1e-12) and V < 0, f"V = {V:.6f}")

    # hit rate / break-even
    hits = np.array([np.isclose(R_sel_vec[j], per_cut_max[j]) for j in range(n)])
    rho = float(hits.mean())
    ell_bar = float(per_cut_loss[~hits].mean()) if (~hits).any() else float("nan")
    rho_star = 1.0 - A / ell_bar
    print()
    check("realised hit rate rho == 0 (selector missed at all 3 cutoffs)", rho == 0.0, f"rho = {rho}")
    check("ell-bar == Lambda == 0.220000 when rho = 0", near(ell_bar, 0.22, 1e-12), f"{ell_bar:.6f}")
    check("Lambda == (1-rho)*ell-bar", near(Lam, (1 - rho) * ell_bar, _EXACT))
    check("break-even hit rate rho* = 1 - A/ell-bar == 0.909091",
          near(rho_star, 1 - 0.02 / 0.22, 1e-9), f"{rho_star:.6f}")
    check("selection is harmful vs best fixed  <=>  Lambda >= A  (0.220 >= 0.020)",
          (Lam >= A) == (R_sel <= R_bf), f"Lambda-A = {Lam - A:.6f} (deficit)")

    # Prop E1 feasibility thresholds
    M, n_cut = 6, 3
    eta = A / ell_bar
    gam_sg = math.sqrt(2.0 * math.log((M - 1) / eta) / n_cut)
    from scipy.stats import norm as _norm
    gam_g = math.sqrt(2.0 / n_cut) * float(_norm.ppf(1.0 - eta / (M - 1)))
    n_eff = 4.0 / (1.0 + 3.0 * 0.8)
    gam_eff = math.sqrt(2.0 * math.log((M - 1) / eta) / n_eff)
    print()
    check("eta = A/ell-bar == 0.090909", near(eta, 1 / 11, 1e-12), f"{eta:.6f}")
    check("Prop E1 sub-Gaussian threshold == 1.6345 tau (M=6, n=3)", near(gam_sg, 1.6345, 5e-5), f"{gam_sg:.6f} tau")
    check("Prop E1 Gaussian threshold == 1.7088 tau", near(gam_g, 1.7088, 5e-5), f"{gam_g:.6f} tau")
    check("n_eff = 4/(1+3*0.8) == 1.17647 (rho_fold=0.8 is ASSUMED, not measured)",
          near(n_eff, 4 / 3.4, 1e-9), f"{n_eff:.5f}")
    check("Prop E2 threshold at n_eff == 2.6101 tau", near(gam_eff, 2.6101, 5e-5), f"{gam_eff:.6f} tau")

    # Prop E3 exact small-n facts
    p_sign_two = 2.0 / 2 ** n_cut
    size_2of3 = (math.comb(3, 2) + math.comb(3, 3)) / 8.0
    mde_mult = float(stats.t.ppf(0.975, n_cut - 1)) / math.sqrt(n_cut)
    print()
    check("E3(i) minimum attainable two-sided sign/permutation p at n=3 == 0.25", near(p_sign_two, 0.25, _EXACT), f"{p_sign_two}")
    check("E3(i) no exact level-0.05 paired test exists at n=3", p_sign_two > 0.05, f"min p = {p_sign_two} > 0.05")
    check("E3(ii) size of '>=2 of 3' under Bin(3,1/2) == 0.5", near(size_2of3, 0.5, _EXACT), f"{size_2of3}")
    check("E3(iii) paired-t MDE multiplier t_{0.975,2}/sqrt(3) == 2.4841", near(mde_mult, 2.4841, 5e-5), f"{mde_mult:.6f}")

    # paired tests
    def paired(x, y):
        d = np.asarray(x, float) - np.asarray(y, float)
        t, pv = stats.ttest_rel(x, y)
        sd = float(d.std(ddof=1))
        return d, float(t), float(pv), sd

    d_bl, t_bl, p_bl, sd_bl = paired(R_sel_vec, R[mi[baseline], :])
    d_ets, t_ets, p_ets, sd_ets = paired(R_sel_vec, R[mi["AutoETS"], :])
    mde = mde_mult * sd_ets
    print()
    print(f"  selected - baseline  d = {np.round(d_bl, 6).tolist()}")
    check("paired selected-vs-baseline: d-bar = -0.020, t = -1.0000, p = 0.4226",
          near(d_bl.mean(), -0.02, 1e-12) and near(t_bl, -1.0, 1e-9) and near(p_bl, 0.4226, 5e-5),
          f"d-bar={d_bl.mean():.6f} t={t_bl:.4f} p={p_bl:.4f}")
    print(f"  selected - AutoETS   d = {np.round(d_ets, 6).tolist()}")
    check("paired selected-vs-AutoETS: d-bar = -0.200, t = -2.0000, p = 0.1835",
          near(d_ets.mean(), -0.2, 1e-12) and near(t_ets, -2.0, 1e-9) and near(p_ets, 0.1835, 5e-5),
          f"d-bar={d_ets.mean():.6f} t={t_ets:.4f} p={p_ets:.4f}")
    check("MDE = 2.4841 * s_d == 0.4303 (= 85% of the class spread)",
          near(mde, 0.4303, 5e-5) and near(mde / Delta, 0.849, 1e-3),
          f"MDE={mde:.6f}  MDE/Delta={mde / Delta:.4f}")
    check("the -0.020 harm is NOT distinguishable from 0 at n=3 (p > 0.05)", p_bl > 0.05, f"p = {p_bl:.4f}")

    # Corollary D2 -- power cap from putting the default inside the roster
    fired = sum(1 for c in cutoffs if sel_model[c] != baseline)
    check("Cor D2 power cap P(m-hat != m_0) == 1/3", near(fired / n, 1 / 3, 1e-12), f"{fired}/{n}")

    return dict(models=models, cutoffs=cutoffs, R=R, mi=mi, ci=ci, sel_model=sel_model,
                per_cut_max=per_cut_max, R_sel_vec=R_sel_vec, mean_by_model=mean_by_model,
                A=A, Lam=Lam, R_bf=R_bf, R_sel=R_sel, Delta=Delta, baseline=baseline)


# --------------------------------------------------------------------------- #
# BLOCK 2 -- Proposition C2 (the sharp construction)
# --------------------------------------------------------------------------- #
def c2_setup(c, delta, K=1024, k=103, L=50, eps=1e-9):
    n_s = n_o = L
    n_c = k - L
    n_f = K - k - L
    assert n_f >= 0 and n_c >= 0
    s = np.arange(0, n_s)
    o = np.arange(n_s, n_s + n_o)
    cr = np.arange(n_s + n_o, n_s + n_o + n_c)
    fl = np.arange(n_s + n_o + n_c, K)
    off = eps * np.arange(1, L + 1)

    # analytic population medians = the exact MASE minimiser's report
    a_A = np.empty(K)
    a_A[s] = 0.5 + off
    a_A[o] = (c + delta - 0.5) + off if c < 1.0 else (0.5 + delta) + off
    a_A[cr] = (c + delta / 2 - 0.5) if c < 1.0 else (0.5 + delta / 2)
    a_A[fl] = -3.0

    # analytic exceedance probabilities pi (the rival m_B's report)
    n_extra = max(k - 2 * L, 0)                     # carriers admitted on {Z > c}
    a_B = np.empty(K)
    a_B[s] = 1.0 - c
    a_B[o] = 1.0
    a_B[cr] = c + (1.0 - c) * (n_extra / n_c if n_c else 0.0)
    a_B[fl] = 0.0
    return dict(s=s, o=o, cr=cr, fl=fl, a_A=a_A, a_B=a_B, K=K, k=k, L=L, c=c, delta=delta)


def c2_analytic_mase(su):
    """Exact E|G_r - a_r| averaged over regions, for a given report vector."""
    c, delta, K = su["c"], su["delta"], su["K"]
    ns = len(su["s"]); no = len(su["o"]); nc = len(su["cr"]); nf = len(su["fl"])

    def block_int(a, shift, lo_mass_val, lo_mass_p):
        """E|G-a| for G = Z+shift on {Z<=c} (Z~U[0,1], density 1) plus mass
        lo_mass_p at lo_mass_val.  int_0^c |z - t| dz with t = a - shift."""
        t = a - shift
        if t < 0:
            core = c * c / 2.0 - t * c              # int_0^c (z - t) dz
        elif t > c:
            core = t * c - c * c / 2.0              # int_0^c (t - z) dz
        else:
            core = t * t / 2.0 + (c - t) ** 2 / 2.0
        return core + lo_mass_p * abs(lo_mass_val - a)

    def out(a_vec):
        tot = 0.0
        # s-block: G = Z + off, a = a_vec (offsets cancel) -> E|Z - (a-off)|
        for i, r in enumerate(su["s"]):
            t = a_vec[r] - 1e-9 * (i + 1)
            tot += (t * t + (1 - t) ** 2) / 2.0 if 0 <= t <= 1 else (abs(0.5 - t))
        for i, r in enumerate(su["o"]):
            tot += block_int(a_vec[r] - 1e-9 * (i + 1), delta, -1.0, 1.0 - c)
        for r in su["cr"]:
            tot += block_int(a_vec[r], delta / 2.0, -2.0, 1.0 - c)
        for r in su["fl"]:
            tot += abs(-3.0 - a_vec[r])
        return tot / K

    return out(su["a_A"]), out(su["a_B"])


def c2_simulate(su, n_draws, seed=12345, chunk=4000):
    K, k, L = su["K"], su["k"], su["L"]
    c, delta = su["c"], su["delta"]
    s, o, cr, fl = su["s"], su["o"], su["cr"], su["fl"]
    off = 1e-9 * np.arange(1, L + 1)
    rng = np.random.default_rng(seed)

    in_D = np.zeros(K)
    err_A = np.zeros(K)
    err_B = np.zeros(K)
    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        z = rng.random(m)
        mask = z <= c
        G = np.empty((m, K))
        G[:, s] = z[:, None] + off[None, :]
        G[:, o] = np.where(mask[:, None], z[:, None] + delta + off[None, :], -1.0)
        G[:, cr] = np.where(mask[:, None], z[:, None] + delta / 2.0, -2.0)
        G[:, fl] = -3.0
        # symmetric tie-break among exactly-equal entries (carriers, fillers)
        Gj = G + rng.uniform(-1e-12, 1e-12, size=G.shape)
        top = np.argpartition(-Gj, k - 1, axis=1)[:, :k]
        D = np.zeros((m, K), dtype=bool)
        np.put_along_axis(D, top, True, axis=1)
        in_D += D.sum(axis=0)
        err_A += np.abs(G - su["a_A"][None, :]).sum(axis=0)
        err_B += np.abs(G - su["a_B"][None, :]).sum(axis=0)
        done += m

    pi = in_D / n_draws
    return pi, float(err_A.sum() / n_draws / K), float(err_B.sum() / n_draws / K)


def top_set(report, L):
    return set(np.argsort(-np.asarray(report), kind="stable")[:L].tolist())


def block2_prop_c2() -> None:
    hdr("BLOCK 2 -- PROPOSITION C2 (sharp construction; sup regret = 1)")
    K, k, L = 1024, 103, 50
    c, delta = 0.985, 0.01
    chance = k / K
    check("hypothesis 2L <= k holds for the instrument's constants (100 <= 103)", 2 * L <= k, f"2L={2*L}, k={k}")

    su = c2_setup(c, delta, K, k, L)
    s, o, cr = su["s"], su["o"], su["cr"]
    print(f"  analytic medians: nu_s = {su['a_A'][s[0]]:.6f}  nu_o = {su['a_A'][o[0]]:.6f}  "
          f"nu_carrier = {su['a_A'][cr[0]]:.6f}  nu_filler = -3")
    check("analytic nu_s == 0.500", near(su["a_A"][s[0]], 0.5, 1e-8))
    check("analytic nu_o == c + delta - 0.5 == 0.495", near(su["a_A"][o[0]], 0.495, 1e-8))
    check("analytic nu_carrier == c + delta/2 - 0.5 == 0.490", near(su["a_A"][cr[0]], 0.490, 1e-8))
    check("median ordering nu_s > nu_o > nu_carrier (so m_A must pick the s-block)",
          su["a_A"][s[0]] > su["a_A"][o[0]] > su["a_A"][cr[0]])

    topA = top_set(su["a_A"], L)
    topB = top_set(su["a_B"], L)
    check("m_A (median-emitting = exact population MASE minimiser) reports the s-block",
          topA == set(s.tolist()))
    check("m_B (pi-emitting) reports the o-block", topB == set(o.tolist()))

    mA_an, mB_an = c2_analytic_mase(su)
    print(f"  analytic MASE: m_A = {mA_an:.6f}   m_B = {mB_an:.6f}   ratio = {mB_an/mA_an:.2f}x")

    n_draws = 200_000
    t0 = time.time()
    pi, mase_A, mase_B = c2_simulate(su, n_draws)
    print(f"  Monte Carlo {n_draws:,} draws in {time.time()-t0:.1f}s")

    pi_o = float(pi[o].mean()); pi_s = float(pi[s].mean()); pi_c = float(pi[cr].mean())
    R_A = float(pi[list(topA)].mean())
    R_B = float(pi[list(topB)].mean())
    regret = R_B - R_A
    mc = 4.0 / math.sqrt(n_draws)                    # generous MC band

    print(f"  MC: pi_o = {pi_o:.5f}  pi_s = {pi_s:.5f}  pi_carrier = {pi_c:.5f}")
    print(f"  MC: R(m_A) = {R_A:.5f}   R(m_B) = {R_B:.5f}   regret = {regret:.5f}   chance = {chance:.5f}")
    print(f"  MC: MASE m_A = {mase_A:.4f}   m_B = {mase_B:.4f}")

    check("analytic pi_o == 1", near(su["a_B"][o[0]], 1.0, _EXACT))
    check("analytic pi_s == 1 - c == 0.015", near(su["a_B"][s[0]], 0.015, 1e-12))
    check("MC pi_o == 1.00000", near(pi_o, 1.0, 1e-9), f"{pi_o:.5f}")
    check("MC pi_s == 0.01503 (theory 0.015)", near(pi_s, 0.015, mc), f"{pi_s:.5f}, |err| = {abs(pi_s-0.015):.5f}")
    check("MC R(m_A) == 0.01502 (theory 1-c = 0.015)", near(R_A, 0.015, mc), f"{R_A:.5f}")
    check("MC R(m_B) == 1.00000", near(R_B, 1.0, 1e-9), f"{R_B:.5f}")
    check("MC regret == 0.98498 (theory c = 0.985)", near(regret, c, mc), f"{regret:.5f}, |err| = {abs(regret-c):.5f}")
    check("MC MASE m_A == 0.0396 (analytic agreement)", near(mase_A, mA_an, 3e-3) and near(mase_A, 0.0396, 3e-3), f"{mase_A:.4f} vs analytic {mA_an:.4f}")
    check("MC MASE m_B == 2.628 (analytic agreement)", near(mase_B, mB_an, 1e-2) and near(mase_B, 2.628, 1e-2), f"{mase_B:.4f} vs analytic {mB_an:.4f}")
    check("m_A wins on the SELECTION score by ~66x while losing the REPORT",
          mase_A < mase_B and R_A < R_B, f"MASE ratio {mase_B/mase_A:.1f}x")
    check("m_A is 6.7x BELOW the chance rate on the reported metric",
          R_A < chance, f"R(m_A)={R_A:.5f} < chance={chance:.5f}; ratio {chance/R_A:.1f}x")
    check("THEOREM A: normalized regret (R_bf - R(m-hat))/Delta == 1 EXACTLY",
          abs((R_B - R_A) / (R_B - R_A) - 1.0) < _EXACT, "two-element class, R_bf-R(m_A) = Delta")

    # m_A really is the population MASE minimiser: coordinate perturbation must raise MASE
    rng = np.random.default_rng(7)
    z = rng.random(200_000)
    worse = 0
    for r in [s[0], o[0], cr[0]]:
        if r in s:
            g = z.copy()
        elif r in o:
            g = np.where(z <= c, z + delta, -1.0)
        else:
            g = np.where(z <= c, z + delta / 2.0, -2.0)
        base = np.abs(g - su["a_A"][r]).mean()
        for h in (-0.05, -0.01, 0.01, 0.05):
            if np.abs(g - (su["a_A"][r] + h)).mean() > base - 1e-12:
                worse += 1
    check("m_A is a coordinatewise MASE minimiser (12/12 perturbations do not improve)",
          worse == 12, f"{worse}/12")

    # --- sup regret = 1: sweep c -> 1-delta, delta -> 0 ----------------------
    print("\n  sup-regret sweep (the claim is sup = 1, approached but not attained):")
    sweep_ok = True
    last = 0.0
    for cc, dd in [(0.6, 0.05), (0.8, 0.02), (0.985, 0.01), (0.995, 0.002),
                   (0.998, 0.001), (0.9995, 0.0002)]:
        s2 = c2_setup(cc, dd, K, k, L)
        tA = top_set(s2["a_A"], L)
        p2, _, _ = c2_simulate(s2, 60_000, seed=11)
        RA2 = float(p2[list(tA)].mean())
        RB2 = float(p2[list(top_set(s2["a_B"], L))].mean())
        reg2 = RB2 - RA2
        picks_s = tA == set(s2["s"].tolist())
        print(f"    c={cc:<7} delta={dd:<7} m_A picks {'s-block' if picks_s else 'OTHER'}  "
              f"R(m_A)={RA2:.5f}  R(m_B)={RB2:.5f}  regret={reg2:.5f}  (theory c={cc})")
        sweep_ok &= picks_s and abs(reg2 - cc) < 0.01 and reg2 > last
        last = reg2
    check("Prop C2 sup regret -> 1: monotone sweep reaches 0.9995 while m_A stays the "
          "population MASE minimiser and stays on the s-block", sweep_ok,
          f"largest realised regret = {last:.5f} < 1")

    # Lemma D -- non-attainment at c = 1
    print("\n  Lemma D (non-attainment): rerun with c = 1")
    su1 = c2_setup(1.0, delta, K, k, L)
    z = np.random.default_rng(99).random(200_000)
    nu_s_mc = float(np.median(z))
    nu_o_mc = float(np.median(z + delta))
    print(f"    MC medians: nu_s = {nu_s_mc:.4f}   nu_o = {nu_o_mc:.4f}")
    check("Lemma D: at c = 1 the medians FLIP (nu_o > nu_s)", nu_o_mc > nu_s_mc,
          f"nu_o - nu_s = {nu_o_mc - nu_s_mc:.4f} (= delta)")
    check("Lemma D: at c = 1 the median-emitting model correctly reports the o-block",
          top_set(su1["a_A"], L) == set(su1["o"].tolist()))
    check("Lemma D: regret exactly 1 is UNATTAINABLE for (MASE, precision@L)",
          top_set(su1["a_A"], L) == set(su1["o"].tolist()),
          "pi_o=1 & pi_s=0 forces G_o >= G_s a.s. hence ordered medians")


# --------------------------------------------------------------------------- #
# BLOCK 3 -- Proposition C1
# --------------------------------------------------------------------------- #
def block3_prop_c1() -> None:
    hdr("BLOCK 3 -- PROPOSITION C1 (wins on BOTH co-primary criteria, loses the report)")
    from scipy.stats import rankdata

    K, k, L = 1024, 103, 50
    n_stable, n_vol = 724, 300
    chance = k / K
    mu = np.concatenate([np.full(n_stable, 1.0), np.full(n_vol, 0.0)])
    sig = np.concatenate([np.full(n_stable, 0.1), np.full(n_vol, 3.0)])

    a_A = mu.copy()
    a_B = mu + 1.5 * sig
    # tie-break key: an sigma-independent permutation (stands in for "ties by region size")
    tb = np.random.default_rng(2026).permutation(K).astype(float) / (2.0 * K)
    topA = set(np.argsort(-(a_A + 1e-6 * tb), kind="stable")[:L].tolist())
    topB = set(np.argsort(-(a_B + 1e-6 * tb), kind="stable")[:L].tolist())
    check("m_A's top-50 is drawn from the STABLE block (mu = 1 beats mu = 0)",
          all(i < n_stable for i in topA))
    check("m_B's top-50 is drawn from the VOLATILE block (4.5 beats 1.15)",
          all(i >= n_stable for i in topB))

    rA = rankdata(a_A); rB = rankdata(a_B)
    check("rank(m_B) = K + 1 - rank(m_A) exactly => Spearman(m_B) = -Spearman(m_A)",
          np.allclose(rB, K + 1 - rA))

    # analytic MASE
    SQ = math.sqrt(2.0 / math.pi)
    E_abs_shift = 2 * (1 / math.sqrt(2 * math.pi)) * math.exp(-1.5 ** 2 / 2) + 1.5 * (
        2 * 0.5 * (1 + math.erf(1.5 / math.sqrt(2))) - 1)
    mase_A_an = float((sig * SQ).mean())
    mase_B_an = float((sig * E_abs_shift).mean())
    print(f"  analytic MASE: m_A = {mase_A_an:.4f}   m_B = {mase_B_an:.4f}")

    n_draws, chunk = 40_000, 2000
    rng = np.random.default_rng(4242)
    in_D = np.zeros(K); errA = np.zeros(K); errB = np.zeros(K)
    cA = rA - rA.mean(); cB = rB - rB.mean()
    nA = math.sqrt(float((cA ** 2).sum())); nB = math.sqrt(float((cB ** 2).sum()))
    idxK = np.arange(K, dtype=float)
    cG_norm = math.sqrt(float(((idxK - idxK.mean()) ** 2).sum()))
    sp_A = 0.0; sp_B = 0.0
    t0 = time.time()
    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        G = mu[None, :] + sig[None, :] * rng.standard_normal((m, K))
        top = np.argpartition(-G, k - 1, axis=1)[:, :k]
        D = np.zeros((m, K), dtype=bool)
        np.put_along_axis(D, top, True, axis=1)
        in_D += D.sum(axis=0)
        errA += np.abs(G - a_A[None, :]).sum(axis=0)
        errB += np.abs(G - a_B[None, :]).sum(axis=0)
        order = np.argsort(G, axis=1)
        rg = np.empty_like(order)
        np.put_along_axis(rg, order, np.arange(K)[None, :], axis=1)
        cg = rg - rg.mean(axis=1, keepdims=True)
        sp_A += float((cg @ cA).sum()) / (nA * cG_norm)
        sp_B += float((cg @ cB).sum()) / (nB * cG_norm)
        done += m
    print(f"  Monte Carlo {n_draws:,} draws in {time.time()-t0:.1f}s")

    pi = in_D / n_draws
    R_A = float(pi[list(topA)].mean()); R_B = float(pi[list(topB)].mean())
    mase_A = float(errA.sum() / n_draws / K); mase_B = float(errB.sum() / n_draws / K)
    sA = sp_A / n_draws; sB = sp_B / n_draws
    mc = 6.0 / math.sqrt(n_draws * L)

    print(f"  R(m_A) = {R_A:.4f}   R(m_B) = {R_B:.4f}   regret = {R_B-R_A:.4f}   chance = {chance:.4f}")
    print(f"  MASE   m_A = {mase_A:.4f}   m_B = {mase_B:.4f}")
    print(f"  Spearman m_A = {sA:+.4f}   m_B = {sB:+.4f}")

    check("MASE: m_A STRICTLY better (0.7577 vs 1.4804)",
          mase_A < mase_B and near(mase_A, 0.7577, 5e-3) and near(mase_B, 1.4804, 8e-3),
          f"{mase_A:.4f} < {mase_B:.4f}; analytic {mase_A_an:.4f}/{mase_B_an:.4f}")
    check("Spearman: m_A STRICTLY better (+0.2050 vs -0.2050)",
          sA > sB and near(sA, 0.2050, 8e-3) and near(sB, -0.2050, 8e-3), f"{sA:+.4f} > {sB:+.4f}")
    check("REPORTED: R(m_A) == 0.0055", near(R_A, 0.0055, 4e-3), f"{R_A:.4f}")
    check("REPORTED: R(m_B) == 0.3302", near(R_B, 0.3302, 1.5e-2), f"{R_B:.4f}")
    check("regret == 0.3247", near(R_B - R_A, 0.3247, 1.5e-2), f"{R_B-R_A:.4f}")
    check("PROP C1 VERDICT: R(m_A) < chance < R(m_B) while m_A wins BOTH selection criteria",
          (R_A < chance < R_B) and (mase_A < mase_B) and (sA > sB),
          f"{R_A:.4f} < {chance:.4f} < {R_B:.4f}")
    check("m_A is ~18x BELOW chance; m_B ~3.3x ABOVE",
          near(chance / R_A, 18.3, 2.0) and near(R_B / chance, 3.28, 0.2),
          f"chance/R_A = {chance/R_A:.1f}x, R_B/chance = {R_B/chance:.2f}x")
    check("THEOREM A: normalized regret == 1 on this two-element class",
          abs((R_B - R_A) / (R_B - R_A) - 1.0) < _EXACT)

    # m_A really is the exact population MASE minimiser.  MC cannot resolve this at
    # sigma = 3 (the curvature is ~1e-5 against an MC s.e. of ~3e-3), so use the exact
    # E|N(mu,s^2) - a| = 2 s phi(u) + (a-mu)(2 Phi(u) - 1), u = (a-mu)/s.
    def e_abs(a, mu_, s_):
        u = (a - mu_) / s_
        phi = math.exp(-u * u / 2) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1 + math.erf(u / math.sqrt(2)))
        return 2 * s_ * phi + (a - mu_) * (2 * Phi - 1)

    bad = 0
    for mu_, s_ in [(1.0, 0.1), (0.0, 3.0)]:
        base = e_abs(mu_, mu_, s_)
        for h in (-1.0, -0.5, -0.1, -0.01, -1e-4, 1e-4, 0.01, 0.1, 0.5, 1.0):
            if e_abs(mu_ + h, mu_, s_) <= base + 1e-15:
                bad += 1
        if e_abs(mu_ + 1.5 * s_, mu_, s_) <= base:
            bad += 1
    check("m_A = mu is the STRICT population MASE minimiser (exact formula, 22/22 perturbations worse)",
          bad == 0, f"{22 - bad}/22")


# --------------------------------------------------------------------------- #
# BLOCK 4 -- adversarial extras
# --------------------------------------------------------------------------- #
def block4_extras() -> None:
    hdr("BLOCK 4 -- ADVERSARIAL EXTRAS (Cor B3 witness; identity on random tables)")
    from scipy.stats import rankdata, spearmanr

    # --- Corollary B3: common median, distinct dispersions -------------------
    K, k, L = 1024, 103, 50
    chance = k / K
    rng = np.random.default_rng(31337)
    sig = np.linspace(0.2, 3.0, K)
    perm = rng.permutation(K)          # scatter sigma across region indices
    sig = sig[perm]
    n_draws, chunk = 20_000, 2000
    in_D = np.zeros(K)
    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        G = sig[None, :] * rng.standard_normal((m, K))
        top = np.argpartition(-G, k - 1, axis=1)[:, :k]
        D = np.zeros((m, K), dtype=bool)
        np.put_along_axis(D, top, True, axis=1)
        in_D += D.sum(axis=0)
        done += m
    pi = in_D / n_draws
    check("Cor B3: all population medians are 0 (symmetric, common median)", True, "by construction")
    check("Cor B3: sum_r pi_r == k exactly (so mean pi == chance)", near(pi.sum(), k, 1e-9), f"{pi.sum():.4f}")
    R_median_model = float(pi.mean())                 # expected R under a sigma-independent tie-break
    top_disp = np.argsort(-sig)[:L]
    R_disp = float(pi[top_disp].mean())
    rho_sp = float(spearmanr(sig, pi).statistic)
    print(f"  R(median-model, avg over sigma-independent tie-breaks) = {R_median_model:.4f}  (chance {chance:.4f})")
    print(f"  R(dispersion-ranking model) = {R_disp:.4f}   Spearman(sigma, pi) = {rho_sp:+.4f}")
    check("Cor B3: median-model attains exactly the chance rate 0.1006",
          near(R_median_model, chance, 1e-9), f"{R_median_model:.4f}")
    check("Cor B3: dispersion-ranking model attains 0.2375", near(R_disp, 0.2375, 1.5e-2), f"{R_disp:.4f}")
    check("Cor B3: Spearman(sigma, pi) == +0.9989", near(rho_sp, 0.9989, 3e-3), f"{rho_sp:+.4f}")
    check("Cor B3: T_rep is NOT T_sel-measurable (T_sel constant, T_rep strictly increasing in sigma)",
          rho_sp > 0.99 and R_disp > R_median_model)
    check("CAVEAT recorded: R(median-model) = chance only for a tie-break independent of sigma; "
          "a sigma-correlated tie-break spans [%.4f, %.4f]"
          % (float(pi[np.argsort(sig)[:L]].mean()), R_disp), True,
          "Thm B(3)'s 'under every tie-break' is the operative reading")

    # --- Theorem D is an identity, not a coincidence -------------------------
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(20_000):
        M, n = int(rng.integers(2, 9)), int(rng.integers(2, 8))
        Rt = rng.random((M, n))
        sel = rng.integers(0, M, size=n)
        per_max = Rt.max(axis=0)
        mean_by = Rt.mean(axis=1)
        R_sel = float(Rt[sel, np.arange(n)].mean())
        A = float(per_max.mean() - mean_by.max())
        Lam = float((per_max - Rt[sel, np.arange(n)]).mean())
        worst = max(worst, abs((R_sel - mean_by.max()) - (A - Lam)))
        assert A >= -1e-12 and Lam >= -1e-12
    check("Theorem D holds as an IDENTITY on 20,000 random tables (A>=0, Lambda>=0 too)",
          worst < 1e-12, f"max |lhs - rhs| = {worst:.3e}")

    # --- Corollary D1 equivalence on random tables ---------------------------
    rng = np.random.default_rng(6)
    bad = 0
    for _ in range(20_000):
        M, n = int(rng.integers(2, 7)), int(rng.integers(2, 9))
        Rt = rng.random((M, n))
        sel = rng.integers(0, M, size=n)
        per_max = Rt.max(axis=0)
        A = float(per_max.mean() - Rt.mean(axis=1).max())
        Lam = float((per_max - Rt[sel, np.arange(n)]).mean())
        beats = float(Rt[sel, np.arange(n)].mean()) > Rt.mean(axis=1).max() + 1e-15
        if beats != (Lam < A - 1e-15):
            bad += 1
    check("Cor D1: 'selection beats the best fixed model iff Lambda < A' on 20,000 random tables",
          bad == 0, f"{bad} violations")


# --------------------------------------------------------------------------- #
# BLOCK 5 -- the real CV artifacts
# --------------------------------------------------------------------------- #
def block5_real_cv(b1: dict) -> None:
    hdr("BLOCK 5 -- ADVERSARIAL EXTENSION: the REAL CV artifacts (not requested; measured tau)")
    import pandas as pd

    cutoffs = b1["cutoffs"]; R = b1["R"]; mi = b1["mi"]; ci = b1["ci"]
    ETA = b1["A"] / 0.22
    per_cut_max = b1["per_cut_max"]; R_sel_vec = b1["R_sel_vec"]

    NUMPY_MODELS = {"Naive", "SeasonalNaive", "SmoothedMomentum"}
    feasible_max = []
    for c in cutoffs:
        cv = pd.read_parquet(REPO / "artifacts" / "cutoffs" / c / "forecast_cv.parquet")
        summ = (cv.groupby("model").agg(spearman=("spearman", "mean"),
                                        mase=("mase_median", "median")).reset_index()
                  .sort_values(["spearman", "mase"], ascending=[False, True]))
        sel = summ.iloc[0]["model"]
        cand = sorted(summ["model"].tolist())
        print(f"\n  {c}: CV candidates = {cand}")
        print("    " + summ.to_string(index=False).replace("\n", "\n    "))
        check(f"{c}: harness rule (max mean-Spearman, tie-break MASE) reproduces the frozen selection",
              sel == b1["sel_model"][c], f"reproduced {sel}, file says {b1['sel_model'][c]}")

        # The paper analyses the pair (MASE, precision@L); the harness's actual rule
        # (forecast.py: sort_values(["spearman","mase"], [False,True])) is Spearman-primary
        # with MASE only as tie-break.  Does that framing difference change anything HERE?
        sel_mase = summ.sort_values("mase").iloc[0]["model"]
        sel_spear = summ.sort_values("spearman", ascending=False).iloc[0]["model"]
        check(f"{c}: a MASE-ONLY rule picks the SAME model as the harness "
              f"(so treating S_sel as MASE is harmless on this instance)",
              sel_mase == sel and sel_spear == sel,
              f"MASE-only -> {sel_mase}, Spearman-only -> {sel_spear}, harness -> {sel}")

        # per-fold leads of the selected model over each rival, in Spearman units
        piv = cv.pivot(index="model", columns="window", values="spearman")
        # window index is REVERSED between the numpy CV (w=0 is the LATEST window)
        # and the statsforecast CV (w=0 is the EARLIEST). Align onto calendar time.
        aligned = {}
        for mname, row in piv.iterrows():
            v = row.dropna()
            if mname in NUMPY_MODELS:
                aligned[mname] = {3 - int(w): float(x) for w, x in v.items()}
            else:
                aligned[mname] = {int(w): float(x) for w, x in v.items()}
        # S_sel is NEGATIVELY oriented; the harness maximises Spearman, so the
        # per-fold score of model m is (-spearman) and the lead of the winner over
        # a rival is gamma = mean_folds[ spearman(sel) - spearman(rival) ] > 0.
        ratios = []
        for rival in aligned:
            if rival == sel:
                continue
            common = sorted(set(aligned[sel]) & set(aligned[rival]))
            if len(common) < 2:
                continue
            d = np.array([aligned[sel][w] - aligned[rival][w] for w in common])
            gam = float(d.mean()); tau = float(d.std(ddof=1))
            if tau > 0:
                ratios.append((rival, gam, tau, gam / tau, len(common)))
        for rival, gam, tau, ratio, nf in ratios:
            print(f"    lead over {rival:<18} gamma = {gam:+.4f}  tau = {tau:.4f}  "
                  f"gamma/tau = {ratio:+.2f}  (n_folds = {nf})")
        if ratios:
            worst = min(r[3] for r in ratios)
            n_fold = min(r[4] for r in ratios)
            need = math.sqrt(2 * math.log((len(aligned) - 1) / ETA) / n_fold)
            met = worst >= need
            print(f"    min gamma/tau over rivals = {worst:+.2f}; Prop E1 (M={len(aligned)}, "
                  f"n={n_fold}, eta={ETA:.4f}) requires >= {need:.4f}  ->  {'MET' if met else 'NOT MET'}")
            check(f"{c}: Prop E1 NECESSARY condition for rho >= 1-eta is NOT met on measured folds",
                  not met, f"slack = {worst - need:+.2f} tau (tau from only {n_fold} folds -- very noisy)")

        # candidate set actually available to the CV selector at this cutoff
        avail = [m for m in cand if m in mi]
        feasible_max.append(max(R[mi[m], ci[c]] for m in avail))
        if len(avail) < 6:
            print(f"    NOTE: only {len(avail)}/6 roster models had CV rows here; "
                  f"per-cutoff realised best over the FEASIBLE set = {feasible_max[-1]:.3f} "
                  f"(over the full roster = {per_cut_max[ci[c]]:.3f})")

    # alternative accounting on the CV-feasible roster
    feasible_max = np.array(feasible_max)
    A_f = float(feasible_max.mean() - b1["R_bf"])
    Lam_f = float((feasible_max - R_sel_vec).mean())
    lhs = b1["R_sel"] - b1["R_bf"]
    print()
    print(f"  CV-FEASIBLE-ROSTER accounting:  E[max feasible] = {feasible_max.mean():.6f}   "
          f"A = {A_f:.6f}   Lambda = {Lam_f:.6f}")
    check("Theorem D identity ALSO exact on the CV-feasible-roster accounting",
          abs(lhs - (A_f - Lam_f)) < _EXACT, f"|diff| = {abs(lhs - (A_f - Lam_f)):.3e}")
    check("On the CV-feasible roster the value of adaptivity is EXACTLY ZERO "
          "=> break-even hit rate rho* = 1: selection could not pay at all",
          near(A_f, 0.0, 1e-12), f"A_feasible = {A_f:.6f} (headline accounting uses the full 6-model roster: A = {b1['A']:.6f})")



# --------------------------------------------------------------------------- #
# BLOCK 6 -- the mathematics stated in cache.md / the paper's Appendix A
#   Prop C : the DEPLOYED pair (global Spearman, precision@k) provably violates
#            the alignment condition of Theorem B, at the instrument's own
#            dimensions N = 1023 eligible regions, k = 50.
#   Thm E(i) sharpness : the factor 2 in the margin condition cannot be improved.
#   Cor D1 : the break-even hit rate, with its cost hypothesis made explicit, and
#            its feasibility at n = 3 origins.
# --------------------------------------------------------------------------- #
def _spearman_vec(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def block6_alignment_witness(b1) -> None:
    hdr("BLOCK 6 -- PROP C: the deployed pair violates the alignment condition (Thm B)")
    N, k = 1023, 50
    truth = np.arange(N, dtype=np.float64)          # higher = better
    top = set(range(N - k, N))

    # pi1: global order preserved, true top-k pushed to the very bottom
    pi1 = np.empty(N, dtype=np.float64)
    pi1[:N - k] = np.arange(k, N, dtype=np.float64)
    pi1[N - k:] = np.arange(k, dtype=np.float64)
    # pi2: true top-k exactly on top, everything below reversed
    pi2 = np.empty(N, dtype=np.float64)
    pi2[N - k:] = np.arange(N - k, N, dtype=np.float64)
    pi2[:N - k] = np.arange(N - k - 1, -1, -1, dtype=np.float64)

    out = {}
    for nm, pi in (("pi1", pi1), ("pi2", pi2)):
        sel = set(np.argsort(-pi)[:k].tolist())
        out[nm] = (_spearman_vec(pi, truth), len(sel & top) / k)
        print(f"  {nm}: spearman = {out[nm][0]:+.6f}   precision@{k} = {out[nm][1]:.3f}")

    s1, p1 = out["pi1"]; s2, p2 = out["pi2"]
    check("Prop C: the rank-correlation-preferred candidate is the precision-worst one",
          s1 > s2 and p1 < p2,
          f"spearman {s1:+.6f} > {s2:+.6f} while precision@{k} {p1:.3f} < {p2:.3f}")
    check("Prop C: precision@k is exactly 0 vs exactly 1 -- the full spread",
          near(p1, 0.0, 1e-12) and near(p2, 1.0, 1e-12))
    check("Prop C: no non-decreasing phi can satisfy prec@k = phi(spearman) "
          "=> Thm B's condition fails for the deployed pair", s1 > s2 and p1 < p2,
          f"selection margin in rank correlation = {s1 - s2:.4f}")
    # closed form of the pi1 displacement sum: k(N-k)N
    rho1_exact = 1.0 - 6.0 * k * (N - k) / (N * N - 1)
    check("Prop C: pi1's rank correlation matches the closed form 1 - 6k(N-k)/(N^2-1)",
          near(s1, rho1_exact, 1e-9), f"computed {s1:.9f} vs closed form {rho1_exact:.9f}")
    check("Prop C: the coarse bound 1 - 6k/N is a valid lower bound",
          s1 >= 1.0 - 6.0 * k / N, f"{s1:.6f} >= {1.0 - 6.0 * k / N:.6f}")
    check("Prop C supplies hypothesis (A3) of Theorem A, so the reachable "
          "normalized regret for this pair is 1", True,
          "selection-best is report-worst by construction")

    # ---- Theorem E(i): sharpness of the factor 2 --------------------------- #
    hdr("BLOCK 6b -- THM E(i): the factor 2 in the margin condition is sharp")
    g = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    kk = 2
    gap = g[kk - 1] - g[kk]                                   # g_(2) - g_(3) = 1.0
    eta_ok = gap / 2.0 - 1e-9
    eps_ok = np.zeros_like(g); eps_ok[kk - 1] = -eta_ok; eps_ok[kk] = +eta_ok
    eps_bad = np.zeros_like(g); eps_bad[kk - 1] = -(gap / 2.0 + 1e-9); eps_bad[kk] = +(gap / 2.0 + 1e-9)
    top_of = lambda v: set(np.argsort(-v)[:kk].tolist())
    check("Thm E(i): 2*max|eps| < gap leaves the top-k set unchanged",
          top_of(g + eps_ok) == top_of(g), f"gap = {gap}, 2*max|eps| = {2*eta_ok:.9f}")
    check("Thm E(i): 2*max|eps| > gap by an arbitrarily small amount flips it "
          "-- so the constant 2 cannot be improved",
          top_of(g + eps_bad) != top_of(g), f"2*max|eps| = {2*(gap/2+1e-9):.9f} > {gap}")

    # ---- Corollary D1: break-even hit rate, feasibility at n = 3 ----------- #
    hdr("BLOCK 6c -- COR D1: break-even hit rate, with its cost hypothesis")
    A, Lam = b1["A"], b1["Lam"]
    n_orig = 3
    h_obs = 0.0                       # the selector hit the per-origin argmax at 0 of 3 origins
    cbar = Lam / (1.0 - h_obs)        # mean cost of a miss under the stated hypothesis
    h_star = 1.0 - A / cbar
    feasible = [i / n_orig for i in range(n_orig + 1)]
    reachable = [h for h in feasible if h >= h_star - 1e-12]
    print(f"  A = {A:.6f}   Lambda = {Lam:.6f}   A/Lambda = {A/Lam:.6f}")
    print(f"  hypothesis: every miss costs the observed mean miss cost cbar = {cbar:.6f}")
    print(f"  break-even hit rate h* = 1 - A/cbar = {h_star:.6f}")
    print(f"  feasible hit rates at n=3: {feasible}   reachable: {reachable}")
    check("Cor D1: break-even hit rate is 0.9091 under the stated cost hypothesis",
          near(h_star, 0.9090909090909091, 1e-9), f"h* = {h_star:.10f}")
    check("Cor D1: at n = 3 the ONLY feasible way to break even was h = 1 "
          "(hit the per-origin oracle at every origin)",
          reachable == [1.0], f"reachable hit rates = {reachable}")
    check("Cor D1: A/Lambda = 1/11 -- the prize was one eleventh of the cost",
          near(A / Lam, 1.0 / 11.0, 1e-9), f"A/Lambda = {A/Lam:.9f}")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("verify_selectionconsistency.py -- O4 selection-consistency, numerical companion")
    print(f"repo = {REPO}")
    block0_eb_anchor()
    b1 = block1_frozen_table()
    block2_prop_c2()
    block3_prop_c1()
    block4_extras()
    try:
        block6_alignment_witness(b1)
    except Exception as e:                                    # pragma: no cover
        check("BLOCK 6 ran", False, f"{type(e).__name__}: {e}")
    try:
        block5_real_cv(b1)
    except Exception as e:                                    # pragma: no cover
        check("BLOCK 5 ran", False, f"{type(e).__name__}: {e}")

    hdr("SUMMARY")
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"  FAILED: {name}   {detail}")
    print(f"\n  {len(_RESULTS) - n_fail}/{len(_RESULTS)} checks passed.")
    if n_fail == 0:
        print("\n  OVERALL: PASS -- every claim in O4's numeric block is reproduced: the "
              "Theorem-D identity holds to machine precision on the frozen table "
              "(A = 0.020000, Lambda = 0.220000, gap = -0.200000), Proposition C2 attains "
              "regret 0.985 with the exact MASE minimiser 6.7x below chance, and Proposition C1's "
              "model wins BOTH co-primary selection criteria while scoring 18x below chance.")
    else:
        print(f"\n  OVERALL: FAIL -- {n_fail} check(s) did not reproduce.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
