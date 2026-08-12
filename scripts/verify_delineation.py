"""verify_delineation.py — numerical companion for O5, "A delineation-invariance
calculus for bibliometric statistics" (slug: delineation).

Adversarial re-derivation, from the raw counts, of every numeric claim A1..E25 in
the O5 answer, on the PRODUCTION origin of this repo (K=1024, 64 supers, n=234
months, add-alpha=1.0, kappa=3.147961).

Nothing in the repo is modified. Run from the repo root:

    python scripts/verify_delineation.py

Gate 0 reproduces series.run_shares from data/papers.parquet + the committed region
labels and refuses to continue unless kappa == 3.147960961354892 and
max|p_hat - share_smoothed| == 0.0 exactly.

Conventions pinned by that gate and by matching the claim's own anchor numbers:
  x        = share_smoothed (p_hat), 1024 x 234
  balances = SBP contrast matrix of artifacts/linkage.f64.npy, b = Psi . log x
  D_v      = Var_t[Delta b_v], ddof=1        (reproduces sum_v D_v to 12 s.f.)
  alpha_h  = -OLS slope of log2 D_v on log2 h_v   (Theorem 16's sign convention)
  windows  = T0 2024-06: H = 2026-01..2026-06 (idx 228..233),
                         T = 2023-07..2024-06 (idx 198..209)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brontide import config, series  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K = config.N_REGIONS            # 1024
NS = config.N_SUPERREGIONS      # 64
A = config.ADD_ALPHA            # 1.0
KAPPA0 = 3.147960961354892
H_IDX = np.arange(228, 234)     # 2026-01 .. 2026-06
T_IDX = np.arange(198, 210)     # 2023-07 .. 2024-06

_RESULTS: list[tuple[str, str, bool, str]] = []
_NOTES: list[str] = []


def check(tag: str, desc: str, ok: bool, detail: str) -> bool:
    _RESULTS.append((tag, desc, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag:<5s} {desc}\n         {detail}")
    return bool(ok)


def note(txt: str) -> None:
    _NOTES.append(txt)
    print(f"  [NOTE]       {txt}")


def close(a: float, b: float, rtol: float) -> bool:
    return abs(a - b) <= rtol * max(abs(b), 1e-300)


# --------------------------------------------------------------------------
# Gate 0 — reproduce the EB pipeline exactly
# --------------------------------------------------------------------------
def build_counts() -> tuple[np.ndarray, list[str]]:
    import pyarrow.parquet as pq

    labels = np.load(ROOT / "data" / "labels" / "region_labels.i16.npy").astype(np.int32)
    months = pq.read_table(config.RAW_PARQUET, columns=["month"]).column("month").to_pylist()
    if len(labels) != len(months):
        raise SystemExit("row-order invariant broken")

    def ordinal(m: str) -> int:
        y, mo = m.split("-")
        return int(y) * 12 + int(mo) - 1

    o0, o1 = ordinal(config.CORPUS_START), ordinal(config.CORPUS_END)
    month_labels = [f"{o // 12:04d}-{o % 12 + 1:02d}" for o in range(o0, o1 + 1)]
    idx = np.fromiter((ordinal(m) - o0 if o0 <= ordinal(m) <= o1 else -1 for m in months),
                      dtype=np.int32, count=len(months))
    keep = idx >= 0
    C = np.zeros((K, len(month_labels)))
    np.add.at(C, (labels[keep], idx[keep]), 1.0)
    return C, month_labels


def eb(C: np.ndarray, rows: np.ndarray, sup: np.ndarray, trail: np.ndarray,
       kappa: float | None) -> dict:
    """Exact reimplementation of series.run_shares on the subcomposition `rows`."""
    Cx = C[rows]
    Kx = rows.size
    den = Cx.sum(axis=0) + A * Kx                      # M_t + a K
    p = (Cx + A) / den[None, :]
    sx = sup[rows]
    ns = np.bincount(sx, minlength=NS)
    P_S = np.zeros((NS, C.shape[1]))
    np.add.at(P_S, sx, p)
    m = (P_S / np.where(ns > 0, ns, 1)[:, None])[sx]
    tr = trail[rows]
    kap = series._estimate_kappa(tr, sx) if kappa is None else kappa
    w = tr / (tr + kap)
    return {"p": p, "m": m, "w": w, "kappa": kap, "den": den,
            "phat": w[:, None] * p + (1.0 - w[:, None]) * m}


def sbp_basis(Z: np.ndarray) -> np.ndarray:
    members: list[list[int]] = [[i] for i in range(K)]
    Psi = np.zeros((K - 1, K))
    for v in range(K - 1):
        L = members[int(Z[v, 0])]
        R = members[int(Z[v, 1])]
        r, s = len(L), len(R)
        Psi[v, L] = math.sqrt(s / (r * (r + s)))
        Psi[v, R] = -math.sqrt(r / (s * (r + s)))
        members.append(L + R)
    return Psi


def alpha_h(Z: np.ndarray, dlx: np.ndarray) -> tuple[float, float, float, float, float, np.ndarray]:
    D = (sbp_basis(Z) @ dlx).var(axis=1, ddof=1)
    x, y = np.log2(Z[:, 2]), np.log2(D)
    slope, icpt = np.polyfit(x, y, 1)
    r2 = 1.0 - ((y - (slope * x + icpt)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return -slope, r2, float(np.var(x, ddof=1)), float(np.cov(x, y, ddof=1)[0, 1]), \
        float(np.corrcoef(x, y)[0, 1]), D


def topk(vals: np.ndarray, rows: np.ndarray, k: int = 50) -> set:
    return set(rows[np.argsort(-vals)[:k]].tolist())


def main() -> int:
    t_start = time.time()
    print("=" * 78)
    print("verify_delineation.py — O5 delineation-invariance calculus, numerical audit")
    print("=" * 78)

    # ---------------- Gate 0 ------------------------------------------------
    print("\nGATE 0 — reimplementation of the EB pipeline")
    import pyarrow.parquet as pq
    C, month_labels = build_counts()
    n = C.shape[1]
    sup = np.load(ROOT / "artifacts" / "super_region.i16.npy").astype(np.int32)
    trail = C[:, n - 12:n].sum(axis=1)
    allrows = np.arange(K)
    base = eb(C, allrows, sup, trail, None)

    tab = pq.read_table(ROOT / "artifacts" / "region_shares.parquet")
    ss = np.asarray(tab.column("share_smoothed")).reshape(K, n)
    pa = np.asarray(tab.column("share_addalpha")).reshape(K, n)
    d_ph = float(np.abs(base["phat"] - ss).max())
    d_p = float(np.abs(base["p"] - pa).max())
    g_ok = check("G0", "kappa and share_smoothed reproduce exactly",
                 base["kappa"] == KAPPA0 and d_ph == 0.0 and d_p == 0.0,
                 f"kappa={base['kappa']!r} (target {KAPPA0!r}); "
                 f"max|p_hat-share_smoothed|={d_ph}; max|p-share_addalpha|={d_p}; "
                 f"n_months={n} ({month_labels[0]}..{month_labels[-1]})")
    if not g_ok:
        print("\nABORT: pipeline reimplementation does not match. No number below is trustworthy.")
        return 1

    phat0, p0, m0, w0, den0 = base["phat"], base["p"], base["m"], base["w"], base["den"]
    drops = [np.where(sup != s)[0] for s in range(NS)]

    # ---------------- PART A ------------------------------------------------
    print("\nPART A — restriction is a rank-one closure field (Lemmas 1-2, Thm 3)")
    a1 = a2 = a4 = 0.0
    kaps = []
    for rows in drops:
        r = eb(C, rows, sup, trail, KAPPA0)
        lam = den0 / r["den"]
        a1 = max(a1, float(np.abs(r["p"] / p0[rows] - lam[None, :]).max()))
        a2 = max(a2, float(np.abs(r["phat"] / (lam[None, :] * phat0[rows]) - 1.0).max()))
        rr = eb(C, rows, sup, trail, None)
        pred = lam[None, :] * (phat0[rows] + (rr["w"] - w0[rows])[:, None] * (p0[rows] - m0[rows]))
        a4 = max(a4, float(np.abs(rr["phat"] / pred - 1.0).max()))
        kaps.append(rr["kappa"])
    kaps = np.array(kaps)

    check("A1", "max_{r,t} |p'/p - lambda_t| <= 1e-15 over 64 super-drops",
          a1 <= 1e-15, f"measured {a1:.4e} (claim 2.220e-16); slack x{1e-15 / max(a1, 1e-300):.1f}")
    check("A2", "kappa frozen: max |p_hat'/(lambda_t p_hat) - 1| <= 1e-14",
          a2 <= 1e-14, f"measured {a2:.4e} (claim 6.601e-16 / 6.66e-16); slack x{1e-14 / max(a2, 1e-300):.1f}")
    check("A4", "exact factorisation p_hat' = lam (p_hat + (w'-w)(p-m)), rel err <= 1e-14",
          a4 <= 1e-14, f"measured {a4:.4e} (claim 3.651e-16)")
    check("A5", "kappa' over the 64 drops inside [3.0597, 3.3379]",
          3.0597 <= kaps.min() and kaps.max() <= 3.3380,
          f"measured [{kaps.min():.10f}, {kaps.max():.10f}]; "
          f"max |dkappa|/kappa = {np.abs(kaps - KAPPA0).max() / KAPPA0:.4f} (claim <= 6.1%)")

    # A3 — saturation is necessary
    rng = np.random.default_rng(0)
    s7 = np.where(sup == 7)[0]
    nonsat = {
        "16 random regions (default_rng(0))": np.setdiff1d(allrows, rng.choice(K, 16, replace=False)),
        "half of super 7 (low index half)": np.setdiff1d(allrows, s7[:s7.size // 2]),
        "half of super 7 (high index half)": np.setdiff1d(allrows, s7[s7.size // 2:]),
    }
    a3 = {}
    for nm, rows in nonsat.items():
        r = eb(C, rows, sup, trail, KAPPA0)
        lam = den0 / r["den"]
        a3[nm] = float(np.abs(r["phat"] / (lam[None, :] * phat0[rows]) - 1.0).max())
    check("A3", "saturation NECESSARY: non-saturated restrictions break covariance (>= 5e-2)",
          min(a3.values()) >= 5e-2,
          "; ".join(f"{k}: {v:.4e}" for k, v in a3.items()) +
          "  (claim 5.926e-02 / 8.443e-02)")
    sat = {}
    for nm, rows in (("super 7", np.where(sup != 7)[0]),
                     ("supers 7+11", np.where((sup != 7) & (sup != 11))[0])):
        r = eb(C, rows, sup, trail, KAPPA0)
        lam = den0 / r["den"]
        sat[nm] = float(np.abs(r["phat"] / (lam[None, :] * phat0[rows]) - 1.0).max())
    check("A3b", "saturated + frozen kappa stays at machine precision (<= 1e-14)",
          max(sat.values()) <= 1e-14, "; ".join(f"{k}: {v:.4e}" for k, v in sat.items()))
    note("A3: the claim's '8.443e-02' for 'half of super 7' is not reproducible without the "
         "half-selection rule; both natural halves respect the stated >= 5e-2 bound.")

    # ---------------- PART B ------------------------------------------------
    print("\nPART B — prescription P1: arithmetic vs geometric window means")
    lg_geo0 = np.log(phat0[:, H_IDX]).mean(1) - np.log(phat0[:, T_IDX]).mean(1)
    lg_ari0 = np.log(phat0[:, H_IDX].mean(1)) - np.log(phat0[:, T_IDX].mean(1))
    ds0 = phat0[:, H_IDX].mean(1) - phat0[:, T_IDX].mean(1)
    sT0 = phat0[:, T_IDX].mean(1)

    sp_ari = sp_geo = 0.0
    sp_ari_rf = sp_geo_rf = 0.0
    sd_geo_fr = []
    sd_ds_fr = []
    sd_geo_rf = []
    sd_ds_rf = []
    taus = []
    tilt_l2 = []
    tilt_term = []
    tilt_ptw = []
    act_mv = pred_mv = resid = 0.0
    env_ok = 0
    flips = []
    deltas = np.zeros(NS)

    from scipy.stats import kendalltau

    for s, rows in enumerate(drops):
        fr = eb(C, rows, sup, trail, KAPPA0)
        lam = den0 / fr["den"]
        deltas[s] = lam[T_IDX].mean() / lam[H_IDX].mean() - 1.0
        ph = fr["phat"]
        ga = np.log(ph[:, H_IDX].mean(1)) - np.log(ph[:, T_IDX].mean(1))
        gg = np.log(ph[:, H_IDX]).mean(1) - np.log(ph[:, T_IDX]).mean(1)
        da, dg = ga - lg_ari0[rows], gg - lg_geo0[rows]
        sp_ari = max(sp_ari, float(da.max() - da.min()))
        sp_geo = max(sp_geo, float(dg.max() - dg.min()))
        sd_geo_fr.append(len(topk(lg_geo0[rows], rows) ^ topk(gg, rows)) / 50)
        dsA = ph[:, H_IDX].mean(1) - ph[:, T_IDX].mean(1)
        sd = len(topk(ds0[rows], rows) ^ topk(dsA, rows)) / 50
        sd_ds_fr.append(sd)
        if sd > 0:
            flips.append((s, sorted(topk(ds0[rows], rows) - topk(dsA, rows)),
                          sorted(topk(dsA, rows) - topk(ds0[rows], rows))))
        # Theorem 8' tilt law
        lamH, lamT = lam[H_IDX].mean(), lam[T_IDX].mean()
        sH, sT = phat0[rows][:, H_IDX].mean(1), phat0[rows][:, T_IDX].mean(1)
        pred_ds = lamH * sH - lamT * sT
        tilt_l2.append(float(np.linalg.norm(pred_ds - dsA) / np.linalg.norm(dsA)))
        tilt_term.append(float(max(np.abs(lamH * sH / ph[:, H_IDX].mean(1) - 1).max(),
                                   np.abs(lamT * sT / ph[:, T_IDX].mean(1) - 1).max())))
        tilt_ptw.append(float(np.abs(pred_ds / dsA - 1).max()))

        # full EB re-run, kappa refit
        rf = eb(C, rows, sup, trail, None)
        phB = rf["phat"]
        ggB = np.log(phB[:, H_IDX]).mean(1) - np.log(phB[:, T_IDX]).mean(1)
        gaB = np.log(phB[:, H_IDX].mean(1)) - np.log(phB[:, T_IDX].mean(1))
        dgB, daB = ggB - lg_geo0[rows], gaB - lg_ari0[rows]
        sp_geo_rf = max(sp_geo_rf, float(dgB.max() - dgB.min()))
        sp_ari_rf = max(sp_ari_rf, float(daB.max() - daB.min()))
        sd_geo_rf.append(len(topk(lg_geo0[rows], rows) ^ topk(ggB, rows)) / 50)
        dsB = phB[:, H_IDX].mean(1) - phB[:, T_IDX].mean(1)
        sd_ds_rf.append(len(topk(ds0[rows], rows) ^ topk(dsB, rows)) / 50)
        taus.append(float(kendalltau(lg_geo0[rows], ggB).statistic))
        # Theorem 8 first-order law (centred by the exact closure shift)
        c_exact = float(np.log(lam[H_IDX]).mean() - np.log(lam[T_IDX]).mean())
        mv = dgB - c_exact
        theta = m0[rows] / p0[rows]
        phi = (1 - theta) / (w0[rows][:, None] + (1 - w0[rows][:, None]) * theta)
        Wt = phat0[rows]
        pbar = ((phi[:, H_IDX] * Wt[:, H_IDX]).sum(1) / Wt[:, H_IDX].sum(1)
                - (phi[:, T_IDX] * Wt[:, T_IDX]).sum(1) / Wt[:, T_IDX].sum(1))
        pr = (rf["w"] - w0[rows]) * pbar
        act_mv = max(act_mv, float(np.abs(mv).max()))
        pred_mv = max(pred_mv, float(np.abs(pr).max()))
        resid = max(resid, float(np.abs(mv - pr).max()))
        Phi, dl = lg_geo0[rows], np.abs(pr)
        o = np.argsort(-Phi)
        if (Phi[o[:50]] - dl[o[:50]]).min() > (Phi[o[50:]] + dl[o[50:]]).max():
            env_ok += 1

    check("B6", "arithmetic window means: max region-spread of log g shift > 1e-3",
          sp_ari > 1e-3 and close(sp_ari, 9.0814e-03, 2e-3),
          f"measured {sp_ari:.4e} (claim 9.0814e-03)")
    check("B7", "geometric window means, kappa frozen: max region-spread < 1e-12",
          sp_geo < 1e-12,
          f"measured {sp_geo:.4e} (claim 1.0880e-14); arith/geo ratio = {sp_ari / sp_geo:.2e}")
    check("B7b", "P1 buys >= 11 orders of magnitude (kappa frozen)",
          sp_ari / sp_geo > 1e11, f"ratio {sp_ari / sp_geo:.3e} (claim ~8.3e+11)")
    check("B7c", "ADVERSARIAL: with kappa REFIT, P1 alone buys nothing",
          sp_geo_rf > 1e-3,
          f"refit spreads: arithmetic {sp_ari_rf:.4e}, geometric {sp_geo_rf:.4e} "
          f"(ratio {sp_ari_rf / sp_geo_rf:.3f}) -- geometric is NOT better; the 11 orders "
          f"of magnitude in B7 exist only jointly with P2 (freeze kappa)")

    # ---------------- PART C ------------------------------------------------
    print("\nPART C — top-k certificate and the decisive negative (Thms 7, 8, 8')")
    srt = np.sort(lg_geo0)[::-1]
    gamma50 = float(srt[49] - srt[50])
    check("C8", "gamma_50 for geometric log g = 1.3682e-02, certificate gamma_50 > 2*delta",
          close(gamma50, 1.3682e-02, 1e-4) and gamma50 > 2 * sp_geo,
          f"gamma_50 = {gamma50:.6e}; 2*delta = {2 * sp_geo:.4e}; "
          f"margin factor {gamma50 / (2 * sp_geo):.3e} (claim 6.2e+11)")
    check("C9", "geometric log g, kappa frozen: top-50 symmetric difference 0 on 0/64",
          max(sd_geo_fr) == 0.0,
          f"{sum(1 for v in sd_geo_fr if v > 0)}/64 drops churn; max symdiff {max(sd_geo_fr):.3f}")

    n_bad = sum(1 for v in sd_ds_fr if v > 0)
    det = f"{n_bad}/64 drops churn; max symdiff {max(sd_ds_fr):.3f}"
    if flips:
        s, out_, in_ = flips[0]
        det += (f"; super {s}: out={out_} in={in_}; "
                f"ds[884]={ds0[884]:.6e} sbarT[884]={sT0[884]:.6e} "
                f"ds[333]={ds0[333]:.6e} sbarT[333]={sT0[333]:.6e}")
    ok10 = (n_bad == 1 and max(sd_ds_fr) == 0.04 and flips
            and flips[0][0] == 3 and flips[0][1] == [884] and flips[0][2] == [333]
            and close(ds0[884], 9.1925e-04, 1e-4) and close(sT0[884], 1.0961e-03, 1e-4)
            and close(ds0[333], 9.1764e-04, 1e-4) and close(sT0[333], 1.7337e-03, 1e-4))
    check("C10", "share difference: top-50 churns on exactly 1/64 (super 3), 884 out / 333 in",
          bool(ok10), det)

    ratio = (ds0[884] - ds0[333]) / (sT0[884] - sT0[333])
    check("C10b", "Theorem 8' flip criterion: (ds_i-ds_j)/(sbarT_i-sbarT_j) between delta and 0",
          deltas[3] < ratio < 0,
          f"ratio = {ratio:.6e}; delta(super 3) = {deltas[3]:.6e}; "
          f"delta range over 64 drops [{deltas.min():.4e}, {deltas.max():.4e}] "
          f"(claim [-2.494e-02, 6.181e-03])")

    check("C11", "Theorem 8' tilt law: relative error small but NOT zero",
          max(tilt_l2) <= 5e-3,
          f"L2 rel err of Delta s' vector: max {max(tilt_l2):.4e}, median {np.median(tilt_l2):.4e}; "
          f"rel err of the sbar terms: max {max(tilt_term):.4e}; "
          f"POINTWISE rel err on Delta s': max {max(tilt_ptw):.4e}  <-- the law is NOT exact "
          f"(claim quotes 3.386e-04 without naming the norm)")
    note("C11: Theorem 8' says 'exactly', but lambda_t varies WITHIN each window "
         "(up to 1.8% across H, 3.0% across T), so Delta s' = lam_H sbar^H - lam_T sbar^T "
         "is a first-order approximation, not an identity. Its CONCLUSION (the flip) is exact.")

    check("C12", "full EB re-run, kappa refit, geometric log g: no top-50 churn; tau-b ~ 0.9999",
          max(sd_geo_rf) == 0.0 and np.median(taus) > 0.9999 and min(taus) > 0.999,
          f"symdiff {sum(1 for v in sd_geo_rf if v > 0)}/64 (max {max(sd_geo_rf):.3f}); "
          f"tau-b median {np.median(taus):.6f} min {min(taus):.6f} "
          f"(claim 0.999941 / 0.999469); Delta s under refit churns "
          f"{sum(1 for v in sd_ds_rf if v > 0)}/64")

    check("C13", "Theorem 8 first-order law explains the residual move to ~80%",
          close(pred_mv, act_mv, 0.10) and resid / act_mv < 0.30,
          f"max|actual centred| {act_mv:.4e} (claim 2.0582e-02); max|predicted| {pred_mv:.4e} "
          f"(claim 2.1019e-02); max second-order residual {resid:.4e} "
          f"({100 * resid / act_mv:.1f}% of actual; claim 4.6413e-03 = 22.6%)")
    check("C13b", "Theorem 7(c) envelope certificate holds on >= 59/64 drops",
          env_ok >= 59, f"holds on {env_ok}/64 with delta_r = |first-order predicted move| "
                        f"(claim 59/64)")
    eps = KAPPA0 / (trail + KAPPA0)
    check("C13c", "the two small factors of the mechanism",
          close(float(np.median(eps)), 0.0121, 0.02),
          f"median eps_r = kappa/(tau_r+kappa) = {np.median(eps):.6f} (claim 0.0121); "
          f"max |dkappa|/kappa = {np.abs(kaps - KAPPA0).max() / KAPPA0:.4f} (claim <= 6.1%)")

    # ---------------- PART D ------------------------------------------------
    print("\nPART D — amalgamation identities (Thms 9, 10, 12)")
    err9 = 0.0
    kl_sd = []
    for b in range(NS):
        idx = np.where(sup == b)[0]
        nB = idx.size
        xB = phat0[idx].sum(0)
        q = phat0[idx] / xB[None, :]
        KL = np.log(1.0 / nB) - np.log(q).mean(0)          # KL(u_B || q_B)
        err9 = max(err9, float(np.abs(np.log(xB) - (np.log(nB) + np.log(phat0[idx]).mean(0) + KL)).max()))
        kl_sd.append(float(np.diff(KL).std(ddof=0)))
    check("D14", "Thm 9: log x_B = log n_B + mean_r log x_r + KL(u_B||q_B) to <= 1e-13",
          err9 <= 1e-13, f"max |defect| = {err9:.4e} over 64 supers x {n} months (claim 3.553e-15)")

    def shannon(xv: np.ndarray) -> tuple[float, float, float, float]:
        xf = xv / xv.sum()
        Hf = float(-(xf * np.log(xf)).sum())
        xc = np.bincount(sup, weights=xf, minlength=NS)
        Hc = float(-(xc * np.log(xc)).sum())
        inner = 0.0
        for b in range(NS):
            idx = np.where(sup == b)[0]
            qq = xf[idx] / xf[idx].sum()
            inner += xc[b] * float(-(qq * np.log(qq)).sum())
        return Hf, Hc, inner, abs(Hf - (Hc + inner))

    e10 = max(shannon(v)[3] for v in (C.sum(1), phat0.mean(1), phat0[:, -1]))
    check("D15", "Thm 10: H_fine = H_coarse + sum_B x_B H(q_B) to <= 1e-13",
          e10 <= 1e-13, f"max |defect| over 3 compositions = {e10:.4e} (claim 1.776e-15)")

    Xf = phat0 / phat0.sum(0, keepdims=True)
    Hm = -(Xf * np.log(Xf)).sum(0)
    Xc = np.zeros((NS, n))
    np.add.at(Xc, sup, Xf)
    Hcm = -(Xc * np.log(Xc)).sum(0)
    Hf24, Hc24 = float(Hm[210:].mean()), float(Hcm[210:].mean())
    check("D16", "coarsening 1024->64 destroys ~2.7 nats; ~93% of N_eff is a resolution artefact",
          abs(Hf24 - Hc24 - 2.7197) < 0.02 and abs(Hf24 - 6.7478) < 0.02,
          f"mean monthly entropy over the last 24 months: H_fine {Hf24:.4f} (N_eff {math.exp(Hf24):.1f}), "
          f"H_coarse {Hc24:.4f} (N_eff {math.exp(Hc24):.1f}), loss {Hf24 - Hc24:.4f} "
          f"(claim 6.7478 / 852.2, 4.0280 / 56.2, 2.7197); "
          f"1 - N_eff_coarse/N_eff_fine = {1 - math.exp(Hc24 - Hf24):.3f}")
    note("D16: the claim does not say which composition the entropies are taken on; the closest "
         "natural convention reproduces them to ~1e-3 nats. The identity D15 is exact regardless.")

    sH0, sT0b = phat0[:, H_IDX].mean(1), phat0[:, T_IDX].mean(1)
    rat, dif = sH0 / sT0b, sH0 - sT0b
    v_rat = v_dif_hi = v_dif_lo = 0
    for b in range(NS):
        idx = np.where(sup == b)[0]
        rB = sH0[idx].sum() / sT0b[idx].sum()
        if rB < rat[idx].min() - 1e-15 or rB > rat[idx].max() + 1e-15:
            v_rat += 1
        dB = dif[idx].sum()
        if dB > dif[idx].max():
            v_dif_hi += 1
        elif dB < dif[idx].min():
            v_dif_lo += 1
    check("D17", "Thm 12: ratio 0/64 mediant violations, difference 55/64 range violations",
          v_rat == 0 and v_dif_hi + v_dif_lo == 55,
          f"share ratio {v_rat}/64; share difference {v_dif_hi + v_dif_lo}/64 "
          f"(= {v_dif_hi} above max + {v_dif_lo} below min)")
    note("D17: the claim's stated mechanism -- 'Delta s_B strictly exceeds max_r Delta s_r whenever "
         "two members share a sign' -- accounts for only 14 of the 55; the other 41 fall BELOW the "
         "block minimum. The count is right, the stated reason is not.")

    Z0 = np.load(ROOT / "artifacts" / "linkage.f64.npy")
    Psi0 = sbp_basis(Z0)
    dlx = np.diff(np.log(phat0), axis=1)
    db = Psi0 @ dlx
    med_kl, med_db = float(np.median(kl_sd)), float(np.median(db.std(axis=1, ddof=0)))
    check("D18", "the amalgamation defect is time-VARYING (blocks are not internally static)",
          med_kl > 0.03 and abs(med_kl / med_db - 0.132) < 0.02,
          f"median sd_t[Delta KL_B] = {med_kl:.5f} (claim 0.06089); "
          f"median sd_t[Delta b_v] = {med_db:.5f} (claim 0.45969); ratio {med_kl / med_db:.3f} "
          f"(claim 0.132)")

    hx, hy = math.log(3), math.log(2)
    hax = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    check("D19", "Counterexample 11: one merge on three atoms reverses the diversity order",
          hx > hy and hax < hy,
          f"fine H(x)={hx:.4f} > H(y)={hy:.4f}; coarse H(Ax)={hax:.4f} < H(Ay)={hy:.4f}")

    # ---------------- PART E ------------------------------------------------
    print("\nPART E — hierarchy (Thms 14, 15, 16, 17) — the sharpest claim")
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage
    from sklearn.metrics import adjusted_rand_score

    TARGET = 236.768651962737
    cent = np.load(ROOT / "artifacts" / "centroids.f32.npy").astype(np.float64)
    clr = dlx - dlx.mean(axis=0, keepdims=True)
    tr_clr = float(clr.var(axis=1, ddof=1).sum())
    S = np.cov(dlx, ddof=1)
    pair = float((2 * K * np.diag(S).sum() - 2 * S.sum()) / (2 * K))

    sums = {}
    stats = {}
    for meth in ("ward", "complete", "weighted", "average", "median", "centroid", "single"):
        Zm = linkage(cent, method=meth)
        a, r2, vh, cv, cr, D = alpha_h(Zm, dlx)
        stats[meth] = (a, r2, vh, cv, cr)
        sums[meth] = float(D.sum())

    def rand_tree(seed: int) -> np.ndarray:
        rg = np.random.default_rng(seed)
        Z = np.zeros((K - 1, 4))
        active = list(range(K))
        size = {i: 1 for i in range(K)}
        for v in range(K - 1):
            i, j = rg.choice(len(active), size=2, replace=False)
            a_, b_ = active[i], active[j]
            for q in sorted([i, j], reverse=True):
                active.pop(q)
            Z[v] = [a_, b_, v + 1.0, size[a_] + size[b_]]
            size[K + v] = size[a_] + size[b_]
            active.append(K + v)
        return Z

    for sd in range(3):
        sums[f"random{sd}"] = float((sbp_basis(rand_tree(sd)) @ dlx).var(axis=1, ddof=1).sum())

    devs = {k: abs(v - TARGET) / TARGET for k, v in sums.items()}
    lo, hi = min(sums.values()), max(sums.values())
    # 1023-term float64 sums: one ulp at 236.77 is 2.8e-14, so ~1e-14 RELATIVE
    # (i.e. agreement to >= 13 significant figures) is the tightest honest bar here.
    # The claim's own stated precision is "identical to 12 significant figures".
    check("E20", "sum_v D_v is hierarchy-FREE and equals 236.768651962737 to >=12 s.f.",
          max(devs.values()) <= 1e-14
          and abs(tr_clr - TARGET) / TARGET <= 1e-14 and abs(pair - tr_clr) / tr_clr <= 1e-14
          and f"{lo:.12g}" == f"{hi:.12g}" == f"{TARGET:.12g}",
          f"all 10 bases agree to 12 s.f. at {lo:.12g}; full spread [{lo!r}, {hi!r}]; "
          f"max relative deviation over 7 linkage rules + 3 random binary trees = "
          f"{max(devs.values()):.3e} (~{max(devs.values()) / 2.22e-16:.0f} ulp, expected for a "
          f"1023-term sum); tr Sigma_clr = {tr_clr!r}; "
          f"(1/2K) sum_ij Var[Dlog(x_i/x_j)] = {pair!r} (rel dev vs tr {abs(pair - tr_clr) / tr_clr:.3e})")

    claim21 = {"ward": -0.3486, "complete": -0.5427, "weighted": -0.7893, "average": -0.8129,
               "median": -1.0825, "centroid": -1.1996, "single": -1.5542}
    worst = max(abs(stats[k][0] - v) for k, v in claim21.items())
    aa = np.array([stats[k][0] for k in claim21])
    check("E21", "alpha_h moves 4.46x across linkage rules on the SAME corpus",
          worst <= 5e-4 and close(aa.min() / aa.max(), 4.46, 0.01),
          "; ".join(f"{k} {stats[k][0]:+.4f}" for k in claim21) +
          f"  | max deviation from claim {worst:.1e}; spread "
          f"{aa.min() / aa.max():.3f}x; ward R^2 {stats['ward'][1]:.3f}")

    vh = np.array([stats[k][2] for k in claim21])
    cr = np.array([stats[k][4] for k in claim21])
    check("E22", "the alpha_h spread lives in the DENOMINATOR Var(log2 h), not the correlation",
          close(vh.max() / vh.min(), 14.1, 0.02) and cr.max() / cr.min() < 1.25,
          f"Var(log2 h) in [{vh.min():.4f}, {vh.max():.4f}] ({vh.max() / vh.min():.1f}x); "
          f"corr(log D, log h) in [{cr.min():.4f}, {cr.max():.4f}] ({cr.max() / cr.min():.2f}x) "
          f"(claim 14.1x vs 1.20x)")

    D0 = np.sort((Psi0 @ dlx).var(axis=1, ddof=1))[::-1]
    ev = np.linalg.eigvalsh(np.cov(clr, ddof=1))[::-1]
    viol = float((np.cumsum(D0) - np.cumsum(ev[:K - 1])).max())
    check("E24", "Schur-Horn: D majorized by spec(Sigma); Ward realises only ~33% of lambda_max",
          D0[0] <= ev[0] and viol <= 1e-12,
          f"max_v D_v = {D0[0]:.4f} <= lambda_max = {ev[0]:.4f} "
          f"(claim 2.0354 <= 6.2132); max partial-sum violation {viol:.3e}; "
          f"max_v D_v / lambda_max = {D0[0] / ev[0]:.3f}")

    u0 = cophenet(Z0)
    c0 = fcluster(Z0, t=NS, criterion="maxclust")
    a0 = stats["ward"][0]
    jit = {}
    for sig in (0.05, 0.50):
        dis, ari, als = [], [], []
        for sd in range(25):
            rg = np.random.default_rng(sd)
            Zp = linkage(cent + rg.normal(0.0, sig, cent.shape), method="ward")
            dis.append(float(np.abs(cophenet(Zp) - u0).max() / u0.max()))
            ari.append(adjusted_rand_score(c0, fcluster(Zp, t=NS, criterion="maxclust")))
            als.append(alpha_h(Zp, dlx)[0])
        jit[sig] = (float(np.median(dis)), float(np.median(ari)),
                    float(np.mean(als)), float(np.std(als, ddof=1)))
    d5, r5, m5, s5 = jit[0.05]
    d50, r50, m50, s50 = jit[0.50]
    check("E23", "Thm 17: cophenetic distortion ~0.97 / ARI64 ~0.62 yet alpha_h moves <2%",
          d5 > 0.9 and abs(r5 - 0.6225) < 0.03 and abs(m5 - a0) / abs(a0) < 0.02,
          f"sigma=0.05 (25 seeds): distortion median {d5:.4f} (claim 0.9747), ARI64 median "
          f"{r5:.4f} (claim 0.6225), alpha_h {m5:+.4f} +/- {s5:.4f} (claim -0.3443 +/- 0.0025) "
          f"= {100 * abs(m5 - a0) / abs(a0):.1f}% from {a0:+.4f}; "
          f"sigma=0.50: ARI64 {r50:.4f} (claim 0.393), alpha_h {m50:+.4f} (claim -0.4169). "
          f"Linkage rule moves alpha_h 4.46x; input noise moves it <2%.")

    # E25 — matched-N multinomial null, as pre-registered
    M_t = C.sum(axis=0)
    q_time = (C / M_t[None, :]).mean(axis=1)
    q_time /= q_time.sum()
    q_pool = C.sum(axis=1) / C.sum()
    bases = {mth: linkage(cent, method=mth) for mth in ("ward", "average", "single")}
    NREP = 25
    null = {}
    for qname, q in (("time-averaged share (as pre-registered)", q_time),
                     ("pooled share", q_pool)):
        acc = {mth: [] for mth in bases}
        for sd in range(NREP):
            rg = np.random.default_rng(sd)
            Cs = np.empty((K, n))
            for t in range(n):
                Cs[:, t] = rg.multinomial(int(M_t[t]), q)
            tr_s = Cs[:, n - 12:n].sum(axis=1)
            r = eb(Cs, allrows, sup, tr_s, None)
            dl_s = np.diff(np.log(r["phat"]), axis=1)
            for mth, Zm in bases.items():
                acc[mth].append(alpha_h(Zm, dl_s)[0])
        null[qname] = {mth: (float(np.mean(v)), float(np.std(v, ddof=1))) for mth, v in acc.items()}

    zs = {}
    lines = []
    for qname, dd in null.items():
        for mth, (mu, sg) in dd.items():
            z = (stats[mth][0] - mu) / sg
            zs[(qname, mth)] = z
            lines.append(f"[{qname}] {mth}: obs {stats[mth][0]:+.4f} vs null {mu:+.4f} +/- {sg:.4f}, z = {z:+.1f}")
    pre = null["time-averaged share (as pre-registered)"]
    check("E25", "the multinomial-null rejection survives under every hierarchy; no VALUE does",
          all(z < -6 for z in zs.values())
          and abs(pre["ward"][0] + 0.1450) < 0.004
          and abs(zs[("time-averaged share (as pre-registered)", "ward")] + 33.7) < 1.5,
          " | ".join(lines) + f"  (claim: ward -0.1450 +/- 0.0060, z -33.7; average -24.9; single -6.6)")
    note("E25: the answer's parenthetical '(The Ward row reproduces the pre-registered "
         "-0.1151 +/- 0.0072)' is FALSE. Under the pre-registration's own q (time-averaged share) "
         f"the null mean is {pre['ward'][0]:+.4f}; the -0.1151 figure corresponds to a DIFFERENT q "
         f"(pooled share), for which this run gives {null['pooled share']['ward'][0]:+.4f}. "
         "The two differ by ~5 null sd. The rejection is unaffected either way.")

    # ---------------- summary ----------------------------------------------
    npass = sum(1 for _, _, ok, _ in _RESULTS if ok)
    print("\n" + "=" * 78)
    for tag, desc, ok, _ in _RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {tag:<5s} {desc}")
    print("-" * 78)
    if _NOTES:
        print("  Qualifications recorded (none of these is a failed check):")
        for t in _NOTES:
            print(f"    - {t}")
        print("-" * 78)
    print(f"  {npass}/{len(_RESULTS)} checks PASS   ({time.time() - t_start:.0f}s)")
    print("=" * 78)
    return 0 if npass == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
