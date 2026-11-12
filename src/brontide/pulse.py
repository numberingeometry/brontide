"""Reorganization pulse: how fast the composition of research turns.

A composition p (shares of papers across fields) maps to the unit sphere by
p -> sqrt(p). The angle between two mapped compositions,

    theta = arccos(sum_k sqrt(p_k q_k)),

is the Bhattacharyya angle (half the Fisher-Rao distance). It is zero when every
field kept its share and grows when fields' relative growth factors are unequal:
1 - cos(theta) = 1 - (1 + CV_p(sqrt(q/p))^2)^(-1/2). It depends only on relative
growth, so a common change in publication volume does not move it.

The squared Hellinger distance 1 - cos(theta) = sum_k c_k splits exactly into
nonnegative field terms c_k = (sqrt(q_k) - sqrt(p_k))^2 / 2. Counts add sampling
noise that inflates every c_k. Each window is treated as a multinomial draw from
its observed composition, and the exact bootstrap expectation of c_k (it needs
only binomial marginals) gives a deterministic first-order bias correction. The
corrected terms still sum to the corrected total, but small ones can be negative.

Nested partitions split exactly on the 1 - cos scale, not in degrees:
1 - cos(theta_micro) = (1 - cos(theta_territory)) + sum_J sqrt(P_J Q_J) (1 - cos(theta_J)).
The within-territory angle reported here is arccos of one minus that last sum.
"""
from __future__ import annotations

import math

import numpy as np


def expected_sqrt_share(n: int, prob: float, width: float = 12.0) -> float:
    """E[sqrt(X / n)] for X ~ Binomial(n, prob), exact up to ~1e-30 of tail mass."""
    if n <= 0:
        raise ValueError("n must be positive")
    if prob <= 0.0:
        return 0.0
    if prob >= 1.0:
        return 1.0
    mu, sd = n * prob, math.sqrt(n * prob * (1.0 - prob))
    lo = max(0, math.floor(mu - width * sd - 10))
    hi = min(n, math.ceil(mu + width * sd + 10))
    x = np.arange(lo, hi + 1, dtype=float)
    first = (math.lgamma(n + 1) - math.lgamma(lo + 1) - math.lgamma(n - lo + 1)
             + lo * math.log(prob) + (n - lo) * math.log1p(-prob))
    steps = np.log((n - x[:-1]) / (x[:-1] + 1)) + math.log(prob) - math.log1p(-prob)
    logpmf = np.concatenate([[first], first + np.cumsum(steps)])
    return float((np.exp(logpmf) * np.sqrt(x / n)).sum())


def corrected_terms(before: np.ndarray, after: np.ndarray) -> dict[str, np.ndarray | float]:
    """Raw and bias-corrected Hellinger terms between two count vectors."""
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    na, nb = round(a.sum()), round(b.sum())
    if na <= 0 or nb <= 0:
        raise ValueError("both windows need papers")
    p, q = a / na, b / nb
    raw = 0.5 * (np.sqrt(q) - np.sqrt(p)) ** 2
    ep = np.array([expected_sqrt_share(na, float(v)) for v in p])
    eq = np.array([expected_sqrt_share(nb, float(v)) for v in q])
    corrected = 2.0 * raw - (0.5 * (p + q) - ep * eq)
    pooled = (a + b) / (na + nb)
    epp = np.array([expected_sqrt_share(na, float(v)) for v in pooled])
    eqp = np.array([expected_sqrt_share(nb, float(v)) for v in pooled])
    floor = float((pooled - epp * eqp).sum())
    return {"raw": raw, "corrected": corrected, "floor": floor, "p": p, "q": q}


def angle_deg(h2: float) -> float:
    """Angle between square-root compositions from 1 - cos(angle), clipped at zero."""
    return math.degrees(math.acos(min(1.0, max(0.0, 1.0 - h2))))


def trailing_sums(counts: np.ndarray, window: int = 12) -> np.ndarray:
    """Trailing-window sums along the month axis (columns)."""
    c = np.cumsum(np.pad(np.asarray(counts, dtype=np.int64), ((0, 0), (1, 0))), axis=1)
    out = np.empty((counts.shape[0], counts.shape[1]), dtype=np.int64)
    for t in range(counts.shape[1]):
        out[:, t] = c[:, t + 1] - c[:, max(0, t - window + 1)]
    return out


def build_pulse(territory_counts: np.ndarray, micro_counts: np.ndarray, months: list[str],
                years: list[int], movers: int = 6) -> dict:
    """Year-over-year turning of the trailing-year composition, at two resolutions.

    For each month t with two complete, disjoint 12-month windows, compare the
    window ending at t with the window ending at t - 12. Returns angles in degrees
    (bias-corrected, raw and the no-change sampling floor), the leading territory
    terms, and a flat projection of the calendar-year trajectory on the sphere.
    """
    tc = np.asarray(territory_counts, dtype=np.int64)
    mc = np.asarray(micro_counts, dtype=np.int64)
    if not np.array_equal(tc.sum(axis=0), mc.sum(axis=0)):
        raise ValueError("Territory and microregion counts disagree")
    window = 12
    t_trail, m_trail = trailing_sums(tc, window), trailing_sums(mc, window)
    ends = list(range(2 * window - 1, len(months)))
    out_months, between, total, within, between_raw, total_raw, between_floor, total_floor, leaders = (
        [], [], [], [], [], [], [], [], [])
    for t in ends:
        s = t - window
        tt = corrected_terms(t_trail[:, s], t_trail[:, t])
        mt = corrected_terms(m_trail[:, s], m_trail[:, t])
        hb, hm = float(tt["corrected"].sum()), float(mt["corrected"].sum())
        out_months.append(months[t])
        between.append(angle_deg(hb)); total.append(angle_deg(hm)); within.append(angle_deg(hm - hb))
        between_raw.append(angle_deg(float(tt["raw"].sum())))
        total_raw.append(angle_deg(float(mt["raw"].sum())))
        between_floor.append(angle_deg(tt["floor"])); total_floor.append(angle_deg(mt["floor"]))
        terms = np.asarray(tt["corrected"])
        order = np.argsort(-terms, kind="stable")[:movers]
        leaders.append([[int(k), round(float(100 * terms[k] / hb), 2) if hb > 0 else 0.0,
                         1 if tt["q"][k] > tt["p"][k] else -1] for k in order])

    # Calendar-year trajectory of sqrt-compositions, projected on its two
    # principal axes. The projection only shortens distances on the sphere.
    year_counts = np.stack([tc[:, [i for i, m in enumerate(months) if int(m[:4]) == y]].sum(axis=1)
                            for y in years])
    roots = np.sqrt(year_counts / year_counts.sum(axis=1, keepdims=True))
    centered = roots - roots.mean(axis=0)
    _, sing, vt = np.linalg.svd(centered, full_matrices=False)
    # Fix the arbitrary sign of each axis: the latest year lies on the positive side.
    axes = []
    for j in range(2):
        if centered[-1] @ vt[j] < 0:
            vt[j] = -vt[j]
        axes.append(vt[j])
    coords = centered @ np.stack(axes).T
    explained = (sing ** 2 / np.sum(sing ** 2))[:2]
    steps = []
    for i in range(1, len(years)):
        steps.append(angle_deg(float(corrected_terms(year_counts[i - 1], year_counts[i])["corrected"].sum())))
    direct = angle_deg(float(corrected_terms(year_counts[0], year_counts[-1])["corrected"].sum()))
    axis_ends = [{"negative": [int(k) for k in np.argsort(a, kind="stable")[:3]],
                  "positive": [int(k) for k in np.argsort(-a, kind="stable")[:3]]} for a in axes]
    # Prominent local maxima of the territory angle, for orientation labels only.
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(np.round(np.asarray(between), 4), prominence=1.0, distance=24)
    highlights = [{"month": out_months[i], "index": int(i), "leader": leaders[i][0][0]} for i in peaks]
    r = lambda v: [round(float(x), 4) for x in v]
    return {
        "months": out_months, "window_months": window,
        "comparison": "trailing 12 months versus the preceding, non-overlapping 12 months",
        "highlights": highlights,
        "territories": {"angle_deg": r(between), "raw_angle_deg": r(between_raw),
                        "floor_angle_deg": r(between_floor), "leaders": leaders},
        "microregions": {"angle_deg": r(total), "raw_angle_deg": r(total_raw),
                         "floor_angle_deg": r(total_floor)},
        "within": {"angle_deg": r(within),
                   "definition": "arccos(1 - [(1 - cos microregion angle) - (1 - cos territory angle)]), both corrected"},
        "trajectory": {"years": list(years), "x": r(coords[:, 0]), "y": r(coords[:, 1]),
                       "explained": r(explained), "axes": axis_ends,
                       "step_angle_deg": r(steps), "path_angle_deg": round(float(sum(steps)), 3),
                       "direct_angle_deg": round(direct, 3)},
    }
