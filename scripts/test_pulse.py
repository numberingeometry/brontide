"""Checks for the pulse geometry against simple reference computations.

    python scripts/test_pulse.py
"""
from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np

from brontide.pulse import angle_deg, corrected_terms, expected_sqrt_share, trailing_sums

ROOT = Path(__file__).resolve().parents[1]
EXPLORER = ROOT / "artifacts/explorer"


def test_binomial_moment() -> None:
    for n, p in ((7, .3), (60, .01), (400, .5), (1500, .002)):
        x = np.arange(n + 1)
        logpmf = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                  + (k * math.log(p) if k else 0.0) + (n - k) * math.log1p(-p) for k in x]
        brute = float((np.exp(logpmf) * np.sqrt(x / n)).sum())
        assert abs(brute - expected_sqrt_share(n, p)) < 1e-12


def test_terms_and_correction() -> None:
    rng = np.random.default_rng(7)
    p = rng.gamma(.6, size=200); p /= p.sum()
    q = p * np.exp(rng.normal(scale=.2, size=200)); q /= q.sum()
    truth = 1 - np.sqrt(p * q).sum()
    raw, cor = [], []
    for _ in range(120):
        t = corrected_terms(rng.multinomial(20000, p), rng.multinomial(24000, q))
        assert abs(t["raw"].sum() - (1 - np.sqrt(t["p"] * t["q"]).sum())) < 1e-12
        raw.append(t["raw"].sum()); cor.append(t["corrected"].sum())
    assert abs(np.mean(cor) - truth) < abs(np.mean(raw) - truth) / 8


def test_published_pulse() -> None:
    pulse = json.loads((EXPLORER / "pulse.json").read_text())
    regions = json.loads((EXPLORER / "regions.json").read_text())
    micro = json.loads((EXPLORER / "microregions.json").read_text())
    months = regions["months"]
    tc = trailing_sums(np.array([r["raw_counts"] for r in regions["regions"]]))
    mc = trailing_sums(np.array([r["raw_counts"] for r in micro["regions"]]))
    for k, month in enumerate(pulse["months"]):
        t = months.index(month)
        for counts, key in ((tc, "territories"), (mc, "microregions")):
            a, b = counts[:, t - 12] / counts[:, t - 12].sum(), counts[:, t] / counts[:, t].sum()
            direct = math.degrees(math.acos(min(1.0, float(np.sqrt(a * b).sum()))))
            assert abs(direct - pulse[key]["raw_angle_deg"][k]) < 6e-5, (month, key)
        leaders = pulse["territories"]["leaders"][k]
        assert all(x[1] >= y[1] for x, y in pairwise(leaders))
        assert sum(x[1] for x in leaders) <= 100.5
    assert pulse["months"][0] == "2008-12" and pulse["months"][-1] == months[-1]
    assert all(a >= 0 for a in pulse["territories"]["angle_deg"] + pulse["microregions"]["angle_deg"])
    assert angle_deg(0.0) == 0.0


if __name__ == "__main__":
    test_binomial_moment(); test_terms_and_correction(); test_published_pulse()
    print("PASS: binomial moments, exact term sums, bias correction, and published pulse angles")
