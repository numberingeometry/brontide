"""Stage 8 — Out-of-time validation . THE headline artifact.

Three replicates at ``config.VALIDATION_CUTOFFS`` = (2017-12, 2020-12, 2022-12):
one quiet window + two regime-shift stress tests (COVID, the LLM explosion).

**Lookahead control:** PCA + k-means fit on papers ≤ T0 only; later papers
assigned nearest-centroid; forecasts from ≤ T0 data; novelty stream reported per
window. κ and the novelty quantile τ are ALSO fit on ≤ T0 (per fold-train-end).
Accepted-and-disclosed leakages: SPECTER2 weights trained on post-T0 literature;
UMAP display fit on all data (display only).

**Scoring** against realized *smoothed*-share growth over (T0, T0+24m] (same
stabilization both sides): **precision@50** vs top-decile realized growth +
**Spearman**(ĝ, realized) + **CSI** (critical success index — K&B-comparable) —
**headline comparison vs the SmoothedMomentum baseline** (persistence and
seasonal-naive also reported).

**Pre-registered negative criterion:** if the roster fails to beat smoothed
momentum on precision@50 in ≥ ``NEGATIVE_CRITERION_MIN_WINS`` of 3 replicates,
the writeup says so plainly ("region-level emergence is momentum-dominated at
24m") — the harness, map, and novelty stream remain the résumé artifact.

Sprint 1 ships the pure scoring functions below (unit-tested). ``run`` (the masked
Run B orchestration) is Sprint 2 — it needs the complete corpus (the ≤T0 re-fits require the
embeddings memmap).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


# --- scoring primitives (pure, unit-tested) ---------------------------------
def precision_at_k(pred_ranked: np.ndarray, realized_positive: set, k: int = config.TOP_N) -> float:
    """Fraction of the top-``k`` predicted regions that are in the realized-positive
    set (e.g. the realized top-decile of growth). ``pred_ranked`` = region ids sorted
    most-emerging first."""
    top = pred_ranked[:k]
    if len(top) == 0:
        return 0.0
    return float(sum(1 for r in top if r in realized_positive) / len(top))


def csi(pred_set: set, realized_set: set) -> float:
    """Critical Success Index = hits / (hits + misses + false alarms) = |P∩R| / |P∪R|
    (K&B-comparable). 1.0 = perfect, 0.0 = disjoint. Empty ∪ → 0.0."""
    p, r = set(pred_set), set(realized_set)
    union = p | r
    return float(len(p & r) / len(union)) if union else 0.0


def spearman_growth(pred_growth: np.ndarray, realized_growth: np.ndarray) -> float:
    """Spearman correlation of predicted vs realized growth over the scored regions."""
    from scipy.stats import spearmanr
    m = np.isfinite(pred_growth) & np.isfinite(realized_growth)
    if m.sum() < 3:
        return 0.0
    val = spearmanr(pred_growth[m], realized_growth[m]).statistic
    return float(val) if np.isfinite(val) else 0.0


def top_decile(growth: np.ndarray, eligible: np.ndarray | None = None) -> set:
    """Region ids in the top decile of ``growth`` (restricted to ``eligible`` if given).
    The realized-positive reference set for precision@50."""
    idx = np.arange(len(growth))
    if eligible is not None:
        idx = idx[eligible]
    g = growth[idx]
    finite = np.isfinite(g)
    idx, g = idx[finite], g[finite]
    if len(idx) == 0:
        return set()
    thresh = np.quantile(g, 0.9)
    return set(int(r) for r in idx[g >= thresh])


def evaluate_negative_criterion(model_wins: int, n_replicates: int = 3,
                                min_wins: int = config.NEGATIVE_CRITERION_MIN_WINS) -> dict:
    """Did the roster beat SmoothedMomentum on precision@50 in ≥ ``min_wins`` of
    ``n_replicates`` cutoffs? Returns the verdict + the pre-registered sentence."""
    beat = model_wins >= min_wins
    return {
        "wins": int(model_wins), "of": int(n_replicates), "min_wins": int(min_wins),
        "beats_baseline": bool(beat),
        "verdict": ("roster beats SmoothedMomentum" if beat
                    else "region-level emergence is momentum-dominated at 24m"),
    }


def _month_add(ym: str, k: int) -> str:
    y, m = map(int, ym.split("-"))
    o = y * 12 + (m - 1) + k
    return f"{o // 12:04d}-{o % 12 + 1:02d}"


def _score_cutoff(T0: str, shares_path: Path, forecasts_path: Path) -> tuple[dict, dict]:
    """Score one cutoff: predicted ĝ (per model) vs realized (T0, T0+24m] growth.

    Requires ``shares_path`` to have been produced with ``score_through >= T0+24m``
    (so the realized months exist). Realized growth mirrors ĝ exactly — realized
    smoothed-share avg over horizon months 19–24, over the SAME trailing-12m-at-T0
    denominator. Returns (per-model {precision50, spearman, csi}, info). Pure + testable.
    """
    import numpy as np
    import pandas as pd

    from . import emergence
    lo_h, hi_h = config.EMERGENCE_HORIZON_MONTHS
    K = config.N_REGIONS
    sh = pd.read_parquet(shares_path)
    smonths = sorted(sh["month"].unique())
    S = sh.pivot(index="region", columns="month", values="share_smoothed").reindex(range(K))[smonths].to_numpy(np.float64)
    C = sh.pivot(index="region", columns="month", values="count").reindex(range(K))[smonths].to_numpy(np.float64)
    n_le = sum(mn <= T0 for mn in smonths)
    if n_le + hi_h > len(smonths):
        raise RuntimeError(f"realized window T0+{hi_h}m exceeds shares at T0={T0} "
                           f"(need score_through>={_month_add(T0, hi_h)})")

    trailing12_share = S[:, n_le - 12:n_le].mean(axis=1)
    trailing12_count = C[:, n_le - 12:n_le].sum(axis=1)
    eligible = trailing12_count >= config.ELIGIBILITY_FLOOR
    realized_share = S[:, n_le + lo_h - 1:n_le + hi_h].mean(axis=1)
    realized_g = realized_share / np.where(trailing12_share > 0, trailing12_share, np.nan)

    realized_pos = top_decile(realized_g, eligible)                       # precision@50 reference
    elig = np.where(eligible)[0]
    realized_top50 = set(int(r) for r in elig[np.argsort(-np.nan_to_num(realized_g[elig]))[:config.TOP_N]])

    fc = pd.read_parquet(forecasts_path)
    scores = {}
    for model in sorted(fc["model"].unique()):
        fm = fc[fc["model"] == model]
        fmonths = sorted(fm["month"].unique())
        fmeans = fm.pivot(index="region", columns="month", values="mean").reindex(range(K))[fmonths].to_numpy(np.float64)
        gh = emergence.g_hat(fmeans, trailing12_share)
        gh_e = np.where(eligible, np.nan_to_num(gh, nan=-np.inf), -np.inf)
        pred_ranked = np.argsort(-gh_e)                                   # eligible, most-emerging first
        pred_emerging = set(int(r) for r in pred_ranked[:config.TOP_N] if eligible[r] and gh[r] >= config.G_EMERGING)
        scores[model] = {
            "precision50": precision_at_k(pred_ranked[:config.TOP_N], realized_pos, config.TOP_N),
            "spearman": spearman_growth(gh[eligible], realized_g[eligible]),
            "csi": csi(pred_emerging, realized_top50),
        }
    return scores, {"eligible": int(eligible.sum()), "realized_positive": len(realized_pos)}


def _results_table(per_cutoff: dict, nc: dict, prehash: str, date: str) -> str:
    lines = [f"_Scored {date}. Pre-registration content-hash (thresholds+clarification): `{prehash}`._", "",
             "| cutoff | model | precision@50 | Spearman | CSI |", "|---|---|---|---|---|"]
    for T0, d in per_cutoff.items():
        for model, s in d["scores"].items():
            tag = " **(selected)**" if model == d["selected"] else (" _(baseline)_" if model == config.BASELINE_MODEL else "")
            lines.append(f"| {T0} | {model}{tag} | {s['precision50']:.3f} | {s['spearman']:.3f} | {s['csi']:.3f} |")
    lines += ["", f"**Negative criterion:** roster beat {config.BASELINE_MODEL} on precision@50 in "
                  f"**{nc['wins']}/{nc['of']}** cutoffs (need ≥{nc['min_wins']}). "
                  f"Verdict: **{nc['verdict']}**."]
    return "\n".join(lines)


def run(cutoffs: tuple[str, ...] = config.VALIDATION_CUTOFFS,
        out: Path = config.ARTIFACTS_DIR / "validation.md") -> Path:
    """Run the full out-of-time harness across cutoffs, append the results table to
    the (already pre-registered) validation.md, and evaluate the negative criterion.

    IMPORTANT: the thresholds section of validation.md must already be committed
    (dated, content-hashed) BEFORE this scoring run — pre-registration discipline.

    Full validation: the per-cutoff ≤T0 re-fits (steps 1–2) need the embeddings memmap.
    Every cutoff uses cutoff-scoped paths — the production all-data memmaps/centroids
    are never reused, so PCA, k-means, κ, and τ are all fit on ≤T0 only.
    """
    import datetime
    import hashlib
    import json

    import numpy as np
    import pyarrow.parquet as pq

    from . import emergence, forecast, reduce, regions, series

    text = out.read_text(encoding="utf-8")
    prehash = hashlib.sha256(text.split("## Results")[0].encode("utf-8")).hexdigest()[:16]
    months = np.array(pq.read_table(config.RAW_PARQUET, columns=["month"]).column("month").to_pylist())

    per_cutoff: dict = {}
    for T0 in cutoffs:
        cdata = config.DATA_DIR / "cutoffs" / T0
        cart = config.ARTIFACTS_DIR / "cutoffs" / T0
        cdata.mkdir(parents=True, exist_ok=True)
        cart.mkdir(parents=True, exist_ok=True)
        fit_mask = months <= T0                                           # bool over all papers (row-aligned)
        t24 = _month_add(T0, config.HORIZON_MONTHS)
        pca_mm = cdata / "pca50.f32.memmap"

        reduce.run_pca(fit_mask=fit_mask, out=pca_mm)                     # <=T0 fit, transform all
        regions.run(pca_memmap=pca_mm, fit_mask=fit_mask, out_dir=cart)   # <=T0 fit, assign all
        shares = cart / "region_shares.parquet"
        series.run_shares(cutoff=T0, regions_dir=cart, score_through=t24, out=shares)
        series.run_novelty(cutoff=T0, regions_dir=cart, pca_memmap=pca_mm, out=cart / "novelty_stream.parquet")
        fcp = cart / "forecasts.parquet"
        forecast.run(cutoff=T0, shares=shares, out=fcp)
        emergence.run(forecasts=fcp, shares=shares, cutoff=T0, out=cart / "emergence.parquet")

        scores, info = _score_cutoff(T0, shares, fcp)
        selected = json.loads((fcp.with_suffix(fcp.suffix + ".meta.json")).read_text())["selected_model"]
        per_cutoff[T0] = {"selected": selected, "scores": scores, "info": info}

    wins = sum(1 for d in per_cutoff.values()
               if d["scores"].get(d["selected"], {}).get("precision50", 0.0)
               > d["scores"].get(config.BASELINE_MODEL, {}).get("precision50", 0.0))
    nc = evaluate_negative_criterion(wins, len(cutoffs))

    table = _results_table(per_cutoff, nc, prehash, datetime.date.today().isoformat())
    placeholder = "_(appended by `validate.run()` at scoring time — mid-August. Empty until then.)_"
    out.write_text(text.replace(placeholder, table), encoding="utf-8")
    print(f"\nvalidate.run -> {out}\n  cutoffs={list(cutoffs)}\n  {nc['verdict']} ({nc['wins']}/{nc['of']})\n"
          f"  pre-registration hash: {prehash}")
    return out
