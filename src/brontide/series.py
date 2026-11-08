"""Stage 5 — Series .

Two signals per (micro-region × month), ``CORPUS_START``→``CORPUS_END`` (the last
COMPLETE month; the trailing partial month is dropped — its raw count deflates the
eligibility floor):

1. **Primary target — EB-smoothed region share** of monthly submissions (controls
   the arXiv-growth confound). Each micro-region's share ``p_r`` is shrunk toward its
   super-region's **per-region share level ``P_S/n_S``** (NOT the raw super total
   ``P_S`` — that is ~16× larger, breaks scale, and inverts the ranking), with weight
   ``w_r = n_r/(n_r+κ)`` where ``n_r`` is the region's **trailing-12m** count fixed
   per origin (NOT the raw monthly count, which collapses ĝ to the super ranking).
   ``κ`` is empirical-Bayes **estimated** (symmetric Dirichlet-Multinomial
   concentration, method-of-moments, one global pool), refit on ``<= T0`` per origin.
   add-``ADD_ALPHA`` before any log/logit so zero months are defined. **No small-
   region deletion** — ``ACTIVITY_FLOOR_TRAILING12M`` is a *reporting* filter only.
   Region set **frozen per forecast origin**. Full spec: config.py EB block +
   artifacts/validation.md (2026-07-23 clarification).

2. **Novelty stream** — per month, count + fraction of new papers whose nearest-
   centroid distance exceeds ``NOVELTY_FIT_QUANTILE`` of the *fit-set's* nearest-
   centroid distances (per super-region and global). Leak status follows the fit
   set: leak-free ``<= T0`` in the masked run; "novelty vs the final frozen map"
   (display-only, like UMAP) in the unmasked production run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config


def _json_dump(obj: dict) -> str:
    return json.dumps(obj, indent=2)


def _month_index(months: list[str], start: str, end: str) -> tuple[np.ndarray, list[str]]:
    """Map "YYYY-MM" strings to contiguous month indices over [start, end].

    Returns (idx array aligned to `months`, list of month labels). Papers outside
    [start, end] get idx -1.
    """
    def ordinal(m: str) -> int:
        y, mo = m.split("-")
        return int(y) * 12 + (int(mo) - 1)

    o_start, o_end = ordinal(start), ordinal(end)
    labels = []
    o = o_start
    while o <= o_end:
        labels.append(f"{o // 12:04d}-{o % 12 + 1:02d}")
        o += 1
    arr = np.fromiter((ordinal(m) - o_start if o_start <= ordinal(m) <= o_end else -1
                       for m in months), dtype=np.int32, count=len(months))
    return arr, labels


def _load_regions(regions_dir: Path):
    """region_labels (per paper), super_region map (region->super), n_S per super."""
    labels = np.load(regions_dir / "region_labels.i16.npy").astype(np.int32)
    super_of_region = np.load(regions_dir / "super_region.i16.npy").astype(np.int32)
    n_s = np.bincount(super_of_region, minlength=config.N_SUPERREGIONS)  # frozen # micro-regions per super
    return labels, super_of_region, n_s


def _estimate_kappa(trailing_counts: np.ndarray, super_of_region: np.ndarray) -> float:
    """Symmetric Dirichlet-Multinomial concentration (per-region prior count) by
    method-of-moments, ONE global pool over within-super trailing-12m counts.

    Reading (requiring independent review — config/validation.md fix the *form*
    w_r=n_r/(n_r+kappa) and "symmetric-DM MoM, single global pool", not the closed
    form): treat each super S's micro-region counts as symmetric DM(N_S; kappa·1_{n_S}).
    Per super, the Pearson-vs-uniform statistic G_S = Σ_i (x_i − N_S/n_S)² / (N_S/n_S)
    has DM expectation (n_S−1)·(N_S + kappa·n_S)/(kappa·n_S + 1). Solve
    Σ_S G_S(obs) = Σ_S E[G_S](kappa) for the single kappa (monotone decreasing in
    kappa → unique root). kappa is the per-region prior count, exactly the units of
    w_r=n_r/(n_r+kappa).
    """
    from scipy.optimize import brentq

    n_super = config.N_SUPERREGIONS
    g_obs = 0.0
    terms = []  # (n_S, N_S) per non-trivial super
    for s in range(n_super):
        idx = np.where(super_of_region == s)[0]
        n_s = idx.size
        if n_s < 2:
            continue
        x = trailing_counts[idx].astype(np.float64)
        big_n = x.sum()
        if big_n <= 0:
            continue
        exp = big_n / n_s
        g_obs += float(((x - exp) ** 2 / exp).sum())
        terms.append((n_s, big_n))

    if not terms:
        return 1.0

    def g_exp(kappa: float) -> float:
        return sum((n_s - 1) * (big_n + kappa * n_s) / (kappa * n_s + 1.0) for n_s, big_n in terms)

    lo, hi = 1e-6, 1e9
    # g_exp is decreasing in kappa: g_exp(lo) ~ max overdispersion, g_exp(hi) -> Σ(n_S-1).
    if g_obs >= g_exp(lo):
        return lo                     # more dispersed than the model's floor -> minimal shrink
    if g_obs <= g_exp(hi):
        return hi                     # at/under multinomial -> heavy shrink
    return float(brentq(lambda k: g_exp(k) - g_obs, lo, hi, xtol=1e-3, rtol=1e-6))


def run_shares(sample: int | None = None,
               cutoff: str | None = None,
               regions_dir: Path = config.ARTIFACTS_DIR,
               score_through: str | None = None,
               out: Path = config.ARTIFACTS_DIR / "region_shares.parquet") -> Path:
    """(region × month) counts → EB-smoothed shares (long parquet). ~2 MB committed.

    ``cutoff`` = forecast origin T0 (``"YYYY-MM"``); ``None`` = production origin =
    ``config.CORPUS_END``. Months used = ``CORPUS_START..T0`` (masking for run B).
    ``regions_dir`` supplies the cutoff-specific labeling from ``regions.run``
    (``region_labels``, ``super_region`` map, ``centroids``) — the source of the
    region set + n_S, so a masked validation origin points here at its own
    ``artifacts/cutoffs/<T0>/`` dir, never the production one.

    ``score_through`` (validation only) extends the month window past T0 so the
    realized (T0, T0+24m] shares are available for scoring — but κ and the
    trailing-12m weight w_r stay anchored at **T0** (fit on ≤T0), so the realized
    side is smoothed with the same, leak-free stabilization as the forecast side.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = cutoff or config.CORPUS_END
    t_end = score_through or t0                                 # realized side may extend past T0 (leak-safe)
    labels_all, super_of_region, n_s = _load_regions(regions_dir)

    _cols = ["month"]
    _names = pq.ParquetFile(config.RAW_PARQUET).schema_arrow.names
    if "source" in _names:
        _cols.append("source")
    _tbl = pq.read_table(config.RAW_PARQUET, columns=_cols)
    months_all = _tbl.column("month").to_pylist()
    if sample is not None:
        months_all = months_all[:sample]
        labels_all = labels_all[:sample]
    if len(months_all) != len(labels_all):
        raise RuntimeError(f"row-order invariant broken: {len(months_all)} months vs {len(labels_all)} labels")

    # THE SCORED STATISTIC IS A SHARE OVER ARXIV MONTHS.
    #
    # The corpus is allowed to grow past arXiv -- journal-only work in the focus areas is
    # navigable and useful -- but a non-arXiv paper must never reach here. It would land in
    # C[region, month] as a numerator AND in M_t as a denominator, so every one of the 1,024
    # regions' shares would shift, and the pre-registered forecast numbers would move
    # because of an ingest decision rather than because of anything about research. There is
    # deliberately no flag to turn this off.
    #
    # On the frozen corpus every row is arXiv, so this is a no-op and the published numbers
    # are bit-identical -- which is how it is tested.
    if "source" in _names:
        src_all = _tbl.column("source").to_pylist()
        if sample is not None:
            src_all = src_all[:sample]
        arx = np.fromiter((str(x) == "arxiv" for x in src_all), dtype=bool,
                          count=len(src_all))
        n_drop = int((~arx).sum())
        if n_drop:
            print(f"  series: excluding {n_drop:,} non-arXiv rows from the scored share "
                  f"(navigation only; see update.series_mask)")
            months_all = [m for m, k in zip(months_all, arx) if k]
            labels_all = labels_all[arx]

    midx, month_labels = _month_index(months_all, config.CORPUS_START, t_end)   # papers month > t_end -> -1
    keep = midx >= 0
    reg = labels_all[keep]
    mon = midx[keep]
    n_months = len(month_labels)
    n_le_t0 = sum(ml <= t0 for ml in month_labels)             # T0 index within the window (== n_months unless scoring)
    K = config.N_REGIONS

    # (region x month) counts; monthly totals.
    C = np.zeros((K, n_months), dtype=np.float64)
    np.add.at(C, (reg, mon), 1.0)
    M_t = C.sum(axis=0)                                         # papers per month (all regions)

    # add-alpha share p_{r,t}
    a = config.ADD_ALPHA
    p = (C + a) / (M_t + a * K)[None, :]                        # (K, n_months)

    # super per-region mean m_{r,t} = P_{S,t}/n_S  (P over the SAME add-alpha shares)
    P_S = np.zeros((config.N_SUPERREGIONS, n_months), dtype=np.float64)
    np.add.at(P_S, super_of_region, p)                         # sum p over regions in each super
    m = (P_S / n_s[:, None])[super_of_region]                  # (K, n_months), broadcast super mean back to regions

    # trailing-12m count n_r at T0 (last 12 months ending at T0 inclusive), fixed per origin.
    # Anchored at n_le_t0 (NOT the window end) so score_through never leaks post-T0 data into w_r/kappa.
    trail = C[:, max(0, n_le_t0 - 12):n_le_t0].sum(axis=1)     # (K,)
    kappa = _estimate_kappa(trail, super_of_region)
    w = trail / (trail + kappa)                                # (K,) FIXED across months

    p_hat = w[:, None] * p + (1.0 - w[:, None]) * m            # (K, n_months) EB-smoothed share

    # long parquet: region, month, count, share_addalpha, super_region, share_smoothed
    reg_col = np.repeat(np.arange(K, dtype=np.int32), n_months)
    mon_col = np.tile(np.array(month_labels, dtype=object), K)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "region": reg_col,
        "month": mon_col,
        "count": C.reshape(-1).astype(np.int32),
        "share_addalpha": p.reshape(-1),
        "super_region": np.repeat(super_of_region.astype(np.int32), n_months),
        "share_smoothed": p_hat.reshape(-1),
    }), out, compression="zstd")

    # metadata: kappa + weight distribution (the grain-trap tell)
    (out.with_suffix(out.suffix + ".meta.json")).write_text(_json_dump({
        "cutoff_T0": t0, "score_through": t_end, "n_le_t0": n_le_t0,
        "n_months": n_months, "months": f"{month_labels[0]}..{month_labels[-1]}",
        "K": K, "kappa": float(kappa), "add_alpha": a,
        "weight_w_min": float(w.min()), "weight_w_median": float(np.median(w)), "weight_w_max": float(w.max()),
        "trailing12m_count_min": float(trail.min()), "trailing12m_median": float(np.median(trail)),
        "eb_shrink_target": config.EB_SHRINK_TARGET, "eb_shrink_weight": config.EB_SHRINK_WEIGHT,
        "eb_kappa_estimator": config.EB_KAPPA_ESTIMATOR,
    }))
    print(f"\nseries.run_shares[{config.resolve_sample(sample)}] -> {out}\n"
          f"  T0={t0}  months={n_months} ({month_labels[0]}..{month_labels[-1]})\n"
          f"  kappa (global EB) = {kappa:.2f}\n"
          f"  weight w_r  min/med/max = {w.min():.3f}/{np.median(w):.3f}/{w.max():.3f}\n"
          f"  rows = {K * n_months:,}  size = {out.stat().st_size / 1e6:.1f} MB")
    return out


def run_novelty(sample: int | None = None,
                cutoff: str | None = None,
                regions_dir: Path = config.ARTIFACTS_DIR,
                pca_memmap: Path = config.PCA_MEMMAP,
                out: Path = config.ARTIFACTS_DIR / "novelty_stream.parquet") -> Path:
    """Per-month new-topic influx: fraction beyond the fit-set distance quantile.

    ``cutoff``/``regions_dir`` as in ``run_shares``. ``pca_memmap`` is the PCA space
    to measure distances in — validate points it at the cutoff-scoped ``<=T0`` PCA so
    τ, the centroids, and the distances are all leak-free (never the production
    all-data memmap). The distance threshold is ``NOVELTY_FIT_QUANTILE`` of the
    FIT-SET nearest-centroid distances, inheriting the fit set's leak profile
    (leak-free <= T0 in run B; vs-final-map in run A).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = cutoff or config.CORPUS_END
    labels_all, super_of_region, _ = _load_regions(regions_dir)
    centroids = np.load(regions_dir / "centroids.f32.npy").astype(np.float32)   # (K, 50)

    pca_meta = json.loads((pca_memmap.with_suffix(pca_memmap.suffix + ".meta.json")).read_text())
    n_pca = int(pca_meta["n"])
    pca = np.memmap(pca_memmap, dtype=np.float32, mode="r", shape=(n_pca, config.PCA_DIM))

    months_all = pq.read_table(config.RAW_PARQUET, columns=["month"]).column("month").to_pylist()
    if sample is not None:
        months_all, labels_all, n_pca = months_all[:sample], labels_all[:sample], sample
    if not (len(months_all) == len(labels_all) == n_pca):
        raise RuntimeError(f"row-order invariant: months={len(months_all)} labels={len(labels_all)} pca={n_pca}")

    midx, month_labels = _month_index(months_all, config.CORPUS_START, t0)

    # nearest-centroid distance = distance to the ASSIGNED centroid (k-means assigns
    # to the nearest, so this IS the min distance), streamed in chunks.
    dist = np.empty(n_pca, dtype=np.float64)
    CH = 200_000
    for lo in range(0, n_pca, CH):
        hi = min(lo + CH, n_pca)
        diff = np.asarray(pca[lo:hi], dtype=np.float32) - centroids[labels_all[lo:hi]]
        dist[lo:hi] = np.sqrt((diff.astype(np.float64) ** 2).sum(axis=1))

    keep = midx >= 0
    d, mon, sup = dist[keep], midx[keep], super_of_region[labels_all[keep]]
    # tau = fit-set quantile. Run A fit set = all <= T0 (unmasked); run B = <= T0 masked.
    tau = float(np.quantile(d, config.NOVELTY_FIT_QUANTILE))
    novel = d > tau
    n_months = len(month_labels)

    # aggregate per (super, month) and global (super_region = -1)
    tot_sm = np.zeros((config.N_SUPERREGIONS, n_months))
    nov_sm = np.zeros((config.N_SUPERREGIONS, n_months))
    np.add.at(tot_sm, (sup, mon), 1.0)
    np.add.at(nov_sm, (sup, mon), novel.astype(np.float64))
    tot_g = tot_sm.sum(axis=0)
    nov_g = nov_sm.sum(axis=0)

    rows_super = config.N_SUPERREGIONS * n_months
    sr = np.concatenate([np.repeat(np.arange(config.N_SUPERREGIONS, dtype=np.int32), n_months),
                         np.full(n_months, -1, dtype=np.int32)])
    mo = np.concatenate([np.tile(np.array(month_labels, dtype=object), config.N_SUPERREGIONS),
                         np.array(month_labels, dtype=object)])
    tot = np.concatenate([tot_sm.reshape(-1), tot_g])
    nov = np.concatenate([nov_sm.reshape(-1), nov_g])
    frac = np.divide(nov, tot, out=np.zeros_like(nov), where=tot > 0)

    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "super_region": sr, "month": mo,
        "n_papers": tot.astype(np.int32), "n_novel": nov.astype(np.int32), "frac_novel": frac,
    }), out, compression="zstd")
    (out.with_suffix(out.suffix + ".meta.json")).write_text(_json_dump({
        "cutoff_T0": t0, "leak_status": "display_only_vs_final_map" if cutoff is None else "leak_free_leq_T0",
        "tau": tau, "novelty_fit_quantile": config.NOVELTY_FIT_QUANTILE,
        "n_months": n_months, "rows": int(rows_super + n_months),
        "note": "run A: centroids see all data -> novelty vs final frozen map; show leading-edge, not 2007->2026 curve",
    }))
    print(f"\nseries.run_novelty[{config.resolve_sample(sample)}] -> {out}\n"
          f"  T0={t0}  tau(q{config.NOVELTY_FIT_QUANTILE})={tau:.3f}  ({'display-only' if cutoff is None else 'leak-free'})\n"
          f"  global frac_novel recent (last 6mo): "
          f"{', '.join(f'{month_labels[-6 + i]} {nov_g[-6 + i] / max(tot_g[-6 + i], 1):.3f}' for i in range(6))}")
    return out
