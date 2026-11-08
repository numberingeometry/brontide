"""Stage 10b — Figures and the dashboard payload.

Everything here is built from **committed artifacts only** — no raw corpus, no embeddings
memmap, no GPU, no network. That is the point: a reader who clones the repo can regenerate
every figure in the README in about a minute.

Outputs (all under ``figures/``):

===========================  ==========================================================
``map-evolution.gif``        the frontier igniting, 2007 → 2026, one frame per year
``frontier-map.png``         hero still of the same map with the largest regions labelled
``validation-precision.png`` precision@50 by model × cutoff against the 0.1007 chance rate
``forecast-vs-realized.png`` small multiples: history, forecast band, and what happened
``region-hierarchy.png``     the Ward tree, 1024 microregions cut to 64 super-regions
``novelty-stream.png``       share of papers landing far from every known centroid
``emerging-regions.png``     the current top-20 by ĝ, named
===========================  ==========================================================

Plus ``artifacts/region_figs.json`` — the per-super-region history + forecast payload the
dashboard reads.

Region names come from ``artifacts/region_names.json`` (see ``name_regions.py``); they are
cosmetic and never enter a scored quantity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config

FIGURES_DIR = config.REPO_ROOT / "figures"
MAP_DIR = config.ARTIFACTS_DIR / "map" / "data"

# --- palette (validated: dataviz six-checks, light surface #fcfcfb, all-pairs) -----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
S1 = "#2a78d6"   # blue   — history / primary magnitude
S2 = "#eb6834"   # orange — baseline (SmoothedMomentum)
S3 = "#1baf7a"   # aqua   — realized (always direct-labelled: contrast relief)
DARK_SURFACE = "#0d1117"

CHANCE_P50 = 0.1007   # 103 realized-positive of 1021-1023 eligible; see LIMITATIONS.md


def _style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": "#c3c2b7",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.titleweight": "bold", "figure.dpi": 130,
    })


def _names() -> dict[int, str]:
    from . import name_regions
    try:
        return name_regions.load()
    except FileNotFoundError:
        return {}


def _map_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(points Nx2 int16, cluster N uint8, year N uint8, meta) from the committed export."""
    meta = json.loads((MAP_DIR / "meta.json").read_text(encoding="utf-8"))
    n = meta["n"]
    pts = np.fromfile(MAP_DIR / "points.bin", dtype=np.int16).reshape(n, 2)
    cl = np.fromfile(MAP_DIR / "cluster.bin", dtype=np.uint8)
    yr = np.fromfile(MAP_DIR / "year.bin", dtype=np.uint8)
    return pts, cl, yr, meta


def _cluster_rgb(meta: dict) -> np.ndarray:
    """(64, 3) uint8 colours, taken from the committed map export so figures, the dashboard,
    and the site map all agree."""
    from matplotlib.colors import to_rgb
    rgb = np.zeros((64, 3), np.float64)
    for c in meta["clusters"]:
        rgb[c["id"]] = to_rgb(c["color"])
    return (rgb * 255).astype(np.uint8)


def _rasterize(pts: np.ndarray, cl: np.ndarray, rgb: np.ndarray, sel: np.ndarray,
               w: int, h: int, pad: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate selected points into (colour_sum, count) rasters of shape (h, w, 3)/(h, w).

    Colour is the density-weighted mean of the contributing clusters, so overlapping regions
    blend rather than the last-drawn winning.
    """
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    px = ((x - x0) / (x1 - x0) * (w - 2 * pad) + pad).astype(np.int32)
    py = ((y1 - y) / (y1 - y0) * (h - 2 * pad) + pad).astype(np.int32)
    idx = (py * w + px)[sel]
    cols = rgb[cl[sel]].astype(np.float64)

    cnt = np.bincount(idx, minlength=w * h).astype(np.float64)
    acc = np.stack([np.bincount(idx, weights=cols[:, k], minlength=w * h) for k in range(3)], -1)
    return acc.reshape(h, w, 3), cnt.reshape(h, w)


def _compose(acc: np.ndarray, cnt: np.ndarray, gamma: float = 0.45,
             base: str = DARK_SURFACE) -> np.ndarray:
    """Blend the accumulated colour over the dark surface with a gamma-compressed alpha, so
    sparse early years stay visible without saturating dense ones."""
    from matplotlib.colors import to_rgb
    bg = np.array(to_rgb(base)) * 255
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt[..., None] > 0, acc / np.maximum(cnt[..., None], 1), 0.0)
    alpha = np.clip((cnt / max(cnt.max(), 1)) ** gamma * 2.6, 0, 1)[..., None]
    return np.clip(mean * alpha + bg * (1 - alpha), 0, 255).astype(np.uint8)


def _gif_fonts():
    """(year, subtitle, footer) fonts. matplotlib ships DejaVuSans, which — unlike PIL's
    default bitmap font — is scalable and has the arrow and middot glyphs the caption uses."""
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        return (ImageFont.truetype(path, 30), ImageFont.truetype(path, 13),
                ImageFont.truetype(path, 11))
    except Exception:                                    # pragma: no cover - font-less box
        d = ImageFont.load_default()
        return d, d, d


# --- 1. the hero GIF ---------------------------------------------------------------------
def map_evolution_gif(out: Path = None, w: int = 900, h: int = 680,
                      ms_per_year: int = 420) -> Path:
    """Animate the map year by year: everything published so far stays lit, the current
    year's papers flare white. 20 frames, 2007 → 2026."""
    from PIL import Image, ImageDraw

    out = Path(out or FIGURES_DIR / "map-evolution.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    pts, cl, yr, meta = _map_arrays()
    rgb = _cluster_rgb(meta)
    y0 = meta["year0"]
    years = sorted(set(yr.tolist()))

    frames = []
    for y in years:
        acc, cnt = _rasterize(pts, cl, rgb, yr <= y, w, h)
        img = _compose(acc, cnt)
        # this year's arrivals, flared on top
        acc_n, cnt_n = _rasterize(pts, cl, rgb, yr == y, w, h)
        flare = np.clip((cnt_n / max(cnt_n.max(), 1)) ** 0.4 * 1.5, 0, 1)[..., None]
        img = np.clip(img * (1 - flare) + 255 * flare, 0, 255).astype(np.uint8)

        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        f_year, f_sub, f_foot = _gif_fonts()
        d.text((26, 16), f"{y0 + y}", fill="#ffffff", font=f_year)
        d.text((28, 58), f"{int((yr <= y).sum()):,} papers mapped", fill="#9aa4b2", font=f_sub)
        d.text((26, h - 28), "arXiv 2007–2026 · SPECTER2 → PCA → UMAP · 64 super-regions",
               fill="#6b7280", font=f_foot)
        # timeline
        x0b, x1b, yb = 26, w - 26, h - 46
        d.line([(x0b, yb), (x1b, yb)], fill="#2c333d", width=3)
        frac = (y - years[0]) / max(years[-1] - years[0], 1)
        d.line([(x0b, yb), (x0b + int((x1b - x0b) * frac), yb)], fill="#ffffff", width=3)
        frames.append(pil.convert("P", palette=Image.ADAPTIVE, colors=192))

    durations = [ms_per_year] * len(frames)
    durations[-1] = 2200          # hold the final state
    frames[0].save(out, save_all=True, append_images=frames[1:], loop=0,
                   duration=durations, optimize=True, disposal=2)
    return out


# --- 2. hero still -----------------------------------------------------------------------
def map_static(out: Path = None, w: int = 1500, h: int = 1100, label_top: int = 22) -> Path:
    """The full map with the largest super-regions directly labelled."""
    import matplotlib.pyplot as plt
    _style()
    out = Path(out or FIGURES_DIR / "frontier-map.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    pts, cl, yr, meta = _map_arrays()
    rgb = _cluster_rgb(meta)
    acc, cnt = _rasterize(pts, cl, rgb, np.ones(len(cl), bool), w, h)
    img = _compose(acc, cnt, gamma=0.42)

    names = _names()
    sizes = np.bincount(cl, minlength=64)
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    x0, x1, y0v, y1v = x.min(), x.max(), y.min(), y.max()
    pad = 36

    fig, ax = plt.subplots(figsize=(w / 130, h / 130))
    fig.patch.set_facecolor(DARK_SURFACE)
    ax.imshow(img)
    ax.set_axis_off()

    # Greedy label placement: biggest regions first, skip anything that would collide with
    # an already-placed label, and keep text inside the frame. Unlabelled regions are not
    # hidden — they are simply too crowded to name at this size.
    placed: list[tuple[float, float, float]] = []
    for cid in np.argsort(-sizes):
        if len(placed) >= label_top:
            break
        m = cl == cid
        # median is steadier than mean for the bimodal regions
        cx = ((np.median(x[m]) - x0) / (x1 - x0)) * (w - 2 * pad) + pad
        cy = ((y1v - np.median(y[m])) / (y1v - y0v)) * (h - 2 * pad) + pad
        label = names.get(int(cid), str(cid))
        # imshow puts 1 data unit = 1 image pixel, so estimate the rendered half-width in px:
        # fontsize 7.2pt at 130 dpi ≈ 13 px tall, ≈ 6.5 px average advance per character.
        half = len(label) * 3.4
        cx = float(np.clip(cx, half + 6, w - half - 6))
        if any(abs(cx - px) < (half + phalf) and abs(cy - py) < 24
               for px, py, phalf in placed):
            continue
        placed.append((cx, cy, half))
        ax.text(cx, cy, label, color="white", fontsize=7.2,
                ha="center", va="center", weight="bold",
                path_effects=_outline())
    ax.set_title("The arXiv research frontier — 2.7M papers, 64 super-regions",
                 color="white", pad=14)
    fig.tight_layout()
    fig.savefig(out, facecolor=DARK_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def _outline():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=2.4, foreground="#0d1117")]


# --- 3. the headline validation figure ---------------------------------------------------
def validation_figure(out: Path = None) -> Path:
    """precision@50 by model, faceted by cutoff, against the 0.1007 chance rate.

    Position carries model identity (one bar per model per facet), so colour is free to encode
    the thing that matters: which model the CV selected, and which one is the baseline.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from . import validate
    _style()
    out = Path(out or FIGURES_DIR / "validation-precision.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    cutoffs = config.VALIDATION_CUTOFFS
    per = {}
    for T0 in cutoffs:
        cart = config.ARTIFACTS_DIR / "cutoffs" / T0
        scores, _ = validate._score_cutoff(T0, cart / "region_shares.parquet",
                                           cart / "forecasts.parquet")
        sel = json.loads((cart / "forecasts.parquet.meta.json").read_text())["selected_model"]
        per[T0] = (scores, sel)

    models = ["AutoETS", "AutoARIMA", "AutoTheta", "SmoothedMomentum", "SeasonalNaive", "Naive"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), sharey=True)
    labels = {"2017-12": "2017-12  (quiet window)", "2020-12": "2020-12  (COVID)",
              "2022-12": "2022-12  (LLM explosion)"}

    for ax, T0 in zip(axes, cutoffs):
        scores, sel = per[T0]
        vals = [scores[m]["precision50"] for m in models]
        ypos = np.arange(len(models))[::-1]
        colors = [S2 if m == config.BASELINE_MODEL else S1 for m in models]
        ax.barh(ypos, vals, height=0.66, color=colors, zorder=3)
        for yp, v, m in zip(ypos, vals, models):
            tag = "   CV-selected" if m == sel else ""
            ax.text(v + 0.014, yp, f"{v:.3f}{tag}", va="center", fontsize=8.5,
                    color=INK if m == sel else INK_2,
                    weight="bold" if m == sel else "normal")
        ax.axvline(CHANCE_P50, color=MUTED, ls="--", lw=1.4, zorder=4)
        ax.set_yticks(ypos, models, fontsize=8.8)
        ax.set_xlim(0, 1.0)
        ax.set_title(labels[T0], fontsize=10.5, pad=10)
        ax.set_xlabel("precision@50")
        ax.grid(axis="y", visible=False)
        ax.text(CHANCE_P50 + 0.012, -0.72, "chance 0.101", fontsize=7.6, color=MUTED,
                va="center")
    fig.legend(handles=[Patch(color=S2, label="SmoothedMomentum (pre-registered baseline)"),
                        Patch(color=S1, label="forecast roster")],
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Pre-registered criterion: 0/3 — and why that verdict needs its scope note",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.text(0.5, 0.945, "Every non-Naive model clears chance at every cutoff. The CV selected "
                         "the baseline itself at two of three cutoffs.",
             ha="center", fontsize=9.2, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --- 4. forecast vs realized -------------------------------------------------------------
def _super_series(shares_path: Path, value: str = "share_smoothed"):
    """Aggregate the 1024 microregion series up to the 64 Ward super-regions."""
    import pandas as pd
    sh = pd.read_parquet(shares_path)
    g = sh.groupby(["super_region", "month"], as_index=False)[value].sum()
    return g.pivot(index="super_region", columns="month", values=value)


def forecast_panels(cutoff: str = "2020-12", out: Path = None, n_panels: int = 6) -> Path:
    """History → forecast band → what actually happened, for the super-regions that grew most.

    Uses the ``2020-12`` masked cutoff by default, because that is a cutoff where the CV
    selected the baseline itself — so the panels show the selection failure, not just a forecast.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    _style()
    out = Path(out or FIGURES_DIR / "forecast-vs-realized.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cart = config.ARTIFACTS_DIR / "cutoffs" / cutoff

    S = _super_series(cart / "region_shares.parquet")
    months = list(S.columns)
    le = [m for m in months if m <= cutoff]
    post = [m for m in months if m > cutoff][:config.HORIZON_MONTHS]

    fc = pd.read_parquet(cart / "forecasts.parquet")
    names = _names()

    trailing = S[le[-12:]].mean(axis=1)
    realized = S[post[18:24]].mean(axis=1)
    growth = (realized / trailing.replace(0, np.nan)).sort_values(ascending=False)
    picks = [int(s) for s in growth.index[:n_panels]]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.4), sharex=True)
    hist_tail = le[-60:]
    for ax, sid in zip(axes.ravel(), picks):
        # history is context, not a competing series — it wears ink, so the three forward
        # lines are the only things carrying series colour
        xs_h = np.arange(len(hist_tail))
        ax.plot(xs_h, S.loc[sid, hist_tail].to_numpy() * 100, color="#6f6e6a", lw=1.5,
                label="history (≤ cutoff)")

        xs_f = np.arange(len(hist_tail), len(hist_tail) + len(post))
        for model, colr, lbl in ((config.BASELINE_MODEL, S2, "momentum baseline"),
                                 ("AutoETS", S1, "AutoETS forecast")):
            fm = fc[fc["model"] == model]
            agg = (fm.merge(
                pd.read_parquet(cart / "region_shares.parquet")[["region", "super_region"]]
                  .drop_duplicates(), on="region")
                .query("super_region == @sid").groupby("month")[["mean", "lo", "hi"]].sum()
                .reindex(post))
            ls = "--" if model == config.BASELINE_MODEL else "-"
            ax.plot(xs_f, agg["mean"].to_numpy() * 100, color=colr, lw=1.9, ls=ls, label=lbl)
            if model == "AutoETS" and agg["lo"].notna().any():
                ax.fill_between(xs_f, agg["lo"].to_numpy() * 100, agg["hi"].to_numpy() * 100,
                                color=S1, alpha=0.14, lw=0)

        ax.plot(xs_f, S.loc[sid, post].to_numpy() * 100, color=S3, lw=2.4, label="realized")
        ax.text(xs_f[-1], S.loc[sid, post[-1]] * 100, "  realized", color=S3, fontsize=8,
                va="center", weight="bold")            # direct label (contrast relief)
        ax.axvline(len(hist_tail) - 0.5, color=MUTED, lw=1, ls=":")
        ax.set_title(names.get(sid, f"super-region {sid}"), fontsize=9.6)
        ax.set_xticks([0, len(hist_tail) - 1, xs_f[-1]],
                      [hist_tail[0], cutoff, post[-1]], fontsize=8)
        ax.tick_params(labelsize=8)

    for ax in axes[:, 0]:
        ax.set_ylabel("share of monthly arXiv (%)", fontsize=9)
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Forecast vs realized at the {cutoff} cutoff — fastest-growing super-regions",
                 fontsize=13.5, weight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --- 5. the Ward hierarchy ---------------------------------------------------------------
def region_dendrogram(out: Path = None) -> Path:
    """The Ward tree that cuts 1024 microregions down to 64 named super-regions."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    _style()
    out = Path(out or FIGURES_DIR / "region-hierarchy.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    Z = np.load(config.ARTIFACTS_DIR / "linkage.f64.npy")
    # the pipeline's own cut, not a re-derived one
    sup = np.load(config.ARTIFACTS_DIR / "super_region.i16.npy").astype(int)
    names = _names()

    fig, ax = plt.subplots(figsize=(13.6, 7.0))
    dn = dendrogram(Z, no_labels=True, ax=ax, color_threshold=0,
                    above_threshold_color="#9a9992", orientation="top")
    order = np.asarray(dn["leaves"], int)          # microregion ids, left to right
    sup_ordered = sup[order]

    # a Ward cut is contiguous in leaf order, so each super-region is one run; label its centre
    bounds = np.flatnonzero(np.diff(sup_ordered)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(sup_ordered)]))
    ticks, labels_ = [], []
    for s, e in zip(starts, ends):
        if e - s < 4:                               # too narrow to letter at this width
            continue
        ticks.append(((s + e) / 2) * 10)
        labels_.append(names.get(int(sup_ordered[s]), str(sup_ordered[s])))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels_, rotation=90, fontsize=6.2, color=INK_2)

    ax.set_ylabel("Ward merge distance")
    ax.set_title("1,024 embedding microregions → 64 super-regions (Ward linkage)", pad=10)
    ax.grid(axis="x", visible=False)
    note = ("The first split separates computation from the physical sciences — nothing told it "
            "to; the tree is built from SPECTER2 geometry alone.")
    if len(labels_) < 64:
        note += f" {64 - len(labels_)} narrow super-regions are unlabelled at this width."
    fig.text(0.5, -0.02, note, ha="center", fontsize=8.2, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --- 6. novelty --------------------------------------------------------------------------
def novelty_figure(out: Path = None) -> Path:
    """Share of papers landing beyond the 99th percentile of the fit-set centroid distance."""
    import matplotlib.pyplot as plt
    import pandas as pd
    _style()
    out = Path(out or FIGURES_DIR / "novelty-stream.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    nv = pd.read_parquet(config.ARTIFACTS_DIR / "novelty_stream.parquet")
    tot = nv.groupby("month", as_index=False)[["n_papers", "n_novel"]].sum()
    tot["frac"] = tot["n_novel"] / tot["n_papers"].replace(0, np.nan)
    tot = tot[tot["n_papers"] > 200]

    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    xs = np.arange(len(tot))
    ax.fill_between(xs, tot["frac"] * 100, color=S1, alpha=0.16, lw=0)
    ax.plot(xs, tot["frac"] * 100, color=S1, lw=2)
    step = max(len(tot) // 10, 1)
    ax.set_xticks(xs[::step], [m for m in tot["month"][::step]], fontsize=8, rotation=45)
    ax.set_ylabel("papers beyond the 99th-pct centroid distance (%)")
    ax.set_title("Novelty stream — arrivals far from every known region")
    fig.text(0.5, -0.06, "Display-only: computed against the final frozen map, so it shows the "
                         "leading edge rather than a leak-free 2007→2026 curve. Never scored.",
             ha="center", fontsize=8.4, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --- 7. what the model says right now ----------------------------------------------------
def emergence_ranking(out: Path = None, top: int = 18) -> Path:
    """Super-regions ranked by forecast growth at the production origin.

    Aggregated to super-regions rather than the 1024 microregions: ``labels.json`` names only
    the 64 supers, so a microregion ranking would repeat one name a dozen times and say
    nothing. ĝ here is the ratio of summed forecast share (horizon months 19–24) to summed
    trailing-12m share, matching the emergence statistic's own definition one level up.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    _style()
    out = Path(out or FIGURES_DIR / "emerging-regions.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    S = _super_series(config.ARTIFACTS_DIR / "region_shares.parquet")
    months = list(S.columns)
    trailing = S[months[-12:]].mean(axis=1)

    fc = pd.read_parquet(config.ARTIFACTS_DIR / "forecasts.parquet")
    sel = json.loads((config.ARTIFACTS_DIR / "forecasts.parquet.meta.json")
                     .read_text())["selected_model"]
    xwalk = (pd.read_parquet(config.ARTIFACTS_DIR / "region_shares.parquet")
             [["region", "super_region"]].drop_duplicates())
    f = fc[fc["model"] == sel].merge(xwalk, on="region")
    fmonths = sorted(f["month"].unique())
    lo, hi = config.EMERGENCE_HORIZON_MONTHS
    horizon = fmonths[lo - 1:hi]
    fwd = (f[f["month"].isin(horizon)].groupby(["super_region", "month"])["mean"].sum()
           .groupby("super_region").mean())

    g = (fwd / trailing.replace(0, np.nan)).dropna().sort_values(ascending=False).head(top)
    lab = json.loads((config.ARTIFACTS_DIR / "labels.json").read_text(encoding="utf-8"))
    names = _names()

    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    ypos = np.arange(len(g))[::-1]
    ax.barh(ypos, g.to_numpy(), height=0.66, color=S1, zorder=3)
    for yp, (sid, v) in zip(ypos, g.items()):
        ax.text(v + 0.06, yp, f"{v:.2f}×", va="center", fontsize=8.6, color=INK_2)
    ax.set_yticks(ypos, [names.get(int(s), str(s)) for s in g.index], fontsize=8.6)
    ax.set_xlabel("ĝ  —  forecast share (horizon months 19–24) ÷ trailing-12m share")
    ax.set_title(f"Super-regions by forecast growth, production origin {config.CORPUS_END}",
                 pad=10)
    ax.axvline(1.0, color=MUTED, ls="--", lw=1.2, zorder=4)
    ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.04, f"Produced by the CV-selected model at this origin — which is {sel}, "
                         "the baseline that won the validation. A ranking, not a promise; "
                         "the dashed line is flat share.",
             ha="center", fontsize=8.4, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --- 8. dashboard payload ----------------------------------------------------------------
def region_figs_json(out: Path = None) -> Path:
    """``[{id, name, terms, history, forecast, g_hat, class}]`` — what dashboard.html reads."""
    import pandas as pd
    out = Path(out or config.ARTIFACTS_DIR / "region_figs.json")
    S = _super_series(config.ARTIFACTS_DIR / "region_shares.parquet")
    months = list(S.columns)

    fc = pd.read_parquet(config.ARTIFACTS_DIR / "forecasts.parquet")
    sel = json.loads((config.ARTIFACTS_DIR / "forecasts.parquet.meta.json")
                     .read_text())["selected_model"]
    xwalk = (pd.read_parquet(config.ARTIFACTS_DIR / "region_shares.parquet")
             [["region", "super_region"]].drop_duplicates())
    fsel = (fc[fc["model"] == sel].merge(xwalk, on="region")
            .groupby(["super_region", "month"], as_index=False)["mean"].sum())

    em = pd.read_parquet(config.ARTIFACTS_DIR / "emergence.parquet")
    gs = em.groupby("super_region").agg(g_hat=("g_hat", "median"),
                                        n_emerging=("class", lambda c: int((c == "emerging").sum())))
    lab = json.loads((config.ARTIFACTS_DIR / "labels.json").read_text(encoding="utf-8"))
    names = _names()

    recs = []
    for sid in sorted(S.index):
        f = fsel[fsel["super_region"] == sid].sort_values("month")
        recs.append({
            "id": int(sid),
            "name": names.get(int(sid), f"super-region {sid}"),
            "terms": lab.get(str(sid), {}).get("terms", [])[:8],
            "sample_title": (lab.get(str(sid), {}).get("sample_titles") or [""])[0][:120],
            "history": [round(float(v) * 100, 5) for v in S.loc[sid].to_numpy()],
            "forecast": [round(float(v) * 100, 5) for v in f["mean"].to_numpy()],
            "g_hat": round(float(gs.loc[sid, "g_hat"]), 3) if sid in gs.index else None,
            "n_emerging": int(gs.loc[sid, "n_emerging"]) if sid in gs.index else 0,
        })

    payload = {"months": months, "forecast_months": sorted(f["month"].unique().tolist()),
               "selected_model": sel, "regions": recs}
    Path(out).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return Path(out)


def run(out_dir: Path = None) -> list[Path]:
    """Build every figure plus the dashboard payload. Committed artifacts only."""
    global FIGURES_DIR
    if out_dir:
        FIGURES_DIR = Path(out_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (region_figs_json, validation_figure, forecast_panels, emergence_ranking,
               novelty_figure, region_dendrogram, map_static, map_evolution_gif):
        made.append(fn())
        print(f"  {made[-1].relative_to(config.REPO_ROOT)}")
    return made
