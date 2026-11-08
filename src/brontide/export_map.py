"""Stage 10a — Export map.

Write the portfolio site's existing binary contract from REAL coordinates, so the
map page needs **zero changes** (it already reads these four files). Replaces the
synthetic stand-in produced by the site's ``gen_standin_map_data.py``.

Contract (must match byte-for-byte — see the site's map/index.html reader):
  points.bin   int16 LE, interleaved (x, y), quantized over meta.bbox to
               [-MAP_QUANT_OFFSET, +MAP_QUANT_OFFSET]:
               q = round((v - vmin)/(vmax - vmin) * MAP_QUANT_RANGE - MAP_QUANT_OFFSET)
  cluster.bin  uint8 super-region id per point (255 = noise)   [<= 64 « 255 cap]
  year.bin     uint8 offset from meta.year0 (MAP_YEAR0) per point
  meta.json    {synthetic:false, n, bbox:[xmin,ymin,xmax,ymax], year0, year1,
                noise_color, clusters:[{id,label,trend,color,cx,cy}], files:{...}}

For real data: ``trend`` = sign of the realized share slope
(emerging/steady/declining); ``synthetic`` = false; drop the synthetic badge.
"""
from __future__ import annotations

import colorsys
import json
from pathlib import Path

import numpy as np

from . import config


def _palette(n: int) -> list[str]:
    """n visually-distinct hex colors (evenly-spaced hue)."""
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.62, 0.90)
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


def _super_trend(shares_parquet: Path, super_of_region: np.ndarray) -> dict[int, str]:
    """emerging/steady/declining per super = sign of realized smoothed-share slope
    over the trailing 24 months (aggregated region shares → super)."""
    import pyarrow.parquet as pq

    t = pq.read_table(shares_parquet, columns=["region", "month", "super_region", "share_smoothed"])
    reg = t.column("region").to_numpy()
    sup = t.column("super_region").to_numpy()
    sm = t.column("share_smoothed").to_numpy()
    months = np.array(t.column("month").to_pylist())
    uniq_months = sorted(set(months.tolist()))
    last24 = set(uniq_months[-24:])
    midx = {m: i for i, m in enumerate(uniq_months[-24:])}
    # super-total smoothed share per recent month
    agg = np.zeros((config.N_SUPERREGIONS, len(last24)))
    for i in range(len(reg)):
        if months[i] in last24:
            agg[sup[i], midx[months[i]]] += sm[i]
    trend = {}
    xs = np.arange(agg.shape[1], dtype=np.float64)
    for s in range(config.N_SUPERREGIONS):
        y = agg[s]
        slope = np.polyfit(xs, y, 1)[0] if y.sum() > 0 else 0.0
        rel = slope * len(xs) / (y.mean() + 1e-12)          # relative change over window
        trend[s] = "emerging" if rel > 0.15 else "declining" if rel < -0.15 else "steady"
    return trend


def run(umap_memmap: Path = config.UMAP_MEMMAP,
        out_dir: Path = config.SITE_MAP_DATA_DIR,
        downsample: int = config.MAP_DOWNSAMPLE) -> Path:
    """Downsample, quantize, and write the four map files. Returns ``out_dir``.

    Downsample stratified per micro-region (proportional, min 20/region) to
    ~``downsample`` points; keep the site's <5 MB transfer budget.
    """
    art = config.ARTIFACTS_DIR
    umeta = json.loads((umap_memmap.with_suffix(umap_memmap.suffix + ".meta.json")).read_text())
    n = int(umeta["n"])
    xy = np.array(np.memmap(umap_memmap, dtype=np.float32, mode="r", shape=(n, 2)))
    labels = np.load(art / "region_labels.i16.npy").astype(np.int32)[:n]
    super_of_region = np.load(art / "super_region.i16.npy").astype(np.int32)
    months = pq_months(n)
    years = np.array([int(m[:4]) for m in months], dtype=np.int32)

    # 1. stratified downsample per micro-region (proportional, min 20)
    rng = np.random.default_rng(0)
    picks = []
    for r in range(config.N_REGIONS):
        idx = np.where(labels == r)[0]
        if idx.size == 0:
            continue
        take = min(idx.size, max(20, round(downsample * idx.size / n)))
        picks.append(rng.choice(idx, take, replace=False))
    sel = np.sort(np.concatenate(picks))
    p_xy = xy[sel]
    p_super = super_of_region[labels[sel]]
    p_year = years[sel]

    # 2. bbox (2% pad) + int16 quantize
    xmin, ymin = p_xy.min(axis=0)
    xmax, ymax = p_xy.max(axis=0)
    padx, pady = (xmax - xmin) * 0.02, (ymax - ymin) * 0.02
    xmin, xmax, ymin, ymax = xmin - padx, xmax + padx, ymin - pady, ymax + pady

    def quant(v, vmin, vmax):
        q = np.round((v - vmin) / (vmax - vmin) * config.MAP_QUANT_RANGE - config.MAP_QUANT_OFFSET)
        return np.clip(q, -32768, 32767).astype("<i2")

    qx = quant(p_xy[:, 0], xmin, xmax)
    qy = quant(p_xy[:, 1], ymin, ymax)
    points = np.empty(p_xy.shape[0] * 2, dtype="<i2")
    points[0::2] = qx
    points[1::2] = qy

    cluster = p_super.astype(np.uint8)                       # <=64, no noise here
    year = (p_year - config.MAP_YEAR0).astype(np.uint8)

    # 5. per-super meta: label, color, trend, cx/cy
    labels_json = json.loads((art / "labels.json").read_text())
    trend = _super_trend(art / "region_shares.parquet", super_of_region)
    colors = _palette(config.N_SUPERREGIONS)
    clusters = []
    for s in range(config.N_SUPERREGIONS):
        m = p_super == s
        cx = float(p_xy[m, 0].mean()) if m.any() else 0.0
        cy = float(p_xy[m, 1].mean()) if m.any() else 0.0
        terms = labels_json.get(str(s), {}).get("terms", [])
        clusters.append({"id": s, "label": ", ".join(terms[:3]) if terms else f"region {s}",
                         "trend": trend.get(s, "steady"), "color": colors[s],
                         "cx": round(cx, 3), "cy": round(cy, 3)})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "points.bin").write_bytes(points.tobytes())
    (out_dir / "cluster.bin").write_bytes(cluster.tobytes())
    (out_dir / "year.bin").write_bytes(year.tobytes())
    meta = {
        "synthetic": False, "n": int(sel.size),
        "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "year0": config.MAP_YEAR0, "year1": int(years.max()),
        "noise_color": "#3a3a3a", "clusters": clusters,
        "files": {"points": "points.bin", "cluster": "cluster.bin", "year": "year.bin"},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    total = sum((out_dir / f).stat().st_size for f in ("points.bin", "cluster.bin", "year.bin", "meta.json"))
    assert total < config.MAP_BUDGET_MB * 1e6, f"map export {total / 1e6:.1f} MB exceeds {config.MAP_BUDGET_MB} MB"
    print(f"\nexport_map.run -> {out_dir}\n"
          f"  points   : {sel.size:,} (downsampled from {n:,})\n"
          f"  bytes    : {total / 1e6:.2f} MB (budget {config.MAP_BUDGET_MB})\n"
          f"  years    : {config.MAP_YEAR0}..{years.max()}   supers labelled: {len(clusters)}")
    return out_dir


def pq_months(n: int) -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_table(config.RAW_PARQUET, columns=["month"]).column("month").to_pylist()[:n]
