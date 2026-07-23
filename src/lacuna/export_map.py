"""Stage 10a — Export map (CPU-only environment).

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

from pathlib import Path

import numpy as np

from . import config


def run(umap_memmap: Path = config.UMAP_MEMMAP,
        out_dir: Path = config.SITE_MAP_DATA_DIR,
        downsample: int = config.MAP_DOWNSAMPLE) -> Path:
    """Downsample, quantize, and write the four map files. Returns ``out_dir``.

    Downsample stratified per micro-region (proportional, min 20/region) to
    ~``downsample`` points; keep the site's <5 MB transfer budget.
    """
    # TODO:
    #   1. stratified downsample indices (proportional per micro-region, min 20).
    #   2. bbox with 2% pad; quantize UMAP x,y to int16 (formula above); interleave.
    #   3. cluster.bin = super_region id (uint8; 255 for noise/unassigned).
    #   4. year.bin = paper_year - MAP_YEAR0 (uint8).
    #   5. meta.json: bbox, year0/year1, per-super-region {label(from labels.json),
    #      color, trend(realized slope), cx/cy in UMAP space}; synthetic=false.
    #   6. assert total transferred bytes < MAP_BUDGET_MB.
    raise NotImplementedError("export_map.run")
