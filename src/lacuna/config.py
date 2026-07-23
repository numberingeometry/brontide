"""Frozen configuration for the lacuna pipeline.

Single source of truth for paths, dimensions, and the **pre-registered**
modeling + emergence constants. Anything the validation harness scores against
lives here and is mirrored, dated, into ``artifacts/validation.md`` before the
first scoring run (pre-registration discipline — see the v1 plan).

"lacuna" is a placeholder name (final name deferred; candidate dossier in
local workspace/projects/lacuna/2026-07-21-rename-vetting.md).
"""
from __future__ import annotations

from pathlib import Path

# --- paths ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"          # gitignored — big intermediates live here (host only)
ARTIFACTS_DIR = REPO_ROOT / "artifacts"  # small committed outputs (parquet/json/map/reports)

RAW_PARQUET = DATA_DIR / "papers.parquet"          # ingest -> id, date, categories, title, abstract
EMBED_MEMMAP = DATA_DIR / "embeddings.f16.memmap"  # embed  -> float16 (N, EMBED_DIM)
PCA_MEMMAP = DATA_DIR / "pca50.f32.memmap"         # reduce -> float32 (N, PCA_DIM)
UMAP_MEMMAP = DATA_DIR / "umap2.f32.memmap"        # reduce -> float32 (N, 2) display-only

# --- corpus / embedding -----------------------------------------------------
EMBED_MODEL = "allenai/specter2_base"      # + proximity adapter "allenai/specter2"
EMBED_DIM = 768
EMBED_MAX_TOKENS = 256
EMBED_BATCH = 128                          # host GPU; tune to VRAM
CORPUS_START = "2007-01"                   # first monthly bin

# --- geometry ---------------------------------------------------------------
PCA_DIM = 50                               # clustering/metric space (NOT umap — avoids manifold distortion)
UMAP_FIT_SAMPLE = 500_000                  # fit UMAP on a stratified sample, transform the rest
N_REGIONS = 1024                           # MiniBatchKMeans microclusters (leaves of the v1.5 tree)
N_SUPERREGIONS = 64                        # Ward-linkage cut for map display + labels (<= 255 uint8 cap)

# --- series -----------------------------------------------------------------
ADD_ALPHA = 1.0                            # add-alpha smoothing before any log/logit (zero months defined)
# Empirical-Bayes: each region-share shrunk toward its super-region share,
# shrinkage weight from region size. Region set is FROZEN per forecast origin.
ACTIVITY_FLOOR_TRAILING12M = 20            # REPORTING filter only (small regions stay in the universe)

# --- forecast ---------------------------------------------------------------
HORIZON_MONTHS = 24
CV_TEST_WINDOW_MONTHS = 24
CV_N_ORIGINS = 4
CV_MIN_TRAIN_MONTHS = 120
EMERGENCE_HORIZON_MONTHS = (19, 24)        # months of the horizon the emergence call reads
# statsforecast roster; "smoothed-momentum" is the serious baseline (custom).
FORECAST_MODELS = ("Naive", "SeasonalNaive", "SmoothedMomentum",
                   "AutoETS", "AutoARIMA", "AutoTheta")
BASELINE_MODEL = "SmoothedMomentum"        # trailing-12m vs prior-12m share growth, carried forward

# --- emergence rules (PRE-REGISTERED — freeze in artifacts/validation.md) ----
TOP_N = 50
G_EMERGING = 1.5                           # emerging = top-50 by g-hat with g-hat >= 1.5
G_DECLINING = 0.75                         # declining = bottom-50 with g-hat <= 0.75 (share decline, not saturation)
ELIGIBILITY_FLOOR = 20                     # >= 20 papers in trailing 12m at T0
SATURATING_GROWTH_24M = 1.3                # flattening-after-growth: trailing-24m growth >= 1.3 ...
SATURATING_G_BAND = (0.9, 1.1)             # ... AND 0.9 <= g-hat <= 1.1

# --- out-of-time validation -------------------------------------------------
VALIDATION_CUTOFFS = ("2017-12", "2020-12", "2022-12")  # one quiet + two regime-shift stress tests
# Pre-registered negative criterion: if the roster fails to beat SmoothedMomentum
# on precision@50 in >= 2 of 3 replicates, the writeup says so plainly.
NEGATIVE_CRITERION_MIN_WINS = 2

# --- map export (matches the site's binary contract exactly) ----------------
# points.bin int16 LE interleaved (x,y) quantized over meta.bbox to [-32000, 32000];
# cluster.bin uint8 super-region id (255 = noise); year.bin uint8 offset from year0.
MAP_QUANT_RANGE = 64000                     # (x - xmin)/(xmax-xmin) * 64000 - 32000
MAP_QUANT_OFFSET = 32000
MAP_YEAR0 = 2007
MAP_DOWNSAMPLE = 150_000                     # proportional per micro-region, min 20/region
MAP_BUDGET_MB = 5.0                          # transferred before first zoom (DevTools check)

# Site repo (sibling under [configured directory] — export_map.py writes here directly.
SITE_MAP_DATA_DIR = (REPO_ROOT.parent / "numberingeometry.github.io"
                     / "projects" / "lacuna" / "map" / "data")


def resolve_sample(sample: int | None) -> str:
    """Human tag for a run's sample size ('full' or e.g. '50k')."""
    if sample is None:
        return "full"
    return f"{sample // 1000}k" if sample >= 1000 else str(sample)
