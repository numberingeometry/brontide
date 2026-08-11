"""Frozen configuration for the lacuna pipeline.

Single source of truth for paths, dimensions, and the **pre-registered**
modeling + emergence constants. Anything the validation harness scores against
lives here and is mirrored, dated, into ``artifacts/validation.md`` before the
first scoring run (pre-registration discipline — see the v1 plan).

"lacuna" is a placeholder name; the final name is deferred (see the README).
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
CORPUS_END = "2026-06"                     # last COMPLETE monthly bin = production forecast origin T0.
# A data-window marker (mirror of CORPUS_START), NOT a scored threshold: the trailing
# partial month is dropped because its RAW count would deflate the eligibility floor
# (shares themselves are ~completeness-invariant). Bump per production run. INVARIANT:
# must stay >= the latest VALIDATION_CUTOFFS T0 + 24m (>= 2024-12), so the scored
# December cutoffs are never truncated. Applied via the production run's T0 origin, NOT
# a global month-filter inside series (the scored masked harness shares that code path).

# --- geometry ---------------------------------------------------------------
PCA_DIM = 50                               # clustering/metric space (NOT umap — avoids manifold distortion)
UMAP_FIT_SAMPLE = 500_000                  # fit UMAP on a stratified sample, transform the rest
N_REGIONS = 1024                           # MiniBatchKMeans microclusters (leaves of the v1.5 tree)
N_SUPERREGIONS = 64                        # Ward-linkage cut for map display + labels (<= 255 uint8 cap)

# --- series -----------------------------------------------------------------
ADD_ALPHA = 1.0                            # add-alpha smoothing before any log/logit (zero months defined)
# Empirical-Bayes region-share smoothing (disambiguated 2026-07-23 — see
# artifacts/validation.md). Each micro-region's monthly share p_r is shrunk toward
# its super-region's PER-REGION share level P_S / n_S (n_S = frozen # of micro-regions
# in that super, from the 64-cut map) — NOT the raw super total P_S, which is ~16x
# larger, breaks scale, and inverts the emerging/declining ranking. Weight
# w_r = n_r / (n_r + kappa) with n_r the region's TRAILING-12m count fixed per origin
# (NOT the raw monthly count ~11 — that drives w->0 every month and collapses g-hat to
# the super-region ranking). kappa is EMPIRICAL-BAYES ESTIMATED, never a hardcoded
# value (that is why no kappa constant lives here) — refit on <= T0 data per origin.
# Region set FROZEN per forecast origin.
EB_SHRINK_TARGET = "super_region_per_region_mean"   # prior mean m_r = P_S / n_S (scale-consistent with p_r)
EB_SHRINK_WEIGHT = "region_trailing12m"             # w_r = n_r/(n_r+kappa), n_r = trailing-12m count, per-origin fixed
EB_KAPPA_ESTIMATOR = "dirichlet_multinomial_mom"    # symmetric-DM concentration, method-of-moments, single global pool
EB_KAPPA_REFIT = "per_forecast_origin_leq_T0"       # refit on <= (fold-train-end) data; NEVER a frozen number
ACTIVITY_FLOOR_TRAILING12M = 20            # REPORTING filter only (small regions stay in the universe)
# Novelty stream: a paper is "novel" when its nearest-centroid distance exceeds the
# NOVELTY_FIT_QUANTILE quantile of the FIT-SET's nearest-centroid distances (registered
# 2026-07-23; was an unregistered "calibrated threshold"). Dimensionless, so its leak
# status = the fit set's: leak-free in the masked run (centroids + quantile from <= T0),
# "novelty vs the final frozen map" (display-only, like UMAP) in the unmasked run.
NOVELTY_FIT_QUANTILE = 0.99

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
ELIGIBILITY_FLOOR = 20                     # >= 20 RAW papers in trailing 12m at T0 (raw count, not smoothed/add-alpha)
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
