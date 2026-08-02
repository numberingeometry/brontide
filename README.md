# lacuna — research-frontier forecasting *(placeholder name)*

Forecast the **temporal dynamics of the research frontier in embedding space**: which regions of the arXiv SPECTER2 embedding cloud are **emerging vs saturating**. This is the v2 of `arxiv-heatmap` — a résumé / tool build, headlined by a **pre-registered, multi-cutoff out-of-time harness against a smoothed-momentum baseline**.

> **Name is a placeholder.** "lacuna" is the internal codename; the final name is chosen from the vetted dossier once the build takes shape (`local workspace/projects/lacuna/notes/lacuna/lacuna-2026-07-21-rename-vetting.md`). Rename = one `gh repo rename` later.

## Start here (60 seconds)

1. **[LEDGER.md](LEDGER.md)** — the claim ledger (generated; the pre-registered L1 validation rows live here). Hand-edited source: [`blueprint/claims.yaml`](blueprint/claims.yaml); gate: `python blueprint/check.py` (also fires as the pre-commit hook).
2. **The work** — [`src/lacuna/`](src/lacuna/) (the 10 stages + `config.py`) · [`scripts/`](scripts/) (thin runners) · [`notes/`](notes/) (repo-claim evidence) · [`notebooks/`](notebooks/) (EDA only) · [`artifacts/`](artifacts/) (committed small outputs + the pre-registration record).
3. **Decisions** — [`decisions/`](decisions/) (dated ADRs, immutable; index in its README).
4. **The local editor configuration** — [*_LINK.md](*_LINK.md) → the repository's lacuna hub (positioning dossier, v1 plan, rename vetting).

## Positioning (honest — no "first/only" language)

Large-scale research maps already exist — **Lacuna** (Weiss et al., 2026) charts 734K ML papers into 27K LLM-derived directions, and **CSET/ETO's Map of Science** clusters tens of millions of documents. Region-level growth forecasting with out-of-time evaluation was demonstrated by **Klavans, Boyack & Murdick (PLOS ONE 2020)**, whose model still powers the Map of Science's "extreme growth" predictions. This project claims **no new formulation**: it combines an all-arXiv embedding map with corpus-wide region-**share** time-series forecasts and holds them to a standard none of those systems apply — pre-registered, multi-cutoff out-of-time validation against a smoothed-momentum baseline. Cite K&B 2020 + CSET/ETO as direct predecessors everywhere; scale is not a claim (K&B is ~15× bigger).

## Pipeline (stage → where it runs)

| # | Stage | Module | Runs on | Output |
|---|---|---|---|---|
| 1 | Ingest | `ingest.py` | compute host (~1h) | `data/papers.parquet` (id, month, categories, title, abstract) |
| 2 | Embed | `embed.py` | **compute host GPU (4–8h)** | float16 memmap `N×768` (SPECTER2 + proximity adapter) |
| 3 | Reduce | `reduce.py` | compute host | IncrementalPCA 768→50 (metric); UMAP 50→2 (display only) |
| 4 | Regions | `regions.py` | compute host (~30m) | KMeans **K=1024** microclusters + Ward tree → 64 super-regions |
| 5 | Series | `series.py` | compute host | EB-smoothed region-**share** monthly series + novelty stream |
| 6 | Forecast | `forecast.py` | compute host/CPU-only environment | statsforecast roster, 24m horizon, rolling-origin CV |
| 7 | Emergence | `emergence.py` | CPU-only environment | rank by ĝ; emerging / declining / saturating (pre-registered) |
| 8 | Validate | `validate.py` | CPU-only environment | 3 cutoffs, precision@50 + Spearman + CSI vs smoothed-momentum |
| 9 | Labels | `labels.py` | compute host | c-TF-IDF top terms per super-region |
| 10 | Export | `export_map.py`, `export_figs.py` | CPU-only environment | site's `map/data/*.bin` + per-region forecast JSON |

All numeric knobs are frozen in **`src/lacuna/config.py`** and mirrored, dated, into **`artifacts/validation.md`** (the pre-registration record) **before** any scoring run.

## Cross-device workflow (CPU-only environment ↔ compute host)

- **GitHub is the sync bus**, not OneDrive (this repo lives off OneDrive; OneDrive syncs files-not-commits and would silently diverge).
- **CPU-only environment** (storage-limited): scaffold, module code, `--sample` smoke tests, exports, site integration, git orchestration.
- **compute host** (GPU, ≥100 GB): the heavy stages 1–5 (+9). Reports SPECTER2 throughput after embed so we can trigger the S2-bulk fallback if needed.
- **Only small `artifacts/` cross the wire** (region-series parquet ~2 MB, map export ~1–2 MB, forecast JSON, validation report). The 4 GB raw + 3.8 GB embeddings are gitignored — they live **only on the compute host**; the CPU-only environment works from committed artifacts.

## Layout

```
src/lacuna/    ingest embed reduce regions series forecast emergence
               validate labels export_map export_figs  + config.py
scripts/       run_<stage>.py thin CLI wrappers  + smoke_test.py
notebooks/     EDA / reports only
data/          gitignored — big intermediates (compute host)
artifacts/     committed small outputs + validation.md (pre-registration)
```

## Running

```sh
conda env create -f environment.yml && conda activate lacuna
pip install -e .

# Every stage supports --sample N (CPU-only environment smoke test before the compute host full run):
python scripts/run_ingest.py   --sample 50000
python scripts/smoke_test.py   --sample 50000   # whole pipeline on a sample

# compute host full run (no --sample):
python scripts/run_ingest.py && python scripts/run_embed.py && ...
```

## Milestones (~1–2 days/week; the quantum D′ paper keeps priority hours)

- **End of July** — repo scaffolded + private remote *(done)*; ingest + embed on compute host; first real UMAP quick-look.
- **Mid-August** — regions/series/forecast harness; **first out-of-time validation numbers** (double go/no-go: beat smoothed momentum? citable writeup or portfolio-only?); site map swapped to real data.
- **Mid-September** — emergence briefs + forecast figure on the site; validation writeup in README; pick the final name; shared tooling bookkeeping (build stub via change-control + integrity checker).
- Buffer to **Sep 30, 2026**.

Full approved plan: `local workspace/projects/lacuna/notes/lacuna/lacuna-2026-07-22-v1-plan.md`.
