---
type: readme
status: active
opened: 2026-08-01
---

# data/ — compute host-only bulk intermediates (gitignored)

**Nothing here is committed, and nothing here is required to read the results.** This directory is `device-local: true` in the `structure:` block: the heavy stages run on the compute host, and only the small derived outputs cross the wire in [`../artifacts/`](../artifacts/). An ignored directory still documents its shape and its rebuild command — an unexplained empty dir is indistinguishable from an oversight (repository setup, repository organization).

## What belongs here

| path | what | size (order) | produced by |
|---|---|---|---|
| `papers.parquet` | one row per arXiv paper: id, month, categories, title, abstract | ~4 GB | `python scripts/run_ingest.py` |
| `embeddings/*.f16.memmap` | SPECTER2 embeddings, `N×768` float16 | ~3.8 GB | `python scripts/run_embed.py` (GPU) |
| `reduced/` | exact PCA 768→50 (metric) + UMAP 2-D (display only) | ~1 GB | `python scripts/run_reduce.py` |
| `regions/` | KMeans K=1024 microclusters + Ward tree | small | `python scripts/run_regions.py` |

## Rebuild

```sh
python scripts/run_ingest.py && python scripts/run_embed.py && python scripts/run_reduce.py && python scripts/run_regions.py
```

CPU-only environment smoke test without the bulk: every stage accepts `--sample N` (see the repo README).

## Why it is ignored

Regenerable from code plus the public arXiv corpus; the artifacts that record a decision (region series, forecasts, the pre-registration in `artifacts/validation.md`) are committed instead. The distinction is *record vs regenerable intermediate*, not size.
