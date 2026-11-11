# Rebuilding the atlas data

The website runs from the committed bundle in `artifacts/explorer/`; no bulk
corpus download is needed. Check that bundle against its source artifacts with:

```sh
python -m pip install -e .
python scripts/build_explorer.py --check
```

A full pipeline rebuild needs separately acquired arXiv metadata and the optional
`pipeline` dependencies. Its large intermediates stay in this ignored directory:

| Output | Stage |
|---|---|
| `papers.parquet` — identifiers, dates, categories, titles and abstracts | `scripts/run_ingest.py` |
| `embeddings/` — SPECTER2 representations | `scripts/run_embed.py` |
| `reduced/` — PCA metric space and UMAP display | `scripts/run_reduce.py` |
| `regions/` — microregions and hierarchy | `scripts/run_regions.py` |

After those stages, the series and export runners build the aggregate tables and
browser data. Read the runners' command-line help and `src/brontide/config.py`
before running them. A complete rebuild requires multi-GB storage and substantial
embedding compute; it is not part of serving or verifying the included snapshot.

Keep new corpus generations separate from the frozen inputs. A wider date range
needs matching date encodings, embeddings, groups and aggregate counts; changing
only the visible axis would misrepresent coverage. [Data and methods](../docs/DATA.md).
