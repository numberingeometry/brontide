# brontide

A visual atlas of 2.7 million arXiv papers.

[![Explore research through space and time](assets/atlas.gif)](https://numberingeometry.github.io/brontide/)

Find a field, follow its history, and explore the smaller ideas within it.

**[Open the atlas](https://numberingeometry.github.io/brontide/)** ·
[Field guide](docs/EXPLORER.md) · [Methods & results](docs/FORECASTING.md)

## Run locally

```sh
python -m http.server 8000
```

Open `http://localhost:8000`. The included snapshot needs no corpus download or backend.

The map shows 149,994 sampled papers from January 2007–June 2026. Counts and histories
use the full aggregate data; density describes the display sample. Forecasts are archived
experimental outputs. [Data and limitations](docs/EXPLORER.md).

[Python pipeline](data/README.md) · [Prior work](RELATED-WORK.md) · [MIT license](LICENSE)
