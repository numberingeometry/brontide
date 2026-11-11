# Data and methods

Brontide explores a frozen arXiv snapshot from **January 2007–June 2026**.
The [manifest](../artifacts/explorer/manifest.json) records the source hashes,
date range, encodings and sampling limits.

| Layer | Coverage |
|---|---|
| Indexed corpus | 2,705,992 papers |
| Monthly count tables | 2,691,288 papers across 234 months |
| Display sample | 149,994 papers, with annual dates |
| Grouping | 1,024 microregions within 64 territories |

The committed records do not identify the 14,704 indexed papers outside the
monthly tables. June 2026 is the final observed month; the last year is partial.

## Reading the views

**Currents:** each ribbon's thickness is its territory's paper count over the
trailing 12 months divided by the corpus count over the same window. The first
11 months use the available shorter history. The ribbons sum to 100%; their order
is fixed. Their vertical placement is a layout choice, and neighbors do not imply
transfers, citations or influence. Curves interpolate observations for display.
Hover or focus to read the exact window and its counts alongside the raw month.

**Landscape:** paper representations use SPECTER2, reduced to 50 dimensions by
PCA for clustering and to two by UMAP for display. K-means microregions are grouped
with Ward linkage. Projected distances and gaps are exploratory cues, not calibrated
measures. Density describes the stratified display sample, not full-corpus intensity.

The geometry and grouping were fitted to the full snapshot. Earlier windows are
retrospective views, not reconstructions of what a model could have known then.
Topic names are descriptive labels. Representative links search arXiv by title;
they are not verified per-dot paper identifiers. Microregion statistics are available,
but individual displayed dots have only territory assignments.

## Archived projections

The selected-territory chart also shows a SmoothedMomentum experiment originating
in June 2026, extending to June 2028. It is separate from the observed Currents view.
Regions were forecast independently, so projected shares need not sum to 100%.
The tool does not display calibrated prediction intervals.

The historical selection pipeline beat its momentum comparator at **0 of 3**
evaluation cutoffs. Its validation had inconsistent fold construction and a
selection/evaluation objective mismatch. This is a limitation of that pipeline,
not evidence that scientific change is inherently unpredictable. The frozen
[evaluation record](../artifacts/validation.md) and
[reproduction check](../scripts/verify_results.py) remain available.

## Sources and reproduction

Source metadata: [arXiv](https://arxiv.org/) and its
[metadata access documentation](https://info.arxiv.org/help/bulk_data.html).
Methods: [SPECTER2](https://github.com/allenai/SPECTER2),
[UMAP](https://github.com/lmcinnes/umap) and standard PCA, k-means and Ward clustering.
Brontide is an independent tool and is not affiliated with arXiv.

`python scripts/build_explorer.py --check` verifies the public bundle against its
committed inputs without fitting models or downloading a corpus. See the
[data pipeline](../data/README.md) for local rebuilds and the
[field guide](EXPLORER.md) for controls.
