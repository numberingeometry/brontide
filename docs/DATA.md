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

**Pulse:** for each month from December 2008, compare the 12 months ending
there with the 12 months before. Field shares $p$ and $q$ map to the unit sphere
by square roots, and the displayed angle is $\arccos\sum_k\sqrt{p_kq_k}$, the
Bhattacharyya angle (half the Fisher–Rao distance). It depends only on relative
growth: if every field grew by the same factor, the angle is zero however many
papers were posted. It equals $1-\cos\theta=\tfrac12\sum_k(\sqrt{q_k}-\sqrt{p_k})^2$,
so it splits into nonnegative field terms; “What moved” lists the largest of
these as shares of the territory-level turn.

Counts are finite, so two samples from an unchanged mix still differ, most
visibly across 1,024 small microregions. Each window is treated as a multinomial
draw from its observed shares; the expectation of each field term under that
model needs only binomial moments of $\sqrt{X/n}$, computed exactly. Subtracting
the implied bias is a deterministic first-order correction. In simulations from
the real 2014→2015 microregion counts it removes about 98% of the bias. The
dotted line shows the expected angle under no change, for scale. Raw angles are
kept in [`pulse.json`](../artifacts/explorer/pulse.json).

The darker series uses the 64 territories and the lighter one the 1,024
microregions; the difference is turnover inside territories. The long-turn panel
projects the calendar-year compositions for 2007–2025 onto their two principal
directions; it shortens distances and is for orientation. Peak labels name the
largest contributor at prominent local maxima (prominence at least 1°).

Pulse measures arXiv's composition. It also moves when communities begin or stop
posting to arXiv, when categories change, or when a field's preprint habits
change. It is not a measure of progress, quality, novelty or influence.

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
committed inputs without fitting models or downloading a corpus.
`python scripts/test_pulse.py` checks the pulse arithmetic against direct
recomputation and brute-force binomial sums. See the
[data pipeline](../data/README.md) for local rebuilds and the
[field guide](EXPLORER.md) for controls.
