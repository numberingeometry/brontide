# Explore the research atlas

[Open Brontide](https://numberingeometry.github.io/brontide/). The atlas runs entirely
in the browser from a frozen, versioned data bundle.

[Watch the recorded showcase](https://numberingeometry.github.io/brontide/assets/atlas-showcase.mp4)
for a short introduction to the views and controls.

## Start with a field

Search a territory name or descriptive keyword in the topic browser. Select a territory
to highlight its sampled papers and inspect its monthly publication share, terms and
representative title. The title opens an arXiv search; it is a corpus example and need
not fall inside the selected time window.

The list can be sorted by corpus size, observed change or name. Counts in the inspector
come from aggregate records. The number of dots on screen is a display sample.

## Choose a view

| View | Horizontal position | Height |
|---|---|---|
| Space-time | Fixed semantic projection | Publication year |
| Density | The same projection | Smoothed relative concentration of displayed points |
| 2D map | The same projection | None |

Drag to orbit the 3D scene, shift-drag or right-drag to pan, and scroll or pinch to zoom.
The map tools provide zoom and reset buttons. Arrow keys orbit, `+` and `−` zoom, and
`Home` resets the camera. Press `/` to search. The topic list provides a keyboard-accessible
way to select regions without picking points in the canvas.

Density describes the **stratified display sample**. Its normalization and the sampling
policy mean that it is not an estimate of full-corpus publication intensity. Neither
height nor area measures scientific importance. The projected distances are useful for
exploration, with the usual distortions of dimensionality reduction.

## Follow time

The From and To sliders filter publication years. Play advances the timeline; playback
speed is adjustable. Compare retains earlier papers as a muted baseline while highlighting
the selected window. The activity chart marks the same window.

Dots have annual dates; the chart retains monthly resolution. The last year ends in
June 2026. Percentage-point changes compare the mean observed monthly shares of the
endpoint years, so the partial final year has fewer observed months.

The geometry is fixed from the full frozen corpus. Selecting an earlier year does not
reconstruct a model trained only on literature available then.

## Inspect smaller areas

Within a territory, open **Explore microregions** to load its statistical subareas.
Choose a microregion to inspect its history. The detailed series are loaded on demand.

The public display sample contains broad-region assignments only. Microregion selection
therefore changes the statistical detail without inventing a new assignment for map dots.

## Read the forecast carefully

Observed monthly activity and experimental forecasts are visually separated. Forecasts
start after June 2026 and come from the archived SmoothedMomentum model. Regions were
forecast independently: the forecast shares are **not jointly normalized to sum to 100%**.
The explorer does not show the archived intervals as calibrated uncertainty.

The registered selection procedure met its forecasting criterion at zero of three
evaluation cutoffs. Read [the methods and results](FORECASTING.md) and
[the full limitations](../LIMITATIONS.md) for the scientific interpretation.

## Tour, share and export

**Take a short tour** steps through the views with explanatory captions. **Share view**
copies a URL that restores the selected scene. **Save image** downloads a PNG of the map.
The application respects reduced-motion preferences and retains manual controls.

To run a local copy, serve the repository with `python -m http.server 8000` and open
`http://localhost:8000`. The [data manifest](../artifacts/explorer/manifest.json) records
the snapshot, source artifacts and display limits. Regenerate the bundle with
`python scripts/build_explorer.py`; verify it with the `--check` option.
