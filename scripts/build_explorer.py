"""Build the public atlas bundle from committed scientific artifacts.

This exporter never fits a model or edits the frozen inputs. Paper coordinates and
annual dates are copied byte-for-byte. Monthly statistics are full aggregate
tables, not estimates from the display sample. Run with ``--check`` in CI to
verify that the checked-in bundle is reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pyarrow.parquet as pq

from brontide.pulse import build_pulse

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "explorer"
INPUTS = (
    "BUILD.json", "region_figs.json", "labels.json", "region_names.json",
    "regions_summary.json", "region_shares.parquet", "forecasts.parquet",
    "emergence.parquet", "super_region.i16.npy", "linkage.f64.npy",
    "map/data/meta.json", "map/data/points.bin", "map/data/cluster.bin",
    "map/data/year.bin",
)


def read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def encode_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def month_sequence(start: str, end: str) -> list[str]:
    y0, m0 = map(int, start.split("-"))
    y1, m1 = map(int, end.split("-"))
    return [f"{i // 12:04d}-{i % 12 + 1:02d}"
            for i in range(y0 * 12 + m0 - 1, y1 * 12 + m1)]


def rounded(values: np.ndarray) -> list[float]:
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values cannot enter the explorer bundle")
    return np.round(values, 7).tolist()


def build_bundle() -> dict[str, bytes]:
    source_hashes = {name: hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest()
                     for name in INPUTS}
    build_id = "atlas-" + hashlib.sha256(encode_json(source_hashes)).hexdigest()[:16]
    build = read_json("BUILD.json")
    figures = read_json("region_figs.json")
    labels = read_json("labels.json")
    names = read_json("region_names.json")["names"]
    meta = read_json("map/data/meta.json")
    summary = read_json("regions_summary.json")
    supers = np.load(ARTIFACTS / "super_region.i16.npy").astype(np.int32)
    linkage = np.load(ARTIFACTS / "linkage.f64.npy")
    k, n_super = len(supers), len(meta["clusters"])
    months = figures["months"]
    forecast_months = figures["forecast_months"]
    if months != month_sequence(months[0], months[-1]):
        raise ValueError("Monthly observation axis must be contiguous")
    if forecast_months != month_sequence(forecast_months[0], forecast_months[-1]):
        raise ValueError("Forecast axis must be contiguous")
    if month_sequence(months[-1], forecast_months[0]) != [months[-1], forecast_months[0]]:
        raise ValueError("Forecast must begin the month after observations end")
    if k != summary["k_regions"] or n_super != summary["n_super"]:
        raise ValueError("Region mappings disagree with the frozen summary")

    # Fill by explicit IDs, never by assuming Parquet row order.
    shares = pq.read_table(ARTIFACTS / "region_shares.parquet")
    row_regions = shares["region"].to_numpy()
    month_index = {month: i for i, month in enumerate(months)}
    row_months = np.array([month_index[m] for m in shares["month"].to_pylist()])
    if len(row_regions) != k * len(months):
        raise ValueError("Incomplete monthly microregion grid")
    cells = row_regions * len(months) + row_months
    if len(np.unique(cells)) != len(cells):
        raise ValueError("Duplicate microregion-month observations")
    if not np.array_equal(supers[row_regions], shares["super_region"].to_numpy()):
        raise ValueError("Monthly records disagree with the hierarchy")
    counts = np.zeros((k, len(months)), dtype=np.int64)
    smoothed = np.zeros_like(counts, dtype=np.float64)
    counts[row_regions, row_months] = shares["count"].to_numpy()
    smoothed[row_regions, row_months] = shares["share_smoothed"].to_numpy() * 100
    if (counts < 0).any() or (smoothed < 0).any():
        raise ValueError("Negative observations in source data")
    monthly_totals = counts.sum(axis=0)
    if (monthly_totals <= 0).any():
        raise ValueError("Empty monthly denominator")

    table = pq.read_table(ARTIFACTS / "forecasts.parquet")
    selected_model = figures["selected_model"]
    chosen = [row for row in table.to_pylist() if row["model"] == selected_model]
    forecast_index = {month: i for i, month in enumerate(forecast_months)}
    forecast = np.full((k, len(forecast_months)), np.nan)
    seen: set[tuple[int, int]] = set()
    for row in chosen:
        index = row["region"], forecast_index[row["month"]]
        if index in seen:
            raise ValueError("Duplicate microregion forecast")
        seen.add(index)
        forecast[index] = row["mean"] * 100
    if len(seen) != k * len(forecast_months) or not np.isfinite(forecast).all():
        raise ValueError("Incomplete or non-finite forecast grid")
    emergence = {int(row["region"]): row
                 for row in pq.read_table(ARTIFACTS / "emergence.parquet").to_pylist()}
    if set(emergence) != set(range(k)):
        raise ValueError("Emergence table must cover each microregion once")

    points = (ARTIFACTS / "map/data/points.bin").read_bytes()
    clusters = (ARTIFACTS / "map/data/cluster.bin").read_bytes()
    years = (ARTIFACTS / "map/data/year.bin").read_bytes()
    n_points = meta["n"]
    if len(points) != n_points * 4 or len(clusters) != n_points or len(years) != n_points:
        raise ValueError("Point binary sizes do not match the original metadata")
    point_cluster = np.frombuffer(clusters, dtype=np.uint8)
    point_year = np.frombuffer(years, dtype=np.uint8)
    if int(point_cluster.max()) >= n_super or int(point_year.max()) + meta["year0"] > meta["year1"]:
        raise ValueError("Point metadata is outside the declared range")
    if not np.array_equal(np.unique(point_cluster), np.arange(n_super)):
        raise ValueError("The map sample must represent every superregion")

    style = {entry["id"]: entry for entry in meta["clusters"]}
    regions = []
    for record in figures["regions"]:
        rid = record["id"]
        children = np.flatnonzero(supers == rid)
        raw = counts[children].sum(axis=0)
        aggregate_history = smoothed[children].sum(axis=0)
        aggregate_forecast = forecast[children].sum(axis=0)
        # Existing figure exports round to five decimal percentage points.
        if not np.allclose(aggregate_history, record["history"], rtol=0, atol=5.1e-6):
            raise ValueError(f"Superregion {rid} history disagrees with the published figures")
        if not np.allclose(aggregate_forecast, record["forecast"], rtol=0, atol=5.1e-6):
            raise ValueError(f"Superregion {rid} forecast disagrees with the published figures")
        representatives = [
            {"title": title,
             "url": "https://arxiv.org/search/?" + urlencode({"query": title,
                         "searchtype": "title", "abstracts": "show", "order": "-announced_date_first"}),
             "link_type": "title_search"}
            for title in labels[str(rid)]["sample_titles"]
        ]
        regions.append({
            **record,
            "name": names[str(rid)], "terms": labels[str(rid)]["terms"],
            "representatives": representatives,
            "color": style[rid]["color"], "cx": style[rid]["cx"], "cy": style[rid]["cy"],
            "trend": style[rid]["trend"], "n_papers": labels[str(rid)]["n_papers"],
            "series_n_papers": int(raw.sum()), "sample_points": int((point_cluster == rid).sum()),
            "raw_counts": raw.tolist(), "raw_share": rounded(raw / monthly_totals * 100),
            "microregion_count": len(children), "microregion_ids": children.tolist(),
        })
    regions.sort(key=lambda r: r["id"])

    microregions = []
    for rid in range(k):
        row = emergence[rid]
        microregions.append({
            "id": rid, "superregion_id": int(supers[rid]), "name": f"Microregion {rid}",
            "history": rounded(smoothed[rid]), "forecast": rounded(forecast[rid]),
            "raw_counts": counts[rid].tolist(),
            "n_papers": int(counts[rid].sum()), "g_hat": float(row["g_hat"]),
            "growth_24m": float(row["growth_24m"]), "classification": row["class"],
            "eligible": bool(row["eligible"]),
        })

    # Preserve the true Ward tree rather than inventing semantic subdivisions.
    if linkage.shape != (k - 1, 4):
        raise ValueError("Unexpected Ward linkage shape")
    members = {rid: {rid} for rid in range(k)}
    tree_counts = {rid: int(counts[rid].sum()) for rid in range(k)}
    super_roots = {int(supers[rid]): rid for rid in range(k)}
    nodes = []
    for j, (left, right, distance, n_members) in enumerate(linkage):
        node_id, left, right = k + j, int(left), int(right)
        if left >= node_id or right >= node_id or distance < 0:
            raise ValueError("Malformed Ward tree")
        members[node_id] = members[left] | members[right]
        if len(members[node_id]) != int(n_members):
            raise ValueError("Ward node size does not match its descendants")
        tree_counts[node_id] = tree_counts[left] + tree_counts[right]
        groups = {int(supers[r]) for r in members[node_id]}
        group = next(iter(groups)) if len(groups) == 1 else None
        if group is not None:
            super_roots[group] = node_id
        nodes.append({"id": node_id, "left": left, "right": right,
                      "distance": round(float(distance), 7), "n_microregions": int(n_members),
                      "n_papers": tree_counts[node_id], "superregion_id": group})
    for sid, node_id in super_roots.items():
        if members[node_id] != set(np.flatnonzero(supers == sid)):
            raise ValueError("Superregion labels are not a cut of the saved Ward tree")

    regions_payload = {
        "schema_version": 1, "build_id": build_id,
        "months": months, "forecast_months": forecast_months,
        "selected_model": selected_model, "units": "percent_of_monthly_arxiv",
        "history_kind": "empirical_bayes_smoothed_share",
        "raw_share_kind": "observed_count_divided_by_monthly_total",
        "monthly_totals": monthly_totals.tolist(), "regions": regions,
    }
    micro_payload = {
        "schema_version": 1, "build_id": build_id,
        "months": months, "forecast_months": forecast_months,
        "selected_model": selected_model, "units": "percent_of_monthly_arxiv",
        "monthly_totals": monthly_totals.tolist(),
        "n_papers_scope": "sum_of_published_monthly_counts",
        "map_point_membership_available": False,
        "regions": microregions,
        "hierarchy": {"root_id": 2 * k - 2, "nodes": nodes, "superregion_roots": super_roots},
    }
    territory_counts = np.stack([np.asarray(r["raw_counts"], dtype=np.int64) for r in regions])
    pulse_years = sorted({int(m[:4]) for m in months if months.count(f"{m[:4]}-12")})
    pulse_payload = {
        "schema_version": 1, "build_id": build_id,
        "measure": "Bhattacharyya angle between square-root trailing-year compositions, degrees",
        "correction": "exact first-order bootstrap bias correction from binomial marginals",
        **build_pulse(territory_counts, counts, months, pulse_years),
    }
    files = {"points": "points.bin", "cluster": "cluster.bin", "year": "year.bin",
             "regions": "regions.json", "microregions": "microregions.json", "pulse": "pulse.json"}
    bundle = {
        "points.bin": points, "cluster.bin": clusters, "year.bin": years,
        "regions.json": encode_json(regions_payload), "microregions.json": encode_json(micro_payload),
        "pulse.json": encode_json(pulse_payload),
    }
    corpus_n = int(build["frozen_result"]["n_papers"])
    if sum(r["n_papers"] for r in regions) != corpus_n:
        raise ValueError("Region corpus totals disagree with the frozen build")
    if sum(r["series_n_papers"] for r in regions) != int(monthly_totals.sum()):
        raise ValueError("Monthly counts do not aggregate across superregions")
    manifest = {
        "schema_version": 1, "build_id": build_id,
        "title": "Brontide research atlas", "snapshot_kind": "frozen_observations",
        "frozen_result": {"n_papers": corpus_n, "corpus_start": months[0], "corpus_end": months[-1]},
        "sample": {
            "n_points": n_points, "year0": meta["year0"], "year1": meta["year1"],
            "encoding_xy": "int16-le-interleaved", "bbox": meta["bbox"],
            "quantization": {"offset": 32000, "range": 64000},
            "encoding_cluster": "uint8-superregion-id", "encoding_year": "uint8-offset-from-year0",
            "time_resolution": "year", "paper_ids_available": False,
            "microregion_membership_available": False,
            "method": "Original stratified sample proportional to microregion size, with at least 20 points per microregion; no second subsample.",
        },
        "monthly_series": {"start": months[0], "end": months[-1], "n_months": len(months),
                           "n_papers": int(monthly_totals.sum()),
                           "corpus_records_outside_series": corpus_n - int(monthly_totals.sum())},
        "geometry": {"embedding": "SPECTER2 proximity adapter", "pca_dimensions": 50,
                     "display_projection": "UMAP 2D", "fit_scope": "full frozen corpus",
                     "n_microregions": k, "n_superregions": n_super,
                     "hierarchy": "Ward linkage over microregion centroids"},
        "forecast": {"origin": months[-1], "start": forecast_months[0], "end": forecast_months[-1],
                     "n_months": len(forecast_months), "model": selected_model,
                     "historical_criterion_passes": 0, "historical_cutoffs": 3,
                     "intervals_displayed": False},
        "files": files,
        "payloads": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                     for name, data in bundle.items()},
        "initial_payload_bytes": sum(len(data) for name, data in bundle.items()
                                     if name not in ("microregions.json", "pulse.json")),
        "sources": {name: {"sha256": digest} for name, digest in source_hashes.items()},
        "notes": [
            "Dots are a display sample; all reported monthly statistics use the published aggregate tables.",
            "Point dates have annual resolution. Monthly controls apply exactly to aggregate series and by year to map dots.",
            "Historical views use a projection fitted on the complete frozen corpus; they are retrospective maps, not reconstructions of what was knowable then.",
            f"The published monthly table covers {int(monthly_totals.sum()):,} records, while the indexed corpus contains {corpus_n:,}. The committed artifacts do not identify the {corpus_n - int(monthly_totals.sum()):,} records outside that table.",
            "Region names are display labels generated from terms and example titles. Representative links search arXiv by title; they are not asserted paper-ID matches.",
            "Microregion statistics and Ward ancestry are available. Per-dot microregion assignments and per-paper identifiers are not in the public map export.",
            "The later append described by BUILD.json is not included in this frozen explorer snapshot.",
            "Forecast curves are historical model projections, not calibrated probabilities or guaranteed future growth.",
            "Pulse angles compare non-overlapping trailing years of the same aggregate counts. They describe arXiv's composition, which also changes when communities begin or stop posting there; they are not a measure of scientific progress.",
        ],
    }
    bundle["manifest.json"] = encode_json(manifest)
    if sum(len(v) for key, v in bundle.items() if key not in ("microregions.json", "pulse.json")) >= 5_000_000:
        raise ValueError("Initial atlas payload exceeds the 5 MB budget")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify checked-in outputs without writing")
    args = parser.parse_args()
    bundle = build_bundle()
    if args.check:
        stale = [name for name, value in bundle.items()
                 if not (OUTPUT / name).exists() or (OUTPUT / name).read_bytes() != value]
        if stale:
            raise SystemExit("Explorer bundle needs rebuilding: " + ", ".join(stale))
        print("PASS: explorer files reproduce exactly from the committed scientific artifacts")
    else:
        OUTPUT.mkdir(exist_ok=True)
        for name, value in bundle.items():
            (OUTPUT / name).write_bytes(value)
        initial = sum(len(v) for key, v in bundle.items() if key not in ("microregions.json", "pulse.json"))
        print(f"Built {len(bundle)} explorer files; {initial:,} initial bytes, "
              f"{len(bundle['microregions.json']):,} lazy microregion bytes, "
              f"{len(bundle['pulse.json']):,} lazy pulse bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
