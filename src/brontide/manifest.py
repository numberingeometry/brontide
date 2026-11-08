"""Build identity — one small file that says which brontide build is on disk.

brontide was written as a publication: the contract it meets is *reproduce the frozen
number*, and ``scripts/verify_results.py`` meets it. It is now also a **dependency** —
downstream tools project new papers into this build's frozen geometry and store the
resulting row indices and region ids alongside their own records. That is a different
contract, and it needs one thing this repo did not publish: a way to tell whether the
geometry underfoot has moved.

Why a manifest rather than a version constant. The stages run independently and are
re-runnable one at a time — re-projecting UMAP without re-ingesting is normal — so a
single id stamped at ingest would go stale the moment a later stage ran alone. The
manifest instead describes *what is actually on disk right now*, and derives ``build_id``
from the artifact digests, so it changes if and only if an artifact changed.

What a consumer should do:

    m = json.load(open(BRONTIDE / "artifacts" / "BUILD.json"))
    if m["build_id"] != my_cached_build_id:
        ...  # every row index, region id and map coordinate you cached is void

**Row indices and region ids are build-scoped.** A row index addresses
``embeddings.f16.memmap`` / ``pca50.f32.memmap`` / ``umap2.f32.memmap``; a rebuild on a
grown corpus renumbers every one of them. Region and super-region ids are worse, because
they look stable: ``MiniBatchKMeans`` is seeded (``random_state=0``) and so is exactly
reproducible **given identical input** — but a refreshed corpus is not identical input,
so "super-region 55" is an artifact of one fit, not a name. Cache the *name* alongside
the id and re-resolve both when ``build_id`` moves.

Digests. Small files are hashed whole. The memmaps are 0.5-4 GB and hashing them costs
minutes for a guarantee nothing here needs, so they get a *quick digest* — size plus
sha256 of the head, middle and tail megabyte. It catches a rebuild, a truncation and a
resumed-but-incomplete write; it is not a full content hash and is labelled ``quick`` so
no one mistakes it for one. ``--full`` hashes everything if you want the real thing.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import config

# artifacts/, not data/: `/data/*` is gitignored (regenerable intermediates), and the whole
# point of this file is that an installation without the raw corpus -- including a reader of
# the GitHub repository -- can still tell which build the ids and coordinates it holds came
# from. It is ~2 KB and describes the build, not the device; artifacts missing locally
# are recorded `present: false` rather than omitted.
MANIFEST = config.ARTIFACTS_DIR / "BUILD.json"
SCHEMA = 1

# Files a downstream consumer indexes into or projects against. Paths are recorded
# relative to the repo root so the manifest is readable from a checkout on any machine.
_ARTIFACTS = [
    ("papers.parquet", config.RAW_PARQUET),
    ("embeddings.f16.memmap", config.EMBED_MEMMAP),
    ("pca50.f32.memmap", config.PCA_MEMMAP),
    ("umap2.f32.memmap", config.UMAP_MEMMAP),
    ("centroids.f32.npy", config.ARTIFACTS_DIR / "centroids.f32.npy"),
    ("super_region.i16.npy", config.ARTIFACTS_DIR / "super_region.i16.npy"),
    ("linkage.f64.npy", config.ARTIFACTS_DIR / "linkage.f64.npy"),
    ("region_names.json", config.ARTIFACTS_DIR / "region_names.json"),
    ("regions_summary.json", config.ARTIFACTS_DIR / "regions_summary.json"),
]

_QUICK_THRESHOLD = 64 * 1024 * 1024     # hash whole files up to here
_QUICK_WINDOW = 1024 * 1024             # head/middle/tail bytes for the quick digest


def _digest(path: Path, full: bool = False) -> dict:
    size = path.stat().st_size
    h = hashlib.sha256()
    if full or size <= _QUICK_THRESHOLD:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return {"bytes": size, "sha256": h.hexdigest(), "digest": "full"}
    with open(path, "rb") as fh:
        for off in (0, max(0, size // 2 - _QUICK_WINDOW // 2), max(0, size - _QUICK_WINDOW)):
            fh.seek(off)
            h.update(fh.read(_QUICK_WINDOW))
    h.update(str(size).encode())
    return {"bytes": size, "sha256": h.hexdigest(), "digest": "quick"}


def _n_papers() -> int | None:
    """Row count, from the PCA sidecar and then the parquet footer. Never a literal."""
    meta = config.PCA_MEMMAP.with_suffix(config.PCA_MEMMAP.suffix + ".meta.json")
    try:
        return int(json.loads(meta.read_text())["n"])
    except (OSError, KeyError, ValueError, TypeError):
        pass
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(config.RAW_PARQUET).metadata.num_rows)
    except Exception:
        return None


def build(full: bool = False) -> dict:
    """Assemble the manifest for whatever is on disk. Missing artifacts are recorded
    as missing rather than skipped — a consumer must be able to see a partial build."""
    artifacts: dict[str, dict] = {}
    for name, path in _ARTIFACTS:
        rel = os.path.relpath(path, config.REPO_ROOT).replace("\\", "/")
        if not path.exists():
            artifacts[name] = {"path": rel, "present": False}
            continue
        d = _digest(path, full)
        d.update({"path": rel, "present": True, "mtime_ns": path.stat().st_mtime_ns})
        artifacts[name] = d

    # build_id is derived, never assigned: it is a digest of the artifact digests, so it
    # moves if and only if something a consumer depends on moved. Sorted for determinism.
    acc = hashlib.sha256()
    for name in sorted(artifacts):
        a = artifacts[name]
        acc.update(f"{name}:{a.get('sha256', 'missing')}:{a.get('bytes', 0)}".encode())

    return {
        "schema": SCHEMA,
        "build_id": acc.hexdigest()[:16],
        "corpus_start": config.CORPUS_START,
        "corpus_end": config.CORPUS_END,
        "n_papers": _n_papers(),
        "embed_model": config.EMBED_MODEL,
        "embed_adapter": "allenai/specter2",
        "embed_dim": config.EMBED_DIM,
        "pca_dim": config.PCA_DIM,
        "n_regions": config.N_REGIONS,
        "n_superregions": config.N_SUPERREGIONS,
        "ids_are_build_scoped": True,
        "note": ("Row indices address the memmaps of THIS build; region and super-region "
                 "ids are the labels of THIS k-means fit. Both are renumbered by any "
                 "rebuild on a changed corpus. Compare build_id before reusing either."),
        "artifacts": artifacts,
    }


def write(full: bool = False, out: Path = MANIFEST) -> Path:
    m = build(full)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2) + "\n")
    missing = [k for k, v in m["artifacts"].items() if not v.get("present")]
    print(f"\nmanifest -> {out}\n"
          f"  build_id : {m['build_id']}\n"
          f"  corpus   : {m['corpus_start']} .. {m['corpus_end']}  "
          f"n={m['n_papers']:,}" if m["n_papers"] else "  corpus   : n unknown")
    print(f"  artifacts: {len(m['artifacts']) - len(missing)}/{len(m['artifacts'])} present"
          + (f"   MISSING: {', '.join(missing)}" if missing else ""))
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Write artifacts/BUILD.json describing the build on disk.")
    p.add_argument("--full", action="store_true",
                   help="full sha256 of the multi-GB memmaps (minutes) instead of the quick digest")
    write(p.parse_args().full)


if __name__ == "__main__":
    main()
