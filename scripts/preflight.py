"""preflight.py -- run before any job that writes to data/. Cheap; no GPU.

    conda run -n brontide python scripts/preflight.py
    conda run -n brontide python scripts/preflight.py --strict   # exit 1 on warnings too

Checks are the executable half of AGENTS.md's pipeline-safety protocol. Prose is detection;
this is the tier that actually fires. Everything here runs in seconds against artifacts whose
regeneration costs hours.

Exit 0 = safe to proceed. Exit 1 = something a long job would have found the hard way.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from brontide import config  # noqa: E402

FAIL, WARN, OK = "FAIL", "WARN", "ok  "
results: list[tuple[str, str, str]] = []


def add(level: str, name: str, detail: str) -> None:
    results.append((level, name, detail))


def check_sidecars() -> None:
    """Every headerless memmap's byte size must match its declared row count.

    A truncated or half-grown memmap reads as finite zeros rather than raising, so nothing
    downstream would complain -- it would just place papers at the origin. This is the only
    check that catches it.
    """
    specs = [(config.EMBED_MEMMAP, config.EMBED_DIM, 2, "progress"),
             (config.PCA_MEMMAP, config.PCA_DIM, 4, "meta"),
             (config.UMAP_MEMMAP, 2, 4, "meta")]
    for path, width, itemsize, kind in specs:
        if not path.exists():
            add(WARN, f"{path.name} present", "absent (fine on a device without the corpus)")
            continue
        side = path.with_suffix(path.suffix + (".progress.json" if kind == "progress"
                                              else ".meta.json"))
        if not side.exists():
            add(FAIL, f"{path.name} sidecar", f"missing {side.name}")
            continue
        s = json.loads(side.read_text())
        declared = int(s.get("n") or s.get("n_total") or 0)
        actual = path.stat().st_size // (width * itemsize)
        if declared != actual:
            add(FAIL, f"{path.name} size", f"sidecar says {declared:,} rows, file holds {actual:,}")
        else:
            add(OK, f"{path.name} size", f"{actual:,} rows agree with the sidecar")
        if kind == "progress":
            done = int(s.get("n_done", 0))
            if done < declared:
                add(WARN, "embed completeness",
                    f"{done:,}/{declared:,} embedded ({100 * done / max(declared, 1):.1f}%)"
                    " -- a job is in progress or was interrupted")


def check_cross_artifact() -> None:
    """All four artifacts must describe the same number of rows.

    Each sidecar agreeing with its OWN file is not enough, and that gap is not theoretical:
    an append that died in the embed stage left the parquet and embeddings at 2,706,067 with
    pca50 and umap2 at 2,705,992 -- every individual check passed while the corpus was
    internally inconsistent, and re-running the append could not see it. Repair with
    `python -m brontide.update --finish`.
    """
    counts: dict[str, int] = {}
    try:
        import pyarrow.parquet as pq
        counts["papers.parquet"] = pq.ParquetFile(config.RAW_PARQUET).metadata.num_rows
    except Exception:                                          # noqa: BLE001
        pass
    side = config.EMBED_MEMMAP.with_suffix(config.EMBED_MEMMAP.suffix + ".progress.json")
    if side.exists():
        counts["embeddings (n_done)"] = int(json.loads(side.read_text()).get("n_done", 0))
    for path in (config.PCA_MEMMAP, config.UMAP_MEMMAP):
        sc = path.with_suffix(path.suffix + ".meta.json")
        if sc.exists():
            counts[path.name] = int(json.loads(sc.read_text())["n"])
    if len(set(counts.values())) <= 1:
        add(OK, "cross-artifact row counts",
            f"all agree at {next(iter(counts.values()), 0):,}")
    else:
        add(FAIL, "cross-artifact row counts",
            ", ".join(f"{k}={v:,}" for k, v in counts.items())
            + "  -> python -m brontide.update --finish")


def check_zero_rows() -> None:
    """Spot-check for zeroed embedding rows below the progress mark.

    An all-zero embedding is the signature of a truncation, and it is silent: it projects,
    it lands on a centroid, it draws on the map. Only its norm gives it away.
    """
    if not config.EMBED_MEMMAP.exists():
        return
    side = config.EMBED_MEMMAP.with_suffix(config.EMBED_MEMMAP.suffix + ".progress.json")
    if not side.exists():
        return
    done = int(json.loads(side.read_text()).get("n_done", 0))
    if done < 2:
        return
    n_file = config.EMBED_MEMMAP.stat().st_size // (config.EMBED_DIM * 2)
    emb = np.memmap(config.EMBED_MEMMAP, dtype=np.float16, mode="r",
                    shape=(n_file, config.EMBED_DIM))
    rng = np.random.default_rng(0)
    idx = np.unique(rng.integers(0, done, size=min(500, done)))
    norms = np.linalg.norm(np.asarray(emb[idx], dtype=np.float32), axis=1)
    zeros = int((norms == 0).sum())
    if zeros:
        add(FAIL, "embeddings below n_done",
            f"{zeros}/{len(idx)} sampled rows are ALL ZERO -- the memmap was truncated")
    else:
        add(OK, "embeddings below n_done", f"{len(idx)} sampled rows all non-zero")


def check_geometry() -> None:
    """Is the frozen space still re-applicable? (recovers/loads the PCA basis)"""
    # Only rows below n_done have embeddings, so only they can test the basis.
    side = config.EMBED_MEMMAP.with_suffix(config.EMBED_MEMMAP.suffix + ".progress.json")
    ceiling = None
    if side.exists():
        st = json.loads(side.read_text())
        if int(st.get("n_done", 0)) < int(st.get("n_total", 0)):
            ceiling = int(st["n_done"])
    try:
        from brontide import geometry
        r = geometry.verify(600, max_row=ceiling)
    except Exception as e:                                    # noqa: BLE001
        add(WARN, "geometry.verify", f"could not run ({type(e).__name__}: {e})")
        return
    if r["pca_err_median"] > 1e-6:
        add(FAIL, "PCA basis", f"median error {r['pca_err_median']:.2e} against stored pca50")
    else:
        scope = "" if ceiling is None else f" (rows < {ceiling:,}; embed in progress)"
        add(OK, "PCA basis", f"{r['basis_source']}, median error {r['pca_err_median']:.1e}, "
                             f"region agreement {r['region_agreement']:.1%}{scope}")
    add(OK if r["reducer"] else WARN, "UMAP reducer",
        "saved -- transform places new papers to ~0.4% of the map (measured)"
        if r["reducer"]
        else "absent for this build -- new papers place by corpus k-NN (~0.95% of the map)")


def check_batching() -> None:
    p = ROOT / "scripts" / "test_embed_batching.py"
    if not p.exists():
        add(WARN, "batching test", "scripts/test_embed_batching.py missing")
        return
    cp = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, cwd=ROOT)
    add(OK if cp.returncode == 0 else FAIL, "embed index arithmetic",
        "every row written exactly once across 5 resume points" if cp.returncode == 0
        else (cp.stdout or cp.stderr).strip().splitlines()[-1][:90])


def check_backup() -> None:
    """Rule 5: the embeddings cost 14 GPU-hours; a copy costs 4 GB of disk."""
    if not config.EMBED_MEMMAP.exists():
        return
    cands = list(config.DATA_DIR.glob("embeddings.f16.memmap.bak*"))
    gb = config.EMBED_MEMMAP.stat().st_size / 1e9
    if cands:
        add(OK, "embeddings backup", f"{cands[0].name} present")
    else:
        add(WARN, "embeddings backup",
            f"none. {gb:.2f} GB = ~14 GPU-hours with no copy. Before touching embed.py or the "
            f"row count:  copy {config.EMBED_MEMMAP.name} to .bak")


def check_public() -> None:
    """Rule 9: this repo is public, so nothing personal may reach a commit."""
    p = ROOT / "scripts" / "check_public_safe.py"
    if not p.exists():
        add(WARN, "public-safety scan", "scripts/check_public_safe.py missing")
        return
    cp = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, cwd=ROOT)
    if cp.returncode == 0:
        add(OK, "public-safety scan", "no personal or private-project material tracked")
    else:
        n = sum(1 for L in cp.stdout.splitlines() if L.startswith("  ") and "[" in L)
        add(FAIL, "public-safety scan",
            f"{n} finding(s) -- run python scripts/check_public_safe.py")


def check_frozen() -> None:
    """The published artifacts must not have been recomputed."""
    m = config.ARTIFACTS_DIR / "BUILD.json"
    if not m.exists():
        add(WARN, "BUILD.json", "absent -- run python -m brontide.manifest")
        return
    b = json.loads(m.read_text())
    fr = b.get("frozen_result")
    add(OK if fr else WARN, "frozen_result recorded",
        f"{fr['corpus_end']} / {fr['n_papers']:,}" if fr
        else "not yet split from live (set on the first append)")
    add(OK, "generation", str(b.get("generation", "absent (pre-append manifest)")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strict", action="store_true", help="warnings also fail")
    a = ap.parse_args()

    for fn in (check_public, check_sidecars, check_cross_artifact, check_zero_rows, check_geometry, check_batching,
               check_backup, check_frozen):
        try:
            fn()
        except Exception as e:                                # noqa: BLE001
            add(FAIL, fn.__name__, f"{type(e).__name__}: {e}")

    print("\npreflight\n")
    for level, name, detail in results:
        print(f"  [{level}] {name:<28} {detail}")
    fails = sum(1 for l, _, _ in results if l == FAIL)
    warns = sum(1 for l, _, _ in results if l == WARN)
    print(f"\n  {fails} failure(s), {warns} warning(s)")
    if fails:
        print("  DO NOT start a long job until these are resolved.")
    return 1 if (fails or (a.strict and warns)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
