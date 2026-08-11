"""Regenerate the published validation table from committed artifacts and diff it.

This is the whole claim of the project, checkable in about two seconds:

    python scripts/verify_results.py

No raw corpus, no embeddings, no GPU, no network, no API key. It re-scores all three
out-of-time cutoffs from ``artifacts/cutoffs/<T0>/`` using the same
``validate._score_cutoff`` the original run used, rebuilds the results table, and compares
it line by line against what is published in ``artifacts/validation.md``.

Exit 0 = the published numbers reproduce exactly. Exit 1 = drift, with a diff.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lacuna import config, validate  # noqa: E402


def rebuild() -> tuple[dict, list[str]]:
    """Re-score every cutoff from committed artifacts. Returns (per_cutoff, table rows)."""
    per_cutoff: dict = {}
    for T0 in config.VALIDATION_CUTOFFS:
        cart = config.ARTIFACTS_DIR / "cutoffs" / T0
        shares, fcp = cart / "region_shares.parquet", cart / "forecasts.parquet"
        if not shares.exists():
            raise SystemExit(f"missing committed artifact: {shares}")
        scores, info = validate._score_cutoff(T0, shares, fcp)
        selected = json.loads(
            fcp.with_suffix(fcp.suffix + ".meta.json").read_text(encoding="utf-8")
        )["selected_model"]
        per_cutoff[T0] = {"selected": selected, "scores": scores, "info": info}

    rows = []
    for T0, d in per_cutoff.items():
        for model, s in d["scores"].items():
            tag = (" **(selected)**" if model == d["selected"]
                   else (" _(baseline)_" if model == config.BASELINE_MODEL else ""))
            rows.append(f"| {T0} | {model}{tag} | {s['precision50']:.3f} | "
                        f"{s['spearman']:.3f} | {s['csi']:.3f} |")
    return per_cutoff, rows


def published() -> list[str]:
    """The table rows currently in artifacts/validation.md."""
    text = (config.ARTIFACTS_DIR / "validation.md").read_text(encoding="utf-8")
    body = text.split("## Results", 1)[1]
    return [ln.strip() for ln in body.splitlines()
            if ln.startswith("| 20")]           # data rows begin with a cutoff year


def main() -> int:
    per_cutoff, rebuilt = rebuild()
    pub = published()

    wins = sum(1 for d in per_cutoff.values()
               if d["scores"].get(d["selected"], {}).get("precision50", 0.0)
               > d["scores"].get(config.BASELINE_MODEL, {}).get("precision50", 0.0))
    nc = validate.evaluate_negative_criterion(wins, len(per_cutoff))

    print(f"re-scored {len(per_cutoff)} cutoffs from committed artifacts "
          f"({', '.join(per_cutoff)})\n")
    print("| cutoff | model | precision@50 | Spearman | CSI |")
    print("|---|---|---|---|---|")
    for r in rebuilt:
        print(r)
    print(f"\nnegative criterion: {nc['wins']}/{nc['of']} "
          f"(need >={nc['min_wins']}) -> {nc['verdict']}")

    if rebuilt == pub:
        print(f"\nOK: all {len(rebuilt)} published rows reproduce exactly.")
        print("See LIMITATIONS.md for what this verdict does and does not license.")
        return 0

    print("\nDRIFT: rebuilt table does not match artifacts/validation.md")
    for a, b in zip(rebuilt, pub):
        if a != b:
            print(f"  rebuilt:   {a}\n  published: {b}")
    if len(rebuilt) != len(pub):
        print(f"  row count: rebuilt {len(rebuilt)} vs published {len(pub)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
