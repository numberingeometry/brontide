"""Stage 9b — Human-readable names for the 64 super-regions (CPU-only environment).

`labels.py` produces c-TF-IDF term lists per super-region — accurate but not readable:
``pose, depth, scene, reconstruction, rendering`` is a fingerprint, not a name. Every figure,
axis label, and tooltip downstream wants ``3D Reconstruction & Neural Rendering`` instead.

This stage asks an LLM to compress each region's top terms plus a few sample titles into a short
noun-phrase name. The output is **committed** (``artifacts/region_names.json``) so that nothing
downstream — figures, dashboard, README — needs an API key or network access to render. Re-running
is only necessary if the region map itself is recut.

Naming is cosmetic by construction: names are never inputs to the forecast, the emergence
statistic, or the scored validation. A wrong name is a legibility bug, not a result bug.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import config

MODEL = "claude-opus-5"

PROMPT = """\
You are naming clusters of arXiv papers for a research-frontier map.

Below are {n} clusters. Each has its top c-TF-IDF terms and one representative paper title.
Give each cluster a short, human-readable name that a researcher outside that subfield would
recognise.

Rules:
- 2-5 words, title case, noun phrase. No trailing punctuation.
- Be specific enough to distinguish neighbours: "Medical Image Segmentation" not "Medical AI";
  "Diffusion & Generative Models" not "Generative AI".
- Use the field's own vocabulary, not a lay paraphrase.
- Use "&" rather than "and" when joining two distinct topics.
- Do not number the names or restate the cluster id in the name.

Return ONLY a JSON object mapping each cluster id (as a string) to its name.

{clusters}
"""


def _render_clusters(labels: dict) -> str:
    """Render the cluster block of the prompt from a labels.json mapping."""
    lines = []
    for k in sorted(labels, key=int):
        terms = ", ".join(labels[k]["terms"][:10])
        titles = labels[k].get("sample_titles") or []
        lines.append(f"{k}| terms: {terms}")
        if titles:
            lines.append(f"   example title: {titles[0][:100]}")
    return "\n".join(lines)


def build_prompt(labels: dict) -> str:
    """The exact prompt used to produce the committed artifacts/region_names.json."""
    return PROMPT.format(n=len(labels), clusters=_render_clusters(labels))


def run(labels_path: Path = config.ARTIFACTS_DIR / "labels.json",
        out: Path = config.ARTIFACTS_DIR / "region_names.json",
        model: str = MODEL) -> Path:
    """Regenerate region names via the Anthropic API. Requires ANTHROPIC_API_KEY.

    The committed artifact was produced with this prompt and model; downstream code reads the
    committed file and never calls this. Returns the output path.
    """
    import anthropic  # deferred — not needed to read the committed artifact

    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    prompt = build_prompt(labels)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    names = json.loads(text)

    missing = set(labels) - set(names)
    if missing:
        raise RuntimeError(f"model omitted {len(missing)} regions: {sorted(missing)[:5]}")

    payload = {
        "model": model,
        "n_regions": len(names),
        "provenance": "LLM-generated from c-TF-IDF terms + sample titles; cosmetic only, never "
                      "an input to any scored quantity",
        "names": {k: names[k] for k in sorted(names, key=int)},
    }
    Path(out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return Path(out)


def load(path: Path = config.ARTIFACTS_DIR / "region_names.json") -> dict[int, str]:
    """Read the committed names as {super_region_id: name}. No API key, no network."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): v for k, v in payload["names"].items()}
