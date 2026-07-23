"""Stage 2 — Embed (host GPU, ~4–8h).

Local SPECTER2: ``allenai/specter2_base`` + the proximity adapter
(``allenai/specter2``). Input = ``title [SEP] abstract``, fp16, max 256 tokens,
batch ~128. Output = a float16 memmap of shape ``(N, 768)`` (~3.8 GB at 2.6M).

Report throughput (papers/sec) so we can trigger the fallback: if GPU
throughput disappoints, download S2 bulk embeddings matched on arXiv IDs
(bigger download, no GPU) instead.
"""
from __future__ import annotations

from pathlib import Path

from . import config


def run(sample: int | None = None,
        parquet: Path = config.RAW_PARQUET,
        out: Path = config.EMBED_MEMMAP) -> Path:
    """Embed titles+abstracts to a float16 memmap. Returns the memmap path.

    Prints papers/sec on completion (the S2-bulk fallback trigger).
    """
    # TODO:
    #   1. transformers.AutoTokenizer/AutoModel.from_pretrained(EMBED_MODEL);
    #      adapters.init(model); model.load_adapter("allenai/specter2",
    #      source="hf", set_active=True). model.half().cuda().eval().
    #   2. text = title + tokenizer.sep_token + abstract; truncate EMBED_MAX_TOKENS.
    #   3. batch EMBED_BATCH; CLS/pooler -> float16; write into
    #      np.memmap(out, dtype=float16, mode="w+", shape=(N, EMBED_DIM)).
    #   4. time it; print N / elapsed papers/sec.
    raise NotImplementedError("embed.run")
