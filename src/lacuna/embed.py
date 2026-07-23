"""Stage 2 — Embed (host GPU, ~4–8h).

Local SPECTER2: ``allenai/specter2_base`` + the proximity adapter
(``allenai/specter2``). Input = ``title [SEP] abstract``, fp16, max 256 tokens,
batch ~128. Output = a float16 memmap of shape ``(N, 768)`` (~3.8 GB at 2.6M).

Report throughput (papers/sec) so we can trigger the fallback: if GPU
throughput disappoints, download S2 bulk embeddings matched on arXiv IDs
(bigger download, no GPU) instead.

Measured on the host's GTX 1060 6GB (sm_61): **52.8 papers/sec fp16** (fp32 is
30.7 -- fp16 wins ~1.7x because this workload is memory-bandwidth bound, not
FP16-ALU bound, so Pascal's crippled native fp16 rate does not bite). That puts
2.7M rows at ~14h, which is why this run is designed to be RESUMABLE: a 14-hour
unattended job that cannot restart is a job that loses a whole night to one
driver hiccup. Progress is checkpointed to a JSON sidecar next to the memmap;
re-running picks up at the last completed row instead of starting over.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config

_ADAPTER = "allenai/specter2"
_ADAPTER_NAME = "proximity"

# Rows pulled from parquet per read. Independent of EMBED_BATCH (the GPU batch);
# this only controls how much text sits in host RAM at once.
_READ_CHUNK = 8192
# Checkpoint cadence. Cheap (a flush + a small JSON write), so keep it frequent:
# this is the maximum work a crash can cost.
_CHECKPOINT_EVERY = 50_000


def _progress_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".progress.json")


def _load_progress(out: Path, n_total: int) -> int:
    """Rows already embedded, or 0 to start fresh.

    Restarts from scratch if the sidecar describes a different run (row count or
    dimension changed), since the memmap layout would no longer line up.
    """
    p = _progress_path(out)
    if not (p.exists() and out.exists()):
        return 0
    try:
        state = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    if state.get("n_total") != n_total or state.get("dim") != config.EMBED_DIM:
        print(f"  progress sidecar is for a different run "
              f"(n_total={state.get('n_total')}, dim={state.get('dim')}) -- starting fresh")
        return 0
    return int(state.get("n_done", 0))


def _save_progress(out: Path, n_total: int, n_done: int) -> None:
    _progress_path(out).write_text(json.dumps({
        "n_total": n_total, "n_done": n_done, "dim": config.EMBED_DIM,
        "model": config.EMBED_MODEL, "adapter": _ADAPTER, "dtype": "float16",
    }, indent=2))


def _load_model():
    """SPECTER2 base + proximity adapter, fp16 on GPU, eval mode."""
    import torch
    from transformers import AutoTokenizer

    from adapters import AutoAdapterModel

    tok = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    model = AutoAdapterModel.from_pretrained(config.EMBED_MODEL)
    model.load_adapter(_ADAPTER, source="hf", load_as=_ADAPTER_NAME, set_active=True)
    # set_active on load_adapter does not always stick -- transformers then warns
    # "adapters available but none are activated" and silently embeds with the
    # BASE model, which would be wrong embeddings rather than a crash. Force it
    # and verify.
    model.set_active_adapters(_ADAPTER_NAME)
    active = str(model.active_adapters)
    if _ADAPTER_NAME not in active:
        raise RuntimeError(
            f"proximity adapter did not activate (active_adapters={active!r}); "
            "refusing to embed with the bare base model"
        )
    print(f"  adapter active: {active}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable -- this stage is the GPU job")
    model = model.half().cuda().eval()
    return tok, model


def run(sample: int | None = None,
        parquet: Path = config.RAW_PARQUET,
        out: Path = config.EMBED_MEMMAP) -> Path:
    """Embed titles+abstracts to a float16 memmap. Returns the memmap path.

    Prints papers/sec on completion (the S2-bulk fallback trigger).
    """
    import numpy as np
    import pyarrow.parquet as pq
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(parquet)
    n_total = pf.metadata.num_rows if sample is None else min(sample, pf.metadata.num_rows)

    n_done = _load_progress(out, n_total)
    mode = "r+" if n_done else "w+"
    if n_done:
        print(f"  RESUMING at row {n_done:,}/{n_total:,} "
              f"({100 * n_done / n_total:.1f}% already embedded)")
    if n_done >= n_total:
        print("  already complete -- nothing to do")
        return out

    tok, model = _load_model()
    mm = np.memmap(out, dtype=np.float16, mode=mode, shape=(n_total, config.EMBED_DIM))

    row = 0            # absolute row index in the parquet
    n_new = 0          # rows embedded in THIS invocation (for honest throughput)
    t0 = time.perf_counter()
    last_ckpt = n_done

    try:
        for chunk in pf.iter_batches(batch_size=_READ_CHUNK, columns=["title", "abstract"]):
            if row >= n_total:
                break
            titles = chunk.column("title").to_pylist()
            abstracts = chunk.column("abstract").to_pylist()

            for i in range(0, len(titles), config.EMBED_BATCH):
                lo = row + i
                hi = min(lo + config.EMBED_BATCH, n_total)
                if hi <= lo:
                    break
                if hi <= n_done:            # already embedded on a previous run
                    continue
                k = hi - lo
                texts = [f"{t}{tok.sep_token}{a}"
                         for t, a in zip(titles[i:i + k], abstracts[i:i + k])]
                enc = tok(texts, padding=True, truncation=True,
                          max_length=config.EMBED_MAX_TOKENS, return_tensors="pt")
                with torch.no_grad():
                    hidden = model(input_ids=enc["input_ids"].cuda(),
                                   attention_mask=enc["attention_mask"].cuda()
                                   ).last_hidden_state[:, 0, :]   # CLS
                mm[lo:hi] = hidden.to(torch.float16).cpu().numpy()
                n_new += k

                if lo + k - last_ckpt >= _CHECKPOINT_EVERY:
                    mm.flush()
                    last_ckpt = lo + k
                    _save_progress(out, n_total, last_ckpt)
                    rate = n_new / (time.perf_counter() - t0)
                    eta_h = (n_total - last_ckpt) / rate / 3600 if rate else float("nan")
                    print(f"  {last_ckpt:,}/{n_total:,} "
                          f"({100 * last_ckpt / n_total:5.1f}%)  "
                          f"{rate:5.1f} papers/sec  ETA {eta_h:4.1f} h", flush=True)

            row += len(titles)
    finally:
        # Even on Ctrl-C / crash, persist what we have so the restart is cheap.
        mm.flush()
        done = max(last_ckpt, min(row, n_total))
        _save_progress(out, n_total, done)
        del mm

    elapsed = time.perf_counter() - t0
    rate = n_new / elapsed if elapsed else 0.0
    _save_progress(out, n_total, n_total)
    print(f"\nembed[{config.resolve_sample(sample)}] -> {out}\n"
          f"  rows        : {n_total:,} ({n_new:,} this run)\n"
          f"  shape/dtype : ({n_total}, {config.EMBED_DIM}) float16\n"
          f"  size        : {out.stat().st_size / 1e9:.2f} GB\n"
          f"  throughput  : {rate:.1f} papers/sec  ({elapsed / 3600:.2f} h this run)")
    return out
