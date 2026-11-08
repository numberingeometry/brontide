"""Stage 2 — Embed (GPU embedding).

Local SPECTER2: ``allenai/specter2_base`` + the proximity adapter
(``allenai/specter2``). Input = ``title [SEP] abstract``, fp16, max 256 tokens,
batch ~128. Output = a float16 memmap of shape ``(N, 768)`` (~3.8 GB at 2.6M).

Report throughput (papers/sec) so we can trigger the fallback: if GPU
throughput disappoints, download S2 bulk embeddings matched on arXiv IDs
(bigger download, no GPU) instead.

Reference benchmark on a GTX 1060 6GB (sm_61): **52.8 papers/sec fp16** (fp32 is
30.7 -- fp16 wins ~1.7x because this workload is memory-bandwidth bound, not
FP16-ALU bound, so Pascal's crippled native fp16 rate does not bite). That puts
2.7M rows at ~14h, which is why this run is designed to be RESUMABLE: a 14-hour
unattended job that cannot restart is a job that loses a whole night to one
driver hiccup. Progress is checkpointed to a JSON sidecar next to the memmap;
re-running picks up at the last completed row instead of starting over.
"""
from __future__ import annotations

import os

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

    A GROWN corpus is an append, not a different run. update.py appends new papers to the
    END of the parquet and never reorders, so rows [0, n_done) still hold the embeddings of
    exactly the papers they held before and only the tail is missing.

    Treating any change in n_total as "different run" was correct while the only way to get
    here was a fresh build, and became a 14-hour bug the moment appends existed: adding 75
    papers to 2.7M made n_total differ, which restarted the embed from row 0. Caught by a
    75-paper trial rather than by the 3-month catch-up, which is the whole point of trying
    the small one first.

    What still forces a restart is a change that invalidates the layout or the meaning of a
    row: a different dimension, a different model or adapter, or a corpus that SHRANK.
    """
    p = _progress_path(out)
    if not (p.exists() and out.exists()):
        return 0
    try:
        state = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return 0

    prev_total = state.get("n_total")
    if state.get("dim") != config.EMBED_DIM:
        print(f"  embedding dim changed ({state.get('dim')} -> {config.EMBED_DIM})"
              " -- starting fresh")
        return 0
    if state.get("model") not in (None, config.EMBED_MODEL) or             state.get("adapter") not in (None, _ADAPTER):
        print(f"  model/adapter changed ({state.get('model')} + {state.get('adapter')})"
              " -- starting fresh")
        return 0
    if not isinstance(prev_total, int) or prev_total > n_total:
        print(f"  corpus shrank or sidecar unusable (n_total={prev_total} -> {n_total})"
              " -- starting fresh")
        return 0
    n_done = int(state.get("n_done", 0))
    if prev_total < n_total:
        print(f"  corpus grew {prev_total:,} -> {n_total:,} (append); "
              f"{n_done:,} rows already embedded are still valid")
    return min(n_done, n_total)


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

    # A "w+" memmap TRUNCATES. That is correct for a first build and catastrophic for
    # anything else, and on 2026-09-04 it was catastrophic: an append made _load_progress
    # return 0 (see the note there), mode became "w+", and opening the memmap zeroed 4.16 GB
    # -- 14 hours of GPU -- in one call, silently, before a single paper was embedded.
    #
    # The resume bug is fixed above. This is the guard that makes the failure impossible
    # rather than merely unlikely: never truncate a file that already holds data unless the
    # caller says so in as many words. A restart that really does need to start over passes
    # BRONTIDE_EMBED_TRUNCATE=1, and then it is a decision rather than an accident.
    if mode == "w+" and out.exists() and out.stat().st_size > 0:
        if os.environ.get("BRONTIDE_EMBED_TRUNCATE") != "1":
            raise RuntimeError(
                f"refusing to truncate {out.name} ({out.stat().st_size / 1e9:.2f} GB of "
                f"existing embeddings) to start from row 0." + "\n" +
                f"  If a full re-embed is genuinely intended, set "
                f"BRONTIDE_EMBED_TRUNCATE=1." + "\n" +
                f"  If this is an APPEND, the progress sidecar should have resumed -- check "
                f"n_total/dim/model in {_progress_path(out).name}."
            )
        print(f"  BRONTIDE_EMBED_TRUNCATE=1 -- discarding "
              f"{out.stat().st_size / 1e9:.2f} GB and starting from row 0")

    tok, model = _load_model()
    mm = np.memmap(out, dtype=np.float16, mode=mode, shape=(n_total, config.EMBED_DIM))

    row = 0            # absolute row index in the parquet
    n_new = 0          # rows embedded in THIS invocation (for honest throughput)
    t0 = time.perf_counter()
    last_ckpt = n_done

    # Skip whole ROW GROUPS that are entirely below n_done rather than reading and
    # discarding them. iter_batches from row 0 decompresses every title and abstract on the
    # way to the resume point: measured appending 75 papers to a 2.7M corpus, that was
    # ~14 minutes of CPU to reach the first row that needed work. A monthly append pays it
    # every time, for nothing. Row groups carry their own row counts in the footer, so the
    # skip is metadata-only.
    start_group = 0
    if n_done:
        cum = 0
        for gi in range(pf.num_row_groups):
            g = pf.metadata.row_group(gi).num_rows
            if cum + g > n_done:
                break
            cum += g
            start_group = gi + 1
        row = cum
        if start_group:
            print(f"  skipping {start_group} row group(s) = {cum:,} rows already embedded")

    def _batches():
        for gi in range(start_group, pf.num_row_groups):
            t = pf.read_row_group(gi, columns=["title", "abstract"])
            for b in t.to_batches(max_chunksize=_READ_CHUNK):
                yield b

    try:
        for chunk in _batches():
            if row >= n_total:
                break
            titles = chunk.column("title").to_pylist()
            abstracts = chunk.column("abstract").to_pylist()

            for i in range(0, len(titles), config.EMBED_BATCH):
                lo = row + i
                # Clamp to the rows ACTUALLY IN THIS CHUNK, not just to n_total. The old
                # reader handed back chunks of exactly _READ_CHUNK (a multiple of
                # EMBED_BATCH), so `titles[i:i+k]` was always full and this was fine by
                # accident. Reading per ROW GROUP makes the last batch of each group short,
                # and then k said 128 while only 32 texts existed -- numpy caught it as
                # "could not broadcast (32,768) into (128,768)" after the model had already
                # run. Three-way min, so the slice, the texts and k can never disagree.
                hi = min(lo + config.EMBED_BATCH, n_total, row + len(titles))
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
