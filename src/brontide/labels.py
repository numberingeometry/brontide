"""Stage 9 — Labels.

c-TF-IDF top terms per super-region (class-based TF-IDF over the 64 super-region
document-aggregates), plus nearest-centroid paper titles as a sanity check.
LLM label polish is a later phase — v1 ships the c-TF-IDF terms.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config

# Vocabulary is fit on a random sample (bounds peak RAM — a full-corpus unigram
# fit is ~3.7 GB and bigrams are a hard OOM on this 16 GB box), then ALL papers
# are streamed to accumulate per-super term counts. Unigram only.
_VOCAB_FIT_SAMPLE = 400_000
_MAX_FEATURES = 30_000
_TOP_TERMS = 15
_SAMPLE_TITLES = 4


def run(out: Path = config.ARTIFACTS_DIR / "labels.json") -> Path:
    """Write ``{super_region_id: {terms: [...], sample_titles: [...]}}``. Returns path."""
    import pyarrow.parquet as pq
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer

    regions_dir = config.ARTIFACTS_DIR                          # regions.run's default out_dir
    labels = np.load(regions_dir / "region_labels.i16.npy").astype(np.int32)
    super_of_region = np.load(regions_dir / "super_region.i16.npy").astype(np.int32)
    paper_super = super_of_region[labels]
    n = labels.shape[0]

    tbl = pq.read_table(config.RAW_PARQUET, columns=["title", "abstract"])
    titles = tbl.column("title").to_pylist()
    abstracts = tbl.column("abstract").to_pylist()
    assert len(titles) == n, f"row-order invariant: {len(titles)} texts vs {n} labels"

    def text(i: int) -> str:
        return f"{titles[i]} {abstracts[i]}"

    # 1. vocabulary on a random sample (deterministic seed)
    rng = np.random.default_rng(0)
    fit_idx = rng.choice(n, size=min(_VOCAB_FIT_SAMPLE, n), replace=False)
    vec = CountVectorizer(ngram_range=(1, 1), lowercase=True, stop_words="english",
                          min_df=25, max_features=_MAX_FEATURES,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b")
    vec.fit(text(i) for i in fit_idx)
    vocab = np.array(vec.get_feature_names_out())
    V = vocab.shape[0]

    # 2. stream ALL papers -> per-super term-count matrix (64 x V)
    counts = np.zeros((config.N_SUPERREGIONS, V), dtype=np.float64)
    CH = 100_000
    for lo in range(0, n, CH):
        hi = min(lo + CH, n)
        X = vec.transform(text(i) for i in range(lo, hi))       # (chunk, V) sparse
        onehot = sparse.csr_matrix((np.ones(hi - lo), (paper_super[lo:hi], np.arange(hi - lo))),
                                   shape=(config.N_SUPERREGIONS, hi - lo))
        counts += (onehot @ X).toarray()                        # sum term counts by super

    # c-TF-IDF (BERTopic form): per-class normalized tf x log(1 + A / f_t)
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    f_t = counts.sum(axis=0)
    avg_words = counts.sum() / config.N_SUPERREGIONS
    ctfidf = tf * np.log(1.0 + avg_words / np.maximum(f_t, 1.0))[None, :]

    # 3. representative titles: papers nearest each super's centroid-of-centroids
    centroids = np.load(regions_dir / "centroids.f32.npy").astype(np.float32)
    pca_meta = json.loads((config.PCA_MEMMAP.with_suffix(config.PCA_MEMMAP.suffix + ".meta.json")).read_text())
    pca = np.memmap(config.PCA_MEMMAP, dtype=np.float32, mode="r",
                    shape=(int(pca_meta["n"]), config.PCA_DIM))

    result = {}
    for s in range(config.N_SUPERREGIONS):
        top = np.argsort(-ctfidf[s])[:_TOP_TERMS]
        rs = np.where(super_of_region == s)[0]
        center = centroids[rs].mean(axis=0) if rs.size else np.zeros(config.PCA_DIM, np.float32)
        members = np.where(paper_super == s)[0]
        sample_titles = []
        if members.size:
            pool = members if members.size <= 50_000 else rng.choice(members, 50_000, replace=False)
            d = ((np.asarray(pca[pool]) - center) ** 2).sum(axis=1)
            sample_titles = [titles[int(j)] for j in pool[np.argsort(d)[:_SAMPLE_TITLES]]]
        result[str(s)] = {"terms": [str(vocab[t]) for t in top],
                          "sample_titles": sample_titles, "n_papers": int(members.size)}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nlabels.run -> {out}\n"
          f"  vocab {V:,} terms (fit on {len(fit_idx):,} sample), {config.N_SUPERREGIONS} supers\n"
          f"  super 0 terms: {', '.join(result['0']['terms'][:8])}")
    return out
