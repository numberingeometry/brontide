"""When does the hierarchy start to matter? A quantified scope condition for the H2 null.

verify_hierarchy_controls.py found H2 null: replacing the learned Ward hierarchy with a
human-built one derived from the 152 arXiv categories -- two trees whose 64-cut partitions
share an ARI of only 0.1989 -- leaves the top-k verdict identical through k=400.

That null is NOT unconditional, and saying so without bounding it would be the sort of
overreach this line of work objects to. The hierarchy enters `share_smoothed` through
exactly one term:

    p_hat = w * p + (1 - w) * m,        w_r = trail_r / (trail_r + kappa)

and m is the only hierarchy-dependent quantity. On this instrument w has median 0.9879, so
the hierarchy contributes a 1.2% correction. The null is a statement about an instrument
that barely shrinks.

PRE-STATED PREDICTION (exploratory; NOT one of the frozen H1-H3 quantities):
    Raising kappa lowers w and hands the hierarchy more weight. The top-k symmetric
    difference between the Ward-verdict and the category-verdict should therefore rise
    monotonically as median w falls, and there should be a threshold w below which the two
    hierarchies give materially different answers (symdiff >= 0.10, the H2 criterion).
    If instead symdiff stays ~0 even at w -> 0, the hierarchy is irrelevant for a deeper
    reason than weighting, and the explanation above is wrong.

Everything here is EXPLORATORY: kappa is an estimated nuisance parameter in the frozen
pipeline, and overriding it is a sensitivity analysis, not a re-run of a frozen quantity.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from brontide import config                                              # noqa: E402
from verify_margin import g_ratio                                        # noqa: E402
from verify_reassignment_bootstrap import top_n                          # noqa: E402
from verify_hierarchy_controls import aitchison_tree, category_profiles  # noqa: E402

T0 = "2024-06"
KS = [5, 10, 25, 50, 100, 200, 400, 600]
KAPPAS = [3.147960961354892, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 100000.0]


def eb_fixed(C, sup, nsup, n_anchor, kappa):
    """The EB map with kappa SUPPLIED rather than estimated."""
    a, K = config.ADD_ALPHA, C.shape[0]
    p = (C + a) / (C.sum(axis=0) + a * K)[None, :]
    P_S = np.zeros((nsup, C.shape[1]), np.float64)
    np.add.at(P_S, sup, p)
    n_s = np.bincount(sup, minlength=nsup)
    m = (P_S / np.maximum(n_s, 1)[:, None])[sup]
    trail = C[:, max(0, n_anchor - 12):n_anchor].sum(axis=1)
    w = trail / (trail + kappa)
    return w[:, None] * p + (1.0 - w[:, None]) * m, w


def main() -> int:
    art, data = REPO / "artifacts", REPO / "data"
    K, nsup = config.N_REGIONS, config.N_SUPERREGIONS
    sh = pd.read_parquet(art / "region_shares.parquet")
    months = sorted(sh["month"].unique())
    i_t0, i_anchor = months.index(T0) + 1, len(months)
    Cnt = (sh.pivot(index="region", columns="month", values="count")
             .reindex(range(K))[months].to_numpy(np.float64))

    labels = np.load(data / "labels" / "region_labels.i16.npy")
    P, vocab, groups = category_profiles(labels, K)
    cutW = fcluster(np.load(art / "linkage.f64.npy"), nsup, criterion="maxclust") - 1
    cutC = fcluster(aitchison_tree(P), nsup, criterion="maxclust") - 1
    elig = np.where(Cnt[:, i_t0 - 12:i_t0].sum(axis=1) >= config.ELIGIBILITY_FLOOR)[0]

    print(f"{len(vocab)} categories, {len(groups)} groups; {len(elig)} eligible regions\n")
    print(f"{'kappa':>10} {'median w':>9} {'1-w':>7}  " + "  ".join(f"k={k}" for k in KS) + "   median")
    print("-" * 96)

    rows = []
    for kap in KAPPAS:
        XW, w = eb_fixed(Cnt, cutW, nsup, i_anchor, kap)
        XC, _ = eb_fixed(Cnt, cutC, nsup, i_anchor, kap)
        gW, gC = g_ratio(XW, i_t0), g_ratio(XC, i_t0)
        sym = {k: len(top_n(gW, elig, k) ^ top_n(gC, elig, k)) / (2 * k) for k in KS}
        med = float(np.median(list(sym.values())))
        mw = float(np.median(w))
        rows.append({"kappa": kap, "median_w": mw, "symdiff": sym, "median_symdiff": med})
        print(f"{kap:>10.1f} {mw:>9.4f} {1-mw:>7.4f}  "
              + "  ".join(f"{sym[k]:>5.3f}" for k in KS) + f"   {med:>6.3f}")

    meds = [r["median_symdiff"] for r in rows]
    monotone = all(meds[i] <= meds[i + 1] + 1e-9 for i in range(len(meds) - 1))
    crossed = [r for r in rows if r["median_symdiff"] >= 0.10]
    print()
    print(f"monotone non-decreasing in kappa: {monotone}")
    if crossed:
        c = crossed[0]
        print(f"H2 criterion (median symdiff >= 0.10) first met at kappa = {c['kappa']:.0f}, "
              f"median w = {c['median_w']:.4f}")
        print(f"  i.e. the hierarchy becomes decisive once the shrinkage term (1-w) exceeds "
              f"{1-c['median_w']:.3f}, versus {1-rows[0]['median_w']:.3f} on the real instrument "
              f"({(1-c['median_w'])/(1-rows[0]['median_w']):.0f}x more shrinkage than it applies).")
        verdict = ("PASS: the H2 null is a bounded statement about a lightly-shrunk instrument, "
                   "not a general claim that hierarchy does not matter.")
    else:
        verdict = ("The verdict never diverges even at extreme shrinkage -- the weighting "
                   "explanation is wrong and the hierarchy-independence has a deeper cause.")
    print("\n" + "=" * 96); print(verdict); print("=" * 96)
    print("EXPLORATORY: a sensitivity analysis over an estimated nuisance parameter, not a frozen quantity.")

    (REPO / "scripts" / "hierarchy_shrinkage_sweep_out.json").write_text(json.dumps({
        "T0": T0, "rows": [{**r, "symdiff": {str(k): v for k, v in r["symdiff"].items()}} for r in rows],
        "monotone": bool(monotone),
        "first_kappa_meeting_H2": (crossed[0]["kappa"] if crossed else None),
        "first_median_w_meeting_H2": (crossed[0]["median_w"] if crossed else None),
        "status": "EXPLORATORY - scope condition for the H2 null",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
