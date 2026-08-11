# Related work

Prior art taken seriously, with no "first" or "only" language. Where a predecessor owns a claim, it is said plainly. Citations were verified against primary sources — arXiv versions distinguished, because two of the entries below have materially different v1 and v2 framings.

## The two decisive predecessors

**Klavans, Boyack & Murdick.** *PLOS ONE* 15(9):e0239177 (2020), and the deployed [CSET/ETO Map of Science](https://sciencemap.eto.tech/). Out-of-time growth forecasting on citation-derived research communities at 38–43M documents, >1M forecasts, ≥8% CAGR in publication share, 3-year horizon, CSI as the scoring metric — and still in production. **Owns scale and deployment outright** (~15× the corpus here). Its model is a cross-sectional probit over four hand-crafted features: no time-series model, no trajectory, no baseline comparison of any kind (no momentum, no persistence), no pre-registration. The deployed forecaster is Rahkovsky et al. (2021), rooted in the 2020 method.

**PreScience v2.** arXiv:2602.20459v2 (2 Jul 2026), Ai2 + UChicago Knowledge Lab + HUJI + Northwestern. §3.2 defines $\Delta s(z) = s_{\text{test}}(z) - s_{\text{train}}(z)$ over 202 topics and 4,334 topic pairs, scored by $R^2$/Pearson; 97,820 target papers (501,866 with companions), six cs categories, Oct 2023 – Sep 2025, one cutoff. Five heuristic baselines (best unary $R^2 = 0.17$; best pair $R^2 = 0.19$). **This defines the share-formulation target publicly**, so that formulation is not a differentiator here — and it never was a strong one: share-of-output as a relative activity index is decades-old standard scientometrics.

> **Version trap.** v1 (24 Feb 2026) is *"PreScience: A Benchmark for Forecasting Scientific Contributions"* with **4 tasks and no aggregate task**. v2 (2 Jul 2026) has **7 tasks**, adding the two aggregate topic-trend variants. The project's GitHub README, the Ai2 blog, and every v1 metadata record still describe the 4-task framing — a secondary source will wrongly suggest PreScience has no topic-trend task. Cite v2. (The paper says **six** arXiv categories — cs.CL, cs.LG, cs.AI, cs.CV, cs.IR, cs.NE; the blog's "seven" is incorrect.)

What PreScience does *not* do on the topic-trend task: it runs zero statistical time-series models — no ARIMA/ETS/Theta/SeasonalNaive — and, uniquely among its seven tasks, no LLM or agentic baseline either. No multi-step horizon (one scalar $\Delta$ between two 12-month aggregates), one cutoff, no rolling-origin CV, no pre-registration. Its topics are a GPT-5.4-proposed, hand-refined flat taxonomy — multilabel at 3.15 labels/paper, ~6.3% "Other" excluded, and **shares do not sum to 1**.

**PreScience does not corroborate the null reported here.** It never ran a statistical forecaster on the topic-trend task, so it is silent on whether momentum beats forecasters. On topic *pairs*, Linear Extrapolation — a trend fit — beats every momentum variant, mildly *against* a momentum-dominance reading. The accurate sentence is: *PreScience finds simple historical signals predictive but incomplete on an analogous share-change target, with no statistical forecaster tested.*

**Timing.** v2 predates this project's run by 21 days. For the topic-trend formulation that is **concurrent work**, not established prior art.

## Others in the neighbourhood

| Work | What it does | What it does not do |
|---|---|---|
| **Ofer, Kaufman & Linial**, *Heliyon* 2023 (arXiv:2305.04133) | Identical share target (topic pubs / total PubMed pubs), 125 topics, 40 years, 5-year horizon, step-forward temporal CV, lag baseline. Levels: lag $R^2 \approx 0.97$. **% change: lag $R^2 \approx 0.004$ vs best model $\approx 0.45$** | Hand-picked topic list, annual resolution, single domain, no multi-cutoff, no pre-registration, no embedding regions |
| **Abolfadl et al.**, arXiv:2606.27394 | Same pipeline shape: SPECTER → AutoCluster → per-cluster ARIMA/STL/Prophet/LSTM, 127,820 abstracts, RMSE 36.76 | Absolute counts rather than shares, one 80/20 split, one narrow domain, 49 topics, one-step-ahead, no momentum baseline, no precision@k |
| **"Lacuna"**, arXiv:2606.26246 (Mila) | Static LLM-built research map of 734K ML papers | No forecasting. **Name collision only** — unrelated to this repo's working codename |
| Impact4Cast, Science4Cast, FOS, ScienceMeter, Proof of Time, ForeSci, CUSP, SciPaths, SciImpact | Link prediction over concept pairs; idea-judgment, event, or paper-level forecasting | None forecasts a quantity at topic or region level |

Ofer et al. cuts both ways and is worth stating: their %-change result implies momentum *should* be beatable on a growth target, which makes the null reported here more interesting, not less.

## What this project does and does not claim

**No new formulation is claimed.** The combination assembled here — unsupervised embedding-derived regions over a whole multi-field corpus at $10^6$ scale, monthly share trajectories, a statistical roster with rolling-origin CV selection, multiple masked out-of-time cutoffs, and a pre-committed go/no-go — does not appear as a unit in the works above. That is a statement about the *conjunction*, not about any conjunct, and three of the four conjuncts are individually occupied.

**Scale is not a claim.** Klavans/Boyack is ~15× larger. Against PreScience: 27.7× on target papers (2.71M vs 97,820) but **5.4×** against their full 501,866-paper corpus — both numbers given. Time span 234 vs 24 months.

**What survives as the contribution is the harness and the finding, not the target**: a pre-registered, leak-controlled, multi-cutoff out-of-time protocol with a published negative result and a full defect disclosure ([`LIMITATIONS.md`](LIMITATIONS.md)).

## On the word "pre-registered"

The freeze here is a local content hash plus git commit timestamps in this repository. That is verifiable by anyone who can read the history, but it is not a third-party deposit. Until the pre-registration is deposited externally (OSF, or a public signed tag), the accurate phrasing is **"pre-committed / frozen before scoring"** rather than an unqualified "pre-registered."

Pre-registration in this area is genuinely scarce: the NeurIPS pre-registration workshop ran at ICCV 2019, NeurIPS 2020, and NeurIPS 2021 with no edition since (PMLR v148 has 23 papers, v181 has 3), and Registered Report adoption sits at 0–7% of journals across major fields (Lin et al., *Scientometrics* 129(4):2123–2130, 2024). Bracher et al., *Nat Commun* 12:5173 (2021) is the closest genuine precedent — a pre-specified forecast evaluation reporting an honest null. Salganik et al., *PNAS* 117(15):8398–8403 (2020) is the template for publishing "a simple benchmark was hard to beat under an honest protocol."
