#!/usr/bin/env python3
"""audit_code.py -- write artifacts/audit/coding.json (coder 1's calls).

The calls are human judgement made from the stored full texts under the rubric of
decisions/2026-08-13-select-report-audit-preregistration.md as amended by
decisions/2026-08-13-audit-rubric-amendment.md. This script is the *record* of
those calls, and it refuses to write the file unless every quote it carries is
found in the corresponding stored text.

Whitespace note: the stored texts come from HTML/PDF extraction and carry
arbitrary line breaks, so quotes are matched after collapsing all runs of
whitespace to a single space, on BOTH sides. That still catches a fabricated or
misattributed quote; it only tolerates line wrapping.

  python scripts/audit_code.py   ->  artifacts/audit/coding.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "audit"
TEXT = OUT / "text"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# has_selection: yes | no | unstated
# mismatch: aligned | mismatch | undetermined | no_reported_functional | n/a
# screen_on_full_text: confirmed | "fails S1" | "fails S2" | "fails S3" | "fails S4"
CODING = [
 dict(
  id="1907.09395", screen_on_full_text="confirmed",
  has_selection="unstated",
  T_sel="not stated -- results are reported for several epoch counts and the paper never says which one feeds the classifier",
  T_rep="AUCROC and accuracy (link-prediction classification); normalized RMSE for the feature-forecasting stage",
  mismatch="n/a",
  locator="Section 5.1 Feature Forecasting Performance / 5.2 (link prediction results)",
  quote="We trained the LSTM model for different number of iterations (i.e. epochs). The effect of iteration number is visible in the rmse values.",
  quote2="The AUCROC (i.e., Area Under ROC Curve) is commonly used for evaluating such imbalanced classification problems.",
  note="Two stages with different functionals -- RMSE for forecasting features, AUCROC for the downstream link prediction. If a selection rule were stated this would be a candidate mismatch, but no basis for choosing the epoch count carried into Table 7 is given, so the call cannot be made and the paper is excluded from the rate."),

 dict(
  id="1911.02890", screen_on_full_text="confirmed",
  has_selection="unstated",
  T_sel="not stated quantitatively -- 'a good compromise between efficiency and accuracy', with no criterion or number reported",
  T_rep="ROC-AUC and best F1-score",
  mismatch="n/a",
  locator="Methods (embedding dimensionality) / Section II.2 (prediction of new PACS pairs)",
  quote="The optimal dimensionality, depending on the complexity of the problem under exam, and in particular on the size of the dataset and the vocabulary, is usually determined by a trial and error procedure, W2V ; W2Vdeep , and our tests suggest that 8 is a good compromise between efficiency and accuracy.",
  quote2="We evaluate the effectiveness of contex similarity to forecast unseen PACS couples using standard performance metrics such as the ROC-AUC and the best F1-score.",
  note="Configuration choices (embedding dimension 8, 500 PACS per training set) are made from 'our tests', but no criterion, split or number is reported, so a reader cannot check consistency. Coded unstated."),

 dict(
  id="2106.12875", screen_on_full_text="confirmed",
  has_selection="yes",
  T_sel="F1 (with precision and recall) under 10-fold cross-validation",
  T_rep="F1 (with precision and recall)",
  mismatch="aligned",
  locator="Section 5.2 The ResearchFlow Forecaster",
  quote="We ran each of them on 17 possible combinations of the four time series in order to assess which set of features would yield the best results.",
  quote2="performed a 10-fold cross-validation of the data and measured the performance of the classifiers by computing the average precision ( P ), recall ( R ), and F1 ( F ).",
  upstream_selection="topic extraction uses the elbow method to select the set of most relevant topics (Section 3); reported separately, not folded into the rate",
  note="Selection and reporting are the same functional (F1), so the benchmark is selection-consistent in the sense of the definition. The chapter also reports Augur's trend detection on precision/recall against a 1,408-item gold standard."),

 dict(
  id="2109.06675", screen_on_full_text="confirmed",
  has_selection="no",
  T_sel="n/a -- a single logistic-regression specification, no candidate set",
  T_rep="accuracy, recall, precision, F-measure and Critical Success Index",
  mismatch="n/a",
  locator="Results, 'Predicting emerging topics' (Table 12)",
  quote="Logistic regression models were trained to use topic characteristics to predict emerging topics as in equations (1) and (2). Five-fold cross-validation was used to obtain evaluation metrics in Table 12.",
  note="Cross-validation is used to estimate performance, not to choose among candidates. No selection step, so no select/report pair exists and the paper is outside the rate denominator."),

 dict(
  id="2111.03496", screen_on_full_text="confirmed",
  has_selection="yes",
  T_sel="not named -- a sliding-window size is chosen from {3,5,10,15} as 'best', with no metric stated",
  T_rep="precision, recall and F-measure (Table 3), and AUC for the ranking variant (Table 6)",
  mismatch="undetermined",
  locator="Section 6.6 Parameter tuning",
  quote="For the size of the sliding window n n , we tested with 4 values: 3 , 5 , 10 , 15 3,5,10,15 and a sliding window of size 5 is best for NYTAC as we have enough time-step to compute it.",
  quote2="Our method CEND depends on several parameters that are manually controlled: the threshold value k k and the size of the sliding window for computing the correlation n n . For each corpus, we experimented and measured the optimal values for these parameters.",
  note="An explicit choice from an explicit candidate set, but the criterion is never named. The charitable reading is F-measure, since the surrounding tables report it; the rubric's standard is whether the methods section permits the call, and it does not. Corpus is mixed: New York Times plus a scientific abstracts corpus (SCI), so S2 is satisfied by the SCI half."),

 dict(
  id="2207.12093", screen_on_full_text="fails S3",
  has_selection="no",
  T_sel="n/a",
  T_rep="none -- topics are ranked by Mann-Kendall Z score and Theil-Sen slope, with no comparison against any reference",
  mismatch="n/a",
  locator="Section 3 Materials and Methods / Section 4.1 Topic trends",
  quote="To discover trends in topics, we used Mann -Kendall's test analysis after creati ng a topic using entity linking method.",
  quote2="We also used the slope to find the top 20 quickly increasing trends.",
  note="Descriptive trend statistics only. There is no held-out data, no ground truth and no candidate set, so S3 in fact fails once the full text is read; the abstract screen's uncertainty rule had admitted it. Excluded from the rate and counted as a screen resolution."),

 dict(
  id="2209.10224", screen_on_full_text="confirmed",
  has_selection="yes",
  T_sel="classification score (accuracy) of a per-patent binary value classifier, chosen automatically by auto-sklearn",
  T_rep="none for the headline output -- the ranked list of emerging technologies and leading companies is never compared with any reference; only the component classifier's accuracy (0.6737) is reported",
  mismatch="no_reported_functional",
  locator="Section 4.2 Machine Learning layer / Section 5 Empirical analysis and results",
  quote="we search for the best-in-class by testing a wide sample of general classifiers on our data set and comparing performance levels. In this task, we are supported by the auto-sklearn framework, which automatically selects the best possible model and calibrates its hyperparameters to maximise classification scores with patents that already have a 5YFC value.",
  quote2="the algorithm opted for a mix of differently parametrised random forest models, attaining an accuracy value of 0.6737, or about two in three patents.",
  note="The paper's product is a technology-emergingness ranking aggregated from per-patent predictions. Selection is on per-patent classification accuracy; the ranking itself is never scored. Under the amendment this is no_reported_functional, excluded from the rate. Coding it as a mismatch would have raised the rate."),

 dict(
  id="2211.01348", screen_on_full_text="fails S3",
  has_selection="no",
  T_sel="n/a",
  T_rep="none -- an Emergence Score is defined as a hand-constructed sum of normalised indicators and topics are ranked by it, with no reference set",
  mismatch="n/a",
  locator="Section 4.5 Emergence_Score",
  quote="We finally calculate the Emergence Score with the indicators that we defined previously.",
  quote2="In this study, we tried to normalize the emergence metrics in a way that they can finally produce a comparable s core of emergence.",
  note="No candidate set, no fitting, no held-out evaluation, no gold standard. The 'evaluation of emergence' in the abstract refers to the construction of the score, not to scoring it against anything. S3 fails on the full text."),

 dict(
  id="2211.15334", screen_on_full_text="confirmed",
  has_selection="yes",
  T_sel="forecast accuracy -- the paper's accuracy metrics throughout are RMSE and MAPE, both of which it reports",
  T_rep="MAPE (headline) and RMSE",
  mismatch="aligned",
  locator="Section 3.2.3 RNN / Section 4 Results and Discussion",
  quote="We conduct a hyperparameter search to find the best training parameters and the most accurate model configuration.",
  quote2="Our autoencoder approach improves the MAPE by 13.5% on average over the second-best result.",
  upstream_selection="S-curve hyperparameters chosen by grid search 'to find the hyperparameters that yield the most accurate fit'",
  note="'Most accurate' is not pinned to one metric, but both candidates (RMSE, MAPE) are reported, so the frozen multiple-metrics rule makes this aligned either way. Worth noting for the paper: RMSE is consistent for the mean and MAPE for a weighted median, so selecting on one and reporting only the other WOULD be a mismatch -- this paper avoids it by reporting both."),

 dict(
  id="2305.04133", screen_on_full_text="confirmed",
  has_selection="yes",
  T_sel="explained variance (R^2) on the test set under step-forward temporal cross-validation",
  T_rep="explained variance (R^2)",
  mismatch="aligned",
  locator="Section 2.4 Methods (model evaluation) / Section 3.4 Predictive model results",
  quote="Model performance was evaluated using step -forward temporal cross validation, implemented in scikit-learn, over all topic time series simultaneously.",
  quote2="The combined tree and embedding models have the best explained variance (R2) scores and overall results. All results are reported for the test set (see Methods).",
  note="Models were trained with default hyperparameters, so the only selection is the choice of the winning model family, made on the same functional that is reported. Aligned. Separately notable: the winner is picked on the test set rather than a validation split, which is a different problem from selection-consistency and is not what this audit measures."),

 dict(
  id="2403.16157", screen_on_full_text="fails S3",
  has_selection="no",
  T_sel="n/a",
  T_rep="none -- a software toolkit paper with no evaluation of a detection or forecast against a reference",
  mismatch="n/a",
  locator="Abstract / Section 3 (module descriptions)",
  quote="pyKCN , a Python toolkit, addresses this gap by automating keyword cleaning, extraction and trend analysis from extensive academic corpora.",
  note="Tool description. No candidate set and no quantitative evaluation against a reference, so S3 fails on the full text."),

 dict(
  id="2408.06839", screen_on_full_text="confirmed",
  has_selection="yes",
  T_sel="R-squared, RMSFE and MAFE under 10-fold cross-validation repeated 100 times",
  T_rep="test R-squared",
  mismatch="aligned",
  locator="Section 'Predicting citation paths'",
  quote="In our study, we compared the Linear Regression (LR) and Multi-Level Model (MLM), subjecting both to rigorous cross-validation to ensure the robustness and reliability of our findings.",
  quote2="Finally, we compared key metrics including R-squared (R2), Root Mean Square Forecasting Error (RMSFE), and Mean Absolute Forecasting Error (MAFE) on both training and testing sets derived during the cross-validation",
  note="R^2 is both a selection metric and the headline ('the test R2 of LR2 has improved 100% (from 0.29 to 0.58)'), so the pair is aligned."),

 dict(
  id="2411.05930", screen_on_full_text="fails S3",
  has_selection="yes",
  T_sel="qualitative -- hyperparameters chosen to 'produce the most fine-grained topics'; thresholds 'empirically fixed'",
  T_rep="none -- the paper states that no metrics exist for the task and reports no quantitative evaluation of weak-signal detection",
  mismatch="no_reported_functional",
  locator="Section 4.2 Algorithm parameters / Section 9.3 Evaluation Challenges",
  quote="To the best of our knowledge, there are currently no established metrics for comparing weak signal detection performance within large volumes of data.",
  quote2="We chose BERTopic hyperparameters to produce the most fine-grained topics since larger topics will hinder the early detection process",
  note="The most honest paper in the sample about its own unevaluability: it says outright that the task has no established metrics. That is exactly the no_reported_functional case, and it also means S3 fails on the full text, so it is excluded from the rate twice over. It should be cited as a disclosed limitation, never as a criticism."),

 dict(
  id="2503.09894", screen_on_full_text="confirmed",
  has_selection="yes",
  T_sel="precision, recall and F1 for exact matches, computed on the development set",
  T_rep="precision and recall of extraction",
  mismatch="aligned",
  locator="Section 2 (prompt iteration) / Section 3 (final development-set results)",
  quote="To guide this iteration, we used a comprehensive set of metrics calculated on our development set, including: precision, recall and F-1 scores, for exact matches.",
  quote2="The final results on our development set were: precision of 44%",
  note="Selection and reporting share the functional, so aligned. Two things worth recording that the rubric does not measure: the reported numbers are on the SAME development set used to choose the prompt, and the downstream products (temporal trends, modality distribution) are not evaluated at all."),

 dict(
  id="2503.23285", screen_on_full_text="confirmed",
  has_selection="no",
  T_sel="n/a -- hyperparameters carried over from the authors' prior work, with a sensitivity table rather than a selection",
  T_rep="cosine-similarity based closeness measures and Kendall tau in the sensitivity analysis",
  mismatch="n/a",
  locator="Section IV.2 Model / Table S4",
  quote="Based on our previous research Peng et al. 2021 , we set the following hyperparameters: context window size w = 10 w=10 , embedding dimensions D = 100 D=100 , and number of sampled negative pairs for each positive input pair k = 5 k=5 . We also conduct validation experiments with different hyperparameter configurations (see Table S4).",
  note="The configuration is fixed a priori from prior work and the alternatives are reported as a robustness check, not used to choose. No selection step."),

 dict(
  id="2504.08743", screen_on_full_text="confirmed",
  has_selection="yes",
  T_sel="TC-W2V topic coherence",
  T_rep="TC-W2V topic coherence, with C_V coherence and C_UMass uniformity alongside",
  mismatch="aligned",
  locator="Section IV-B Experiment 1: Selection of Topic Models / Section IV-C",
  quote="Based on the TC-W2V topic coherence measure, the optimal number of topics obtained from the dynamic topic model is 18.",
  quote2="To further compare the performance of topic models on the yearly abstract data, the coherence ( C V C_{V} ) [ 17 ] and uniformity ( C U ​ m ​ a ​ s ​ s C_{U}mass ) [ 18 ] metrics were evaluated for both NMF and LDA models.",
  note="The number of topics is chosen by TC-W2V and TC-W2V is among the reported metrics, so aligned. Noted but not coded: Table II shows TC-W2V falling monotonically with K (0.226 at K=6 down to 0.093 at K=24) while K=18 is selected, which is an internal inconsistency about the direction of the criterion rather than a select/report mismatch."),

 dict(
  id="2507.21125", screen_on_full_text="confirmed",
  has_selection="yes",
  T_sel="entity-level F1 on a validation set (early stopping)",
  T_rep="F1-score (RATE 91.27% vs BERT 53.73%) against an expert-curated gold standard",
  mismatch="aligned",
  locator="Section II-E BERT baseline for technology extraction / Abstract",
  quote="Training was conducted using the Trainer API with the following hyperparameters: 20 epochs, batch size of 16, learning rate of 2e-5, AdamW optimizer with weight decay of 0.01, and early stopping based on F1-score on the validation set.",
  quote2="RATE achieved F1-score of 91.27%, Significantly outperforming BERT with F1-score of 53.73%.",
  note="The terminal selection (early stopping on validation F1) targets the same functional as the headline (F1 against a 70-article expert gold standard). Aligned. The co-occurrence technology map built from the extracted terms is not itself evaluated."),

 dict(
  id="2508.20514", screen_on_full_text="confirmed",
  has_selection="yes",
  T_sel="not named -- baseline hyperparameters are tuned per dataset and topic number, with no criterion stated; the proposed model's own parameters are fixed at alpha=1, lambda=0.5 with no stated basis",
  T_rep="topic coherence (C_V), topic diversity, Calinski-Harabasz index; accuracy/NMI/ARI on labelled corpora",
  mismatch="undetermined",
  locator="Section IV-B Experiment Setup (Implementation Details)",
  quote="We fine-tune the hyperparameters of these baselines under different datasets and topic numbers.",
  quote2="Parameters are set as α = 1 \\alpha=1 and λ = 0.5 \\lambda=0.5 , and experiments are conducted on two NVIDIA A100-80GB GPUs.",
  note="A selection step exists and is disclosed, but its criterion is not, so the pair cannot be assessed. The reported metrics are several and heterogeneous (coherence, diversity, cluster-geometry indices, and label-agreement indices), which is precisely the setting in which naming the selection criterion would matter most."),

 dict(
  id="2508.21329", screen_on_full_text="fails S3",
  has_selection="no",
  T_sel="n/a",
  T_rep="none -- the paper proposes mathematical machinery for a maturity model and does not score it against a reference",
  mismatch="n/a",
  locator="Section 15 Mathematical Foundations for HELIOS Model Improvement",
  quote="To overcome the limitations identified in the original HELIOS model, the integration of advanced mathematical methodologies is proposed to provide the framework with greater accuracy, dynamism and predictive capacity.",
  quote2="The model must undergo a process of continuous validation and refinement. This includes comparison of the model's forecasts with the actual evolution of technologies",
  note="A design proposal. Its own text places validation in future work, so S3 fails on the full text."),

 dict(
  id="2510.15801", screen_on_full_text="fails S3",
  has_selection="no",
  T_sel="n/a",
  T_rep="none -- disruptive innovations are scored 1-3 per attribute by investigators against a threshold of 9, with no reference set",
  mismatch="n/a",
  locator="Methodology (attribute scoring)",
  quote="Investigators score each attribute for a suspected emerging DI.",
  quote2="If the total score of all attributes is 9 or higher, we flag the suspected DI for consideration for inclusion into the data if it meets all other requirements in the methodology.",
  note="Manual expert scoring with a fixed threshold; no candidate set and no evaluation against a reference. S3 fails on the full text."),

 dict(
  id="2601.16990", screen_on_full_text="fails S3",
  has_selection="no",
  T_sel="n/a",
  T_rep="none -- a library paper describing implemented analyses, with no evaluation against a reference",
  mismatch="n/a",
  locator="Section 2/3 (library capabilities, community detection methods table)",
  quote="There exists a large number of methodologies for community detection that can be applied to a wide variety of fields, with different perks and disadvantages.",
  note="Software description. The library implements several community-detection methods and leaves the choice to the user, which is not a selection step performed by the paper. S3 fails on the full text."),

 dict(
  id="2606.27394", screen_on_full_text="confirmed",
  has_selection="yes",
  T_sel="RMSE on a held-out 20% of each topic time series",
  T_rep="RMSE (36.76)",
  mismatch="aligned",
  locator="Section III-F 2 Model Selection Strategies / Abstract and Section IV-C",
  quote="Exhaustive Evaluation (default) : Each of the four candidate forecasting models is trained on the initial 80% of the time series and evaluated on the remaining 20%. Model performance is quantified using the RMSE.",
  quote2="Among them, STL emerged as the top-performing model, achieving the lowest RMSE of 36.76, indicating high prediction accuracy with minimal error.",
  upstream_selection="clustering algorithm selected by silhouette score (Optuna, AutoCluster); topic model selected by coherence score via successive halving (AutoTopicModeling) -- neither is consistent for the reported RMSE",
  note="The terminal selection is RMSE and the headline is RMSE, so the pair is aligned under the rubric. This is the sample's clearest instance of the amendment mattering: two upstream selections (silhouette, coherence) propagate into the reported RMSE and are not consistent for it, and the paper's actual deliverable -- a strong/weak/noise classification of topics -- is never scored at all."),
]


def main() -> int:
    missing = []
    for rec in CODING:
        f = TEXT / f"{rec['id']}.txt"
        if not f.exists():
            missing.append((rec["id"], "NO TEXT FILE"))
            continue
        hay = norm(f.read_text(encoding="utf-8", errors="replace"))
        for key in ("quote", "quote2"):
            q = rec.get(key)
            if q and norm(q) not in hay:
                missing.append((rec["id"], f"{key} NOT FOUND: {q[:90]}..."))
    if missing:
        print("QUOTE CHECK FAILED -- refusing to write coding.json:")
        for pid, why in missing:
            print(f"  {pid}: {why}")
        return 1

    screen = json.loads((OUT / "screen.json").read_text(encoding="utf-8"))
    ids_in = {r["id"] for r in screen["decisions"] if r["decision"] == "IN"}
    coded = {r["id"] for r in CODING}
    if coded != ids_in:
        print(f"coded set != screened-in set; missing {ids_in - coded}, extra {coded - ids_in}")
        return 1

    doc = {
        "preregistration": "decisions/2026-08-13-select-report-audit-preregistration.md",
        "amendment": "decisions/2026-08-13-audit-rubric-amendment.md",
        "coder": "coder1",
        "n_coded": len(CODING),
        "records": CODING,
    }
    (OUT / "coding.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"coding.json written: {len(CODING)} records, all quotes verified in the stored texts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
