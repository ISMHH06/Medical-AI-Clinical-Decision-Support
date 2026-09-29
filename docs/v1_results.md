# V1 Results — Multimodal Chest X-Ray Classification

**Status:** Phases 0–8 complete. This document consolidates the v1 deliverable before any stretch (Phase 9+) work begins.

## 1. Problem & scope

The system predicts five thoracic abnormalities — Cardiomegaly, Edema, Atelectasis, Pleural Effusion, Consolidation — as independent sigmoid probabilities from a frontal chest X-ray plus a small set of clinical features (age, sex, BMI, insurance type, view position). Labels are not mutually exclusive, so this is multi-label binary classification, not multi-class.

Dataset is CheXpert Plus (Stanford AIMI, via Redivis), chosen over MIMIC-CXR for immediate access without a credentialing wait. Splits are patient-level (80/10/10 by `deid_patient_id`) to prevent leakage between a patient's multiple studies. Uncertain labels were resolved with a per-label policy (U-Ones for Atelectasis/Edema, U-Zeros for Cardiomegaly/Pleural Effusion/Consolidation) rather than one blanket rule.

## 2. Final results

| Model | Phase | Description | Val AUROC | Val AUPRC |
| --- | --- | --- | --- | --- |
| E04 | 4–5 | Vision-only, DenseNet-121, denseblock3/4+norm5+classifier trainable, pos_weight balancing (champion vision model) | 0.7267 | 0.5031 |
| E06 | 6 | Clinical-only MLP, 29 engineered features | 0.5616 | 0.3373 |
| E07 | 7 | Late fusion — unweighted average of E04 + E06 probabilities | 0.7248 | 0.5037 |
| E08 | 7 | Early fusion — frozen DenseNet embedding + trainable clinical encoder + fusion MLP (champion fusion model) | 0.7306 | 0.5057 |
| E09 | 7 | Early fusion with partial vision fine-tuning (negative result) | 0.6966 | 0.4666 |

E08 is the strongest model on aggregate threshold-independent metrics and is the model carried into Phase 8's explainability work. E04 remains a fully competitive alternative once both models are given their own fairly optimized per-label threshold (see Section 3).

## 3. Does clinical information help? (Phase 1's core question, answered)

Clinical data alone is a weak predictor (E06 AUROC 0.5616 vs. E04's 0.7267) — images carry the dominant signal for these five labels, which is expected given they are primarily radiographic findings. Cardiomegaly was the one label where clinical data performed meaningfully above chance (0.629 AUROC), consistent with age/BMI/sex being real physiological risk factors for cardiac enlargement. Consolidation was worst (0.538, near-random), consistent with it being an acute, purely visual finding with no strong demographic relationship.

Fusion's effect is small and needs to be stated precisely rather than oversold:

- Late fusion (E07) underperforms vision-only, dragged down by the weak clinical signal. A weighted sweep found no fixed blending weight beat vision-only by more than +0.0006 AUROC.
- Early fusion (E08) shows a modest aggregate AUROC edge over vision-only (+0.0039). But per-label threshold optimization (Youden's J, applied fairly to both models independently) showed this edge is largely an artifact of different probability calibration between the two models, not clearly superior discrimination — once both models get their own optimal threshold, most labels become near-ties.
- **Cardiomegaly is the one label where E08 shows a real, clear advantage** — better-balanced sensitivity/specificity than E04, not just a calibration shift.
- Partial vision fine-tuning during fusion training (E09) made things worse, not better (AUROC 0.6966), confirming a pattern already seen in the vision-only experiments: at this dataset size, unfreezing more of the backbone underfits rather than helps.
- Phase 8's explainability work (Grad-CAM comparison on E04 vs. E08 for the same patient/label, plus SHAP magnitude analysis) converges on the same conclusion from a different angle: E08's small AUROC advantage is not coming from the model seeing something new in the image — the Grad-CAM attention maps for E04 and E08 are visually near-identical on the same case. The fusion gain looks like a confidence adjustment, not a new visual signal.

**Bottom line:** clinical information provides a small, real, label-specific benefit — concentrated in Cardiomegaly — rather than a broad improvement across all five targets. This is a more honest and more useful answer than a single aggregate AUROC number would suggest, and it's the kind of result that's worth reporting as-is rather than searching for a fusion trick that pushes the aggregate metric further.

## 4. Explainability findings (Phase 8)

Two complementary methods were applied to the E08 fusion model: Grad-CAM (vision branch, via a live-embedding wrapper that restores a real gradient path from pixels to the fusion head) and SHAP (clinical branch, via `shap.DeepExplainer` on a wrapper that holds the vision embedding fixed).

- **Sanity-checked, not assumed.** A positive/negative differential Grad-CAM check for Cardiomegaly confirmed the heatmap discriminates the target label — confident positives show cardiac-silhouette-centered attention, confident negatives collapse to a near-zero heatmap (expected ReLU behavior, not a bug). This differential check was run for Cardiomegaly only; broader claims about heatmap quality on the other four labels are qualitative, not sanity-checked to the same standard.
- **SHAP magnitudes are consistently small** (roughly 0.001–0.004), consistent with the clinical branch's weak overall contribution. More informative than the raw magnitude: *which* features dominate shifts by label. For one patient compared directly across labels, `race_White` was the top Cardiomegaly SHAP feature but did not appear in that same patient's Consolidation top-5 at all. `race_White` appeared in the Cardiomegaly top-5 for all four sampled patients, with a consistent sign each time.
- **Fairness consideration, flagged not buried.** That race/insurance-type pattern is correlational (relative to a SHAP background distribution, not causal), but it is a real signal that the clinical branch has picked up demographic correlations present in training data. This is explicitly deferred to Phase 10 (subgroup fairness testing) rather than resolved here.
- **A genuine failure mode, reported honestly.** One patient with ground-truth Pleural Effusion was missed with high confidence (predicted probability 0.06); the Grad-CAM heatmap for that prediction concentrated in the upper chest rather than the lung bases where effusions physiologically pool — the model wasn't just wrong on the label, its own attention wasn't even in the anatomically relevant region.
- **Mixed multi-label cases were reported as-is**, not cherry-picked: one patient had three true-positive labels and the model correctly flagged only one above threshold; another showed a plausible comorbidity-driven false positive (Atelectasis alongside two correct positives it commonly co-occurs with clinically).

## 5. Limitations of v1

- Evaluated on a 12,444-image stratified subset of CheXpert Plus, not the full dataset — absolute performance numbers should not be read as representative of a full-scale training run.
- Atelectasis and Consolidation are structurally weak labels across every experiment (data scarcity), not something training tweaks alone fixed.
- No probability calibration has been applied yet — raw sigmoid outputs are used directly. Reliability diagrams, ECE, and temperature scaling are explicitly out of scope for v1 (Phase 9).
- No systematic subgroup fairness testing has been done. Phase 8 surfaced one demographic-correlation signal in SHAP attributions, but this was incidental to explainability work, not a designed fairness audit (Phase 10).
- Explainability evidence is illustrative (4–5 hand-selected validation patients), not a comprehensive audit across the validation set.
- A single fixed train/val/test split was used throughout, not cross-validation, so metric estimates carry some unquantified split-dependent variance.
- No external validation on a second dataset (e.g. MIMIC-CXR) has been performed; generalization beyond CheXpert Plus's data distribution is untested (Phase 12).

## 6. What's next (stretch scope)

Phases 9–15 are deliberately out of scope for this document and were not needed to answer the core research question in Section 3. If pursued, the natural next step is Phase 9 (calibration and uncertainty), since an uncalibrated model is the most immediate gap identified above — followed by Phase 10 (robustness and bias) as the direct follow-up to the fairness signal surfaced in Phase 8.
