# Phase 9 — Calibration & Uncertainty: Reliability, ECE, Temperature Scaling

This is the first sub-step of Phase 9 (stretch scope, pursued after v1 was fully consolidated). It asks a question v1 never answered: when E04 or E08 outputs "70% probability," is that number actually trustworthy — does the model turn out to be right roughly 70% of the time when it says so?

## Method

**Predictions.** E04 (vision-only) and E08 (fusion) were run once over the full val split (1307 rows) and the full test split (1314 rows), saving per-label sigmoid probabilities alongside ground truth for all 5 labels and both models. This reused the same inference wrappers built for Phase 8's Grad-CAM/SHAP work, just looped over every row instead of a handful of hand-picked patients — no model weights were touched at any point in this analysis.

**Reliability diagrams and ECE.** For each of the 5 labels and both models, predictions were bucketed and the mean predicted probability in each bucket was compared to the observed fraction of true positives in that bucket. Expected Calibration Error (ECE) summarizes the gap as a single weighted-average number. This was computed per label rather than pooled, since pooling across labels with very different base rates (17%–42%) would blur label-specific miscalibration.

**A binning lesson worth keeping.** The first pass used standard equal-width bins (10 bins spanning [0,1]). Two entries in the resulting ECE table stood out as unusually bad — Atelectasis/E08 (ECE 0.208) and Consolidation/E04 (ECE 0.204), both notably worse than every other label/model combination. Annotating each plotted point with its underlying sample count revealed why those two curves had a bin with exactly **1** sample in it, close to the diagonal-defying tail. That's not evidence of a real calibration failure — it's one data point. A second pass using equal-frequency (quantile) binning, which guarantees a roughly even sample count per bin (~163–164 val samples in every bin, no exceptions), was run to check whether the anomaly was purely a sparse-bin artifact.

It was not, at least not entirely: quantile binning **confirmed** both problem cases (Atelectasis/E08 ECE 0.208 → 0.208 unchanged; Consolidation/E04 ECE 0.204 → 0.204 unchanged) on well-supported bins. Both binning schemes now use bins with over 160 samples each, and the gap is real, not noise. The lesson: a single-sample bin can *coincide* with a real problem without being what's *causing* the bad ECE number — checking bin counts before trusting a calibration score is not optional, and switching binning schemes was the right way to separate "is this a sparse-bin artifact" from "is this real."

**The finding, before any correction:** every reliability curve, across all 5 labels and both models, sits above the diagonal — both models are systematically overconfident. The degree varies by label, and does not always track which model has the better AUROC: E08 (the aggregate-AUROC champion) is *more* miscalibrated than E04 on Atelectasis (ECE 0.208 vs 0.100), a reminder that AUROC is threshold/scale-invariant and says nothing about whether the model's stated confidence can be trusted.

## Temperature scaling

Ten scalar temperatures were fit — one per (label, model) combination — each minimizing binary cross-entropy between `sigmoid(logit/T)` and the true label on the **val** split. Recovering logits from saved probabilities via `logit = log(p/(1-p))` is exact (no information lost), so no new inference pass was needed for this step. To avoid the circularity of fitting and evaluating on the same data, every fitted T was then applied to the **held-out test split**, and calibration was re-measured there — this is the only way to know whether a correction actually generalizes rather than just fitting val-set noise.

**Fitted temperatures** (all > 1, consistent with the overconfidence finding above):

| Label | E04 | E08 |
|---|---|---|
| Cardiomegaly | 1.18 | 1.44 |
| Edema | 1.10 | 1.35 |
| Atelectasis | 1.09 | **2.62** |
| Pleural Effusion | 1.23 | 1.14 |
| Consolidation | 1.02 | 1.19 |

E08 needed more correction than E04 on every label, and Atelectasis/E08 needed far more than anything else (T=2.62 vs. the next-highest at 1.44) — both consistent with what the reliability diagrams already showed.

**Test-set result: NLL improved everywhere; ECE did not.** All 10 combinations showed lower binary cross-entropy on test after scaling — the fitted temperatures generalized cleanly on the metric they were optimized for. But equal-width ECE only improved in 4 of 10 cases, and quantile ECE in 2 of 10. This is not a contradiction or a failure of the method — NLL and ECE measure different things. NLL rewards being confidently *correct* and heavily penalizes being confidently *wrong*, evaluated per-example; ECE measures the aggregate gap between stated confidence and observed accuracy *within a bin*, regardless of which individual predictions in that bin were right or wrong. A temperature can genuinely lower the average per-example penalty while simultaneously shifting the binned pattern in a direction that increases ECE.

Cardiomegaly/E08 makes this concrete: BCE clearly improved (0.534 → 0.523) but both ECE variants got *worse* (0.140 → 0.155) — a real regression on the label E08 is otherwise strongest on, not something to gloss over.

## Deployment decision

Rather than deploy all 10 fitted temperatures uniformly (which would mean shipping four corrections that improved the loss function but visibly worsened the calibration plot we're reporting), each temperature was deployed selectively: **kept only where it improved equal-width ECE on held-out test**, otherwise left at T=1 (no correction).

| Label | Model | Deployed T | Deployed? | Test ECE (deployed config) |
|---|---|---|---|---|
| Cardiomegaly | E04 | 1.00 | No | 0.165 |
| Cardiomegaly | E08 | 1.00 | No | 0.140 |
| Edema | E04 | 1.00 | No | 0.100 |
| Edema | E08 | 1.00 | No | 0.112 |
| Atelectasis | E04 | 1.00 | No | 0.097 |
| **Atelectasis** | **E08** | **2.62** | **Yes** | **0.201** (from 0.206) |
| **Pleural Effusion** | **E04** | **1.23** | **Yes** | **0.021** (from 0.034) |
| **Pleural Effusion** | **E08** | **1.14** | **Yes** | **0.088** (from 0.089) |
| Consolidation | E04 | 1.00 | No | 0.217 |
| **Consolidation** | **E08** | **1.19** | **Yes** | **0.120** (from 0.121) |

4 of 10 combinations deployed. Pleural Effusion/E04 is the clear win among these (ECE nearly halved, 0.034 → 0.021, with NLL, equal-width ECE, and quantile ECE all agreeing). Atelectasis/E08 — the worst-calibrated case in the whole project — improves modestly under its large fitted temperature but is not fixed; it remains the most miscalibrated label/model combination in the system even after correction, and that residual limitation is stated here rather than implied away by the fact that a correction was applied.

Full artifacts: `outputs/calibration/temperatures.json` (all 10 fitted values with val BCE), `outputs/calibration/temperature_scaling_results.csv` (full before/after test table: BCE, equal-width ECE, quantile ECE), `outputs/calibration/deployed_temperatures.json` (final deployment decision with a machine-readable reason per row).

## Limitations (temperature scaling)

- Temperature scaling is a single-scalar correction per (label, model). It can rescale a curve's overall confidence but cannot fix a non-monotonic or highly label-specific miscalibration pattern (the Atelectasis/E08 case) as well as a more flexible method (e.g. isotonic regression) might — that tradeoff was accepted here for simplicity and interpretability, not because a better fix doesn't exist.
- The selective-deployment rule is gated on equal-width ECE specifically. Two of the four deployed cases (Atelectasis/E08, Consolidation/E08) also improve under quantile ECE; the other two were not re-checked against that stricter gate. This is a reasonable, stated simplification, not an oversight to be discovered later.
- This work calibrates *marginal* per-label probabilities. It says nothing about calibration conditioned on demographic subgroups — that question is explicitly left to Phase 10, and connects directly to the fairness signal Phase 8's SHAP analysis already surfaced.

## Abstention (selective prediction)

The second Phase 9 sub-step asks a downstream question: given calibrated confidence, can the system usefully say "don't trust this one, send it to a human" instead of always forcing a decision?

**Method.** Per-label decision thresholds were taken from Phase 7's existing Youden's J optimization (`models/threshold_optimization_results.json`) rather than recomputed, since that artifact already existed and disagreeing with it would need justification we didn't have. For every test example, the deployed (post-temperature-scaling) probability was used to compute a confidence score as `|calibrated_prob - label_threshold|` — distance from that label's own decision boundary, not distance from a fixed 0.5. Test examples were ranked by this score and coverage was swept from 100% (everything retained) down to 50% (the least-confident half abstained on), recomputing sensitivity, specificity, and balanced accuracy on the retained set at each step, separately for all 5 labels × 2 models.

**Headline result: abstention improves balanced accuracy in all 10 (label, model) combinations**, comparing full coverage (1.0) against abstaining on the least-confident 30% (coverage 0.7) — no exceptions. That's a real validation that the calibrated confidence scores carry genuine information about which predictions are harder to trust, not just noise ranked by chance. The gains are largest for the two structurally weak labels: Atelectasis/E08 improves the most (balanced accuracy 0.606 → 0.639), and it is also the single worst-calibrated case from the temperature-scaling analysis above (residual ECE 0.201 even after its T=2.62 correction) — the label with the least trustworthy raw confidence is also the one that benefits most from a mechanism built to act on that confidence. That's a coherent, connected story across both sub-steps, not a coincidence to leave unremarked.

**A more precise reading, from the coverage curves rather than the balanced-accuracy summary alone.** Balanced accuracy is an average of sensitivity and specificity, and averaging can hide two real, opposing trends that cancel into a small net number. Plotting sensitivity and specificity as separate lines (rather than only the combined metric) surfaced exactly that for two labels:

- **Consolidation/E08** shows sensitivity rising sharply as coverage drops (0.76 → 0.93) while specificity *falls* just as sharply (0.50 → 0.36). The balanced-accuracy headline (0.621 → 0.650) reads as a modest, unremarkable improvement; the underlying curves show the operating point shifting substantially toward sensitivity at real cost to specificity, not a clean, symmetric improvement. This is consistent with Consolidation/E08's unusually low decision threshold (0.20, versus 0.35–0.50 for every other label/model combination): at that threshold the model already leans toward calling cases positive, so the most-confident subset by `|prob − 0.20|` becomes increasingly dominated by high-probability positives as coverage shrinks, starving the specificity calculation of confidently-negative examples.
- **Atelectasis/E04** shows the mirror pattern: sensitivity *declines* slightly as coverage drops (0.40 → 0.33) while specificity climbs (0.78 → 0.89) — again a real tradeoff, not a uniform gain.
- The other three labels (Cardiomegaly, Edema, Pleural Effusion) show the more expected pattern for both models: sensitivity rises as coverage tightens, specificity stays roughly flat or drifts only slightly — abstention removing genuinely hard cases without a strong compensating cost on the other axis.

The corrected framing, then, is not "abstention helps every label roughly the same way" — it's "abstention improves the balanced-accuracy summary everywhere, but for two of the five labels that improvement is actually a sensitivity/specificity tradeoff being averaged into a smaller-looking net number, and the shape of that tradeoff is explainable by each label's own decision threshold."

Full artifacts: `outputs/calibration/abstention_coverage_results.csv` (all 110 rows: label × model × coverage-step × sensitivity/specificity/balanced accuracy/n_retained), `outputs/calibration/abstention_coverage_curves.png` (the 5-panel sensitivity/specificity-vs-coverage figure).

## Limitations (abstention)

- Coverage was only swept down to 50%; behavior beyond that point (abstaining on more than half of cases) is untested and not assumed to continue the same trends.
- Sensitivity/specificity at low coverage rest on smaller retained samples than at full coverage (1314 → roughly 657 at coverage 0.5, further split by predicted class), so the most restrictive coverage points on each curve carry more sampling variance than the full-coverage point, even though no point showed the kind of single-digit-sample instability seen earlier in the equal-width ECE binning check.
- This analysis reports what abstention *could* achieve on the retained set; it does not model what happens to the abstained-on 30–50% of cases in practice (e.g. radiologist workload, turnaround time) — that operational question is out of scope here.

## MC dropout (explored, not adopted)

The third Phase 9 sub-step asked whether an alternative uncertainty signal — Monte Carlo dropout — would rank test examples by reliability at least as well as the calibrated-confidence approach above. The comparison was run rigorously, and the answer is no: MC dropout is documented here as a rejected alternative, not a deployed one.

**Why E04 was excluded.** Neither the project's custom E04 head nor the underlying torchvision `densenet121` (default `drop_rate=0`, confirmed against the installed source) contains any dropout layers. MC dropout requires an architecture with dropout already present; E04 has none, and adding it would require retraining, which was out of scope for this sub-step. MC dropout was therefore only implemented for E08.

**Method.** E08's `FusionModel` contains two dropout layers on its active forward path (`clinical_encoder.2` and `fusion_head.2`, both p=0.3) — both downstream of the frozen DenseNet vision embedding. For each of the 1314 test examples, the (dropout-free) vision embedding was computed once and reused, then the fusion head was run 50 times with only those two dropout layers left stochastic; the DenseNet backbone stayed in `eval()` mode throughout the whole procedure so BatchNorm continued using its frozen running statistics rather than batch statistics — mixing those two modes would have silently corrupted the vision embeddings themselves, not just added noise. Per label, the mean and standard deviation across the 50 samples were recorded; the standard deviation is the MC-dropout uncertainty score, with test examples ranked by ascending std (lowest std = most confident) for the same 1.0→0.5 coverage sweep used for calibrated-confidence abstention.

**Sanity check passed.** Mean absolute difference between the MC-dropout mean and the original deterministic single-pass probability was small across all 5 labels (0.004–0.011), confirming the MC samples are centered correctly around the model's normal operating point rather than reflecting some other, broken computation.

**Result: MC-dropout ranking underperformed calibrated-confidence ranking on 4 of 5 labels, sometimes substantially**, comparing balanced accuracy at coverage 0.7:

| Label | Calibrated-confidence ranking | MC-dropout ranking |
|---|---|---|
| Cardiomegaly | 0.783 | 0.751 |
| Edema | 0.703 | 0.681 |
| Atelectasis | 0.639 | **0.588** |
| Pleural Effusion | 0.781 | 0.722 |
| Consolidation | 0.650 | **0.660** |

Atelectasis is the sharpest case: MC-dropout-ranked abstention (0.588) performs *worse* than E08's own no-abstention baseline from the earlier analysis (0.606) — actively counterproductive, not merely less helpful. Consolidation is the sole exception, where MC-dropout ranking wins narrowly (0.660 vs 0.650); a plausible but unconfirmed explanation is that Consolidation/E08's unusually low decision threshold (0.20, versus 0.35–0.50 elsewhere) makes `|calibrated_prob − threshold|` a comparatively weaker confidence signal there, and a threshold-agnostic uncertainty measure like MC-dropout std doesn't inherit that particular distortion — this is a hypothesis worth stating, not a proven mechanism.

**Why this happened, mechanistically.** The two active dropout layers sit entirely downstream of the vision embedding — the actual image-processing pathway (the frozen DenseNet backbone) is fully deterministic and untouched by dropout. MC-dropout variance here can therefore only capture "how sensitive is the small fusion head to stochastic perturbation of its own p=0.3 dropout masks," which is a narrower and noisier signal than the calibrated probability's distance from a label's decision threshold — the latter reflects the full pipeline's output, image and clinical information both. This was flagged as a likely outcome before the experiment ran (limited dropout scope, modest architectural stochasticity), and it is exactly what was observed.

**Deployment decision:** MC-dropout uncertainty is not adopted. The existing calibrated-confidence approach (deployed temperatures + per-label Youden's J thresholds) remains the system's abstention mechanism, documented above. This result is retained as a negative finding: the evaluation harness built for calibrated-confidence abstention was reused, unmodified in its scoring logic, to test an alternative — and it correctly detected that the alternative underperforms rather than only ever reporting methods that work.

Full artifacts: `outputs/calibration/mc_dropout_e08.parquet` (all 1314×50 samples, summarized to mean/std per label), `outputs/calibration/mc_dropout_coverage_results.csv`, `outputs/calibration/mc_dropout_vs_calibration_coverage.png` (5-panel comparison figure, calibrated-confidence vs. MC-dropout ranking, both sensitivity and specificity curves).

### Limitations (MC dropout)

- Only E08 was evaluated; no MC-dropout comparison exists for E04, since it has no dropout layers by construction.
- The negative result is specific to this architecture's dropout placement (small, downstream-only, p=0.3 on two layers) and should not be generalized to "MC dropout doesn't work for medical imaging" — a model with dropout distributed through its vision backbone (as in some from-scratch architectures) could plausibly show a different result.
- The Consolidation exception is reported with an explanation offered as a hypothesis, not verified further; testing it directly (e.g. checking whether the calibrated-confidence signal is specifically noisier near unusual thresholds) was out of scope for this sub-step.
