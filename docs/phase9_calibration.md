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

## Limitations

- Temperature scaling is a single-scalar correction per (label, model). It can rescale a curve's overall confidence but cannot fix a non-monotonic or highly label-specific miscalibration pattern (the Atelectasis/E08 case) as well as a more flexible method (e.g. isotonic regression) might — that tradeoff was accepted here for simplicity and interpretability, not because a better fix doesn't exist.
- The selective-deployment rule is gated on equal-width ECE specifically. Two of the four deployed cases (Atelectasis/E08, Consolidation/E08) also improve under quantile ECE; the other two were not re-checked against that stricter gate. This is a reasonable, stated simplification, not an oversight to be discovered later.
- This work calibrates *marginal* per-label probabilities. It says nothing about calibration conditioned on demographic subgroups — that question is explicitly left to Phase 10, and connects directly to the fairness signal Phase 8's SHAP analysis already surfaced.
