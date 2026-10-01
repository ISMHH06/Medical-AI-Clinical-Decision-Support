# Phase 10 — Robustness & Bias: Stress-Testing

This is the second Phase 10 sub-step (following subgroup fairness testing, `docs/phase10_fairness.md`). It asks a different question: how much does performance degrade under realistic image and clinical-data corruption, and does that degradation differ between E04 (vision-only) and E08 (fusion)?

## Method

**Image perturbations** were applied to the raw test images (before resizing/normalization) across 8 conditions plus a clean baseline, run through both E04 and E08: brightness scaling (0.7x darken, 1.3x brighten), Gaussian pixel noise at 3 severities (σ = 0.02, 0.05, 0.10 of the [0,1] pixel range), and JPEG re-compression at 3 quality levels (75, 50, 25). Clinical features were left unperturbed in this test, isolating the effect of image corruption alone.

**Clinical feature perturbations** were applied to E08's 29-dim clinical feature vector (E04 has no clinical branch and was excluded from this test) across 7 conditions plus a clean baseline: random zeroing of 10%/30%/50% of feature dimensions per patient (simulating missing data), and Gaussian noise added to the two continuous features (age, BMI) at 0.25x/0.5x/1x/2x their own train-split standard deviation (simulating measurement/entry error). The vision embedding was held fixed, isolating the clinical branch's contribution.

Both tests used AUROC on the full test split (n=1314), with E08 probabilities passed through its deployed calibration temperatures, consistent with every other E08 evaluation in this project.

## Findings: image perturbation

**The system is broadly robust to realistic image degradation.** Across all 8 perturbed conditions, the largest *average* AUROC drop (noise at σ=0.10) was only 0.015–0.020 for both models. JPEG compression — even at quality 25, visibly degraded — barely moved performance at all (E08: 0.7247 clean → 0.7241 at q25). This is a genuinely reassuring result for a system meant to handle real-world image variation.

**But the degradation is not uniform across labels, and the pattern is specific and mechanistically plausible, not random.** The single largest drop in the entire study was `Atelectasis/E08` under high Gaussian noise (0.6470 → 0.6083, a drop of 0.0387) — more than double the next-largest drop, and consistent with Atelectasis already being this project's weakest label across every prior analysis (Phase 9 calibration, Phase 9 abstention/MC-dropout, Phase 10 subgroup fairness). Brightening (not darkening) was the more damaging direction for several labels, particularly Pleural Effusion (−0.032 to −0.033 AUROC) — physiologically plausible, since overexposure can wash out subtle density differences (fluid, tissue findings) in ways that uniform darkening does not.

**Fusion provides no meaningful extra robustness over vision-only.** E04 and E08 track each other closely across every condition; neither consistently degrades less than the other. This is consistent with Phase 7/8's earlier finding that fusion's advantage is mostly a calibration effect rather than new visual understanding — if the clinical branch isn't doing real visual work, there's no reason to expect it to compensate when the image itself is degraded, and it doesn't.

## Findings: clinical feature perturbation

**E08 is essentially insensitive to clinical feature corruption.** Across every one of the 7 perturbed conditions — including zeroing half the 29-dim feature vector (`missing_50`) and doubling the noise on age/BMI (`noise_2x`) — AUROC moved by no more than ±0.0002 for any label. This is not a subtle effect too small to matter; it is, within the resolution of this test, no measurable effect at all.

**A follow-up weight inspection ruled out the simplest explanation and pointed to a better-supported one.** The natural first hypothesis — that the fusion head has learned to architecturally down-weight the clinical embedding — was checked directly by inspecting the trained `FusionModel`'s first fusion layer weights and found **not** to hold: the mean absolute weight magnitude for the clinical-embedding input slice (0.0170) is essentially identical to the vision-embedding slice (0.0171), a ratio of 1.01. The clinical encoder's own layers also have ordinary, non-degenerate weight magnitudes (0.092, 0.063 mean-abs). The clinical branch is not architecturally suppressed.

The better-supported explanation connects directly to Phase 6: the clinical-only model (E06) achieves only 0.56 AUROC standalone — barely above chance. If the clinical features themselves carry weak predictive signal, a fusion head that combines both branches with comparable weight will still show little output sensitivity to clinical perturbation — not because the branch is ignored, but because there is little strong signal in that branch to destroy in the first place. The robustness observed here is a property of the clinical data's limited information content, not evidence of architectural suppression. (The clinical slice's normalized L2 is actually *higher* than the vision slice's — 0.000324 vs. 0.0000569 — but this is an artifact of the clinical slice having far fewer columns (32 vs. 1024), concentrating the same weight budget into a smaller space; it does not contradict the mean-absolute-weight finding and is noted here so the metric isn't selectively reported.)

## How this connects to the rest of the project

This is the third independent line of evidence, now converging with Phase 6/7 and Phase 8, on the same underlying characterization of E08: **the fusion model relies overwhelmingly on the image and treats the clinical branch as a minor adjustment, because the clinical data itself carries weak signal for these five labels** — not because the architecture suppresses it.

1. Phase 6/7: clinical-only model (E06) barely beats chance (AUROC 0.56).
2. Phase 8: Grad-CAM shows E04 and E08 attend to visually near-identical regions; fusion's AUROC gain looks like calibration, not new visual understanding.
3. Phase 10 (here): E08's output is essentially unchanged under severe clinical-feature corruption, and this is now confirmed to be a property of the data's weak signal rather than the architecture's weighting.

Three independently-designed analyses, run at different points in the project with different methods, reaching the same conclusion — this is as strong a form of evidence as this project has produced for any single claim.

## Limitations

- Image perturbation severities were chosen to span a realistic range (mild-to-visible) but are not exhaustive; more extreme corruption, or corruption types not tested here (motion blur, geometric distortion, sensor artifacts specific to certain scanner models), could behave differently.
- Clinical perturbation only targeted the two continuous features (age, BMI) with Gaussian noise, since noise injection is not a meaningful corruption model for one-hot categorical indicators; categorical-feature corruption was only tested via the missingness (zeroing) conditions, not via, e.g., flipping a category to an incorrect-but-valid one.
- The weight-inspection check examined only the first fusion layer's weight magnitudes. It does not rule out more subtle forms of information bottleneck (e.g. the clinical embedding being weighted comparably but located in a part of representation space the fusion head's downstream layers don't exploit) — a more thorough mechanistic study (e.g. ablating the clinical branch entirely and comparing to zeroing its inputs) was out of scope here.
- Both stress tests used the test split only; no comparison to val-split perturbation behavior was made, so it's not established whether these robustness patterns would hold on a different held-out sample.

Full artifacts: `outputs/robustness/image_perturbation_results.csv`, `outputs/robustness/image_perturbation_curves.png`, `outputs/robustness/clinical_perturbation_results.csv`, `outputs/robustness/clinical_perturbation_curves.png`.
