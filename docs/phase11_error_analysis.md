# Phase 11 — Error Analysis

Phases 9 and 10 repeatedly flagged Atelectasis as this project's weakest label using aggregate metrics (AUROC, ECE). Phase 11 asks a more specific question: what does that weakness actually look like at the level of individual errors, and is Atelectasis really distinct from the other four labels, or does it just look that way in aggregate?

## Method

All analysis uses E08's deployed (calibrated) probabilities and per-label Youden's J thresholds (`models/threshold_optimization_results.json`) on the full test split (n=1314) — pure analysis of existing saved predictions, no new inference. Three things were measured for every label, not just Atelectasis, specifically to test whether Atelectasis is genuinely an outlier or whether the earlier aggregate-metric comparisons made it look more distinct than it is:

1. **Confusion structure** — false positive rate (FPR) and false negative rate (FNR) separately, rather than the combined sensitivity/specificity view used elsewhere in this project.
2. **Co-occurrence structure** — for each label's false negatives and false positives, what fraction of those error cases have at least one other true-positive label present (comorbidity-adjacent), compared against the overall baseline co-occurrence rate for that label (so the error-subset percentage is read against the right reference point, not in isolation).
3. **Confidence structure** — how far each error sits from its label's decision threshold: a "near-miss" (within 0.10) versus "confidently wrong" (more than 0.25 away).

A follow-up visual step pulled Grad-CAM overlays for 8 selected cases (6 distinct patients — two patients each legitimately qualified for two categories), chosen specifically to test the findings from the quantitative step rather than sampled randomly.

## Finding 1: Atelectasis is not the worst label on every dimension — three labels fail three different ways

Running the comparison across all 5 labels, rather than assuming Atelectasis would again come out worst, produced a genuinely different picture than Phase 9/10's aggregate metrics implied:

| Label | Worst on |
|---|---|
| Atelectasis | Highest FNR (0.312) |
| Consolidation | Highest FPR (0.513) |
| Edema | Highest combined confidently-wrong rate (0.413) |

These are three distinct failure modes, not one label failing worse across the board. Consolidation's problem is a tendency to over-call (more than half its negative cases get a false positive); Edema's problem is that when it's wrong, it tends to be wrong with real confidence rather than hedging near the boundary; Atelectasis's problem, examined next, turns out to be its own distinct thing.

## Finding 2: Atelectasis's errors are overwhelmingly near-threshold, not confidently wrong — and this explains two earlier results

The most striking single number in this analysis: **89.5% of Atelectasis false negatives and 83.5% of its false positives are near-misses** (within 0.10 of threshold), and **0% of either are confidently wrong** (more than 0.25 away). No other label comes close to this pattern — every other label has a substantial confidently-wrong fraction (Cardiomegaly FN: 50%, Edema FP: 48%, Consolidation FP: 36%). Atelectasis's mean error distance from threshold (0.05–0.06) is roughly a fifth to a tenth of most other labels' (0.13–0.25).

This reframes what "Atelectasis is hard" actually means. It is not that the model is confused or badly wrong about Atelectasis — it is that the model's probability estimate for Atelectasis sits right on the decision boundary for the overwhelming majority of its errors. This single finding retroactively explains two earlier, previously unconnected results:

- **Why abstention helped Atelectasis most** (Phase 9): an abstention mechanism built to catch near-threshold, low-confidence predictions should disproportionately help a label whose errors are almost entirely near-threshold — which is exactly what Phase 9 found (Atelectasis/E08 showed the largest balanced-accuracy gain from abstaining, 0.606→0.639).
- **Why Atelectasis/E08 needed the largest temperature correction** (Phase 9, T=2.62, more than double any other fitted temperature): a probability distribution whose errors cluster tightly around the threshold rather than spreading with some confidently-wrong tail is consistent with a sharper, more concentrated distribution requiring stronger flattening to calibrate.

## Finding 3: co-occurrence patterns are mostly general, not Atelectasis-specific — except one label's false positives

Every label's false-negative co-occurrence rate sits above its own baseline rate (82–85% of FNs have another true-positive label present, vs. a 65–67% baseline across labels) — false negatives are, generally, more likely in patients with other findings, which is true across the board, not a distinguishing Atelectasis trait.

The more differentiating result is on the false-positive side: **Pleural Effusion's false-positive co-occurrence rate (45%) sits below its own baseline (65%)** — the opposite direction from every other label. Pleural Effusion's false positives are disproportionately "unprompted" false alarms on patients with no other true finding, not comorbidity-driven confusion. This became the basis for the visual-inspection step's fourth case category.

## Finding 4: visual inspection complicated, rather than confirmed, the near-threshold hypothesis

Going into the visual-inspection step, the working hypothesis was that Atelectasis near-miss cases would show Grad-CAM attention in anatomically plausible regions, just under-confident — "the model knows where to look but hedges." The actual Grad-CAM overlays for the 4 selected Atelectasis near-miss cases (2 FN, 2 FP) did not clearly support this. Attention locations were inconsistent across the four cases — one FN's attention sat in an upper image corner not clearly on lung tissue at all; the other FN's attention was a narrow band near the upper-left clavicle/apex region; the two FPs showed attention closer to the cardiac/mediastinal border. There was no single, consistent "almost-right" pattern across the four cases.

The more honest reading: Atelectasis near-threshold cases do not share a consistent attention signature. This is still compatible with "these are inherently ambiguous or borderline cases" — but it points toward attention instability on hard cases, rather than confirming a clean story of consistently-correct-but-underconfident localization. This is a case where the quantitative finding (near-threshold errors) held up, but the specific mechanistic story initially proposed to explain it did not survive a direct visual check — a useful reminder to verify a hypothesis against real examples rather than accept it once a statistic is consistent with it.

**The Edema contrast case was more directly informative.** The confidently-wrong Edema false negative (model said p=0.05, essentially certain there was no Edema, when Edema was present) showed Grad-CAM attention in a small, isolated region not clearly on lung tissue — a genuine instance of the model being both wrong and, by its own explanation, not looking anywhere relevant. This is a concrete, reportable failure case in the same spirit as Phase 8's Pleural Effusion miss.

**The Pleural Effusion isolated-false-positive cases were, unexpectedly, the most anatomically coherent of the eight.** Both showed attention concentrated on lower/mid lung regions — physiologically where effusions pool — despite being false alarms. This suggests these false positives are not random noise but the model over-calling something genuinely present in the right anatomical region (overlapping findings, positioning artifacts, or borderline fluid that didn't meet the ground-truth labeling threshold) — a more specific and more useful characterization than "unprompted false alarm" alone.

## Methodological note

Two of the eight visualized cases (row indices 3 and 6) are each the same patient appearing under two different category labels (the patient qualified for both an Atelectasis near-miss FP and a Pleural Effusion isolated FP, independently, by coincidence of their specific probability values). This was a deliberate choice — deduplicating by patient would have discarded valid category matches — but means the visual evidence rests on 6 distinct patients, not 8 independent cases, and should be read with that sample size in mind.

## Limitations

- All quantitative findings are for E08 only; no comparative error-structure analysis was run for E04, so it is not established whether E04 shows the same near-threshold Atelectasis pattern or a different one.
- The near-miss / confidently-wrong distance thresholds (0.10 / 0.25) are reasonable but somewhat arbitrary cutoffs; results are reported as distributions (mean, median) alongside these binary cutoffs specifically so the conclusion doesn't rest entirely on where those lines were drawn.
- The visual-inspection sample (6 distinct patients) is small and intentionally targeted rather than randomly sampled — appropriate for generating or stress-testing a hypothesis, not for establishing a population-level visual pattern. The inconsistent Atelectasis attention finding is suggestive, not conclusive, at this sample size.
- No feedback loop back into model retraining was performed in this sub-step (the "re-evaluate" half of Phase 11's hypothesis→experiment→re-evaluate cycle, per the original project scope) — this analysis characterizes the errors but does not yet test whether any change (e.g. a different loss weighting, more Atelectasis-focused augmentation) would address the near-threshold pattern.

Full artifacts: `outputs/error_analysis/confusion_matrix.csv`, `outputs/error_analysis/error_cooccurrence.csv`, `outputs/error_analysis/error_confidence.csv`, `outputs/error_analysis/error_case_gradcam.png`.
