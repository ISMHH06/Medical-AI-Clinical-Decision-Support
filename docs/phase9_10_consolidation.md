# Phase 9 & 10 (in progress) — Consolidated Summary

This is a short synthesis of the stretch work completed after v1 (`docs/v1_results.md`), pulling the headline findings from `docs/phase9_calibration.md` and `docs/phase10_fairness.md` into one place. Full methodology and detail live in those two documents; this page is the "what does it collectively mean" layer on top.

## What's done

**Phase 9 — Calibration & uncertainty (complete):**
1. **Temperature scaling.** Both E04 and E08 are systematically overconfident across all 5 labels. Per-label temperatures were fit on val and evaluated honestly on held-out test; only 4 of 10 (label, model) combinations showed real, generalizing ECE improvement and were deployed — the rest were left uncorrected rather than deployed on the strength of NLL improvement alone, since NLL and ECE measured different things and diverged for 6 of 10 cases.
2. **Abstention.** Using the deployed calibrated probabilities, ranking test examples by distance from each label's decision threshold and abstaining on the least-confident fraction improves balanced accuracy in all 10 (label, model) combinations — the clearest unambiguous win in Phase 9. Separate sensitivity/specificity curves revealed two cases (Consolidation/E08, Atelectasis/E04) where the balanced-accuracy gain is actually a real sensitivity-vs-specificity tradeoff, not a clean uniform improvement.
3. **MC dropout (explored, rejected).** Only E08 has dropout layers; MC-dropout-ranked abstention underperformed calibrated-confidence ranking on 4 of 5 labels, worst on Atelectasis where it was actually worse than not abstaining at all. Not adopted — reported as a negative result with a clear mechanistic explanation (dropout sits downstream of the vision embedding only).

**Phase 10 — Robustness & bias (subgroup fairness sub-step complete; stress-testing not yet started):**
4. **Subgroup fairness.** Tested AUROC and ECE spread across sex, race, ethnicity, and insurance_type subgroups (n≥50 floor). Found a large, robust performance gap by race on Atelectasis (AUROC spread ≈0.14, present in both E04 and E08, not a fusion artifact) that compounds with a matching ECE gap on the same subgroup/label. This is the most significant single finding across both phases — a real measured disparity, not a feature-attribution hint.

## How the two phases connect

A pattern runs through all four sub-steps: **Atelectasis is this project's consistent weak point, and it fails in compounding ways, not independent ones.** It has the worst raw calibration (Phase 9, ECE 0.208 even after the largest fitted correction in the whole study, T=2.62), it's the label where abstention helps most but MC dropout actively hurts most, and it's now also the label with the largest subgroup fairness gap by race. These aren't four unrelated observations about Atelectasis — they're four different diagnostic lenses converging on the same underlying fact: Atelectasis is the label this system understands least well, and that weakness shows up in every direction it's examined from.

The Phase 10 fairness finding also partially complicates, rather than simply confirms, Phase 8's earlier explainability work. Phase 8's SHAP analysis flagged race/insurance as recurring clinical-branch contributors specifically on **Cardiomegaly** — but Phase 10's direct performance measurement found Cardiomegaly's own race-based AUROC spread is comparatively moderate (≈0.02–0.04), while the large gap is on Atelectasis and Edema, labels SHAP's case studies didn't happen to examine. The honest reading: both analyses independently detect that race-correlated signal exists in this system, but a feature-attribution finding on one label doesn't reliably predict where the largest real performance gap will turn up. That's a useful methodological lesson in its own right — attribution and outcome-level fairness testing are complementary, not substitutes for each other.

## What's still open

- **Stress-testing** (perturbed images — brightness/noise/compression; degraded clinical inputs — missing/noisy values) is the one remaining Phase 10 item, not yet started.
- **Causal mechanism behind the subgroup gap** is explicitly unestablished — Phase 10 shows the gap exists, not why (imaging acquisition differences, label prevalence by subgroup, sample composition, or genuine differential model behavior are all plausible, undistinguished by this analysis).
- **Error analysis (Phase 11) and external validation (Phase 12)** remain unstarted stretch phases; given the Atelectasis pattern above, a natural next question for Phase 11 would be whether its errors look structurally different from the other four labels' errors, not just worse in aggregate.

## Where things stand

Both Phase 9 sub-steps and Phase 10's first sub-step represent real, defensible stretch work: every deployed correction was validated on held-out data, every negative result (MC dropout) was kept as a negative result rather than discarded or spun, and the fairness finding was reported as a measured gap without an unsupported causal claim attached. The project now has a clear, evidence-backed answer to "is this system calibrated, does it know when to abstain, and does it treat patient subgroups equally" — three questions v1 never asked, now answered with the same rigor v1's core modeling work was held to.
