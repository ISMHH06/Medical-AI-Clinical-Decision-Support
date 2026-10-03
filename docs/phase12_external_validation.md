# Phase 12 — External Validation: Status

**Status: blocked, not attempted.** This is a documented limitation, not a skipped step.

## The constraint

Phase 12 as originally scoped requires testing the trained model (E08) on a second, independent dataset — MIMIC-CXR was the intended target — to check generalization beyond CheXpert Plus's own data distribution. CheXpert Plus was chosen as the primary dataset specifically because it offered immediate access, with MIMIC-CXR + MIMIC-IV explicitly flagged from the start of this project as a v2 stretch addition contingent on PhysioNet credentialing coming through. As of this writing, that credentialing has not come through, so no second chest X-ray dataset is available.

## What was checked before accepting the blocker

Rather than skip this silently, a substitute was investigated: could an *internal* distribution-shift check — testing whether E08 generalizes across different sites, scanners, or acquisition periods represented within CheXpert Plus itself — serve as a partial, honestly-weaker stand-in for true external validation?

This was checked directly against the processed manifest (`data/processed/image_subset_manifest.parquet`) and found **not viable**: the manifest contains no site, institution, scanner/equipment, or acquisition-date column. Its only structural fields beyond patient ID and labels are `path_to_image`, `local_image_path`, and `split`; `path_to_image` follows a `split/patient/study/view` naming pattern with no batch or source grouping, and the repeated filename component only distinguishes view type (`view1_frontal.jpg`, `view2_frontal.jpg`, etc.), not source. Project documentation (`README.md`, `docs/v1_results.md`) confirms no multi-site or multi-batch structure is documented for this processed subset.

One raw-data signal was identified and explicitly **not** pursued: the original CheXpert Plus export (`df_chexpert_plus_240401.csv`, outside the processed manifest) contains a `patient_report_date_order` field and free-text radiology report content that may reference institution or date-of-study information. Extracting a clean source-group variable from this would require NLP-based entity extraction from report free text with no guarantee of a clean, well-balanced grouping at the end — a substantially larger and noisier undertaking than this check warrants, with real risk of introducing an artificial or misleading "generalization" result rather than a genuine one. This was correctly ruled out rather than forced.

## What this means for the project

No generalization claim beyond CheXpert Plus's own data distribution can currently be made for E04 or E08. This is stated plainly as an open limitation of the project as a whole, not something implied to have been tested. If PhysioNet credentialing for MIMIC-CXR is granted at a later point, Phase 12 can be revisited with the original design (train/evaluate on CheXpert Plus as done throughout this project, evaluate the frozen E04/E08 checkpoints on MIMIC-CXR's equivalent labels with no retraining, following the same AUROC/ECE/subgroup-fairness/stress-testing methodology already built for CheXpert Plus in Phases 9–11 so results are directly comparable).

## Limitations

- This status may change if PhysioNet credentialing is resolved; this document reflects the state as of 2026-10-02 and should be revisited rather than assumed permanent.
- The investigation confirmed no usable metadata exists in the *processed* manifest; it did not exhaustively mine the raw CheXpert Plus export's free-text report fields, since doing so was judged out of proportion to what this check warranted — a different judgment call could reasonably reach a different conclusion if someone were willing to invest in report-text entity extraction specifically for this purpose.
