# Phase 13 — Production System: API & Frontend

**Status: complete.** Backend (prediction + explanation endpoints) and a React frontend calling both.

## What was built

A FastAPI application (`app/`) wrapping the project's already-validated E04/E08 models and explainability pipeline behind an HTTP API, built incrementally and verified at each step rather than assumed correct from a successful server start.

**`GET /health`** — basic liveness check, returns `{"status": "ok"}`.

**`POST /predict`** — accepts a chest X-ray image upload plus raw clinical fields (age, sex, race, ethnicity, insurance_type, optional recent_bmi) as multipart form data. Returns calibrated probabilities and binary predictions (at the deployed per-label thresholds) for both E04 and E08, for all 5 labels, in one response. Clinical fields are validated against the actual category sets present in the training split (not a hardcoded guess), with clear 422 errors for invalid input rather than opaque failures.

**`POST /explain`** — same input contract as `/predict`. Returns, for all 5 labels in one call: the calibrated E08 probability, a base64-encoded Grad-CAM overlay PNG, and the top 5 SHAP features by absolute value. Reuses the exact Grad-CAM (`generate_gradcam`) and SHAP (`generate_shap_values`) functions built in Phase 8 — no reimplementation. The SHAP background (50 train-split samples) is built once at startup, not per-request.

Both models, the SHAP background, calibration temperatures, and decision thresholds are all loaded once at application startup (via FastAPI's `lifespan` context manager), not per-request — necessary given model loading is expensive and the API is expected to handle repeated calls.

## Verification performed

Every piece of this was checked against ground truth, not assumed correct from "the server didn't crash":

- **`/predict` vs. stored batch predictions**: a direct row-0 comparison against `outputs/test_predictions.parquet` (with E08's deployed temperature applied the same way) matched to within ~1e-7 on all 10 (label, model) comparisons — the live inference path is numerically faithful to the validated Phase 9 batch pipeline, not a parallel implementation that happens to look reasonable.
- **`/explain` correctness and timing**: an in-process call to `PredictionService.explain()` (bypassing the HTTP layer) completed in 1.74 seconds and produced predictions matching `/predict`'s numbers exactly. A separate, real HTTP round-trip (curl for `/health`, a Python `requests` call for `/explain`) returned status 200 in 2.44 seconds with identical probabilities and base64 image lengths — confirming the HTTP layer adds negligible overhead and the result is consistent across two independent invocation paths.

## A debugging note worth keeping

An earlier attempt to verify `/explain` over HTTP from within the coding agent's own tool loop appeared to hang indefinitely (30+ minutes, no response). This was investigated and found to be **a VS Code agent process-management artifact, not an application bug**: the agent's background server process was apparently being affected by its own shell/job-tree lifecycle while it tried to poll for a response, and it spent significant time debugging unrelated process-survival behavior (testing with an unrelated `ping` subprocess, switching language models) rather than the actual endpoint. Running the server and the test request manually in two separate, independently-opened terminals resolved this immediately and confirmed the endpoint was correct and fast all along. Lesson for future sessions: when an agent reports a request "hanging" with no error, checking whether the issue is in the agent's own process supervision (rather than the code under test) — ideally by reproducing manually outside the agent's tool loop — is a fast, high-value diagnostic step before assuming application-level slowness or a bug.

## Known simplifications (not bugs, deliberate scope choices)

- **Input plausibility/missingness flags are not derived from caller input.** `age_implausible`, `recent_bmi_implausible`, and the four `*_missing` flags are currently hardcoded to 0/False for every request, regardless of what the caller actually sends. This means the API does not yet replicate the training pipeline's handling of out-of-range or missing clinical values — a caller sending an implausible age, for instance, is currently treated as if it were a normal value. This is a real gap to close before treating the API as faithfully representative of the training-time data contract, not merely a cosmetic one.
- **Grad-CAM images are base64-embedded directly in the JSON response**, not saved to disk with a separate URL. Simple and self-contained for this stage; if response payload size becomes a real concern this is a natural candidate for Phase 14 (serving/optimization) to revisit.
- **No abstention logic yet.** Phase 9's calibrated-confidence abstention mechanism is not exposed via the API — responses currently always return a binary prediction, with no "uncertain, recommend human review" signal. This is the natural next increment.
- A pandas `FutureWarning` appears during clinical feature BMI handling (`fillna` downcasting behavior that will change in a future pandas version). Not currently affecting correctness, but worth a proactive fix rather than waiting for a pandas upgrade to silently change behavior.

## Frontend

Built as a separate React + Vite application (`frontend/`), not served by FastAPI — a deliberate choice to use a real framework from the start rather than a plain-HTML placeholder, given the frontend's role as part of a portfolio project. Communicates with the FastAPI backend over `fetch()` across origins, requiring `CORSMiddleware` to be added to `app/main.py` (restricted to the Vite dev origin `http://localhost:5173`, not a wildcard).

**Structure:** `App.jsx` holds all state (selected image, clinical form fields, prediction/explanation results, loading and error states for each call) via plain `useState` — appropriate at this scale, no state-management library needed. Three components: `UploadForm` (image + clinical fields, triggers `/predict`), `PredictionResults` (renders the E04/E08 probability table, triggers `/explain`), `ExplanationResults` (renders Grad-CAM images and signed SHAP bars per label). `/predict` and `/explain` are deliberately separate user actions (not triggered together), since `/explain` takes noticeably longer (~1.4–2.4s vs. `/predict`'s sub-second response) and most uses of the tool won't need an explanation for every prediction.

**Verification performed**, not just "the dev servers started": a 13-check verification script confirmed, with real HTTP requests and real test-split data (not mocked): the Vite dev server serves and correctly transforms all 5 source files; CORS preflight (`OPTIONS`) succeeds for both `/predict` and `/explain` with the correct `Access-Control-Allow-Origin` header; a request from an unauthorized origin (`evil.example.com`) does **not** receive that header (negative control, confirming CORS is actually restrictive, not just present); and live `POST /predict` and `POST /explain` calls through the CORS-enabled path return results numerically consistent with every earlier verification in this phase. The dropdown option lists for sex/race/ethnicity/insurance_type were pulled from the actual train-split category sets (the same ones Phase 10's subgroup fairness analysis used), not a guessed or simplified list — so the frontend cannot submit a value the backend would reject.

The disclaimer "Research prototype — not a medical device" is shown in the app header — a responsible addition given the subject matter, included without being explicitly requested.

## Next steps

1. Close the plausibility/missingness flag gap noted above (hardcoded `age_implausible`/`recent_bmi_implausible`/`*_missing` flags).
2. Add an abstention/confidence-flag field to `/predict` and `/explain` responses, reusing Phase 9's logic, and surface it in the frontend.
3. Phase 14 (serving/optimization): latency benchmarking, containerization, and revisiting the base64-image response design if needed.
