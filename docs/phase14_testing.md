# Phase 14 — Serving & Optimization: Automated Testing

Second sub-step of Phase 14, following latency/memory benchmarking (`docs/phase14_benchmarking.md`). Formalizes what had previously only been verified through ad-hoc scripts (Phase 13) into two complementary, repeatable test layers.

## Two layers, deliberately both built

**`tests/test_api.py`** — a `pytest` suite using FastAPI's `TestClient`, running the app in-process. Fast (~10.6s for all 9 tests), no manual server management, suitable for running on every change. Covers: `/health`, `/debug/memory` response shape, `/predict` happy path (with a regression guard against exact values verified in Phase 13 — Cardiomegaly e08_probability and e04_probability, asserted to within 1e-4), four `/predict` validation-failure paths (missing required field, invalid categorical value, empty image file, corrupt/non-image file — all expected to return 422, not an unhandled 500), `/explain` happy path (response shape: base64 Grad-CAM image present, exactly 5 SHAP features per label, correct types throughout), and CORS header presence on both a simple request and an `OPTIONS` preflight.

**`scripts/smoke_test_live_server.py`** — a separate, manually-run script against a real, independently-started `uvicorn` process (not `TestClient`). This exists because an in-process test can never catch problems that only appear when the application runs as its own OS process — a startup failure, a binding issue, a missing file path that only manifests outside the test harness. This becomes especially relevant ahead of Docker packaging, where "does the real process actually start and serve correctly" is exactly the class of problem a containerization step can introduce. Covers a subset of the pytest suite's checks (health, memory endpoint, predict happy path with the same regression guard, one validation failure case, CORS preflight) over real HTTP; `/explain` is skipped by default (slow, and already covered by the pytest suite) but can be included with a `--explain` flag.

## Results

`pytest -v`: **9/9 passed** in 10.56s. Three warnings, both pre-existing and left unsuppressed rather than silenced: an `httpx`/`starlette` deprecation notice (unrelated to this project's code), and the pandas `FutureWarning` in `clinical_features.py`'s BMI handling already tracked as an open item in `docs/phase13_api.md`.

`scripts/smoke_test_live_server.py` against a live server: **8/8 passed**, with the Cardiomegaly regression values matching to full float precision (`0.8648473024368286` / `0.8675761222839355`) — identical, not merely close, to the values independently verified in Phase 13's manual HTTP test and the `pytest` suite's in-process run. This is now the fourth independent confirmation of the same prediction values across this project (stored batch predictions from `test_predictions.parquet`, Phase 13's manual verification, the `TestClient` suite, and this live-process run), giving strong confidence the backend's numerical behavior is stable across every way it has been invoked.

## A deliberate scoping decision caught during setup

Without an explicit `pytest.ini` restricting test discovery to `tests/`, a bare `pytest -v` invocation would have tried to collect `scripts/image_stress_test.py` and `scripts/clinical_stress_test.py` — both match pytest's default `*_test.py` naming pattern, but are standalone GPU-dependent analysis scripts from Phase 10, not test modules, and would fail or behave unexpectedly if pytest tried to import and run them as tests. `pytest.ini` was added with `testpaths = tests` to prevent this, and to register the `slow` marker (`/explain`'s test) so it can be filtered out with `-m "not slow"` for a faster subset run without a warning.

## What this test suite does and does not cover

This is a correctness and regression suite, appropriately scoped to what Phase 14 named ("basic API testing") — not a claim of comprehensive coverage. Explicitly not tested: concurrent/simultaneous requests (the benchmarking sub-step also only tested sequential load), `/explain`'s validation-failure paths separately from `/predict`'s (the underlying validation logic is shared between both endpoints, so this is a deliberate, reasonable gap rather than an oversight), and behavior under resource exhaustion (e.g. GPU out-of-memory). These would be reasonable additions if this moved toward an actual production deployment rather than a portfolio-stage API.

Full artifacts: `tests/test_api.py`, `scripts/smoke_test_live_server.py`, `pytest.ini`.
