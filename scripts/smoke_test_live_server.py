"""Phase 14 step 2b: real-server smoke test.

Unlike tests/test_api.py (in-process, TestClient), this script requires
an actual running uvicorn process and makes real HTTP requests over a
real socket. Run this after the pytest suite passes, and again after any
deployment change (e.g. Docker packaging) to confirm the real process
still starts and serves correctly -- TestClient cannot catch problems
that only appear when the app runs as its own OS process (binding
issues, startup failures, missing files in a container, etc.).

Usage:
    1. Start the server in one terminal:
       python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
    2. Wait for "Application startup complete."
    3. Run this script in a second terminal:
       python scripts/smoke_test_live_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

BACKEND = "http://127.0.0.1:8000"
MANIFEST_PATH = Path("data/processed/image_subset_manifest.parquet")

REGRESSION_CARDIO_E04 = 0.8676
REGRESSION_CARDIO_E08 = 0.8648
REGRESSION_TOLERANCE = 1e-4

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def load_test_row() -> tuple[bytes, dict[str, str]]:
    manifest = pd.read_parquet(MANIFEST_PATH, engine="fastparquet")
    row = manifest.loc[manifest["split"] == "test"].reset_index(drop=True).iloc[0]
    image_path = Path(row["local_image_path"])
    fields = {
        "age": str(float(row["age"])),
        "sex": str(row["sex"]),
        "race": str(row["race"]),
        "ethnicity": str(row["ethnicity"]),
        "insurance_type": str(row["insurance_type"]),
    }
    if pd.notna(row["recent_bmi"]):
        fields["recent_bmi"] = str(float(row["recent_bmi"]))
    return image_path.read_bytes(), fields


def main() -> None:
    try:
        resp = requests.get(f"{BACKEND}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: cannot reach {BACKEND}. Start the server first:")
        print("  python -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    check("GET /health", resp.status_code == 200 and resp.json() == {"status": "ok"},
          f"status={resp.status_code} body={resp.text}")

    resp = requests.get(f"{BACKEND}/debug/memory", timeout=5)
    body = resp.json() if resp.status_code == 200 else {}
    check("GET /debug/memory", resp.status_code == 200 and "cuda_available" in body,
          f"status={resp.status_code} body={body}")

    image_bytes, fields = load_test_row()

    resp = requests.post(
        f"{BACKEND}/predict",
        files={"image": ("test.png", image_bytes, "image/png")},
        data=fields,
        timeout=30,
    )
    predict_ok = resp.status_code == 200
    check("POST /predict (200)", predict_ok, f"status={resp.status_code}")

    if predict_ok:
        preds = resp.json()["predictions"]
        cardio = preds.get("Cardiomegaly", {})
        e08_val = cardio.get("e08_probability")
        e04_val = cardio.get("e04_probability")
        e08_ok = e08_val is not None and abs(e08_val - REGRESSION_CARDIO_E08) < REGRESSION_TOLERANCE
        e04_ok = e04_val is not None and abs(e04_val - REGRESSION_CARDIO_E04) < REGRESSION_TOLERANCE
        check(
            "Cardiomegaly e08_probability regression",
            e08_ok,
            f"got {e08_val}, expected {REGRESSION_CARDIO_E08} +/- {REGRESSION_TOLERANCE}",
        )
        check(
            "Cardiomegaly e04_probability regression",
            e04_ok,
            f"got {e04_val}, expected {REGRESSION_CARDIO_E04} +/- {REGRESSION_TOLERANCE}",
        )
        check("All 5 labels present", set(preds) == {
            "Cardiomegaly", "Edema", "Atelectasis", "Pleural Effusion", "Consolidation"
        }, f"got keys: {sorted(preds)}")

    resp = requests.post(
        f"{BACKEND}/predict",
        files={"image": ("test.png", image_bytes, "image/png")},
        data={**fields, "sex": "NotARealCategory"},
        timeout=30,
    )
    check("POST /predict invalid category -> 422", resp.status_code == 422,
          f"status={resp.status_code}")

    resp = requests.options(
        f"{BACKEND}/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        timeout=10,
    )
    check(
        "CORS preflight /predict",
        resp.status_code == 200 and resp.headers.get("access-control-allow-origin") == "http://localhost:5173",
        f"status={resp.status_code} ACAO={resp.headers.get('access-control-allow-origin')!r}",
    )

    print("\nSkipping /explain in this smoke test (slow; covered by pytest's test_explain_happy_path).")
    print("Run with --explain to include it." if "--explain" not in sys.argv else "")
    if "--explain" in sys.argv:
        resp = requests.post(
            f"{BACKEND}/explain",
            files={"image": ("test.png", image_bytes, "image/png")},
            data=fields,
            timeout=60,
        )
        check("POST /explain (200)", resp.status_code == 200, f"status={resp.status_code}")

    n_total = len(results)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n=== {n_pass}/{n_total} checks passed ===")
    if n_pass != n_total:
        print("FAILURES:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
