"""Phase 13 step 3 verification: CORS headers + /predict + /explain through HTTP."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent
BACKEND = "http://127.0.0.1:8000"
FRONTEND = "http://localhost:5173"
ORIGIN = "http://localhost:5173"
WRONG_ORIGIN = "http://evil.example.com"


def check(name: str, condition: bool, detail: str) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}")


def main() -> None:
    # ---- test row 0 (same source as earlier verification scripts) ----
    manifest = pd.read_parquet(REPO / "data" / "processed" / "image_subset_manifest.parquet", engine='fastparquet')
    row = manifest.loc[manifest["split"] == "test"].reset_index(drop=True).iloc[0]
    image_path = Path(row["local_image_path"])
    image_bytes = image_path.read_bytes()
    clinical = {
        "age": str(float(row["age"])),
        "sex": str(row["sex"]),
        "race": str(row["race"]),
        "ethnicity": str(row["ethnicity"]),
        "insurance_type": str(row["insurance_type"]),
    }
    if pd.notna(row["recent_bmi"]):
        clinical["recent_bmi"] = str(float(row["recent_bmi"]))
    print(f"test row 0 image: {image_path.name} ({len(image_bytes)} bytes), fields: {clinical}")

    # ---- 1. frontend dev server is serving the app ----
    r = requests.get(FRONTEND + "/", timeout=10)
    ok_root = r.status_code == 200 and 'id="root"' in r.text
    check("vite serves index.html", ok_root, f"GET {FRONTEND}/ -> {r.status_code}")

    for src in [
        "/src/App.jsx",
        "/src/api.js",
        "/src/components/UploadForm.jsx",
        "/src/components/PredictionResults.jsx",
        "/src/components/ExplanationResults.jsx",
    ]:
        r = requests.get(FRONTEND + src, timeout=10)
        check(
            f"vite transforms {src}",
            r.status_code == 200 and "Internal Server Error" not in r.text[:200],
            f"{r.status_code}, {len(r.text)} bytes",
        )

    # ---- 2. OPTIONS preflight for /predict ----
    t0 = time.perf_counter()
    r = requests.options(
        BACKEND + "/predict",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        timeout=10,
    )
    preflight_ms = (time.perf_counter() - t0) * 1000
    acao = r.headers.get("Access-Control-Allow-Origin")
    check(
        "preflight /predict",
        r.status_code == 200 and acao == ORIGIN,
        f"status={r.status_code} ACAO={acao!r} in {preflight_ms:.0f} ms "
        f"ACAM={r.headers.get('Access-Control-Allow-Methods')!r}",
    )

    # ---- 3. OPTIONS preflight for /explain ----
    r = requests.options(
        BACKEND + "/explain",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
        timeout=10,
    )
    acao = r.headers.get("Access-Control-Allow-Origin")
    check(
        "preflight /explain",
        r.status_code == 200 and acao == ORIGIN,
        f"status={r.status_code} ACAO={acao!r}",
    )

    # ---- 4. actual GET /health with Origin ----
    t0 = time.perf_counter()
    r = requests.get(BACKEND + "/health", headers={"Origin": ORIGIN}, timeout=10)
    health_ms = (time.perf_counter() - t0) * 1000
    acao = r.headers.get("Access-Control-Allow-Origin")
    check(
        "GET /health with Origin",
        r.status_code == 200 and r.json() == {"status": "ok"} and acao == ORIGIN,
        f"status={r.status_code} body={r.text} ACAO={acao!r} in {health_ms:.0f} ms",
    )

    # ---- 5. negative control: wrong origin gets no ACAO ----
    r = requests.get(BACKEND + "/health", headers={"Origin": WRONG_ORIGIN}, timeout=10)
    bad_acao = r.headers.get("Access-Control-Allow-Origin")
    check(
        "wrong origin rejected",
        bad_acao != WRONG_ORIGIN,
        f"ACAO for {WRONG_ORIGIN} = {bad_acao!r} (must not be the origin)",
    )

    # ---- 6. POST /predict (multipart, with Origin) ----
    t0 = time.perf_counter()
    r = requests.post(
        BACKEND + "/predict",
        files={"image": ("test.png", image_bytes, "image/png")},
        data=clinical,
        headers={"Origin": ORIGIN},
        timeout=60,
    )
    predict_s = time.perf_counter() - t0
    acao = r.headers.get("Access-Control-Allow-Origin")
    check(
        "POST /predict",
        r.status_code == 200 and acao == ORIGIN,
        f"status={r.status_code} ACAO={acao!r} in {predict_s:.3f} s",
    )
    if r.status_code == 200:
        preds = r.json()["predictions"]
        for label, item in preds.items():
            print(
                f"    {label}: e04={item['e04_probability']:.4f} ({item['e04_prediction']}) "
                f"e08={item['e08_probability']:.4f} ({item['e08_prediction']})"
            )
    else:
        print(f"    body: {r.text[:500]}")

    # ---- 7. POST /explain (multipart, with Origin) ----
    t0 = time.perf_counter()
    r = requests.post(
        BACKEND + "/explain",
        files={"image": ("test.png", image_bytes, "image/png")},
        data=clinical,
        headers={"Origin": ORIGIN},
        timeout=60,
    )
    explain_s = time.perf_counter() - t0
    acao = r.headers.get("Access-Control-Allow-Origin")
    check(
        "POST /explain",
        r.status_code == 200 and acao == ORIGIN,
        f"status={r.status_code} ACAO={acao!r} in {explain_s:.3f} s",
    )
    if r.status_code == 200:
        exps = r.json()["explanations"]
        for label, item in exps.items():
            top = ", ".join(
                f"{f['feature_name']}={f['shap_value']:+.3f}" for f in item["top_shap_features"]
            )
            print(
                f"    {label}: calibrated={item['calibrated_probability']:.4f} "
                f"gradcam_b64_len={len(item['gradcam_image_base64'])} top_shap=[{top}]"
            )
    else:
        print(f"    body: {r.text[:500]}")


if __name__ == "__main__":
    main()
