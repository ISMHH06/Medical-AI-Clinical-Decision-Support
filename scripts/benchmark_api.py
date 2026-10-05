"""Phase 14: latency and GPU memory benchmarking for /predict and /explain.

Run this with the FastAPI backend already running at http://127.0.0.1:8000
(e.g. `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000` in a
separate terminal). This script only sends HTTP requests -- it does not
start or stop the server itself.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

BACKEND = "http://127.0.0.1:8000"
MANIFEST_PATH = Path("data/processed/image_subset_manifest.parquet")
OUTPUT_DIR = Path("outputs/benchmarks")
RESULTS_PATH = OUTPUT_DIR / "api_latency_benchmark.json"
PLOT_PATH = OUTPUT_DIR / "latency_distribution.png"

N_PREDICT = 50
N_EXPLAIN = 20


def load_test_fields() -> tuple[Path, dict[str, str]]:
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
    return image_path, fields


def call_endpoint(endpoint: str, image_path: Path, fields: dict[str, str]) -> float:
    """Send one request, return elapsed seconds. Raises on non-200."""
    start = time.perf_counter()
    with image_path.open("rb") as fp:
        resp = requests.post(
            f"{BACKEND}{endpoint}",
            files={"image": (image_path.name, fp, "image/png")},
            data=fields,
            timeout=120,
        )
    elapsed = time.perf_counter() - start
    if resp.status_code != 200:
        raise RuntimeError(f"{endpoint} returned {resp.status_code}: {resp.text[:300]}")
    return elapsed


def get_memory() -> dict:
    resp = requests.get(f"{BACKEND}/debug/memory", timeout=10)
    resp.raise_for_status()
    return resp.json()


def compute_stats(latencies_s: list[float]) -> dict:
    latencies_ms = sorted(x * 1000 for x in latencies_s)
    n = len(latencies_ms)
    def pct(p: float) -> float:
        idx = min(n - 1, int(round(p / 100 * (n - 1))))
        return latencies_ms[idx]
    return {
        "n": n,
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def main() -> None:
    try:
        health = requests.get(f"{BACKEND}/health", timeout=5)
        health.raise_for_status()
        assert health.json() == {"status": "ok"}
    except Exception as error:
        print(f"ERROR: backend not reachable at {BACKEND}/health ({error}).")
        print("Start it first: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    print(f"Backend reachable at {BACKEND}.")
    image_path, fields = load_test_fields()
    print(f"Using test image: {image_path.name}")

    baseline_memory = get_memory()
    print(f"Baseline memory: {baseline_memory}")

    print("\nFirst request (may include residual server-start warmup, not an isolated cold start)...")
    first_request_latency_s = call_endpoint("/predict", image_path, fields)
    print(f"  first_request_latency_seconds = {first_request_latency_s:.4f}")

    print(f"\nWarm /predict benchmark: {N_PREDICT} requests...")
    predict_latencies: list[float] = []
    for i in range(N_PREDICT):
        predict_latencies.append(call_endpoint("/predict", image_path, fields))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_PREDICT} done")
    predict_stats = compute_stats(predict_latencies)
    print(f"  /predict stats (ms): {predict_stats}")

    print(f"\nWarm /explain benchmark: {N_EXPLAIN} requests...")
    explain_latencies: list[float] = []
    for i in range(N_EXPLAIN):
        explain_latencies.append(call_endpoint("/explain", image_path, fields))
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{N_EXPLAIN} done")
    explain_stats = compute_stats(explain_latencies)
    print(f"  /explain stats (ms): {explain_stats}")

    final_memory = get_memory()
    print(f"\nFinal memory (cumulative since server start, not reset per-benchmark): {final_memory}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "first_request": {
            "latency_seconds": first_request_latency_s,
            "note": "may include residual warmup from server start, not a true isolated cold start",
        },
        "predict_stats_ms": predict_stats,
        "explain_stats_ms": explain_stats,
        "memory_baseline": baseline_memory,
        "memory_final": final_memory,
        "memory_note": "max_allocated_mb is cumulative since server process start, not reset per-benchmark run",
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results: {RESULTS_PATH}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [[x * 1000 for x in predict_latencies], [x * 1000 for x in explain_latencies]],
        tick_labels=[f"/predict (n={N_PREDICT})", f"/explain (n={N_EXPLAIN})"],
    )
    ax.set_ylabel("Latency (ms)")
    ax.set_title("API endpoint latency distribution (warm, sequential requests)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {PLOT_PATH}")

    print("\n=== SUMMARY ===")
    print(f"First request: {first_request_latency_s * 1000:.1f} ms")
    print(f"/predict  median={predict_stats['median_ms']:.1f}ms  p95={predict_stats['p95_ms']:.1f}ms  p99={predict_stats['p99_ms']:.1f}ms")
    print(f"/explain  median={explain_stats['median_ms']:.1f}ms  p95={explain_stats['p95_ms']:.1f}ms  p99={explain_stats['p99_ms']:.1f}ms")
    print(f"Peak GPU memory observed: {final_memory.get('max_allocated_mb')} MB")


if __name__ == "__main__":
    main()
