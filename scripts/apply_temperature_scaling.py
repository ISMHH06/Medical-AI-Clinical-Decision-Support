"""Apply selective deployment for Phase 9 temperature scaling.

This script reads outputs/calibration/temperature_scaling_results.csv and
chooses whether to deploy each fitted temperature based on held-out test
ECE improvement under equal-width binning.

Rule:
- deploy fitted temperature only if ece_width_after < ece_width_before
- otherwise deploy T=1.0 (no correction)

Outputs:
- outputs/calibration/deployed_temperatures.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "outputs" / "calibration" / "temperature_scaling_results.csv"
OUTPUT_PATH = REPO_ROOT / "outputs" / "calibration" / "deployed_temperatures.json"


def main() -> None:
    """Select temperatures to deploy based on held-out test ECE improvement."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Missing temperature scaling results: {INPUT_PATH}")

    results = pd.read_csv(INPUT_PATH)
    deployed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    deployed_count = 0
    kept_count = 0

    for _, row in results.iterrows():
        ece_before = float(row["ece_width_before"])
        ece_after = float(row["ece_width_after"])
        fitted_temperature = float(row["temperature"])
        deploy = ece_after < ece_before
        deployed_temperature = fitted_temperature if deploy else 1.0
        if deploy:
            reason = f"ece_width improved on test ({ece_before:.6f} -> {ece_after:.6f})"
            deployed_count += 1
        else:
            reason = f"ece_width did not improve on test ({ece_before:.6f} -> {ece_after:.6f}); kept T=1.0"
            kept_count += 1

        deployed_rows.append(
            {
                "label": row["label"],
                "model": row["model"],
                "fitted_temperature": fitted_temperature,
                "deployed_temperature": deployed_temperature,
                "deployed": bool(deploy),
                "reason": reason,
                "ece_width_before": ece_before,
                "ece_width_after": ece_after,
            }
        )
        summary_rows.append(
            {
                "label": row["label"],
                "model": row["model"],
                "deployed_temperature": deployed_temperature,
                "deployed": bool(deploy),
                "resulting_ece_width": ece_after if deploy else ece_before,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(deployed_rows, fp, indent=2)

    summary_table = pd.DataFrame(summary_rows)
    print(summary_table.to_string(index=False))
    print(f"\nDeployed True: {deployed_count}, False: {kept_count}")
    print(f"Saved deployed temperatures: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
