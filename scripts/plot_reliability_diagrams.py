"""Reliability diagrams and Expected Calibration Error (ECE) for E04 vs E08.

Phase 9, step 1: purely diagnostic. No model weights are touched here.
Reads the predictions saved by generate_val_predictions.py and asks, per
label and per model: when the model says "70% probability", is the true
positive rate in that bucket actually close to 70%?

Outputs:
    outputs/calibration/reliability_diagrams.png  -- 5-panel figure
    outputs/calibration/ece_table.csv              -- per-label, per-model ECE + bin counts
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = REPO_ROOT / "outputs" / "val_predictions.parquet"
OUTPUT_DIR = REPO_ROOT / "outputs" / "calibration"
FIGURE_PATH = OUTPUT_DIR / "reliability_diagrams.png"
ECE_TABLE_PATH = OUTPUT_DIR / "ece_table.csv"
FIGURE_PATH_QUANTILE = OUTPUT_DIR / "reliability_diagrams_quantile.png"
ECE_TABLE_PATH_QUANTILE = OUTPUT_DIR / "ece_table_quantile.csv"

LABELS = ["Cardiomegaly_label", "Edema_label", "Atelectasis_label", "Pleural Effusion_label", "Consolidation_label"]
N_BINS = 10
N_BINS_QUANTILE = 8
MIN_BIN_COUNT_WARN = 15  # flag bins thinner than this rather than silently trust them


def compute_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, bin_mean_pred, bin_frac_positive, bin_counts).

    Bins with zero samples are dropped from the returned arrays (not
    plotted, not counted toward ECE) rather than silently shown as 0.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)

    bin_centers, bin_mean_pred, bin_frac_pos, bin_counts = [], [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2.0)
        bin_mean_pred.append(float(y_prob[mask].mean()))
        bin_frac_pos.append(float(y_true[mask].mean()))
        bin_counts.append(count)

    return (
        np.asarray(bin_centers),
        np.asarray(bin_mean_pred),
        np.asarray(bin_frac_pos),
        np.asarray(bin_counts),
    )


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted-average |mean predicted - fraction positive| across bins."""
    _, bin_mean_pred, bin_frac_pos, bin_counts = compute_calibration_curve(y_true, y_prob, n_bins)
    n_total = bin_counts.sum()
    gaps = np.abs(bin_mean_pred - bin_frac_pos)
    ece = float(np.sum((bin_counts / n_total) * gaps))
    return ece, bin_mean_pred, bin_frac_pos, bin_counts, gaps


def compute_calibration_curve_quantile(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS_QUANTILE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return quantile-binned (bin_centers, bin_mean_pred, bin_frac_positive, bin_counts).

    Uses equal-frequency bins based on predicted probabilities. If duplicated
    values reduce the number of unique quantile bins, fewer bins are returned.
    """
    qcut_bins = pd.qcut(y_prob, q=n_bins, labels=False, duplicates="drop")
    if np.ndim(qcut_bins) != 1:
        qcut_bins = np.asarray(qcut_bins).reshape(-1)

    qcut_bins = np.asarray(qcut_bins, dtype=np.float64)
    valid_mask = ~np.isnan(qcut_bins)
    qcut_bins = qcut_bins[valid_mask].astype(np.int64)
    y_prob_valid = y_prob[valid_mask]
    y_true_valid = y_true[valid_mask]

    bin_mean_pred, bin_frac_pos, bin_counts = [], [], []
    for b in np.unique(qcut_bins):
        mask = qcut_bins == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_mean_pred.append(float(y_prob_valid[mask].mean()))
        bin_frac_pos.append(float(y_true_valid[mask].mean()))
        bin_counts.append(count)

    bin_mean_pred_arr = np.asarray(bin_mean_pred)
    return (
        bin_mean_pred_arr,
        bin_mean_pred_arr,
        np.asarray(bin_frac_pos),
        np.asarray(bin_counts),
    )


def expected_calibration_error_quantile(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS_QUANTILE
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted-average |mean predicted - fraction positive| across quantile bins."""
    _, bin_mean_pred, bin_frac_pos, bin_counts = compute_calibration_curve_quantile(
        y_true, y_prob, n_bins
    )
    n_total = bin_counts.sum()
    gaps = np.abs(bin_mean_pred - bin_frac_pos)
    ece = float(np.sum((bin_counts / n_total) * gaps))
    return ece, bin_mean_pred, bin_frac_pos, bin_counts, gaps


def main() -> None:
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS_PATH}. Run generate_val_predictions.py first."
        )

    df = pd.read_parquet(PREDICTIONS_PATH, engine='fastparquet')
    n_val = len(df)
    print(f"Loaded {n_val} val predictions.")
    print(f"Using {N_BINS} equal-width bins (~{n_val / N_BINS:.0f} samples/bin if evenly spread).\n")

    fig, axes = plt.subplots(1, len(LABELS), figsize=(4.2 * len(LABELS), 4.6), sharey=True)

    ece_rows = []
    low_count_rows = []

    for ax, label in zip(axes, LABELS):
        y_true = df[f"{label}_true"].to_numpy(dtype=np.float64)
        pretty_label = label.replace("_label", "")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")

        for model_key, color in [("e04", "tab:blue"), ("e08", "tab:orange")]:
            y_prob = df[f"{label}_{model_key}"].to_numpy(dtype=np.float64)
            ece, bin_mean_pred, bin_frac_pos, bin_counts, gaps = expected_calibration_error(y_true, y_prob)

            ax.plot(
                bin_mean_pred,
                bin_frac_pos,
                marker="o",
                color=color,
                label=f"{model_key.upper()} (ECE={ece:.3f})",
            )

            for mean_pred, frac_pos, count in zip(bin_mean_pred, bin_frac_pos, bin_counts):
                ax.annotate(
                    str(int(count)),
                    (mean_pred, frac_pos),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=color,
                )
                if int(count) < MIN_BIN_COUNT_WARN:
                    low_count_rows.append(
                        {
                            "label": pretty_label,
                            "model": model_key.upper(),
                            "bin_mean_pred": float(mean_pred),
                            "count": int(count),
                        }
                    )

            thin_bins = int((bin_counts < MIN_BIN_COUNT_WARN).sum())
            ece_rows.append(
                {
                    "label": pretty_label,
                    "model": model_key.upper(),
                    "ece": ece,
                    "n_bins_used": len(bin_counts),
                    "n_bins_thin": thin_bins,
                    "min_bin_count": int(bin_counts.min()) if len(bin_counts) else 0,
                    "true_base_rate": float(y_true.mean()),
                    "mean_predicted_prob": float(y_prob.mean()),
                }
            )

        ax.set_title(pretty_label)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("Observed fraction positive")
    fig.suptitle(f"Reliability diagrams — E04 vs E08 (val, n={n_val})")
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)

    ece_table = pd.DataFrame(ece_rows)
    ece_table.to_csv(ECE_TABLE_PATH, index=False)

    print(ece_table.to_string(index=False))

    if low_count_rows:
        low_count_table = pd.DataFrame(low_count_rows)
        low_count_table = low_count_table.sort_values(
            by=["label", "model", "bin_mean_pred"]
        ).reset_index(drop=True)
        print("\nLow-count bins (<15 samples) by label/model/bin:")
        print(low_count_table.to_string(index=False, formatters={"bin_mean_pred": "{:.6f}".format}))
    else:
        print("\nLow-count bins (<15 samples) by label/model/bin: none")

    any_thin = ece_table["n_bins_thin"].sum()
    if any_thin:
        print(
            f"\nNote: {any_thin} (label, model, bin) combinations had fewer than "
            f"{MIN_BIN_COUNT_WARN} val samples. Those points are still plotted "
            f"but should be read as noisy, not precise."
        )

    print(f"\nSaved figure: {FIGURE_PATH}")
    print(f"Saved ECE table: {ECE_TABLE_PATH}")

    print("\nQuantile-binned ECE")
    print(f"Using {N_BINS_QUANTILE} quantile bins (~equal samples per bin before ties).\n")

    fig_q, axes_q = plt.subplots(1, len(LABELS), figsize=(4.2 * len(LABELS), 4.6), sharey=True)
    ece_rows_q = []

    for ax, label in zip(axes_q, LABELS):
        y_true = df[f"{label}_true"].to_numpy(dtype=np.float64)
        pretty_label = label.replace("_label", "")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")

        for model_key, color in [("e04", "tab:blue"), ("e08", "tab:orange")]:
            y_prob = df[f"{label}_{model_key}"].to_numpy(dtype=np.float64)
            ece, bin_mean_pred, bin_frac_pos, bin_counts, gaps = expected_calibration_error_quantile(
                y_true, y_prob, n_bins=N_BINS_QUANTILE
            )

            ax.plot(
                bin_mean_pred,
                bin_frac_pos,
                marker="o",
                color=color,
                label=f"{model_key.upper()} (ECE={ece:.3f})",
            )

            for mean_pred, frac_pos, count in zip(bin_mean_pred, bin_frac_pos, bin_counts):
                ax.annotate(
                    str(int(count)),
                    (mean_pred, frac_pos),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=color,
                )

            thin_bins = int((bin_counts < MIN_BIN_COUNT_WARN).sum())
            ece_rows_q.append(
                {
                    "label": pretty_label,
                    "model": model_key.upper(),
                    "ece": ece,
                    "n_bins_used": len(bin_counts),
                    "n_bins_thin": thin_bins,
                    "min_bin_count": int(bin_counts.min()) if len(bin_counts) else 0,
                    "true_base_rate": float(y_true.mean()),
                    "mean_predicted_prob": float(y_prob.mean()),
                }
            )

        ax.set_title(pretty_label)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="upper left")

    axes_q[0].set_ylabel("Observed fraction positive")
    fig_q.suptitle(f"Reliability diagrams (quantile bins) — E04 vs E08 (val, n={n_val})")
    fig_q.tight_layout()
    fig_q.savefig(FIGURE_PATH_QUANTILE, dpi=200)
    plt.close(fig_q)

    ece_table_q = pd.DataFrame(ece_rows_q)
    ece_table_q.to_csv(ECE_TABLE_PATH_QUANTILE, index=False)
    print(ece_table_q.to_string(index=False))

    print(f"\nSaved quantile figure: {FIGURE_PATH_QUANTILE}")
    print(f"Saved quantile ECE table: {ECE_TABLE_PATH_QUANTILE}")


if __name__ == "__main__":
    main()
