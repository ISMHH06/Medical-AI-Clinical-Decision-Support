"""Build the reproducible Phase 2 CheXpert-style processed dataset.

The module deliberately limits itself to pandas-based data engineering. It
merges local metadata and pathology labels, applies the agreed label policy,
cleans visible clinical fields, assigns patient-level splits, and writes a
single parquet table plus a data dictionary.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


LOGGER = logging.getLogger(__name__)
TARGET_LABELS = [
    "Cardiomegaly",
    "Edema",
    "Atelectasis",
    "Pleural Effusion",
    "Consolidation",
]
U_ONES_LABELS = {"Atelectasis", "Edema"}
CLINICAL_COLUMNS = ["age", "sex", "race", "ethnicity", "insurance_type", "recent_bmi"]
NUMERIC_CLINICAL_COLUMNS = ["age", "recent_bmi"]
CATEGORICAL_CLINICAL_COLUMNS = ["sex", "race", "ethnicity", "insurance_type"]
REQUIRED_CONFIG_KEYS = {
    "metadata_csv",
    "labels_jsonl",
    "processed_output",
    "data_dictionary_output",
}


def configure_logging() -> None:
    """Configure concise progress logging for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_config(config_path: str | Path) -> tuple[dict[str, str], Path]:
    """Load and validate the YAML configuration, returning its repository root.

    Relative file paths in the YAML are resolved from the repository root (the
    parent of ``configs/``), so the command works from any current directory.
    """
    config_file = Path(config_path).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file was not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a key/value mapping.")

    missing_keys = REQUIRED_CONFIG_KEYS.difference(config)
    if missing_keys:
        raise ValueError(f"Configuration is missing required keys: {sorted(missing_keys)}")

    empty_keys = [key for key in REQUIRED_CONFIG_KEYS if not isinstance(config[key], str) or not config[key].strip()]
    if empty_keys:
        raise ValueError(f"Configuration values must be non-empty paths: {empty_keys}")

    # configs/data_config.yaml lives one level below the repository root.
    return {key: config[key].strip() for key in REQUIRED_CONFIG_KEYS}, config_file.parent.parent


def resolve_config_path(path_value: str, repository_root: Path) -> Path:
    """Resolve an absolute or repository-relative path from the YAML config."""
    path = Path(path_value)
    return path if path.is_absolute() else repository_root / path


def require_columns(frame: pd.DataFrame, columns: list[str], source_name: str) -> None:
    """Fail early with a useful message if an expected source column is absent."""
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def load_and_merge(metadata_path: Path, labels_path: Path) -> pd.DataFrame:
    """Load source files and inner-join them while reporting unmatched rows."""
    LOGGER.info("1. LOAD")
    metadata = pd.read_csv(metadata_path)
    labels = pd.read_json(labels_path, lines=True)
    require_columns(metadata, ["path_to_image"], "Metadata CSV")
    require_columns(labels, ["path_to_image", *TARGET_LABELS], "Labels JSONL")
    LOGGER.info("  Metadata rows loaded: %d", len(metadata))
    LOGGER.info("  Label rows loaded: %d", len(labels))

    # An outer merge exposes rows found on only one side before retaining matches.
    audit = metadata[["path_to_image"]].merge(
        labels[["path_to_image"]], how="outer", on="path_to_image", indicator=True
    )
    metadata_only = int((audit["_merge"] == "left_only").sum())
    labels_only = int((audit["_merge"] == "right_only").sum())
    LOGGER.info("  Rows dropped with no matching labels: %d", metadata_only)
    LOGGER.info("  Rows dropped with no matching metadata: %d", labels_only)

    merged = metadata.merge(labels, how="inner", on="path_to_image")
    LOGGER.info("  Rows after inner merge: %d", len(merged))
    return merged


def filter_frontal_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Print frontal/lateral encodings, then retain only the exact Frontal view."""
    require_columns(frame, ["frontal_lateral"], "Merged dataset")
    LOGGER.info("2. FILTER TO FRONTAL VIEW")
    LOGGER.info("  frontal_lateral unique values (including missing): %s", frame["frontal_lateral"].unique().tolist())
    if "Frontal" not in set(frame["frontal_lateral"].dropna()):
        raise ValueError("Expected frontal_lateral value 'Frontal' was not found; inspect the printed values.")
    before = len(frame)
    filtered = frame.loc[frame["frontal_lateral"] == "Frontal"].copy()
    LOGGER.info("  Rows before/after: %d/%d", before, len(filtered))
    return filtered


def select_project_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain the v1 targets and only the image, patient, and clinical fields."""
    selected_columns = ["path_to_image", "deid_patient_id", *CLINICAL_COLUMNS, *TARGET_LABELS]
    require_columns(frame, selected_columns, "Frontal-view dataset")
    LOGGER.info("3. SELECT TARGET LABELS")
    LOGGER.info("  Rows unchanged: %d", len(frame))
    LOGGER.info("  Retained target labels: %s", ", ".join(TARGET_LABELS))
    return frame.loc[:, selected_columns].copy()


def apply_label_policy(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve raw target values and create binary labels using the agreed policy."""
    LOGGER.info("4. APPLY UNCERTAIN/MISSING LABEL POLICY")
    for label in TARGET_LABELS:
        raw = pd.to_numeric(frame[label], errors="coerce")
        unexpected = raw.dropna()[~raw.dropna().isin([-1.0, 0.0, 1.0])]
        if not unexpected.empty:
            raise ValueError(f"Unexpected values in {label}: {sorted(unexpected.unique().tolist())}")

        # Keep the source value inspectable; only the derived column is binary.
        frame.rename(columns={label: f"{label}_raw"}, inplace=True)
        binary = raw.fillna(0).copy()  # Missing means negative for every target.
        binary.loc[raw == -1.0] = 1 if label in U_ONES_LABELS else 0
        frame[f"{label}_label"] = binary.astype("int8")

        positives = int(frame[f"{label}_label"].sum())
        negatives = len(frame) - positives
        percent = (100 * positives / len(frame)) if len(frame) else 0.0
        LOGGER.info("  %s_label: positive=%d, negative=%d, positive rate=%.2f%%", label, positives, negatives, percent)
    LOGGER.info("  Rows unchanged: %d", len(frame))
    return frame


def standardize_categorical(series: pd.Series) -> pd.Series:
    """Trim whitespace and merge case-only variants without imputing missing data."""
    cleaned = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    non_missing = cleaned.dropna()
    # Pick the most frequent existing spelling for each case-insensitive group.
    canonical_by_casefold: dict[str, str] = {}
    for value, count in non_missing.value_counts().items():
        folded = value.casefold()
        if folded not in canonical_by_casefold:
            canonical_by_casefold[folded] = value
    return cleaned.map(lambda value: canonical_by_casefold.get(value.casefold(), value) if pd.notna(value) else pd.NA)


def clean_clinical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Report clinical-data quality and add flags without dropping or imputing rows."""
    LOGGER.info("5. CLEAN CLINICAL FEATURES")
    LOGGER.info("  Missing values before cleaning:\n%s", frame[CLINICAL_COLUMNS].isna().sum().to_string())

    for column in NUMERIC_CLINICAL_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = numeric
        LOGGER.info("  %s min/max/median: %s / %s / %s", column, numeric.min(), numeric.max(), numeric.median())
        if column == "age":
            frame["age_implausible"] = (numeric > 120) | (numeric <= 0)
        else:
            frame["recent_bmi_implausible"] = numeric <= 0

    for column in CATEGORICAL_CLINICAL_COLUMNS:
        LOGGER.info("  %s unique value counts before standardization:\n%s", column, frame[column].value_counts(dropna=False).to_string())
        frame[column] = standardize_categorical(frame[column])
        frame[f"{column}_missing"] = frame[column].isna()

    LOGGER.info("  Rows unchanged: %d", len(frame))
    return frame


def assign_patient_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign each patient to exactly one reproducible 80/10/10 split."""
    LOGGER.info("6. PATIENT-LEVEL SPLIT")
    if frame["deid_patient_id"].isna().any():
        raise ValueError("deid_patient_id contains missing values; patient-level splitting cannot be verified safely.")

    patient_ids = frame["deid_patient_id"].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    patient_count = len(patient_ids)
    train_end = int(patient_count * 0.8)
    validation_end = train_end + int(patient_count * 0.1)
    split_patients = {
        "train": set(patient_ids[:train_end]),
        "val": set(patient_ids[train_end:validation_end]),
        "test": set(patient_ids[validation_end:]),
    }

    patient_to_split = {patient: split for split, patients in split_patients.items() for patient in patients}
    frame["split"] = frame["deid_patient_id"].map(patient_to_split)

    overlaps = [
        split_patients["train"] & split_patients["val"],
        split_patients["train"] & split_patients["test"],
        split_patients["val"] & split_patients["test"],
    ]
    no_overlap = all(not overlap for overlap in overlaps)
    LOGGER.info("  Zero patient ID overlap between splits: %s", no_overlap)
    if not no_overlap or frame["split"].isna().any():
        raise RuntimeError("Patient-level split assignment failed overlap verification.")

    for split in ("train", "val", "test"):
        subset = frame.loc[frame["split"] == split]
        LOGGER.info("  %s: %d patients, %d rows", split, len(split_patients[split]), len(subset))
        for label in TARGET_LABELS:
            rate = 100 * subset[f"{label}_label"].mean() if len(subset) else 0.0
            LOGGER.info("    %s_label positive rate: %.2f%%", label, rate)
    LOGGER.info("  Rows unchanged: %d", len(frame))
    return frame


def column_description(column: str) -> tuple[str, str]:
    """Return a concise meaning and derivation for a final output column."""
    if column == "path_to_image":
        return "Image path identifier", "Copied from the metadata CSV and used to merge labels."
    if column == "deid_patient_id":
        return "De-identified patient identifier", "Copied from metadata; used for patient-level split assignment."
    if column in CLINICAL_COLUMNS:
        return "Clinical feature", "Copied from metadata; numeric values parsed or categorical whitespace/case variants standardized."
    if column.endswith("_raw"):
        return "Raw pathology label", "Original JSONL pathology value (1, 0, -1, or missing) retained for inspection."
    if column.endswith("_label"):
        return "Binary target label", "Derived from the raw label using the documented missing/uncertain-label policy."
    if column.endswith("_implausible"):
        return "Data-quality flag", "True when the numeric value is outside the defined plausible range; no row was dropped."
    if column.endswith("_missing"):
        return "Missingness flag", "True when the categorical clinical value remains missing after standardization."
    if column == "split":
        return "Dataset split", "Train/val/test assigned by de-identified patient ID using seed 42 and an 80/10/10 patient ratio."
    return "Output field", "Retained or derived by the Phase 2 dataset pipeline."


def write_data_dictionary(frame: pd.DataFrame, output_path: Path) -> None:
    """Write Markdown documentation for every column in the saved parquet file."""
    rows = ["# Processed Dataset Data Dictionary", "", "| Name | Dtype | Meaning | How derived |", "| --- | --- | --- | --- |"]
    for column, dtype in frame.dtypes.items():
        meaning, derivation = column_description(column)
        rows.append(f"| {column} | {dtype} | {meaning} | {derivation} |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_dataset(config_path: str | Path) -> pd.DataFrame:
    """Run all Phase 2 dataset-building steps and persist their final outputs."""
    config, repository_root = load_config(config_path)
    metadata_path = resolve_config_path(config["metadata_csv"], repository_root)
    labels_path = resolve_config_path(config["labels_jsonl"], repository_root)
    parquet_path = resolve_config_path(config["processed_output"], repository_root)
    dictionary_path = resolve_config_path(config["data_dictionary_output"], repository_root)

    frame = load_and_merge(metadata_path, labels_path)
    frame = filter_frontal_rows(frame)
    frame = select_project_columns(frame)
    frame = apply_label_policy(frame)
    frame = clean_clinical_features(frame)
    frame = assign_patient_splits(frame)

    LOGGER.info("7. SAVE OUTPUT")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    write_data_dictionary(frame, dictionary_path)
    LOGGER.info("  Saved processed table: %s", parquet_path)
    LOGGER.info("  Saved data dictionary: %s", dictionary_path)
    return frame


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and build the configured processed dataset."""
    parser = argparse.ArgumentParser(description="Build the Phase 2 processed clinical imaging dataset.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "data_config.yaml"),
        help="Path to YAML data-path configuration (default: configs/data_config.yaml).",
    )
    args = parser.parse_args(argv)
    configure_logging()
    build_dataset(args.config)


if __name__ == "__main__":
    main()
