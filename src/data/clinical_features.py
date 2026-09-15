"""Clinical feature engineering for Phase 6 tabular modeling.

Transforms raw manifest columns into a fixed, ordered, fully-numeric
feature matrix with train-only statistics to avoid leakage.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.image_dataset import LABEL_COLUMNS

LOGGER = logging.getLogger(__name__)

NUMERIC_FEATURES = ["age", "recent_bmi"]
CATEGORICAL_FEATURES = ["sex", "race", "ethnicity", "insurance_type"]
MISSING_FLAGS = ["sex_missing", "race_missing", "ethnicity_missing", "insurance_type_missing"]
IMPL_FLAGS = ["age_implausible", "recent_bmi_implausible"]
TARGET_COLUMNS = list(LABEL_COLUMNS)


def _get_train_medians(train_frame: pd.DataFrame) -> tuple[float, float]:
    """Compute age and BMI medians from the train split only."""
    age_median = train_frame.loc[~train_frame["age_implausible"], "age"].median()
    if pd.isna(age_median):
        age_median = train_frame["age"].median()
    bmi_median = train_frame.loc[~train_frame["recent_bmi_implausible"], "recent_bmi"].median()
    if pd.isna(bmi_median):
        bmi_median = train_frame["recent_bmi"].median()
    return float(age_median), float(bmi_median)


def _get_train_standardization_stats(
    train_frame: pd.DataFrame, age_median: float, bmi_median: float
) -> tuple[float, float, float, float]:
    """Compute train-split mean and std for age and BMI after median imputation.

    Returns (age_mean, age_std, bmi_mean, bmi_std). Guards against
    zero/NaN std by falling back to 1.0 with a warning.
    """
    age_filled = train_frame["age"].fillna(age_median)
    bmi_filled = train_frame["recent_bmi"].fillna(bmi_median)

    age_mean = float(age_filled.mean())
    age_std = float(age_filled.std())
    bmi_mean = float(bmi_filled.mean())
    bmi_std = float(bmi_filled.std())

    if age_std <= 0.0 or pd.isna(age_std):
        LOGGER.warning(
            "Train age std is %.4f (or NaN); falling back to 1.0 for standardization.",
            age_std,
        )
        age_std = 1.0
    if bmi_std <= 0.0 or pd.isna(bmi_std):
        LOGGER.warning(
            "Train BMI std is %.4f (or NaN); falling back to 1.0 for standardization.",
            bmi_std,
        )
        bmi_std = 1.0

    return age_mean, age_std, bmi_mean, bmi_std


def _one_hot_encode_train(train_frame: pd.DataFrame) -> list[str]:
    """Fit one-hot column names from train split categorical features.

    Returns the sorted list of expected one-hot column names so that
    val/test can be aligned to the exact same column set.
    """
    frames = []
    for col in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(train_frame[col].fillna("Unknown"), prefix=col, dtype=np.uint8)
        frames.append(dummies)
    combined = pd.concat(frames, axis=1)
    return sorted(combined.columns.tolist())


def _encode_categorical(
    frame: pd.DataFrame, expected_columns: list[str]
) -> pd.DataFrame:
    """One-hot encode categorical features, aligning to train's column set."""
    frames = []
    for col in CATEGORICAL_FEATURES:
        dummies = pd.get_dummies(
            frame[col].fillna("Unknown"), prefix=col, dtype=np.uint8
        )
        frames.append(dummies)
    combined = pd.concat(frames, axis=1)
    missing = set(expected_columns) - set(combined.columns)
    for col in missing:
        combined[col] = 0
    extra = set(combined.columns) - set(expected_columns)
    if extra:
        combined = combined.drop(columns=list(extra))
    return combined[expected_columns]


def build_clinical_features(
    frame: pd.DataFrame,
    train_frame: pd.DataFrame | None = None,
    feature_columns: list[str] | None = None,
    age_mean: float | None = None,
    age_std: float | None = None,
    bmi_mean: float | None = None,
    bmi_std: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build model-ready clinical feature matrix from manifest rows.

    Args:
        frame: The dataframe to process (any split).
        train_frame: If provided, used to compute medians and one-hot
            column names (i.e. when frame IS the train split). If None,
            `feature_columns` must be provided and `train_frame` is not
            used (i.e. when frame is val/test).
        feature_columns: Pre-computed feature column names from the
            training split. Required when `train_frame` is None.
        age_mean, age_std, bmi_mean, bmi_std: Pre-computed train-split
            standardization statistics. Required when `train_frame` is None.

    Returns:
        Tuple of (feature_dataframe, feature_column_names) where the
        dataframe contains only numeric columns in a fixed order with
        no NaN values.
    """
    is_train = train_frame is not None
    if not is_train and feature_columns is None:
        raise ValueError(
            "feature_columns must be provided when train_frame is None "
            "(i.e. for val/test splits)."
        )
    if not is_train and (age_mean is None or age_std is None or bmi_mean is None or bmi_std is None):
        raise ValueError(
            "age_mean, age_std, bmi_mean, bmi_std must be provided when train_frame is None "
            "(i.e. for val/test splits)."
        )

    df = frame.copy()

    if is_train:
        train_df = train_frame if train_frame is not None else df
        age_median, bmi_median = _get_train_medians(train_df)
        age_mean, age_std, bmi_mean, bmi_std = _get_train_standardization_stats(train_df, age_median, bmi_median)
        one_hot_columns = _one_hot_encode_train(train_df)
        LOGGER.info(
            "Train standardization stats: age_mean=%.4f age_std=%.4f bmi_mean=%.4f bmi_std=%.4f",
            age_mean, age_std, bmi_mean, bmi_std,
        )
    else:
        age_median, bmi_median = _get_train_medians(train_frame)

    df["age"] = df["age"].fillna(age_median)
    df["age"] = (df["age"] - age_mean) / age_std
    df["age_implausible"] = df["age_implausible"].astype(np.uint8)

    df["recent_bmi_was_missing"] = df["recent_bmi"].isna().astype(np.uint8)
    df["recent_bmi"] = df["recent_bmi"].fillna(bmi_median)
    df["recent_bmi"] = (df["recent_bmi"] - bmi_mean) / bmi_std
    df["recent_bmi_implausible"] = df["recent_bmi_implausible"].astype(np.uint8)

    for flag in MISSING_FLAGS:
        df[flag] = df[flag].astype(np.uint8)

    cat_encoded = _encode_categorical(df, one_hot_columns if is_train else feature_columns)

    base_numeric = [
        "age",
        "age_implausible",
        "recent_bmi",
        "recent_bmi_implausible",
        "recent_bmi_was_missing",
        *MISSING_FLAGS,
    ]
    base_df = df[base_numeric].astype(np.float32)

    feature_df = pd.concat([base_df, cat_encoded], axis=1)
    feature_columns_final = base_numeric + (one_hot_columns if is_train else feature_columns)

    if feature_df.isna().any().any():
        nan_cols = feature_df.columns[feature_df.isna().any()].tolist()
        raise ValueError(f"NaN values remain in feature columns: {nan_cols}")

    LOGGER.info("Clinical features built: %d columns", len(feature_columns_final))
    LOGGER.info("Columns: %s", feature_columns_final)

    return feature_df, feature_columns_final


def get_train_feature_columns(train_frame: pd.DataFrame) -> list[str]:
    """Convenience function to get feature columns from train split only."""
    _, cols = build_clinical_features(train_frame, train_frame=train_frame)
    return cols


def get_train_standardization_stats(
    train_frame: pd.DataFrame
) -> tuple[float, float, float, float]:
    """Compute train-split standardization stats (age_mean, age_std, bmi_mean, bmi_std)."""
    age_median, bmi_median = _get_train_medians(train_frame)
    return _get_train_standardization_stats(train_frame, age_median, bmi_median)