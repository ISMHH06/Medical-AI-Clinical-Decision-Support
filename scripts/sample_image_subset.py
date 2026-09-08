"""Create reproducible, multi-label-aware image-download manifests.

The script samples from the already processed parquet dataset. It never
modifies that source file; it only writes a smaller parquet manifest and a
plain-text list of image paths for downstream download tooling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import pandas as pd


LABEL_COLUMNS = [
    "Cardiomegaly_label",
    "Edema_label",
    "Atelectasis_label",
    "Pleural Effusion_label",
    "Consolidation_label",
]
SPLIT_SETTINGS = {
    "train": {"per_label_cap": 1_500, "target_size": 10_000},
    "val": {"per_label_cap": 200, "target_size": 1_500},
    "test": {"per_label_cap": 200, "target_size": 1_500},
}
NEGATIVE_FRACTION = 0.30
SEED = 42


def validate_dataset(frame: pd.DataFrame) -> None:
    """Check that the processed table has the fields needed for sampling."""
    required = {"path_to_image", "split", *LABEL_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {missing}")
    if frame["path_to_image"].isna().any():
        raise ValueError("path_to_image contains missing values and cannot form a download manifest.")

    for label in LABEL_COLUMNS:
        invalid = frame.loc[~frame[label].isin([0, 1]), label].dropna().unique().tolist()
        if invalid or frame[label].isna().any():
            raise ValueError(f"{label} must be a complete binary 0/1 column; found invalid values: {invalid}")


def print_positive_rates(frame: pd.DataFrame, heading: str) -> None:
    """Print counts and positive rates for all target labels in a subset."""
    print(heading)
    for label in LABEL_COLUMNS:
        positive_count = int(frame[label].sum())
        rate = 100 * positive_count / len(frame) if len(frame) else 0.0
        print(f"  {label}: {positive_count}/{len(frame)} positive ({rate:.2f}%)")


def sample_split(
    split_frame: pd.DataFrame,
    split_name: str,
    per_label_cap: int,
    target_size: int,
    rng: random.Random,
) -> pd.DataFrame:
    """Sample one split using unions of label-positive rows and fully negative rows.

    Positive samples are independently drawn for each label then unioned, so a
    multi-label image is represented once rather than consuming several quotas.
    The fully negative sample is sized to make roughly 30% of the final split
    subset negative, subject to the number of available negative rows.
    """
    print(f"\n{split_name.upper()} SPLIT")
    print(f"  Source rows: {len(split_frame)}; target: approximately {target_size}")

    positive_indices: set[object] = set()
    for label in LABEL_COLUMNS:
        eligible_indices = split_frame.index[split_frame[label] == 1].to_numpy()
        sample_size = min(per_label_cap, len(eligible_indices))
        sampled = rng.sample(list(eligible_indices), k=sample_size) if sample_size else []
        positive_indices.update(sampled)
        print(f"  {label}: sampled {sample_size} of {len(eligible_indices)} positive rows")

    fully_negative_mask = (split_frame[LABEL_COLUMNS] == 0).all(axis=1)
    negative_indices = split_frame.index[fully_negative_mask].to_numpy()
    # If P rows are positive-driven, N = P * 0.30 / 0.70 gives a 30% negative share.
    desired_negative_count = round(len(positive_indices) * NEGATIVE_FRACTION / (1 - NEGATIVE_FRACTION))
    negative_sample_size = min(desired_negative_count, len(negative_indices))
    sampled_negative = (
        rng.sample(list(negative_indices), k=negative_sample_size) if negative_sample_size else []
    )
    selected_indices = positive_indices.union(sampled_negative)
    subset = split_frame.loc[sorted(selected_indices)].copy()

    actual_negative_count = int((subset[LABEL_COLUMNS] == 0).all(axis=1).sum())
    negative_rate = 100 * actual_negative_count / len(subset) if len(subset) else 0.0
    print(f"  Unique positive-driven rows: {len(positive_indices)}")
    print(f"  Fully negative rows sampled: {negative_sample_size} of {len(negative_indices)} available")
    print(f"  Final subset size: {len(subset)}; fully negative: {actual_negative_count} ({negative_rate:.2f}%)")
    print_positive_rates(subset, "  Resulting positive rates:")
    return subset


def write_paths(paths: pd.Series, output_path: Path) -> None:
    """Write Redivis PNG paths one per LF-terminated line, without a header."""
    # The parquet manifest retains its original CheXpert paths for joins. Only
    # the download list is adapted to the PNG_train object names on Redivis.
    values = []
    for path in paths.astype(str):
        png_path = path.removeprefix("train/")
        if png_path.endswith(".jpg"):
            png_path = f"{png_path[:-4]}.png"
        values.append(png_path)
    content = "\n".join(values)
    if content:
        content += "\n"
    # Explicit newline="\n" prevents Windows' default CRLF translation.
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def build_subset(input_path: Path, manifest_path: Path, paths_path: Path) -> pd.DataFrame:
    """Read the processed dataset, sample all splits, and save both manifests."""
    print(f"Loading processed dataset: {input_path}")
    frame = pd.read_parquet(input_path)
    validate_dataset(frame)
    print(f"Loaded {len(frame)} rows.")

    rng = random.Random(SEED)
    subsets: list[pd.DataFrame] = []
    for split_name, settings in SPLIT_SETTINGS.items():
        split_frame = frame.loc[frame["split"] == split_name]
        if split_frame.empty:
            raise ValueError(f"The processed dataset has no rows in the '{split_name}' split.")
        subsets.append(sample_split(split_frame, split_name, rng=rng, **settings))

    final_subset = pd.concat(subsets, axis=0, ignore_index=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths_path.parent.mkdir(parents=True, exist_ok=True)
    final_subset.to_parquet(manifest_path, index=False)
    write_paths(final_subset["path_to_image"], paths_path)

    print("\nFINAL SUMMARY")
    print(f"  Total images selected: {len(final_subset)}")
    for split_name in SPLIT_SETTINGS:
        print(f"  {split_name}: {int((final_subset['split'] == split_name).sum())}")
    print_positive_rates(final_subset, "  Combined positive rates:")
    print(f"Saved parquet manifest: {manifest_path}")
    print(f"Saved image paths: {paths_path}")
    return final_subset


def main() -> None:
    """Parse paths and build the reproducible image subset."""
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Sample a multi-label image download subset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=repository_root / "data/processed/chexpert_processed.parquet",
        help="Processed parquet source dataset.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=repository_root / "data/processed/image_subset_manifest.parquet",
        help="Parquet manifest output path.",
    )
    parser.add_argument(
        "--paths-output",
        type=Path,
        default=repository_root / "data/processed/image_subset_paths.txt",
        help="Plain-text image path manifest output path.",
    )
    args = parser.parse_args()
    build_subset(args.input, args.manifest_output, args.paths_output)


if __name__ == "__main__":
    main()
