"""Repair image_subset_manifest.parquet local_image_path values after relocation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def extract_filename(raw_path: object) -> str:
    """Return the final path component from a Windows- or POSIX-style path string."""
    path_text = str(raw_path).strip()
    if not path_text:
        raise ValueError("Encountered an empty local_image_path value.")
    filename = path_text.replace("\\", "/").rsplit("/", 1)[-1]
    if not filename:
        raise ValueError(f"Could not extract a filename from path: {path_text!r}")
    return filename


def rebuild_local_image_paths(manifest_path: Path) -> pd.DataFrame:
    """Rebuild and validate local_image_path values for the full manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    repository_root = Path(__file__).resolve().parents[1]
    images_dir = repository_root / "CheXpertPlus_selected"

    frame = pd.read_parquet(manifest_path)
    if "local_image_path" not in frame.columns:
        raise ValueError("Manifest is missing the required local_image_path column.")

    total_rows = len(frame)
    if total_rows == 0:
        raise ValueError("Manifest is empty; nothing to repair.")

    filenames = frame["local_image_path"].map(extract_filename)
    rebuilt_paths = filenames.map(lambda filename: images_dir / filename)

    sample_size = min(20, total_rows)
    sample_frame = frame.sample(n=sample_size, random_state=42)
    sample_indices = sample_frame.index
    sample_paths = rebuilt_paths.loc[sample_indices]
    sample_exists = sample_paths.map(lambda path: os.path.isfile(path))
    missing_sample_paths = sample_paths.loc[~sample_exists]

    print(f"Loaded manifest: {manifest_path}")
    print(f"Rows in manifest: {total_rows}")
    print(f"Sampled rows for existence check: {sample_size}")
    print(f"Sample exist count: {int(sample_exists.sum())}/{sample_size}")
    if not missing_sample_paths.empty:
        print("Missing sample paths:")
        for missing_path in missing_sample_paths:
            print(f"  {missing_path}")
        raise SystemExit(1)

    all_exists = rebuilt_paths.map(lambda path: os.path.isfile(path))
    missing_all_paths = rebuilt_paths.loc[~all_exists]
    if not bool(all_exists.all()):
        print(f"Full-manifest existence check failed: {int(all_exists.sum())}/{total_rows} paths exist.")
        print("Missing paths:")
        for missing_path in missing_all_paths:
            print(f"  {missing_path}")
        raise SystemExit(1)

    updated = frame.copy()
    updated["local_image_path"] = rebuilt_paths.astype(str)
    updated.to_parquet(manifest_path, index=False)

    print("\nFINAL SUMMARY")
    print(f"Total rows updated: {total_rows}")
    print(f"Verified existing files: {total_rows}/{total_rows}")
    print(f"Saved repaired manifest: {manifest_path}")
    return updated


def main() -> None:
    """Repair image_subset_manifest.parquet after a project relocation."""
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Repair local_image_path values in the image subset manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "data" / "processed" / "image_subset_manifest.parquet",
        help="Path to the parquet manifest to repair in place.",
    )
    args = parser.parse_args()
    rebuild_local_image_paths(args.manifest)


if __name__ == "__main__":
    main()