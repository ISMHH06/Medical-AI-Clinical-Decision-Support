"""Reconcile the selected-image manifest with files verified on local disk."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def expected_flattened_filename(image_path: str) -> str:
    """Map a CheXpert-style image path to its flattened downloaded PNG name."""
    relative_path = image_path.removeprefix("train/")
    flattened_path = relative_path.replace("/", "_")
    return str(Path(flattened_path).with_suffix(".png"))


def write_dropped_paths(paths: pd.Series, output_path: Path) -> None:
    """Write original missing image paths one per LF-terminated line."""
    content = "\n".join(paths.astype(str).tolist())
    if content:
        content += "\n"
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def reconcile_manifest(manifest_path: Path, images_dir: Path, dropped_paths_path: Path) -> pd.DataFrame:
    """Keep manifest rows whose flattened PNG file is present on local disk."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest was not found: {manifest_path}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Downloaded image directory was not found: {images_dir}")

    manifest = pd.read_parquet(manifest_path)
    if "path_to_image" not in manifest.columns:
        raise ValueError("Manifest is missing the required path_to_image column.")

    original_count = len(manifest)
    expected_filenames = manifest["path_to_image"].astype(str).map(expected_flattened_filename)
    local_paths = expected_filenames.map(lambda filename: (images_dir / filename).resolve())
    exists_mask = local_paths.map(Path.is_file)

    dropped = manifest.loc[~exists_mask, "path_to_image"]
    kept = manifest.loc[exists_mask].copy()
    kept["local_image_path"] = local_paths.loc[exists_mask].astype(str).to_numpy()

    print(f"Original manifest rows: {original_count}")
    print(f"Verified-present rows: {len(kept)}")
    print(f"Dropped rows: {len(dropped)}")
    if not dropped.empty:
        print("Dropped path_to_image values:")
        for image_path in dropped:
            print(f"  {image_path}")

    # The user-requested overwrite contains every original column plus local_image_path.
    kept.to_parquet(manifest_path, index=False)
    write_dropped_paths(dropped, dropped_paths_path)

    print("\nFINAL SUMMARY")
    print(f"  Original row count: {original_count}")
    print(f"  Verified-present count: {len(kept)}")
    print(f"  Dropped count: {len(dropped)}")
    print(f"  Reconciled manifest saved: {manifest_path}")
    print(f"  Dropped-path record saved: {dropped_paths_path}")
    return kept


def main() -> None:
    """Parse local paths and reconcile the image subset manifest."""
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Reconcile the image subset manifest with downloaded PNG files.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "data/processed/image_subset_manifest.parquet",
        help="Subset parquet manifest to reconcile in place.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=repository_root / "CheXpertPlus_selected",
        help="Directory containing flattened downloaded PNG files.",
    )
    parser.add_argument(
        "--dropped-paths-output",
        type=Path,
        default=repository_root / "data/processed/image_subset_dropped.txt",
        help="Text file recording original paths that were not found locally.",
    )
    args = parser.parse_args()
    reconcile_manifest(args.manifest, args.images_dir, args.dropped_paths_output)


if __name__ == "__main__":
    main()
