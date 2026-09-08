"""Download the selected CheXpert Plus PNG files from Redivis in safe batches.

This script reads the local, generated image-path manifest and queries the
Redivis PNG_train file-index table in batches. Authentication is supplied only
through the REDIVIS_API_TOKEN environment variable; no credentials are stored
in this repository.
"""

from __future__ import annotations

import argparse
import os
import time
from itertools import islice
from pathlib import Path
from typing import Iterator, Sequence


ORGANIZATION_NAME = "aimi"
DATASET_NAME = "CheXpert Plus"
# Confirmed by a live Redivis client warning for this specific dataset.
TABLE_NAME = "png_train:s6cj"
BATCH_SIZE = 150


def read_filenames(paths_file: Path) -> list[str]:
    """Read non-empty filename entries, stripping whitespace and newlines."""
    if not paths_file.is_file():
        raise FileNotFoundError(f"Image-path manifest was not found: {paths_file}")
    filenames = [line.strip() for line in paths_file.read_text(encoding="utf-8").splitlines()]
    filenames = [filename for filename in filenames if filename]
    if not filenames:
        raise ValueError(f"Image-path manifest contains no filenames: {paths_file}")
    return filenames


def chunked(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    """Yield fixed-size filename chunks without building copies of the full list."""
    iterator = iter(values)
    while chunk := list(islice(iterator, size)):
        yield chunk


def sql_literal(value: str) -> str:
    """Return a SQL string literal, defensively escaping embedded quotes."""
    return "'" + value.replace("'", "''") + "'"


def build_query(filenames: Sequence[str]) -> str:
    """Build a scoped Redivis SQL query for one filename batch."""
    literals = ", ".join(sql_literal(filename) for filename in filenames)
    return f"SELECT * FROM {TABLE_NAME} WHERE file_name IN ({literals})"


def display_name(dataset: object) -> str:
    """Extract a human-readable listed-dataset name for matching and logging."""
    properties = getattr(dataset, "properties", None) or {}
    return str(properties.get("name") or getattr(dataset, "scoped_reference", "<unknown dataset>"))


def get_chexpert_dataset(redivis_module: object) -> object:
    """List accessible AImi datasets and resolve the CheXpert Plus dataset.

    Listing first protects against assuming an out-of-date display name or a
    version suffix. A single case-insensitive CheXpert Plus match is required.
    """
    try:
        organization = redivis_module.organization(ORGANIZATION_NAME)
        datasets = organization.list_datasets()
    except Exception as error:
        raise RuntimeError(
            "Could not list datasets for the Redivis organization 'aimi'. "
            "Check REDIVIS_API_TOKEN permissions, network access, and authentication."
        ) from error

    matches = [dataset for dataset in datasets if DATASET_NAME.casefold() in display_name(dataset).casefold()]
    if len(matches) != 1:
        available = ", ".join(display_name(dataset) for dataset in datasets)
        raise RuntimeError(
            f"Expected one dataset matching '{DATASET_NAME}', found {len(matches)}. "
            f"Accessible datasets: {available or '<none>'}"
        )

    dataset = matches[0]
    print(f"Verified Redivis dataset: {display_name(dataset)} (organization: {ORGANIZATION_NAME})")
    return dataset


def download_batches(
    filenames: Sequence[str], output_dir: Path, redivis_module: object, max_chunks: int | None = None
) -> int:
    """Query and download each filename batch, returning the actual file count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = get_chexpert_dataset(redivis_module)
    batches = list(chunked(filenames, BATCH_SIZE))
    total_chunks = len(batches)
    if max_chunks is not None:
        batches = batches[:max_chunks]
        print(f"TEST MODE: processing only {len(batches)} of {total_chunks} chunks.")
    downloaded_total = 0
    failed_chunks: list[int] = []

    for chunk_number, filename_batch in enumerate(batches, start=1):
        print(f"Chunk {chunk_number}/{len(batches)}: querying {len(filename_batch)} files...")
        for attempt in range(2):
            try:
                query = dataset.query(build_query(filename_batch))
                directory = query.to_directory()
                downloaded_files = directory.list(mode="files", recursive=True)
                directory.download(path=str(output_dir), overwrite=False, progress=True)
                break
            except Exception as error:
                if attempt == 0:
                    print(f"  ERROR: chunk {chunk_number} failed: {error}")
                    print(f"  Retrying chunk {chunk_number}/{len(batches)} after failure...")
                    time.sleep(10)
                else:
                    failed_chunks.append(chunk_number)
                    print(f"  ERROR: chunk {chunk_number} failed again; giving up: {error}")
        else:
            continue

        chunk_downloaded = len(downloaded_files)
        downloaded_total += chunk_downloaded
        print(f"  Downloaded {chunk_downloaded} files; running total: {downloaded_total}")

    expected_total = len(filenames)
    print("\nDOWNLOAD SUMMARY")
    print(f"  Expected filenames: {expected_total}")
    print(f"  Files reported downloaded: {downloaded_total}")
    if failed_chunks:
        print(f"  Failed chunks: {failed_chunks}")
    if downloaded_total != expected_total:
        print(
            f"  DISCREPANCY: {expected_total - downloaded_total} requested files were not downloaded. "
            "Inspect the failed chunks and verify file_name matches in PNG_train."
        )
    else:
        print("  All requested files were downloaded.")
    return downloaded_total


def main() -> None:
    """Validate authentication, then download the locally selected image subset."""
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download the selected CheXpert Plus PNG images from Redivis.")
    parser.add_argument(
        "--paths-file",
        type=Path,
        default=repository_root / "data/processed/image_subset_paths.txt",
        help="LF-delimited filename manifest generated by sample_image_subset.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "data/raw/images",
        help="Directory where Redivis downloads the matching PNG files.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Process only this many chunks for a small test run (default: all chunks).",
    )
    args = parser.parse_args()

    if not os.environ.get("REDIVIS_API_TOKEN"):
        raise EnvironmentError(
            "REDIVIS_API_TOKEN is not set. Set it in your environment before running this downloader; "
            "do not place the token in source code or configuration files."
        )

    try:
        import redivis
    except ImportError as error:
        raise ImportError("The redivis package is required. Install dependencies with: pip install -r requirements.txt") from error

    filenames = read_filenames(args.paths_file)
    print(f"Loaded {len(filenames)} requested filenames from {args.paths_file}")
    download_batches(filenames, args.output_dir, redivis, max_chunks=args.max_chunks)


if __name__ == "__main__":
    main()
