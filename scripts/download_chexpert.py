import os
from pathlib import Path
import redivis

# ============================================================
# CONFIGURATION
# ============================================================

# Fully qualified references resolved from Redivis
DATASET_REF = "aimi.chexpert_plus:5yyj:v1_0"
TABLE_REF = "png_train:s6cj"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PATH_LIST = REPOSITORY_ROOT / "data/processed/image_subset_paths.txt"
OUTPUT_DIR = REPOSITORY_ROOT / "CheXpertPlus_selected"

# Set to None to download all 12,444 images
MAX_IMAGES = None  


# ============================================================
# LOAD PATHS
# ============================================================

if not PATH_LIST.exists():
    raise FileNotFoundError(f"Missing path list: {PATH_LIST.resolve()}")

requested_paths = []
with PATH_LIST.open("r", encoding="utf-8") as f:
    for line in f:
        p = line.strip().replace("\\", "/").lstrip("/")
        if p and p.lower() not in {"png", "path", "file_name", "file_path"}:
            requested_paths.append(p)

requested_paths = list(dict.fromkeys(requested_paths))

if MAX_IMAGES is not None:
    target_paths = requested_paths[:MAX_IMAGES]
else:
    target_paths = requested_paths

print("=" * 70)
print(f"Targeting {len(target_paths):,} images for selective download")
print("=" * 70)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Connect directly to the dataset table using the qualified reference
table = redivis.dataset(DATASET_REF).table(TABLE_REF)


# ============================================================
# DOWNLOAD DIRECTLY WITH UNIQUE FILE NAMES
# ============================================================

downloaded_count = 0
failed_count = 0

for idx, rel_path in enumerate(target_paths, 1):
    # Unique flattened filename: patient00026_study1_view1_frontal.png
    unique_filename = rel_path.replace("PNG_train/", "").replace("/", "_")
    dest_path = OUTPUT_DIR / unique_filename

    if dest_path.exists():
        downloaded_count += 1
        if idx % 500 == 0 or idx == len(target_paths):
            print(f"[{idx}/{len(target_paths)}] Skipped (already exists): {unique_filename}")
        continue

    # Path candidate formats matching Redivis PNG_train storage
    candidates = [
        rel_path,
        f"PNG_train/{rel_path}" if not rel_path.startswith("PNG_train/") else rel_path,
        Path(rel_path).name
    ]

    success = False
    for path_str in candidates:
        try:
            f_obj = table.file(path_str)
            f_obj.download(path=str(dest_path))
            downloaded_count += 1
            
            # Print status every 100 images to prevent terminal clutter during bulk download
            if idx % 100 == 0 or idx == len(target_paths) or idx <= 10:
                print(f"[{idx}/{len(target_paths)}] Downloaded: {unique_filename}")
                
            success = True
            break
        except Exception:
            continue

    if not success:
        failed_count += 1
        print(f"[{idx}/{len(target_paths)}] Failed: {rel_path}")

print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)
print(f"Successfully processsed : {downloaded_count:,}")
print(f"Failed                  : {failed_count:,}")
print(f"Output directory        : {OUTPUT_DIR.resolve()}")
