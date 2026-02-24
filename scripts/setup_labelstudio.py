"""
Label Studio Setup Script — Hemocytometer Cell Counter
========================================================
Run this script AFTER installing Label Studio to:
  1. Verify your image folder structure
  2. Create a pre-filled metadata CSV for your images
  3. Print the step-by-step Label Studio setup instructions

Usage:
    python scripts/setup_labelstudio.py

Requirements:
    pip install label-studio
"""

import os
import csv
import sys
import glob
from pathlib import Path
from datetime import date

# ---------------------------------------------------------------------------
# Configuration — edit these paths if needed
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).parent.parent
RAW_CELLS_DIR = PROJECT_ROOT / "data" / "raw" / "with_cells"
RAW_EMPTY_DIR = PROJECT_ROOT / "data" / "raw" / "empty_grid"
METADATA_OUT  = PROJECT_ROOT / "data" / "metadata.csv"
CONFIG_XML    = PROJECT_ROOT / "label_studio" / "labeling_config.xml"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ---------------------------------------------------------------------------

def find_images(folder: Path) -> list[Path]:
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def check_folder_structure():
    print("\n=== Checking project structure ===")
    ok = True
    for folder in [RAW_CELLS_DIR, RAW_EMPTY_DIR]:
        exists = folder.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {folder.relative_to(PROJECT_ROOT)}")
        if not exists:
            ok = False
    return ok


def build_metadata_csv():
    print("\n=== Building metadata CSV ===")

    cell_images  = find_images(RAW_CELLS_DIR)
    empty_images = find_images(RAW_EMPTY_DIR)
    all_images   = [(img, "with_cells") for img in cell_images] + \
                   [(img, "empty_grid") for img in empty_images]

    if not all_images:
        print("  WARNING: No images found in data/raw/. Add your images and re-run.")
        return 0

    today = date.today().isoformat()

    # Load existing metadata to avoid overwriting manual entries
    existing = {}
    if METADATA_OUT.exists():
        with open(METADATA_OUT, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["filename"]] = row

    rows = []
    new_count = 0
    for img_path, subfolder in all_images:
        fname = img_path.name
        if fname in existing:
            rows.append(existing[fname])  # Keep existing manual metadata
        else:
            # Auto-detect density for empty grid images
            density = "empty" if subfolder == "empty_grid" else ""
            rows.append({
                "filename":         fname,
                "subfolder":        subfolder,
                "density":          density,
                "has_trypan":       "yes",
                "grid_visible":     "",
                "focus_quality":    "",
                "dilution_factor":  "",
                "acquisition_date": today,
                "notes":            "",
            })
            new_count += 1

    fieldnames = [
        "filename", "subfolder", "density", "has_trypan",
        "grid_visible", "focus_quality", "dilution_factor",
        "acquisition_date", "notes"
    ]

    with open(METADATA_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    kept  = total - new_count
    print(f"  {new_count} new image(s) added, {kept} existing row(s) preserved")
    print(f"  Total images tracked: {total}")
    print(f"  Saved to: {METADATA_OUT.relative_to(PROJECT_ROOT)}")
    return total


def print_dataset_summary():
    cell_images  = find_images(RAW_CELLS_DIR)
    empty_images = find_images(RAW_EMPTY_DIR)

    print("\n=== Dataset Summary ===")
    print(f"  Images with cells : {len(cell_images)}")
    print(f"  Empty grid images : {len(empty_images)}")
    print(f"  Total             : {len(cell_images) + len(empty_images)}")

    target = 100
    current = len(cell_images) + len(empty_images)
    remaining = max(0, target - current)
    pct = min(100, int(current / target * 100))
    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
    print(f"\n  Annotation target : {target} images")
    print(f"  Progress          : [{bar}] {pct}%  ({remaining} more needed)")


def print_labelstudio_instructions():
    print("\n" + "=" * 60)
    print("  LABEL STUDIO SETUP — STEP BY STEP")
    print("=" * 60)

    print("""
STEP 1: Install Label Studio (if not already done)
  pip install label-studio

STEP 2: Start Label Studio
  label-studio start
  (Opens in browser at http://localhost:8080)

STEP 3: Create an account
  Sign up with any email/password (local only, no internet needed)

STEP 4: Create a new project
  → Click "Create Project"
  → Name it: "Hemocytometer Cell Counter"
  → Skip to "Labeling Interface" tab

STEP 5: Paste the labeling interface config
  → Click the "Code" tab in the labeling interface editor
  → Delete all existing content
  → Open this file and paste its contents:
""")
    print(f"    {CONFIG_XML}")
    print("""
  → Click "Save"

STEP 6: Import your images
  → Go to your project → click "Import"
  → Option A (easiest): drag and drop images directly from:
""")
    print(f"    {RAW_CELLS_DIR}")
    print("""  → Option B: use the folder path if Label Studio prompts for it

STEP 7: Configure keyboard shortcuts (optional but helpful)
  The config already sets:
    V = viable (green)
    D = dead / non_viable (blue)
    A = ambiguous (yellow)

  In Label Studio UI shortcuts:
    W = create rectangle
    Escape = cancel current box
    Delete = remove selected box

STEP 8: Start annotating!
  → See ANNOTATION_GUIDELINES.md for detailed rules
  → Target: < 5 minutes per image
  → Aim for 20 pilot images first, then continue to 100+

STEP 9: Export annotations (after annotating all images)
  → Project → Export → choose "COCO JSON" format
  → Save to: data/annotated/exports/annotations_coco.json

""")
    print("=" * 60)


def main():
    print("Hemocytometer Cell Counter — Label Studio Setup")
    print(f"Project root: {PROJECT_ROOT}\n")

    structure_ok = check_folder_structure()
    if not structure_ok:
        print("\n  Please create missing folders and add your images, then re-run.")

    total = build_metadata_csv()
    print_dataset_summary()
    print_labelstudio_instructions()

    if total == 0:
        print("\nACTION REQUIRED: Copy your hemocytometer images into:")
        print(f"  {RAW_CELLS_DIR}   ← images with cells")
        print(f"  {RAW_EMPTY_DIR}  ← empty hemocytometer images")
        print("\nThen re-run this script.")


if __name__ == "__main__":
    main()
