"""
Prepare YOLO Dataset from Label Studio Export
===============================================
1. In Label Studio: go to your project → Export → "YOLO with images" → Download
2. Save the zip file to: data/annotated/exports/
3. Run this script: python scripts/prepare_yolo_dataset.py

This script reads the zip directly (no extraction) to avoid Windows filename
issues caused by Label Studio's URL-encoded label filenames (?d=...).
"""

import re
import zipfile
import shutil
import random
from pathlib import Path
from urllib.parse import unquote

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
EXPORT_DIR   = PROJECT_ROOT / "data" / "annotated" / "exports"
YOLO_DATASET = PROJECT_ROOT / "data" / "yolo_dataset"
RAW_IMAGES   = PROJECT_ROOT / "data" / "raw" / "with_cells"

TARGET_CLASSES = ["viable", "non_viable"]  # desired output order → 0, 1
TRAIN_RATIO    = 0.85
RANDOM_SEED    = 42

IMG_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# ---------------------------------------------------------------------------


def find_export_zip() -> Path:
    zips = list(EXPORT_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"No zip found in {EXPORT_DIR}\n"
            "Export from Label Studio: Project → Export → YOLO with images"
        )
    return max(zips, key=lambda z: z.stat().st_mtime)


def decode_ls_label_name(zip_member: str) -> str | None:
    """
    Decode a Label Studio YOLO export label filename to the bare image stem.

    Label Studio uses two known formats:
      New: "{hash}__{subfolder}%5C{imagename}.txt"
           e.g. "0472a687__with_cells%5C20250422_4x_CTR_10_TOP_B2.txt"
      Old: "labels/?d=with_cells%5C{imagename}.txt"

    Returns the image stem (e.g. "20250422_4x_CTR_10_TOP_B2"), or None.
    """
    name = Path(zip_member).name

    # Skip the class-list file
    if name.lower() == "classes.txt":
        return None

    # New format: {8+char hex hash}__{url_encoded_path}.txt
    match = re.match(r'^[a-f0-9]{6,}__(.+)\.txt$', name, re.IGNORECASE)
    if match:
        decoded = unquote(match.group(1))          # "with_cells\imagename"
        stem = re.split(r'[/\\]', decoded)[-1]     # "imagename"
        return stem or None

    # Old format: ?d=subfolder%5Cimagename.txt (inside a labels/ dir in zip)
    match = re.search(r'\?d=(?:[^/\\%]+[/\\%5C]+)?(.+?)\.txt$',
                      zip_member, re.IGNORECASE)
    if match:
        return Path(unquote(match.group(1))).stem

    return None


def build_class_remap(classes_txt: str) -> dict[int, int]:
    """
    Label Studio exports classes.txt with ALL label/choice names from the entire
    interface (cell labels + dropdown options) sorted alphabetically.
    This builds a remapping: exported_id → our_yolo_id (0=viable, 1=non_viable).
    Anything not in TARGET_CLASSES is dropped.
    """
    all_classes = [l.strip() for l in classes_txt.strip().splitlines() if l.strip()]
    remap = {}
    for export_id, name in enumerate(all_classes):
        if name in TARGET_CLASSES:
            remap[export_id] = TARGET_CLASSES.index(name)
    print(f"  Class mapping from export: { {v: TARGET_CLASSES[i] for v, i in remap.items()} }")
    return remap


def filter_label_content(content: str, remap: dict[int, int]) -> str:
    """Remap class IDs and drop any class not in TARGET_CLASSES."""
    kept = []
    for line in content.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        export_id = int(parts[0])
        if export_id in remap:
            parts[0] = str(remap[export_id])
            kept.append(" ".join(parts))
    return "\n".join(kept) + ("\n" if kept else "")


def build_dataset(zip_path: Path) -> tuple[int, int]:
    print(f"Reading zip: {zip_path.name}")

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()

        # --- Read class remapping from classes.txt ---
        if "classes.txt" not in members:
            raise ValueError("classes.txt not found in zip.")
        classes_txt = zf.read("classes.txt").decode("utf-8")
        remap = build_class_remap(classes_txt)

        # --- Find label files and decode image stems ---
        label_map: dict[str, str] = {}   # image_stem → label content
        for m in members:
            if not m.endswith(".txt"):
                continue
            stem = decode_ls_label_name(m)
            if stem:
                label_map[stem] = zf.read(m).decode("utf-8")

        # --- Find image files bundled in the zip ---
        zip_images: dict[str, str] = {}
        for m in members:
            p = Path(m)
            if p.suffix.lower() in IMG_EXTS and "?" not in p.name:
                zip_images[p.stem] = m   # stem → zip member path

        print(f"Found {len(label_map)} label file(s) in zip.")

    if not label_map:
        raise ValueError(
            "No label files could be parsed from the zip.\n"
            "Make sure you exported in 'YOLO with images' format."
        )

    # --- Build list of (stem, image_source, label_content) ---
    entries = []
    missing_images = []

    for stem, lbl_content in label_map.items():
        # Try zip-bundled image first
        if stem in zip_images:
            entries.append((stem, ("zip", zip_images[stem]), lbl_content))
            continue

        # Fall back to raw images folder
        for ext in IMG_EXTS:
            candidate = RAW_IMAGES / (stem + ext)
            if candidate.exists():
                entries.append((stem, ("disk", candidate), lbl_content))
                break
        else:
            missing_images.append(stem)

    if missing_images:
        print(f"WARNING: Could not find images for {len(missing_images)} label(s):")
        for s in missing_images[:5]:
            print(f"  {s}")
        if len(missing_images) > 5:
            print(f"  ... and {len(missing_images) - 5} more")

    if not entries:
        raise ValueError("No image+label pairs found. Check your export.")

    print(f"Matched {len(entries)} image+label pair(s).")

    # --- Train / val split ---
    random.seed(RANDOM_SEED)
    random.shuffle(entries)
    split      = max(1, int(len(entries) * TRAIN_RATIO))
    train_data = entries[:split]
    val_data   = entries[split:]
    print(f"Split: {len(train_data)} train / {len(val_data)} val")

    # --- Create output directories ---
    for part in ("train", "val"):
        (YOLO_DATASET / "images" / part).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET / "labels" / part).mkdir(parents=True, exist_ok=True)

    # --- Write files ---
    with zipfile.ZipFile(zip_path) as zf:
        for part, data in [("train", train_data), ("val", val_data)]:
            for stem, img_source, lbl_content in data:
                kind, src = img_source

                # Copy image
                if kind == "zip":
                    img_bytes = zf.read(src)
                    ext = Path(src).suffix
                    dst_img = YOLO_DATASET / "images" / part / (stem + ext)
                    dst_img.write_bytes(img_bytes)
                else:
                    dst_img = YOLO_DATASET / "images" / part / src.name
                    shutil.copy2(src, dst_img)

                # Write remapped + filtered label
                dst_lbl = YOLO_DATASET / "labels" / part / (stem + ".txt")
                dst_lbl.write_text(filter_label_content(lbl_content, remap))

    return len(train_data), len(val_data)


def write_data_yaml() -> Path:
    yaml_path = YOLO_DATASET / "data.yaml"
    yaml_path.write_text(
        f"path: {YOLO_DATASET.as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: {len(TARGET_CLASSES)}\n"
        f"names: {TARGET_CLASSES}\n"
    )
    return yaml_path


def main():
    print("=== Preparing YOLO Dataset ===\n")

    try:
        zip_path = find_export_zip()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    try:
        n_train, n_val = build_dataset(zip_path)
    except (ValueError, KeyError) as e:
        print(f"ERROR: {e}")
        return

    yaml_path = write_data_yaml()

    print(f"\nDataset ready:")
    print(f"  Train : {n_train} images → data/yolo_dataset/images/train/")
    print(f"  Val   : {n_val} images → data/yolo_dataset/images/val/")
    print(f"  Config: {yaml_path.relative_to(PROJECT_ROOT)}")
    print(f"\nNext step: python scripts/train_yolo.py")


if __name__ == "__main__":
    main()
