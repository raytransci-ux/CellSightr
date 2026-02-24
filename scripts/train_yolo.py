"""
Train YOLOv11 Cell Detectors (Nano + Medium + Large)
=====================================================
Run AFTER prepare_yolo_dataset.py:
    python scripts/train_yolo.py

Trains three models from the same dataset:
  - YOLOv11n (nano)   → fast pre-annotations in Label Studio + live camera feed
  - YOLOv11m (medium) → fallback precise model
  - YOLOv11l (large)  → production precise model (best non_viable recall)

With ~300 images on an RTX 5070 Ti:
  - Nano:   ~3-5 min
  - Medium: ~10-15 min
  - Large:  ~20-30 min
Retrain every ~50 newly confirmed images for progressively better pre-annotations.
"""

from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_YAML    = PROJECT_ROOT / "data" / "yolo_dataset" / "data.yaml"
CHECKPOINT   = PROJECT_ROOT / "checkpoints" / "yolo"
# ---------------------------------------------------------------------------

# Shared training config — tuned for hemocytometer cell detection
TRAIN_ARGS = dict(
    epochs    = 200,
    imgsz     = 640,
    patience  = 40,           # Stop early if no improvement after 40 epochs
    device    = 0,            # GPU (use 'cpu' to force CPU)
    exist_ok  = True,

    # Augmentation — critical for cell images
    hsv_h     = 0.015,        # Hue shift (minimal — preserve trypan blue color)
    hsv_s     = 0.5,          # Saturation
    hsv_v     = 0.4,          # Brightness — simulates lighting variation
    fliplr    = 0.5,          # Horizontal flip
    flipud    = 0.5,          # Vertical flip (cells are rotation invariant)
    degrees   = 180,          # Full rotation (cells look the same at any angle)
    scale     = 0.3,          # Zoom in/out
    mosaic    = 0.5,          # Combine 4 images (good for dense scenes)
    mixup     = 0.1,          # Blend images (helps generalisation)
    copy_paste= 0.1,          # Copy cells between images

    # Optimizer
    lr0       = 0.01,
    lrf       = 0.01,
    warmup_epochs = 5,
)

MODELS = [
    {
        "base": "yolo11n.pt",
        "name": "nano",
        "batch": 16,           # Nano is light — can use bigger batch
        "desc": "Label Studio pre-annotations + live camera feed",
    },
    {
        "base": "yolo11m.pt",
        "name": "medium",
        "batch": 8,            # Medium needs more memory
        "desc": "Fallback precise model",
    },
    {
        "base": "yolo11l.pt",
        "name": "large",
        "batch": 4,            # Large needs the most memory
        "desc": "Production precise model (+5% non_viable recall vs medium)",
    },
]


def main():
    if not DATA_YAML.exists():
        print("ERROR: data.yaml not found.")
        print("Run first: python scripts/prepare_yolo_dataset.py")
        return

    CHECKPOINT.mkdir(parents=True, exist_ok=True)

    print("=== Training YOLOv11 Cell Detectors ===")
    print(f"Dataset : {DATA_YAML}")
    print(f"Output  : {CHECKPOINT}\n")

    for cfg in MODELS:
        print(f"{'─' * 60}")
        print(f"Training {cfg['name'].upper()} ({cfg['base']})")
        print(f"  Purpose: {cfg['desc']}")
        print(f"{'─' * 60}\n")

        model = YOLO(cfg["base"])

        model.train(
            data    = str(DATA_YAML),
            batch   = cfg["batch"],
            project = str(CHECKPOINT),
            name    = cfg["name"],
            **TRAIN_ARGS,
        )

        best = CHECKPOINT / cfg["name"] / "weights" / "best.pt"
        print(f"\n{cfg['name'].upper()} complete: {best.relative_to(PROJECT_ROOT)}\n")

    # Summary
    print("=" * 60)
    print("Training complete. Model locations:")
    for cfg in MODELS:
        best = CHECKPOINT / cfg["name"] / "weights" / "best.pt"
        exists = "OK" if best.exists() else "MISSING"
        print(f"  {cfg['name']:8s} → {best.relative_to(PROJECT_ROOT)}  [{exists}]")
    print(f"\nNext step: python scripts/start_ml_backend.py")


if __name__ == "__main__":
    main()
