"""
Label Studio ML Backend — YOLOv11 Cell Detector
================================================
This runs as a small server that Label Studio calls every time you open
an image. It returns pre-drawn bounding boxes for you to accept/correct.

Uses the nano model for fast pre-annotations during labeling.
"""

import re
import os
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT    = PROJECT_ROOT / "data" / "raw"

# Prefer YOLOv11 nano; fall back to legacy v8 checkpoint
MODEL_PATH_V11  = PROJECT_ROOT / "checkpoints" / "yolo" / "nano" / "weights" / "best.pt"
MODEL_PATH_V8   = PROJECT_ROOT / "checkpoints" / "yolo" / "run" / "weights" / "best.pt"

# Must match the order in data.yaml / Label Studio config
LABELS       = ["viable", "non_viable"]
CONF_THRESH  = 0.25   # Lower = more pre-annotations (more false positives)
                      # Higher = fewer but more confident boxes
                      # Start at 0.25, raise to 0.4 after more training
# ---------------------------------------------------------------------------


class CellDetectorBackend(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

        # Try v11 nano first, then fall back to v8
        model_path = MODEL_PATH_V11 if MODEL_PATH_V11.exists() else MODEL_PATH_V8

        if model_path.exists():
            print(f"Loading model from {model_path}")
            self.model = YOLO(str(model_path))
        else:
            print(f"WARNING: No trained model found.")
            print(f"  Checked: {MODEL_PATH_V11}")
            print(f"  Checked: {MODEL_PATH_V8}")
            print("Run python scripts/train_yolo.py first, then restart this backend.")

    def _resolve_path(self, url: str) -> Path | None:
        """
        Convert a Label Studio image URL to a local file path.
        URL format: /data/local-files/?d=with_cells/image.bmp
        """
        match = re.search(r'\?d=(.+?)(?:&|$)', url)
        if match:
            relative = unquote(match.group(1))
            return DATA_ROOT / relative

        # Fallback: try stripping the URL prefix entirely
        for prefix in ["/data/local-files/?d=", "/data/"]:
            if prefix in url:
                relative = unquote(url.split(prefix, 1)[1])
                return DATA_ROOT / relative

        return None

    def predict(self, tasks, **kwargs):
        if self.model is None:
            return []

        predictions = []

        for task in tasks:
            image_url = task["data"].get("image", "")
            image_path = self._resolve_path(image_url)

            if not image_path or not image_path.exists():
                print(f"Could not resolve image path from URL: {image_url}")
                predictions.append({"result": [], "score": 0.0})
                continue

            # Run inference
            img = Image.open(image_path).convert("RGB")
            img_w, img_h = img.size

            results = self.model(str(image_path), conf=CONF_THRESH, verbose=False)[0]

            result = []
            scores = []

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf    = float(box.conf[0])
                cls_id  = int(box.cls[0])
                label   = LABELS[cls_id] if cls_id < len(LABELS) else "viable"

                # Label Studio expects coordinates as % of image dimensions
                result.append({
                    "from_name": "cells",
                    "to_name":   "image",
                    "type":      "rectanglelabels",
                    "value": {
                        "x":              x1 / img_w * 100,
                        "y":              y1 / img_h * 100,
                        "width":          (x2 - x1) / img_w * 100,
                        "height":         (y2 - y1) / img_h * 100,
                        "rectanglelabels": [label],
                    },
                    "score": conf,
                })
                scores.append(conf)

            avg_score = float(np.mean(scores)) if scores else 0.0
            predictions.append({"result": result, "score": avg_score})

        return predictions

    def fit(self, annotations, **kwargs):
        """
        Called by Label Studio when you click 'Train' in the UI.
        For now, training is handled by scripts/train_yolo.py.
        """
        return {"status": "Training is done via scripts/train_yolo.py"}
