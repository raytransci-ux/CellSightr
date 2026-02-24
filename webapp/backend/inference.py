"""YOLO inference engine for hemocytometer cell detection.

Supports dual-model operation:
  - nano:   fast inference for live camera feed (~5ms/frame on GPU)
  - medium: accurate inference for final counts after capture (~25ms/frame on GPU)
"""

from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import time
from typing import Dict, List, Optional

CLASS_NAMES = {0: "viable", 1: "non_viable"}
CLASS_COLORS_BGR = {0: (142, 199, 72), 1: (68, 68, 239)}  # green, red in BGR


class InferenceEngine:
    """Wraps YOLO models for cell detection with hot-swap and dual-model support."""

    def __init__(self, model_path: str, medium_model_path: Optional[str] = None):
        self._model_path = str(model_path)
        self._model = YOLO(self._model_path)
        self._class_names = dict(CLASS_NAMES)

        # Medium model for accurate final counts
        self._medium_model = None
        self._medium_model_path = None
        if medium_model_path and Path(medium_model_path).exists():
            self._medium_model_path = str(medium_model_path)
            self._medium_model = YOLO(self._medium_model_path)

    @property
    def model_name(self) -> str:
        return Path(self._model_path).name

    @property
    def precise_model_name(self) -> Optional[str]:
        return Path(self._medium_model_path).name if self._medium_model_path else None

    @property
    def has_precise_model(self) -> bool:
        return self._medium_model is not None

    @property
    def class_names(self) -> dict:
        return self._class_names

    def warmup(self):
        """Run a dummy inference to trigger CUDA kernel compilation.

        This moves the ~5-8s first-inference penalty to server startup
        instead of making the user wait on their first image.
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        self._model(dummy, conf=0.5, verbose=False)
        nano_ms = (time.perf_counter() - t0) * 1000
        print(f"  Nano warmup: {nano_ms:.0f}ms")
        if self._medium_model:
            t0 = time.perf_counter()
            self._medium_model(dummy, conf=0.5, verbose=False)
            precise_ms = (time.perf_counter() - t0) * 1000
            print(f"  Precise warmup: {precise_ms:.0f}ms")

    def switch_model(self, new_path: str) -> dict:
        """Hot-swap the primary (nano) model. Returns class info."""
        model = YOLO(new_path)
        names = model.names if hasattr(model, "names") and model.names else CLASS_NAMES
        self._model = model
        self._model_path = str(new_path)
        self._class_names = dict(names) if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
        return {"model": self.model_name, "classes": self._class_names}

    def predict(self, image_path: str, conf: float = 0.25, use_precise: bool = False) -> Dict:
        """
        Run inference on a single image.

        Args:
            use_precise: If True and medium model is loaded, use the more accurate model.

        Returns dict with 'detections' list and 'summary' counts.
        """
        model = self._medium_model if (use_precise and self._medium_model) else self._model
        model_used = "precise" if (use_precise and self._medium_model) else "nano"

        t0 = time.perf_counter()
        results = model(image_path, conf=conf, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections = []
        viable = 0
        non_viable = 0

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            cls_name = self._class_names.get(cls_id, f"class_{cls_id}")
            det = {
                "id": i,
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
                "class": cls_id,
                "class_name": cls_name,
                "confidence": round(float(box.conf[0]), 3),
            }
            detections.append(det)
            if cls_name == "viable":
                viable += 1
            else:
                non_viable += 1

        total = viable + non_viable
        return {
            "detections": detections,
            "summary": {
                "total": total,
                "viable": viable,
                "non_viable": non_viable,
                "viability_pct": round((viable / total * 100) if total > 0 else 0, 1),
            },
            "inference_ms": round(elapsed_ms, 1),
            "model_used": model_used,
            "image_size": {
                "width": results.orig_shape[1],
                "height": results.orig_shape[0],
            },
        }

    def render_overlay(
        self,
        image: np.ndarray,
        detections: List[Dict],
        manual_additions: Optional[List[Dict]] = None,
        manual_removals: Optional[set] = None,
    ) -> np.ndarray:
        """Draw detection boxes and manual annotations onto an image for export."""
        overlay = image.copy()
        removals = manual_removals or set()

        for det in detections:
            if det["id"] in removals:
                # Draw with strikethrough at reduced opacity
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                color = CLASS_COLORS_BGR.get(det["class"], (200, 200, 200))
                sub = overlay[y1:y2, x1:x2]
                tint = np.full_like(sub, color)
                cv2.addWeighted(tint, 0.15, sub, 0.85, 0, sub)
                cv2.line(overlay, (x1, y1), (x2, y2), (128, 128, 128), 1)
                continue

            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color = CLASS_COLORS_BGR.get(det["class"], (200, 200, 200))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        if manual_additions:
            for ann in manual_additions:
                cx, cy = int(ann["x"]), int(ann["y"])
                color = CLASS_COLORS_BGR.get(ann.get("class", 0), (200, 200, 200))
                cv2.circle(overlay, (cx, cy), 14, color, 2)
                cv2.drawMarker(overlay, (cx, cy), color, cv2.MARKER_CROSS, 10, 1)

        return overlay
