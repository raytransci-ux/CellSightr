"""Analysis pipeline: inference -> grid detection -> cell filtering -> concentration.

Orchestrates the full hemocytometer analysis workflow:
  1. YOLO inference for cell detection (viable / non_viable)
  2. Hough-transform grid detection for automatic scale calibration
  3. Spatial filtering of detections to only cells inside the grid
  4. Concentration calculation (cells/mL) from filtered counts
"""

import time
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

from calibration import CalibrationSettings, cells_per_ml
from grid_detection import GridDetector, GridResult
from inference import InferenceEngine

BoundaryRule = Literal["standard", "count_all"]


class CellFilter:
    """Filters YOLO detections based on hemocytometer grid boundary."""

    @staticmethod
    def _rotate_point(
        px: float, py: float, cx: float, cy: float, angle_rad: float
    ) -> Tuple[float, float]:
        """Rotate point (px,py) by -angle_rad around (cx,cy) into grid-local coords."""
        dx = px - cx
        dy = py - cy
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        return dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a

    @staticmethod
    def filter_detections(
        detections: List[Dict],
        grid: GridResult,
        rule: BoundaryRule = "count_all",
    ) -> Tuple[List[Dict], List[Dict]]:
        """Filter detections to only those inside the grid boundary.

        Handles rotated grids by transforming cell centers into the grid's
        local coordinate frame before checking bounds.

        Args:
            detections: List of detection dicts with "bbox" as [x1,y1,x2,y2].
            grid: Grid detection result with boundary coordinates.
            rule: Boundary counting convention.
                "count_all" — include any cell whose center falls within the grid.
                "standard" — include top/left border cells, exclude bottom/right
                             (standard hemocytometer convention).

        Returns:
            Tuple of (inside_detections, outside_detections).
        """
        if not grid.detected or grid.boundary is None:
            return detections, []

        # Use rotated coordinate system when grid center/size available
        angle_rad = np.radians(grid.rotation_deg)
        if grid.grid_center and grid.grid_size:
            gcx, gcy = grid.grid_center
            half_w = grid.grid_size[0] / 2
            half_h = grid.grid_size[1] / 2
        else:
            gx1, gy1, gx2, gy2 = grid.boundary
            gcx = (gx1 + gx2) / 2
            gcy = (gy1 + gy2) / 2
            half_w = (gx2 - gx1) / 2
            half_h = (gy2 - gy1) / 2

        inside: List[Dict] = []
        outside: List[Dict] = []

        # Tolerance in pixels for "touching" a border
        border_tol = max(3, int(half_w * 0.01))

        for det in detections:
            bx1, by1, bx2, by2 = det["bbox"]
            cx = (bx1 + bx2) / 2
            cy = (by1 + by2) / 2

            # Rotate cell center into grid-local coordinates
            lx, ly = CellFilter._rotate_point(cx, cy, gcx, gcy, angle_rad)

            # Check if center is within the grid half-extents
            if abs(lx) > half_w or abs(ly) > half_h:
                outside.append(det)
                continue

            if rule == "count_all":
                inside.append(det)
            elif rule == "standard":
                # Standard hemocytometer rule in grid-local coords:
                #   Include cells touching the TOP or LEFT border
                #   Exclude cells touching the BOTTOM or RIGHT border
                # Rotate bbox corners too for border checks
                bbw = (bx2 - bx1) / 2
                bbh = (by2 - by1) / 2

                touches_bottom = ly + bbh >= half_h - border_tol
                touches_right = lx + bbw >= half_w - border_tol
                touches_top = ly - bbh <= -half_h + border_tol
                touches_left = lx - bbw <= -half_w + border_tol

                if touches_bottom and not touches_top:
                    outside.append(det)
                elif touches_right and not touches_left:
                    outside.append(det)
                else:
                    inside.append(det)

        return inside, outside

    @staticmethod
    def summarize(detections: List[Dict]) -> Dict:
        """Compute summary counts from a list of detections."""
        viable = sum(
            1 for d in detections
            if d.get("class_name") == "viable" or d.get("class") == 0
        )
        total = len(detections)
        non_viable = total - viable
        return {
            "total": total,
            "viable": viable,
            "non_viable": non_viable,
            "viability_pct": round((viable / total * 100) if total > 0 else 0, 1),
        }


class ImageQuality:
    """Lightweight image quality checks for acquisition feedback."""

    @staticmethod
    def assess(image: np.ndarray) -> Dict:
        """Evaluate image quality and return warnings if issues detected.

        Returns dict with:
            ok: bool — True if image quality is acceptable
            warnings: list of {code, message, severity} dicts
            metrics: {brightness, noise_estimate, contrast}
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Brightness: mean pixel intensity (0-255)
        brightness = float(np.mean(gray))

        # Contrast: standard deviation of pixel intensities
        contrast = float(np.std(gray))

        # Noise estimate: median absolute deviation of Laplacian
        # (Laplacian highlights edges + noise; in a noisy image the
        #  MAD of the Laplacian is high even in flat regions)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_estimate = float(np.median(np.abs(laplacian)))

        warnings = []

        # Thresholds calibrated against 346 real hemocytometer images:
        #   brightness 34.8–160.2 (median 92.6)
        #   contrast   10.7–52.9  (median 29.5)
        #   noise      1.0–12.0   (median 3.0)
        # Hemocytometer images are naturally dark/low-contrast at 4X;
        # only flag conditions that genuinely degrade cell detection.

        if brightness < 15:
            warnings.append({
                "code": "too_dark",
                "message": "Image is extremely dark. Increase microscope lamp brightness for reliable counting.",
                "severity": "error",
            })
        elif brightness < 25:
            warnings.append({
                "code": "low_brightness",
                "message": "Image is very dim. Consider increasing lamp brightness for better detection accuracy.",
                "severity": "warning",
            })

        if brightness > 200:
            warnings.append({
                "code": "overexposed",
                "message": "Image is overexposed. Reduce lamp brightness to avoid saturating cell detail.",
                "severity": "warning",
            })

        if contrast < 5:
            warnings.append({
                "code": "low_contrast",
                "message": "Very low contrast. Check focus and illumination — cells may not be distinguishable.",
                "severity": "error",
            })
        elif contrast < 8:
            warnings.append({
                "code": "low_contrast",
                "message": "Low contrast detected. Cell detection may be less reliable.",
                "severity": "warning",
            })

        if noise_estimate > 20:
            warnings.append({
                "code": "noisy",
                "message": "High image noise detected. Reduce camera gain/ISO or increase lamp brightness.",
                "severity": "error",
            })
        elif noise_estimate > 12:
            warnings.append({
                "code": "moderate_noise",
                "message": "Moderate noise. Cell detection may be less accurate. Consider increasing illumination.",
                "severity": "warning",
            })

        return {
            "ok": len([w for w in warnings if w["severity"] == "error"]) == 0,
            "warnings": warnings,
            "metrics": {
                "brightness": round(brightness, 1),
                "noise_estimate": round(noise_estimate, 1),
                "contrast": round(contrast, 1),
            },
        }


class AnalysisPipeline:
    """Orchestrates the full analysis: inference -> grid -> filter -> concentration."""

    def __init__(self, engine: InferenceEngine, grid_detector: GridDetector):
        self.engine = engine
        self.grid_detector = grid_detector

    def run(
        self,
        image_path: str,
        conf: float = 0.25,
        use_precise: bool = False,
        boundary_rule: BoundaryRule = "count_all",
        calibration: Optional[CalibrationSettings] = None,
        override_grid: Optional[Dict] = None,
    ) -> Dict:
        """Run the full analysis pipeline on a single image.

        Args:
            image_path: Path to the image file.
            conf: YOLO confidence threshold.
            use_precise: Use the medium (accurate) model instead of nano.
            boundary_rule: "count_all" or "standard" boundary convention.
            calibration: Calibration settings (dilution factor, squares counted).
            override_grid: If provided, skip auto grid detection and use this
                grid dict instead. Must contain keys matching GridResult fields.

        Returns:
            Dict with inference results, grid detection, filtered counts,
            and concentration calculation.
        """
        t_start = time.perf_counter()

        # Step 1: YOLO inference
        inference_result = self.engine.predict(
            image_path, conf=conf, use_precise=use_precise
        )
        t_inference = time.perf_counter()

        # Step 2: Read image for grid detection + quality check
        image = cv2.imread(image_path)
        t_imread = time.perf_counter()
        if image is None:
            inference_result["grid"] = GridResult(detected=False).to_dict()
            inference_result["filtered"] = {
                "detections": inference_result["detections"],
                "excluded": [],
                "summary": inference_result["summary"],
                "boundary_rule": boundary_rule,
                "concentration_cells_per_ml": None,
            }
            inference_result["image_quality"] = {"ok": False, "warnings": [], "metrics": {}}
            return inference_result

        # Image quality assessment
        quality = ImageQuality.assess(image)
        t_quality = time.perf_counter()
        inference_result["image_quality"] = quality

        # Use override grid if provided, otherwise auto-detect
        if override_grid and override_grid.get("detected"):
            boundary = override_grid.get("boundary")
            if boundary and len(boundary) >= 4:
                boundary = tuple(boundary[:4])
            grid_center = override_grid.get("grid_center")
            if grid_center:
                grid_center = tuple(grid_center)
            grid_size = override_grid.get("grid_size")
            if grid_size:
                grid_size = tuple(grid_size)
            grid_result = GridResult(
                detected=True,
                boundary=boundary,
                pixels_per_mm=override_grid.get("pixels_per_mm"),
                horizontal_lines=override_grid.get("horizontal_lines", []),
                vertical_lines=override_grid.get("vertical_lines", []),
                rotation_deg=override_grid.get("rotation_deg", 0.0),
                confidence=override_grid.get("confidence", 1.0),
                grid_center=grid_center,
                grid_size=grid_size,
            )
        else:
            grid_result = self.grid_detector.detect(image)
        t_grid = time.perf_counter()

        # Step 3: Filter detections to grid interior
        inside, outside = CellFilter.filter_detections(
            inference_result["detections"],
            grid_result,
            rule=boundary_rule,
        )
        filtered_summary = CellFilter.summarize(inside)

        # Step 4: Calculate concentration
        concentration = None
        if grid_result.detected and grid_result.pixels_per_mm:
            cal = calibration or CalibrationSettings()
            concentration = cells_per_ml(
                filtered_summary["total"],
                squares_counted=cal.squares_counted,
                dilution_factor=cal.dilution_factor,
                trypan_blue_dilution=cal.trypan_blue_dilution,
            )

        # High-density warning (added post-inference since we need the count)
        if filtered_summary["total"] >= 100:
            quality["warnings"].append({
                "code": "high_density",
                "message": (
                    "High cell density detected. Counting may be less accurate "
                    "with overlapping cells. Consider diluting the sample further."
                ),
                "severity": "warning",
            })

        # Step 5: Compose final result
        result = {
            **inference_result,
            "grid": grid_result.to_dict(),
            "filtered": {
                "detections": inside,
                "excluded": outside,
                "summary": filtered_summary,
                "boundary_rule": boundary_rule,
                "concentration_cells_per_ml": (
                    round(concentration, 1) if concentration is not None else None
                ),
            },
        }

        if grid_result.pixels_per_mm:
            result["auto_calibration"] = {
                "pixels_per_mm": round(grid_result.pixels_per_mm, 2),
                "source": "grid_detection",
            }

        t_end = time.perf_counter()
        result["timing"] = {
            "inference_ms": round((t_inference - t_start) * 1000, 1),
            "imread_ms": round((t_imread - t_inference) * 1000, 1),
            "quality_ms": round((t_quality - t_imread) * 1000, 1),
            "grid_ms": round((t_grid - t_quality) * 1000, 1),
            "filter_ms": round((t_end - t_grid) * 1000, 1),
            "total_ms": round((t_end - t_start) * 1000, 1),
        }
        print(f"Pipeline timing: {result['timing']}")

        return result

    def detect_grid_only(self, image_path: str) -> GridResult:
        """Run grid detection without inference (for debugging / calibration)."""
        image = cv2.imread(image_path)
        if image is None:
            return GridResult(detected=False)
        return self.grid_detector.detect(image)

    def render_full_overlay(
        self,
        image: np.ndarray,
        result: Dict,
    ) -> np.ndarray:
        """Render an overlay image with both grid lines and detection boxes."""
        overlay = image.copy()

        # Draw grid lines first (underneath detections)
        grid_dict = result.get("grid", {})
        if grid_dict.get("detected"):
            # Use the grid detector's rotated rendering
            from grid_detection import GridResult
            grid_obj = GridResult(
                detected=True,
                boundary=tuple(grid_dict["boundary"]) if grid_dict.get("boundary") else None,
                pixels_per_mm=grid_dict.get("pixels_per_mm"),
                horizontal_lines=grid_dict.get("horizontal_lines", []),
                vertical_lines=grid_dict.get("vertical_lines", []),
                rotation_deg=grid_dict.get("rotation_deg", 0.0),
                confidence=grid_dict.get("confidence", 0.0),
                grid_center=tuple(grid_dict["grid_center"]) if grid_dict.get("grid_center") else None,
                grid_size=tuple(grid_dict["grid_size"]) if grid_dict.get("grid_size") else None,
            )
            overlay = self.grid_detector.render_grid_overlay(overlay, grid_obj)

        # Draw detection overlays via inference engine
        filtered = result.get("filtered", {})
        inside = filtered.get("detections", [])
        excluded = filtered.get("excluded", [])

        # Draw excluded detections in gray (dimmed)
        for det in excluded:
            bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (128, 128, 128), 1)

        # Draw included detections with class colors
        CLASS_COLORS_BGR = {0: (142, 199, 72), 1: (68, 68, 239)}
        for det in inside:
            bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
            color = CLASS_COLORS_BGR.get(det.get("class", 0), (200, 200, 200))
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 2)

        return overlay
