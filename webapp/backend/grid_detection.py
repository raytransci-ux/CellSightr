"""Hemocytometer grid detection using Hough Line Transform.

Detects the 4x4 sub-grid (5 horizontal + 5 vertical lines) of an Improved Neubauer
hemocytometer corner square. From the known physical dimensions (1mm x 1mm), computes
pixels_per_mm for automatic scale calibration.

Algorithm:
  1. Preprocess: grayscale -> CLAHE -> blur -> Canny edges
  2. HoughLinesP to detect line segments
  3. Classify as horizontal or vertical by angle
  4. Cluster segments by intercept to find canonical grid lines
  5. Select best 5H + 5V equidistant lines forming the 4x4 pattern
  6. Compute boundary, pixels_per_mm, and confidence

Key insight: the 4x4 corner square has 5 equidistant lines per axis.  Adjacent
regions (central counting grid) have much finer line spacing.  The algorithm
explicitly enforces equidistant spacing to avoid selecting stray lines from the
neighbouring dense grid.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GridResult:
    """Result of grid detection on a hemocytometer image."""

    detected: bool
    boundary: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) axis-aligned bbox
    pixels_per_mm: Optional[float] = None
    horizontal_lines: List[int] = field(default_factory=list)  # y-intercepts at img center
    vertical_lines: List[int] = field(default_factory=list)    # x-intercepts at img center
    rotation_deg: float = 0.0
    confidence: float = 0.0
    grid_center: Optional[Tuple[float, float]] = None  # (cx, cy) center of grid
    grid_size: Optional[Tuple[float, float]] = None     # (width, height) in pixels

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "boundary": list(self.boundary) if self.boundary else None,
            "pixels_per_mm": round(self.pixels_per_mm, 2) if self.pixels_per_mm else None,
            "horizontal_lines": self.horizontal_lines,
            "vertical_lines": self.vertical_lines,
            "rotation_deg": round(self.rotation_deg, 2),
            "confidence": round(self.confidence, 3),
            "grid_center": list(self.grid_center) if self.grid_center else None,
            "grid_size": list(self.grid_size) if self.grid_size else None,
        }


class GridDetector:
    """Detects hemocytometer grid lines using Hough Line Transform."""

    EXPECTED_LINES_PER_AXIS = 5  # 4x4 grid = 5 lines per axis
    # Maximum allowed deviation from equal spacing, as fraction of mean spacing
    MAX_SPACING_DEVIATION = 0.20

    def __init__(self, grid_square_side_mm: float = 1.0):
        self.grid_square_side_mm = grid_square_side_mm

    def detect(self, image: np.ndarray) -> GridResult:
        """Run full grid detection pipeline on a BGR image."""
        if image is None or image.size == 0:
            return GridResult(detected=False)

        h, w = image.shape[:2]

        # Step 1: Preprocess
        edges, gray = self._preprocess(image)

        # Step 2: Detect line segments via Hough
        min_line_length = int(min(w, h) * 0.25)
        segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=min_line_length,
            maxLineGap=25,
        )

        if segments is None or len(segments) == 0:
            return self._fallback_detection(gray, h, w)

        segments = segments.reshape(-1, 4)

        # Step 3: Classify as horizontal or vertical
        h_segments, v_segments, median_angle = self._classify_segments(segments)

        if len(h_segments) < 2 or len(v_segments) < 2:
            return self._fallback_detection(gray, h, w)

        # Step 4: Cluster line segments into canonical lines
        h_lines = self._cluster_lines(h_segments, axis="horizontal", img_size=(w, h))
        v_lines = self._cluster_lines(v_segments, axis="vertical", img_size=(w, h))

        if len(h_lines) < 2 or len(v_lines) < 2:
            return self._fallback_detection(gray, h, w)

        # Step 5: Select best equidistant lines forming the grid pattern
        h_selected = self._select_equidistant_lines(h_lines, self.EXPECTED_LINES_PER_AXIS)
        v_selected = self._select_equidistant_lines(v_lines, self.EXPECTED_LINES_PER_AXIS)

        if len(h_selected) < 2 or len(v_selected) < 2:
            return self._fallback_detection(gray, h, w)

        # Step 6: Compute boundary, center, and pixels_per_mm
        y_top = min(h_selected)
        y_bottom = max(h_selected)
        x_left = min(v_selected)
        x_right = max(v_selected)

        grid_width_px = x_right - x_left
        grid_height_px = y_bottom - y_top

        if grid_width_px <= 0 or grid_height_px <= 0:
            return self._fallback_detection(gray, h, w)

        pixels_per_mm = ((grid_width_px + grid_height_px) / 2) / self.grid_square_side_mm

        # Grid center (intersection of midlines)
        cx = float(np.mean(v_selected))
        cy = float(np.mean(h_selected))

        # Step 7: Confidence scoring
        confidence = self._compute_confidence(
            h_selected, v_selected, grid_width_px, grid_height_px
        )

        if confidence < 0.3:
            fallback = self._fallback_detection(gray, h, w)
            if fallback.confidence > confidence:
                return fallback

        rot_deg = float(np.degrees(median_angle)) if median_angle else 0.0

        return GridResult(
            detected=True,
            boundary=(x_left, y_top, x_right, y_bottom),
            pixels_per_mm=pixels_per_mm,
            horizontal_lines=sorted(h_selected),
            vertical_lines=sorted(v_selected),
            rotation_deg=rot_deg,
            confidence=confidence,
            grid_center=(cx, cy),
            grid_size=(float(grid_width_px), float(grid_height_px)),
        )

    # ── Preprocessing ───────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to grayscale, enhance contrast, detect edges."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE for contrast normalization across varying illumination
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Blur to reduce noise while preserving grid line edges
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.0)

        # Canny edge detection — adaptive thresholds based on image brightness
        median_val = int(np.median(blurred))
        low_thresh = max(30, int(median_val * 0.5))
        high_thresh = max(80, int(median_val * 1.2))
        edges = cv2.Canny(blurred, low_thresh, high_thresh, apertureSize=3)

        # Dilate edges slightly to connect gaps in grid lines
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return edges, gray

    # ── Line classification ─────────────────────────────────────────────

    def _classify_segments(
        self, segments: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """Split line segments into horizontal and vertical by angle."""
        h_segs = []
        v_segs = []

        for seg in segments:
            x1, y1, x2, y2 = seg
            norm_angle = np.arctan2(y2 - y1, x2 - x1) % np.pi

            if norm_angle < np.radians(15) or norm_angle > np.radians(165):
                h_segs.append(seg)
            elif abs(norm_angle - np.pi / 2) < np.radians(15):
                v_segs.append(seg)

        # Median angle of horizontal lines gives rotation offset
        median_angle = None
        if h_segs:
            h_angles = []
            for seg in h_segs:
                a = np.arctan2(seg[3] - seg[1], seg[2] - seg[0])
                if a > np.pi / 2:
                    a -= np.pi
                elif a < -np.pi / 2:
                    a += np.pi
                h_angles.append(a)
            median_angle = float(np.median(h_angles))

        return np.array(h_segs), np.array(v_segs), median_angle

    # ── Clustering ──────────────────────────────────────────────────────

    def _cluster_lines(
        self,
        segments: np.ndarray,
        axis: str,
        img_size: Tuple[int, int],
    ) -> List[int]:
        """Cluster line segments by intercept position into canonical lines.

        For horizontal lines: cluster by y-intercept (y at x = img_center).
        For vertical lines: cluster by x-intercept (x at y = img_center).

        Returns sorted list of intercept positions (pixel coordinates).
        """
        w, h = img_size

        if len(segments) == 0:
            return []

        intercepts = []
        for seg in segments:
            x1, y1, x2, y2 = seg
            if axis == "horizontal":
                if abs(x2 - x1) > 1:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 + slope * (w / 2 - x1)
                else:
                    intercept = (y1 + y2) / 2
            else:
                if abs(y2 - y1) > 1:
                    slope = (x2 - x1) / (y2 - y1)
                    intercept = x1 + slope * (h / 2 - y1)
                else:
                    intercept = (x1 + x2) / 2
            intercepts.append(int(round(intercept)))

        intercepts = sorted(intercepts)

        # Cluster by proximity
        dim = h if axis == "horizontal" else w
        min_gap = max(10, dim // 20)

        clusters: List[List[int]] = []
        current_cluster = [intercepts[0]]

        for i in range(1, len(intercepts)):
            if intercepts[i] - intercepts[i - 1] < min_gap:
                current_cluster.append(intercepts[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [intercepts[i]]
        clusters.append(current_cluster)

        # Canonical position = median of each cluster, weighted by segment count
        canonical = [int(np.median(c)) for c in clusters]
        return sorted(canonical)

    # ── Equidistant line selection (core fix) ───────────────────────────

    def _select_equidistant_lines(
        self, candidates: List[int], expected: int
    ) -> List[int]:
        """Select the best subset of candidates forming an equidistant grid.

        Strategy:
          1. If <= expected candidates, validate and repair spacing.
          2. If more, try all C(n, expected) subsets (small n) or use a
             spacing-based approach (large n) to find the most equidistant set.
          3. After selection, validate that all spacings are within tolerance.
             If the outermost line is an outlier, replace it by extrapolation
             from the consistent inner lines.
        """
        if len(candidates) < 2:
            return candidates

        if len(candidates) <= expected:
            selected = list(candidates)
        elif len(candidates) <= expected + 6:
            selected = self._best_equidistant_subset(candidates, expected)
        else:
            selected = self._best_equidistant_greedy(candidates, expected)

        # Post-selection: repair any outlier outer lines
        selected = self._repair_outlier_lines(selected)

        return selected

    def _best_equidistant_subset(
        self, candidates: List[int], expected: int
    ) -> List[int]:
        """Exhaustive search for the most equidistant subset of `expected` lines."""
        best_cv = float("inf")
        best_subset = candidates[:expected]

        for subset in combinations(candidates, expected):
            spacings = [subset[i + 1] - subset[i] for i in range(len(subset) - 1)]
            if min(spacings) <= 0:
                continue
            mean_sp = np.mean(spacings)
            if mean_sp <= 0:
                continue
            cv = float(np.std(spacings) / mean_sp)
            if cv < best_cv:
                best_cv = cv
                best_subset = list(subset)

        return list(best_subset)

    def _best_equidistant_greedy(
        self, candidates: List[int], expected: int
    ) -> List[int]:
        """Greedy: find the dominant spacing, then collect lines that match it.

        Computes pairwise spacings between all consecutive candidate lines,
        finds the most common spacing (the mode), and builds a grid from it.
        """
        if len(candidates) < 2:
            return candidates

        # Compute all consecutive spacings
        spacings = [candidates[i + 1] - candidates[i] for i in range(len(candidates) - 1)]

        # Find the dominant spacing via histogram
        # Bin width = 5% of median spacing
        median_sp = float(np.median(spacings))
        if median_sp <= 0:
            return candidates[:expected]

        bin_width = max(5, int(median_sp * 0.1))
        best_spacing = median_sp
        best_count = 0

        for sp in spacings:
            count = sum(1 for s in spacings if abs(s - sp) < bin_width)
            if count > best_count:
                best_count = count
                best_spacing = sp

        # Refine: average all spacings near the dominant one
        near = [s for s in spacings if abs(s - best_spacing) < bin_width]
        target_spacing = float(np.mean(near)) if near else best_spacing

        # Build grid: try each candidate as start, collect lines at multiples of target_spacing
        best_set: List[int] = []
        tol = target_spacing * 0.2

        for start in candidates:
            grid_lines = [start]
            for step in range(1, expected):
                target = start + step * target_spacing
                # Find closest candidate within tolerance
                closest = min(candidates, key=lambda c: abs(c - target))
                if abs(closest - target) <= tol and closest not in grid_lines:
                    grid_lines.append(closest)

            if len(grid_lines) > len(best_set):
                best_set = sorted(grid_lines)
            if len(best_set) == expected:
                break

        return best_set if best_set else candidates[:expected]

    def _repair_outlier_lines(self, lines: List[int]) -> List[int]:
        """Validate equidistant spacing and replace outlier outer lines.

        If the first or last spacing is much larger than the median of the
        inner spacings, replace the outlier with an extrapolated position.
        """
        if len(lines) < 3:
            return lines

        lines = sorted(lines)
        spacings = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]

        if len(spacings) < 2:
            return lines

        # Use the inner spacings (all except first and last) as reference
        if len(spacings) >= 4:
            inner_spacings = spacings[1:-1]
        else:
            inner_spacings = spacings

        median_sp = float(np.median(inner_spacings))
        if median_sp <= 0:
            return lines

        threshold = median_sp * (1 + self.MAX_SPACING_DEVIATION)

        repaired = list(lines)

        # Check first spacing (top / left boundary)
        if spacings[0] > threshold:
            # First line is too far from the second — replace with extrapolation
            repaired[0] = repaired[1] - int(round(median_sp))

        # Check last spacing (bottom / right boundary)
        if spacings[-1] > threshold:
            # Last line is too far from the second-to-last — replace
            repaired[-1] = repaired[-2] + int(round(median_sp))

        return sorted(repaired)

    # ── Confidence ──────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        h_lines: List[int],
        v_lines: List[int],
        grid_width: int,
        grid_height: int,
    ) -> float:
        """Compute confidence score for the detected grid."""
        # Line count score
        h_count_score = max(0, 1.0 - abs(len(h_lines) - self.EXPECTED_LINES_PER_AXIS) * 0.2)
        v_count_score = max(0, 1.0 - abs(len(v_lines) - self.EXPECTED_LINES_PER_AXIS) * 0.2)
        line_score = (h_count_score + v_count_score) / 2

        # Spacing regularity score
        spacing_score = 1.0
        for lines in [h_lines, v_lines]:
            if len(lines) >= 3:
                spacings = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
                mean_sp = np.mean(spacings)
                if mean_sp > 0:
                    cv = float(np.std(spacings) / mean_sp)
                    spacing_score = min(spacing_score, max(0, 1.0 - cv))

        # Aspect ratio score: grid should be roughly square
        aspect = grid_width / grid_height if grid_height > 0 else 0
        aspect_score = max(0, 1.0 - abs(aspect - 1.0) * 3)

        confidence = line_score * spacing_score * aspect_score
        return min(1.0, max(0.0, confidence))

    # ── Fallback ────────────────────────────────────────────────────────

    def _fallback_detection(
        self, gray: np.ndarray, h: int, w: int
    ) -> GridResult:
        """Fallback: adaptive threshold + contour detection for the outer boundary."""
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 8, 1), 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 8, 1)))
        h_lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        v_lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
        grid_mask = cv2.add(h_lines_img, v_lines_img)

        contours, _ = cv2.findContours(
            grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return GridResult(detected=False)

        best_contour = None
        best_area = 0
        min_area = h * w * 0.05

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / ch if ch > 0 else 0
            if 0.5 < aspect < 2.0 and area > best_area:
                best_area = area
                best_contour = cnt

        if best_contour is None:
            return GridResult(detected=False)

        x, y, cw, ch = cv2.boundingRect(best_contour)
        pixels_per_mm = ((cw + ch) / 2) / self.grid_square_side_mm

        return GridResult(
            detected=True,
            boundary=(x, y, x + cw, y + ch),
            pixels_per_mm=pixels_per_mm,
            horizontal_lines=[],
            vertical_lines=[],
            rotation_deg=0.0,
            confidence=0.25,
        )

    # ── Geometry helpers ────────────────────────────────────────────────

    @staticmethod
    def _rotated_rect_corners(
        cx: float, cy: float, half_w: float, half_h: float, angle_rad: float
    ) -> List[Tuple[int, int]]:
        """Compute the 4 corners of a rotated rectangle."""
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        corners = []
        for dx, dy in [(-half_w, -half_h), (half_w, -half_h),
                        (half_w, half_h), (-half_w, half_h)]:
            rx = cx + dx * cos_a - dy * sin_a
            ry = cy + dx * sin_a + dy * cos_a
            corners.append((int(round(rx)), int(round(ry))))
        return corners

    @staticmethod
    def _rotated_line_endpoints(
        intercept: float, cx: float, cy: float,
        half_extent: float, angle_rad: float, axis: str,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Compute endpoints of a grid line rotated about the grid center.

        For horizontal lines: intercept is y-position at x=cx; the line
        extends ±half_extent along the rotated x-axis.
        For vertical lines: intercept is x-position at y=cy.
        """
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        if axis == "horizontal":
            dy = intercept - cy
            # Two endpoints: ±half_extent along the rotated x-direction
            p1x = cx + (-half_extent) * cos_a - dy * sin_a
            p1y = cy + (-half_extent) * sin_a + dy * cos_a
            p2x = cx + half_extent * cos_a - dy * sin_a
            p2y = cy + half_extent * sin_a + dy * cos_a
        else:
            dx = intercept - cx
            p1x = cx + dx * cos_a - (-half_extent) * sin_a
            p1y = cy + dx * sin_a + (-half_extent) * cos_a
            p2x = cx + dx * cos_a - half_extent * sin_a
            p2y = cy + dx * sin_a + half_extent * cos_a
        return (int(round(p1x)), int(round(p1y))), (int(round(p2x)), int(round(p2y)))

    # ── Visualization ───────────────────────────────────────────────────

    def render_grid_overlay(
        self, image: np.ndarray, grid: GridResult
    ) -> np.ndarray:
        """Draw detected grid lines and boundary on an image for visualization."""
        overlay = image.copy()

        if not grid.detected or grid.boundary is None:
            cv2.putText(
                overlay, "Grid not detected", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
            )
            return overlay

        angle_rad = np.radians(grid.rotation_deg)

        # Grid center and half-dimensions
        if grid.grid_center and grid.grid_size:
            cx, cy = grid.grid_center
            half_w = grid.grid_size[0] / 2
            half_h = grid.grid_size[1] / 2
        else:
            x1, y1, x2, y2 = grid.boundary
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            half_w = (x2 - x1) / 2
            half_h = (y2 - y1) / 2

        # Draw rotated outer boundary
        corners = self._rotated_rect_corners(cx, cy, half_w, half_h, angle_rad)
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 180, 0), thickness=3)

        # Draw horizontal grid lines (rotated)
        for y_val in grid.horizontal_lines:
            p1, p2 = self._rotated_line_endpoints(
                y_val, cx, cy, half_w, angle_rad, "horizontal"
            )
            cv2.line(overlay, p1, p2, (255, 255, 0), 1)

        # Draw vertical grid lines (rotated)
        for x_val in grid.vertical_lines:
            p1, p2 = self._rotated_line_endpoints(
                x_val, cx, cy, half_h, angle_rad, "vertical"
            )
            cv2.line(overlay, p1, p2, (255, 255, 0), 1)

        # Draw info text
        info = f"Grid: {grid.confidence:.0%} conf | {grid.pixels_per_mm:.0f} px/mm | {grid.rotation_deg:+.1f} deg"
        text_x = int(cx - half_w)
        text_y = int(cy - half_h) - 10
        cv2.putText(
            overlay, info, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2,
        )

        return overlay
