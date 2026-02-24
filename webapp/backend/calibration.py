"""Hemocytometer calibration and concentration calculations.

Standard Improved Neubauer hemocytometer:
- Each large corner square = 1mm x 1mm
- Chamber depth = 0.1mm
- Volume per corner square = 0.0001 mL (1e-4 mL)
- Concentration (cells/mL) = (count / squares_counted) * dilution_factor * 10,000
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CalibrationSettings:
    pixels_per_mm: float = 0.0
    pixels_per_mm_source: str = "manual"  # "manual" or "grid_detection"
    grid_square_side_mm: float = 1.0
    chamber_depth_mm: float = 0.1
    dilution_factor: int = 1
    squares_counted: int = 1
    trypan_blue_dilution: bool = True  # 1:1 trypan blue mix doubles effective dilution

    @property
    def volume_ml(self) -> float:
        """Volume per counted region in mL."""
        return (
            self.grid_square_side_mm
            * self.grid_square_side_mm
            * self.chamber_depth_mm
            * self.squares_counted
            * 1e-3  # mm^3 to mL
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["volume_ml"] = self.volume_ml
        return d


def cells_per_ml(
    count: int,
    squares_counted: int = 1,
    dilution_factor: int = 1,
    trypan_blue_dilution: bool = True,
) -> float:
    """Calculate cell concentration in cells/mL.

    Args:
        count: Number of cells counted.
        squares_counted: Number of large squares counted (1-4).
        dilution_factor: Sample dilution factor before trypan blue mixing.
        trypan_blue_dilution: If True, accounts for 1:1 trypan blue mix
            (doubles the effective dilution factor).
    """
    if squares_counted <= 0:
        return 0.0
    effective_dilution = dilution_factor * (2 if trypan_blue_dilution else 1)
    return (count / squares_counted) * effective_dilution * 10_000


def bbox_area_mm2(bbox: list, pixels_per_mm: float) -> Optional[float]:
    """Convert a bounding box area from pixels to mm^2."""
    if pixels_per_mm <= 0:
        return None
    x1, y1, x2, y2 = bbox
    w_mm = abs(x2 - x1) / pixels_per_mm
    h_mm = abs(y2 - y1) / pixels_per_mm
    return w_mm * h_mm


def format_concentration(conc: float) -> str:
    """Format concentration for display."""
    if conc <= 0:
        return "N/A"
    if conc >= 1e6:
        return f"{conc / 1e6:.2f} x 10\u2076 cells/mL"
    if conc >= 1e3:
        return f"{conc / 1e3:.1f} x 10\u00B3 cells/mL"
    return f"{conc:.0f} cells/mL"
