"""Experiment session management with JSON persistence."""

import json
import time
import uuid
import csv
import io
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from calibration import CalibrationSettings, cells_per_ml

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


@dataclass
class Sample:
    sample_id: int
    image_id: str
    image_path: str
    timestamp: str
    conf_threshold: float = 0.25
    detections: List[Dict] = field(default_factory=list)
    manual_additions: List[Dict] = field(default_factory=list)
    manual_removals: List[int] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    notes: str = ""
    grid_info: Dict = field(default_factory=dict)
    boundary_rule: str = "count_all"
    filtered_summary: Dict = field(default_factory=dict)

    @property
    def effective_summary(self) -> Dict:
        """Recompute counts accounting for manual edits.

        Uses filtered_summary (grid-aware) as the base when available,
        otherwise falls back to unfiltered summary.
        """
        base = self.filtered_summary if self.filtered_summary else self.summary
        viable = base.get("viable", 0)
        non_viable = base.get("non_viable", 0)

        # Subtract removals
        for det_id in self.manual_removals:
            det = next((d for d in self.detections if d["id"] == det_id), None)
            if det:
                if det.get("class_name") == "viable" or det.get("class") == 0:
                    viable -= 1
                else:
                    non_viable -= 1

        # Add manual additions
        for ann in self.manual_additions:
            if ann.get("class", 0) == 0:
                viable += 1
            else:
                non_viable += 1

        viable = max(0, viable)
        non_viable = max(0, non_viable)
        total = viable + non_viable
        return {
            "total": total,
            "viable": viable,
            "non_viable": non_viable,
            "viability_pct": round((viable / total * 100) if total > 0 else 0, 1),
        }


@dataclass
class SampleGroup:
    group_id: int
    name: str
    images: List[Sample] = field(default_factory=list)

    @property
    def aggregate_summary(self) -> Dict:
        """Sum counts across all images in this group."""
        viable = 0
        non_viable = 0
        for img in self.images:
            eff = img.effective_summary
            viable += eff["viable"]
            non_viable += eff["non_viable"]
        total = viable + non_viable
        return {
            "total": total,
            "viable": viable,
            "non_viable": non_viable,
            "viability_pct": round((viable / total * 100) if total > 0 else 0, 1),
            "image_count": len(self.images),
        }

    def concentration(self, dilution_factor: int = 1, trypan_blue: bool = True) -> float:
        """Calculate cells/mL using standard hemocytometer formula.

        cells/mL = (total count / squares counted) × dilution × 10,000
        squares_counted = number of images in this group.
        """
        agg = self.aggregate_summary
        if not self.images:
            return 0.0
        return cells_per_ml(
            agg["total"],
            squares_counted=len(self.images),
            dilution_factor=dilution_factor,
            trypan_blue_dilution=trypan_blue,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        # Add computed properties
        d["aggregate_summary"] = self.aggregate_summary
        for i, img in enumerate(self.images):
            d["images"][i]["effective_summary"] = img.effective_summary
        return d


@dataclass
class Session:
    id: str
    experiment_name: str
    created_at: str
    calibration: Dict = field(default_factory=lambda: CalibrationSettings().to_dict())
    model_used: str = "best.pt"
    sample_groups: List[SampleGroup] = field(default_factory=list)
    active_group_id: int = 0

    @property
    def next_group_id(self) -> int:
        if not self.sample_groups:
            return 1
        return max(g.group_id for g in self.sample_groups) + 1

    @property
    def next_sample_id(self) -> int:
        """Global sequential image ID across all groups."""
        total = sum(len(g.images) for g in self.sample_groups)
        return total + 1

    @property
    def samples(self) -> List[Sample]:
        """Flattened list of all images across all groups (backward compat)."""
        result = []
        for g in self.sample_groups:
            result.extend(g.images)
        return result

    def active_group(self) -> Optional[SampleGroup]:
        for g in self.sample_groups:
            if g.group_id == self.active_group_id:
                return g
        return None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "experiment_name": self.experiment_name,
            "created_at": self.created_at,
            "calibration": self.calibration,
            "model_used": self.model_used,
            "active_group_id": self.active_group_id,
            "sample_groups": [g.to_dict() for g in self.sample_groups],
            # Backward compat: flat samples list
            "samples": [
                {**asdict(img), "effective_summary": img.effective_summary,
                 "group_id": g.group_id, "group_name": g.name}
                for g in self.sample_groups
                for img in g.images
            ],
        }
        return d


class SessionStore:
    """Manages experiment sessions with JSON file persistence."""

    def __init__(self):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._current: Optional[Session] = None
        self._load_latest()

    def _load_latest(self):
        """Load the most recent session file on startup."""
        files = sorted(SESSIONS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if files:
            try:
                data = json.loads(files[0].read_text())
                session = Session(
                    id=data["id"],
                    experiment_name=data["experiment_name"],
                    created_at=data["created_at"],
                    calibration=data.get("calibration", CalibrationSettings().to_dict()),
                    model_used=data.get("model_used", "best.pt"),
                    active_group_id=data.get("active_group_id", 0),
                )
                # Load sample groups if present (new format)
                if "sample_groups" in data:
                    for gd in data["sample_groups"]:
                        images = []
                        for s in gd.get("images", []):
                            filtered = {k: v for k, v in s.items()
                                        if k not in ("effective_summary",)}
                            images.append(Sample(**filtered))
                        session.sample_groups.append(SampleGroup(
                            group_id=gd["group_id"],
                            name=gd["name"],
                            images=images,
                        ))
                else:
                    # Migrate old flat samples format
                    samples = []
                    for s in data.get("samples", []):
                        filtered = {k: v for k, v in s.items()
                                    if k not in ("effective_summary", "group_id", "group_name")}
                        samples.append(Sample(**filtered))
                    if samples:
                        group = SampleGroup(
                            group_id=1,
                            name="Sample 001",
                            images=samples,
                        )
                        session.sample_groups.append(group)
                        session.active_group_id = 1

                # Ensure there's always at least one sample group
                if not session.sample_groups:
                    gid = 1
                    session.sample_groups.append(SampleGroup(group_id=gid, name="Sample 001"))
                    session.active_group_id = gid

                self._current = session
            except (json.JSONDecodeError, KeyError, TypeError):
                self._current = None

    @property
    def current(self) -> Session:
        if not self._current:
            self.new_session()
        return self._current

    def new_session(self, experiment_name: Optional[str] = None) -> Session:
        if experiment_name is None:
            experiment_name = f"EXP_{time.strftime('%Y%m%d_%H%M')}"
        self._current = Session(
            id=str(uuid.uuid4())[:8],
            experiment_name=experiment_name,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        # Start with first sample group
        self.new_sample_group()
        return self._current

    def new_sample_group(self, name: Optional[str] = None) -> SampleGroup:
        """Create a new sample group and make it active."""
        session = self.current
        gid = session.next_group_id
        if name is None:
            name = f"Sample {gid:03d}"
        group = SampleGroup(group_id=gid, name=name)
        session.sample_groups.append(group)
        session.active_group_id = gid
        self._persist()
        return group

    def rename_group(self, group_id: int, name: str) -> Optional[SampleGroup]:
        for g in self.current.sample_groups:
            if g.group_id == group_id:
                g.name = name
                self._persist()
                return g
        return None

    def update_calibration(self, settings: dict) -> dict:
        session = self.current
        session.calibration.update(settings)
        self._persist()
        return session.calibration

    def update_experiment_name(self, name: str):
        self.current.experiment_name = name
        self._persist()

    def add_sample(
        self,
        image_id: str,
        image_path: str,
        detections: List[Dict],
        summary: Dict,
        conf_threshold: float = 0.25,
        grid_info: Optional[Dict] = None,
        boundary_rule: str = "count_all",
        filtered_summary: Optional[Dict] = None,
    ) -> Sample:
        """Add an image to the active sample group."""
        session = self.current
        group = session.active_group()
        if not group:
            group = self.new_sample_group()

        sample = Sample(
            sample_id=session.next_sample_id,
            image_id=image_id,
            image_path=image_path,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            conf_threshold=conf_threshold,
            detections=detections,
            summary=summary,
            grid_info=grid_info or {},
            boundary_rule=boundary_rule,
            filtered_summary=filtered_summary or {},
        )
        group.images.append(sample)
        self._persist()
        return sample

    def update_annotations(self, image_id: str, additions: List[Dict], removals: List[int]) -> Dict:
        sample = self._find_sample(image_id)
        if not sample:
            return {"error": "Sample not found"}
        sample.manual_additions = additions
        sample.manual_removals = removals
        self._persist()
        return sample.effective_summary

    def update_sample(
        self,
        image_id: str,
        detections: Optional[List[Dict]] = None,
        summary: Optional[Dict] = None,
        grid_info: Optional[Dict] = None,
        filtered_summary: Optional[Dict] = None,
        additions: Optional[List[Dict]] = None,
        removals: Optional[List[int]] = None,
        conf_threshold: Optional[float] = None,
    ) -> Optional[Dict]:
        """Update an existing sample's analysis data in place."""
        sample = self._find_sample(image_id)
        if not sample:
            return None
        if detections is not None:
            sample.detections = detections
        if summary is not None:
            sample.summary = summary
        if grid_info is not None:
            sample.grid_info = grid_info
        if filtered_summary is not None:
            sample.filtered_summary = filtered_summary
        if additions is not None:
            sample.manual_additions = additions
        if removals is not None:
            sample.manual_removals = removals
        if conf_threshold is not None:
            sample.conf_threshold = conf_threshold
        self._persist()
        return sample.effective_summary

    def get_sample(self, image_id: str) -> Optional[Sample]:
        return self._find_sample(image_id)

    def _find_sample(self, image_id: str) -> Optional[Sample]:
        for g in self.current.sample_groups:
            for img in g.images:
                if img.image_id == image_id:
                    return img
        return None

    def _find_group_for_image(self, image_id: str) -> Optional[SampleGroup]:
        for g in self.current.sample_groups:
            for img in g.images:
                if img.image_id == image_id:
                    return g
        return None

    def export_csv(self) -> str:
        """Generate CSV string for current session."""
        session = self.current
        cal = session.calibration
        dilution = cal.get("dilution_factor", 1)
        trypan = cal.get("trypan_blue_dilution", True)
        effective_dilution = dilution * (2 if trypan else 1)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "experiment", "sample_name", "image_num", "image_id", "timestamp",
            "total_cells", "viable", "non_viable", "viability_pct",
            "group_total", "group_viable", "group_non_viable", "group_viability_pct",
            "group_concentration_cells_per_ml", "group_image_count",
            "dilution_factor", "trypan_blue_dilution", "effective_dilution",
            "grid_detected", "grid_confidence",
            "manual_added", "manual_removed",
            "conf_threshold", "notes",
        ])

        for g in session.sample_groups:
            agg = g.aggregate_summary
            conc = g.concentration(dilution, trypan)
            for i, img in enumerate(g.images):
                eff = img.effective_summary
                writer.writerow([
                    session.experiment_name,
                    g.name,
                    i + 1,
                    img.image_id,
                    img.timestamp,
                    eff["total"], eff["viable"], eff["non_viable"], eff["viability_pct"],
                    agg["total"], agg["viable"], agg["non_viable"], agg["viability_pct"],
                    round(conc, 1), len(g.images),
                    dilution, trypan, effective_dilution,
                    img.grid_info.get("detected", False),
                    img.grid_info.get("confidence", 0),
                    len(img.manual_additions), len(img.manual_removals),
                    img.conf_threshold, img.notes,
                ])
        return buf.getvalue()

    def _persist(self):
        if self._current:
            path = SESSIONS_DIR / f"{self._current.id}.json"
            path.write_text(json.dumps(self._current.to_dict(), indent=2))
