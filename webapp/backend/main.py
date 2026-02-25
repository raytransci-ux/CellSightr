"""FastAPI application for hemocytometer cell counting."""

import asyncio
import io
import json
import os
import sys
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import re
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Upload size limits
MAX_IMAGE_SIZE = 50 * 1024 * 1024   # 50 MB
MAX_MODEL_SIZE = 200 * 1024 * 1024  # 200 MB

# Add backend dir to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import InferenceEngine
from camera import CameraManager
from session import SessionStore
from calibration import CalibrationSettings, cells_per_ml
from grid_detection import GridDetector
from pipeline import AnalysisPipeline

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
IMAGES_DIR = UPLOADS_DIR / "images"
MODELS_DIR = UPLOADS_DIR / "models"

# Model paths — prefer YOLOv11, fall back to legacy v8
NANO_MODEL_V11   = PROJECT_ROOT / "checkpoints" / "yolo" / "nano" / "weights" / "best.pt"
MEDIUM_MODEL_V11 = PROJECT_ROOT / "checkpoints" / "yolo" / "medium" / "weights" / "best.pt"
LARGE_MODEL_V11  = PROJECT_ROOT / "checkpoints" / "yolo" / "large" / "weights" / "best.pt"
LEGACY_MODEL_V8  = PROJECT_ROOT / "checkpoints" / "yolo" / "run" / "weights" / "best.pt"


def _resolve_model_path() -> tuple[str, Optional[str]]:
    """Find best available nano and precise model paths.

    Precise model preference: large > medium > None.
    Large gains +5% non_viable recall over medium with only 3ms extra latency.
    """
    nano = NANO_MODEL_V11 if NANO_MODEL_V11.exists() else LEGACY_MODEL_V8
    # Prefer large model for precise inference (better non_viable detection)
    if LARGE_MODEL_V11.exists():
        precise = LARGE_MODEL_V11
    elif MEDIUM_MODEL_V11.exists():
        precise = MEDIUM_MODEL_V11
    else:
        precise = None
    return str(nano), str(precise) if precise else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialize services on startup."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    nano_path, medium_path = _resolve_model_path()
    app.state.engine = InferenceEngine(nano_path, medium_model_path=medium_path)
    print("Warming up YOLO models (CUDA kernel compilation)...")
    app.state.engine.warmup()
    print("Warmup complete — ready for fast inference.")
    app.state.grid_detector = GridDetector(grid_square_side_mm=1.0)
    app.state.pipeline = AnalysisPipeline(app.state.engine, app.state.grid_detector)
    app.state.camera = CameraManager()
    app.state.sessions = SessionStore()
    yield
    app.state.camera.release()


app = FastAPI(
    title="CellCount",
    description="Hemocytometer cell counting and viability analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Root ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": app.state.engine.model_name,
        "camera": app.state.camera.get_status(),
    }


# ── Camera ───────────────────────────────────────────────────────────────

@app.get("/api/camera/status")
async def camera_status():
    return app.state.camera.get_status()


@app.get("/api/camera/devices")
async def camera_devices():
    """List available camera devices on the system."""
    return app.state.camera.list_devices()


@app.post("/api/camera/start")
async def camera_start(device_id: int = 0, backend: Optional[str] = None):
    result = app.state.camera.start(device_id, backend)
    if result["status"] == "error":
        raise HTTPException(503, result["message"])
    return result


@app.post("/api/camera/stop")
async def camera_stop():
    app.state.camera.stop()
    return {"status": "stopped"}


@app.websocket("/ws/camera")
async def camera_ws(ws: WebSocket):
    """Stream camera frames over WebSocket as binary JPEG.

    Supports live cell tracking: every Nth frame, run nano inference and
    overlay detection boxes onto the stream.  The client can toggle this
    by sending a JSON message: {"live_tracking": true/false}.
    """
    await ws.accept()
    cam: CameraManager = app.state.camera
    engine: InferenceEngine = app.state.engine
    live_tracking = False
    frame_counter = 0
    track_interval = 5  # run inference every 5th frame (~3 fps of detections at 15 fps stream)
    last_detections = []

    async def receive_commands():
        """Listen for client commands in background."""
        nonlocal live_tracking
        try:
            while True:
                msg = await ws.receive_text()
                data = json.loads(msg)
                if "live_tracking" in data:
                    live_tracking = data["live_tracking"]
                    last_detections.clear()
        except Exception:
            pass

    # Start background listener for client commands
    cmd_task = asyncio.create_task(receive_commands())

    try:
        while True:
            if cam.is_running:
                frame = cam.get_frame()
                if frame is not None:
                    # Resize for streaming (keep aspect ratio, max 800px wide)
                    h, w = frame.shape[:2]
                    if w > 800:
                        scale = 800 / w
                        frame = cv2.resize(frame, (800, int(h * scale)))
                    else:
                        scale = 1.0

                    # Live tracking: run nano inference periodically
                    if live_tracking:
                        frame_counter += 1
                        if frame_counter >= track_interval:
                            frame_counter = 0
                            try:
                                results = engine._model(frame, conf=0.25, verbose=False)[0]
                                last_detections.clear()
                                for box in results.boxes:
                                    cls_id = int(box.cls[0])
                                    bbox = [int(v) for v in box.xyxy[0].tolist()]
                                    last_detections.append((bbox, cls_id, float(box.conf[0])))
                            except Exception:
                                pass

                        # Draw cached detections onto frame
                        for bbox, cls_id, conf in last_detections:
                            color = (72, 199, 142) if cls_id == 0 else (68, 68, 239)  # BGR
                            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    await ws.send_bytes(jpeg.tobytes())
            await asyncio.sleep(0.066)  # ~15 fps
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        cmd_task.cancel()


# ── Capture ──────────────────────────────────────────────────────────────

@app.post("/api/capture")
async def capture():
    """Capture a still frame from the camera."""
    cam: CameraManager = app.state.camera
    if not cam.is_running:
        raise HTTPException(503, "Camera not running")

    frame = cam.capture_still()
    if frame is None:
        raise HTTPException(500, "Failed to capture frame")

    image_id = str(uuid.uuid4())[:12]
    filename = f"{image_id}.jpg"
    filepath = IMAGES_DIR / filename
    cv2.imwrite(str(filepath), frame)

    return {
        "image_id": image_id,
        "image_url": f"/api/images/{filename}",
        "size": {"width": frame.shape[1], "height": frame.shape[0]},
    }


# ── Security helpers ──────────────────────────────────────────────────────

def _safe_path(base_dir: Path, filename: str) -> Path:
    """Resolve a filename within base_dir and verify it doesn't escape."""
    resolved = (base_dir / filename).resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        raise HTTPException(403, "Access denied")
    return resolved


def _sanitize_filename(name: str) -> str:
    """Strip characters unsafe for filenames and HTTP headers."""
    return re.sub(r'[^\w\s\-.]', '_', name).strip()


# ── Analysis ─────────────────────────────────────────────────────────────

def _find_image(image_id: str) -> Path:
    """Locate an image file by ID, trying multiple extensions."""
    for ext in [".jpg", ".png", ".bmp", ".tiff"]:
        filepath = _safe_path(IMAGES_DIR, f"{image_id}{ext}")
        if filepath.exists():
            return filepath
    raise HTTPException(404, "Image not found")


def _get_calibration_settings(sessions: SessionStore) -> CalibrationSettings:
    """Build CalibrationSettings from the current session calibration dict."""
    cal_dict = sessions.current.calibration
    known_fields = {
        "pixels_per_mm", "pixels_per_mm_source", "grid_square_side_mm",
        "chamber_depth_mm", "dilution_factor", "squares_counted",
        "trypan_blue_dilution",
    }
    return CalibrationSettings(**{k: v for k, v in cal_dict.items() if k in known_fields})


@app.post("/api/analyze")
async def analyze(
    image_id: str,
    conf: float = 0.25,
    use_precise: bool = False,
    boundary_rule: str = "count_all",
    request: Request = None,
):
    """Run full analysis pipeline on a previously captured image.

    Args:
        use_precise: If True, use the medium (accurate) model instead of nano.
        boundary_rule: "count_all" or "standard" (top/left include, bottom/right exclude).
        Body (optional JSON): {"override_grid": {...}} to skip auto grid detection.
    """
    filepath = _find_image(image_id)

    # Parse optional override_grid from request body
    override_grid = None
    if request:
        try:
            body = await request.json()
            override_grid = body.get("override_grid")
        except Exception:
            pass  # No body or not JSON — use auto-detection

    pipeline: AnalysisPipeline = app.state.pipeline
    sessions: SessionStore = app.state.sessions
    calibration = _get_calibration_settings(sessions)

    result = pipeline.run(
        str(filepath),
        conf=conf,
        use_precise=use_precise,
        boundary_rule=boundary_rule,
        calibration=calibration,
        override_grid=override_grid,
    )

    # Auto-update calibration pixels_per_mm if grid detected and not yet set
    auto_cal = result.get("auto_calibration")
    if auto_cal and sessions.current.calibration.get("pixels_per_mm", 0) == 0:
        sessions.update_calibration({
            "pixels_per_mm": auto_cal["pixels_per_mm"],
            "pixels_per_mm_source": "grid_detection",
        })

    return result


@app.post("/api/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    boundary_rule: str = Form("count_all"),
):
    """Upload an image and run the full analysis pipeline."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    image_id = str(uuid.uuid4())[:12]
    ext = Path(file.filename).suffix if file.filename else ".jpg"
    filename = f"{image_id}{ext}"
    filepath = IMAGES_DIR / filename

    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(413, f"Image too large (max {MAX_IMAGE_SIZE // 1024 // 1024} MB)")
    filepath.write_bytes(content)

    pipeline: AnalysisPipeline = app.state.pipeline
    sessions: SessionStore = app.state.sessions
    calibration = _get_calibration_settings(sessions)

    result = pipeline.run(str(filepath), conf=conf, boundary_rule=boundary_rule, calibration=calibration)
    result["image_id"] = image_id
    result["image_url"] = f"/api/images/{filename}"
    return result


@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """Serve a saved image, trying alternate extensions if needed."""
    filepath = _safe_path(IMAGES_DIR, filename)
    if not filepath.exists():
        stem = Path(filename).stem
        for ext in [".jpg", ".png", ".bmp", ".tiff", ".jpeg"]:
            alt = _safe_path(IMAGES_DIR, f"{stem}{ext}")
            if alt.exists():
                filepath = alt
                break
        else:
            raise HTTPException(404, "Image not found")
    return FileResponse(str(filepath))


@app.post("/api/images/{image_id}/overlay")
async def get_overlay_image(image_id: str):
    """Generate and return an image with detection overlay baked in."""
    # Find the image file
    filepath = None
    for f in IMAGES_DIR.glob(f"{image_id}.*"):
        filepath = f
        break
    if not filepath:
        raise HTTPException(404, "Image not found")

    sessions: SessionStore = app.state.sessions
    sample = sessions.get_sample(image_id)
    if not sample:
        raise HTTPException(404, "Sample not found")

    image = cv2.imread(str(filepath))
    engine: InferenceEngine = app.state.engine
    overlay = engine.render_overlay(
        image,
        sample.detections,
        sample.manual_additions,
        set(sample.manual_removals),
    )

    _, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return StreamingResponse(
        iter([jpeg.tobytes()]),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=cellcount_{image_id}.jpg"},
    )


# ── Model Management ────────────────────────────────────────────────────

@app.get("/api/model/info")
async def model_info():
    engine: InferenceEngine = app.state.engine
    return {
        "model": engine.model_name,
        "precise_model": engine.precise_model_name,
        "has_precise_model": engine.has_precise_model,
        "classes": engine.class_names,
    }


@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a custom YOLO .pt model."""
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(400, "File must be a .pt YOLO model")

    safe_name = _sanitize_filename(Path(file.filename).name)
    if not safe_name.endswith(".pt"):
        raise HTTPException(400, "Invalid model filename")
    filepath = _safe_path(MODELS_DIR, safe_name)
    content = await file.read()
    if len(content) > MAX_MODEL_SIZE:
        raise HTTPException(413, f"Model too large (max {MAX_MODEL_SIZE // 1024 // 1024} MB)")
    filepath.write_bytes(content)

    # Validate by attempting to load
    try:
        from ultralytics import YOLO
        test_model = YOLO(str(filepath))
        names = test_model.names if hasattr(test_model, "names") else {}
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(400, f"Invalid model file: {e}")

    return {
        "model": safe_name,
        "path": safe_name,
        "classes": dict(names) if names else {},
    }


@app.post("/api/model/select")
async def select_model(model_path: str):
    """Switch to a different model. Path must resolve under models/ or checkpoints/."""
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    path = Path(model_path)
    if not path.exists():
        path = MODELS_DIR / model_path
    if not path.exists():
        raise HTTPException(404, "Model file not found")

    resolved = path.resolve()
    allowed_dirs = [MODELS_DIR.resolve(), CHECKPOINTS_DIR.resolve()]
    if not any(resolved.is_relative_to(d) for d in allowed_dirs):
        raise HTTPException(403, "Model path not allowed")

    engine: InferenceEngine = app.state.engine
    try:
        info = engine.switch_model(str(resolved))
        return info
    except Exception as e:
        raise HTTPException(400, f"Failed to load model: {e}")


# ── Annotations ──────────────────────────────────────────────────────────

@app.patch("/api/annotations/{image_id}")
async def update_annotations(image_id: str, request: Request):
    """Update manual annotations for a sample."""
    body = await request.json()
    additions = body.get("additions", [])
    removals = body.get("removals", [])
    sessions: SessionStore = app.state.sessions
    result = sessions.update_annotations(image_id, additions, removals)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


# ── Calibration ──────────────────────────────────────────────────────────

@app.get("/api/calibration")
async def get_calibration():
    sessions: SessionStore = app.state.sessions
    return sessions.current.calibration


@app.post("/api/calibration")
async def set_calibration(
    pixels_per_mm: float = 0,
    dilution_factor: int = 1,
    squares_counted: int = 1,
    grid_square_side_mm: float = 1.0,
    trypan_blue_dilution: bool = True,
):
    if pixels_per_mm < 0:
        raise HTTPException(400, "pixels_per_mm must be >= 0")
    if dilution_factor < 1:
        raise HTTPException(400, "dilution_factor must be >= 1")
    if squares_counted < 1:
        raise HTTPException(400, "squares_counted must be >= 1")
    if grid_square_side_mm <= 0:
        raise HTTPException(400, "grid_square_side_mm must be > 0")
    sessions: SessionStore = app.state.sessions
    settings = {
        "pixels_per_mm": pixels_per_mm,
        "dilution_factor": dilution_factor,
        "squares_counted": squares_counted,
        "grid_square_side_mm": grid_square_side_mm,
        "trypan_blue_dilution": trypan_blue_dilution,
    }
    return sessions.update_calibration(settings)


# ── Session ──────────────────────────────────────────────────────────────

@app.get("/api/session/current")
async def get_session():
    sessions: SessionStore = app.state.sessions
    return sessions.current.to_dict()


@app.post("/api/session/new")
async def new_session(experiment_name: Optional[str] = None):
    sessions: SessionStore = app.state.sessions
    session = sessions.new_session(experiment_name)
    return session.to_dict()


@app.post("/api/session/rename")
async def rename_session(name: str):
    sessions: SessionStore = app.state.sessions
    sessions.update_experiment_name(name)
    return {"experiment_name": name}


@app.post("/api/session/sample")
async def add_sample(request: Request):
    body = await request.json()
    image_id = body.get("image_id", "")
    detections = body.get("detections", [])
    summary = body.get("summary", {})
    conf_threshold = body.get("conf_threshold", 0.25)
    grid_info = body.get("grid_info", {})
    boundary_rule = body.get("boundary_rule", "count_all")
    filtered_summary = body.get("filtered_summary", {})

    sessions: SessionStore = app.state.sessions
    # Find the image path
    image_path = ""
    for f in IMAGES_DIR.glob(f"{image_id}.*"):
        image_path = str(f)
        break

    sample = sessions.add_sample(
        image_id, image_path, detections, summary, conf_threshold,
        grid_info=grid_info, boundary_rule=boundary_rule,
        filtered_summary=filtered_summary,
    )
    # Return group info alongside sample info
    group = sessions.current.active_group()
    return {
        "sample_id": sample.sample_id,
        "image_id": sample.image_id,
        "effective_summary": sample.effective_summary,
        "grid_detected": grid_info.get("detected", False),
        "group_id": group.group_id if group else None,
        "group_name": group.name if group else None,
        "group_summary": group.aggregate_summary if group else None,
    }


@app.patch("/api/session/sample/{image_id}")
async def update_sample(image_id: str, request: Request):
    """Update an existing saved sample (re-save after editing)."""
    body = await request.json()
    sessions: SessionStore = app.state.sessions
    result = sessions.update_sample(
        image_id,
        detections=body.get("detections"),
        summary=body.get("summary"),
        grid_info=body.get("grid_info"),
        filtered_summary=body.get("filtered_summary"),
        additions=body.get("additions"),
        removals=body.get("removals"),
        conf_threshold=body.get("conf_threshold"),
    )
    if result is None:
        raise HTTPException(404, "Sample not found")
    return {"effective_summary": result}


@app.post("/api/session/group/new")
async def new_sample_group(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    name = body.get("name") if body else None
    sessions: SessionStore = app.state.sessions
    group = sessions.new_sample_group(name)
    return {
        "group_id": group.group_id,
        "name": group.name,
        "active_group_id": sessions.current.active_group_id,
    }


@app.post("/api/session/group/rename")
async def rename_sample_group(request: Request):
    body = await request.json()
    group_id = body.get("group_id")
    name = body.get("name", "")
    sessions: SessionStore = app.state.sessions
    group = sessions.rename_group(group_id, name)
    if not group:
        raise HTTPException(404, "Sample group not found")
    return {"group_id": group.group_id, "name": group.name}


@app.get("/api/session/samples")
async def list_samples():
    sessions: SessionStore = app.state.sessions
    return sessions.current.to_dict().get("sample_groups", [])


@app.get("/api/session/export")
async def export_csv():
    sessions: SessionStore = app.state.sessions
    csv_content = sessions.export_csv()
    safe_name = _sanitize_filename(sessions.current.experiment_name) or "export"
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.csv"'
        },
    )


@app.get("/api/session/export/zip")
async def export_zip():
    """Export all session images + CSV + annotated images as a ZIP archive."""
    sessions: SessionStore = app.state.sessions
    session = sessions.current
    if not session.samples:
        raise HTTPException(400, "No samples to export")

    pipeline: AnalysisPipeline = app.state.pipeline

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add CSV
        csv_content = sessions.export_csv()
        zf.writestr(f"{session.experiment_name}.csv", csv_content)

        # Add original images + annotated overlays
        for sample in session.samples:
            image_id = sample.image_id
            image_path = None
            for f in IMAGES_DIR.glob(f"{image_id}.*"):
                zf.write(str(f), f"images/{f.name}")
                image_path = str(f)
                break

            # Render annotated overlay image
            if image_path:
                image = cv2.imread(image_path)
                if image is not None:
                    # Build a result dict matching render_full_overlay's expected format
                    grid = sample.grid_info or {}
                    removals_set = set(sample.manual_removals or [])

                    # Split detections into inside/excluded using grid
                    inside = []
                    excluded = []
                    if grid.get("detected") and grid.get("boundary"):
                        from pipeline import CellFilter
                        from grid_detection import GridResult
                        gr = GridResult(
                            detected=True,
                            boundary=tuple(grid["boundary"][:4]) if grid.get("boundary") else None,
                            pixels_per_mm=grid.get("pixels_per_mm"),
                            horizontal_lines=grid.get("horizontal_lines", []),
                            vertical_lines=grid.get("vertical_lines", []),
                            rotation_deg=grid.get("rotation_deg", 0.0),
                            confidence=grid.get("confidence", 0.0),
                            grid_center=tuple(grid["grid_center"]) if grid.get("grid_center") else None,
                            grid_size=tuple(grid["grid_size"]) if grid.get("grid_size") else None,
                        )
                        inside, excluded = CellFilter.filter_detections(
                            sample.detections, gr, rule=sample.boundary_rule,
                        )
                    else:
                        inside = list(sample.detections)

                    # Remove manually-removed detections from inside
                    active_inside = [d for d in inside if d.get("id") not in removals_set]

                    result_dict = {
                        "grid": grid,
                        "filtered": {
                            "detections": active_inside,
                            "excluded": excluded,
                        },
                    }
                    overlay = pipeline.render_full_overlay(image, result_dict)

                    # Draw manual additions as circles
                    for ann in (sample.manual_additions or []):
                        cx, cy = int(ann["x"]), int(ann["y"])
                        color = (142, 199, 72) if ann.get("class", 0) == 0 else (68, 68, 239)
                        cv2.circle(overlay, (cx, cy), 12, color, 2)
                        cv2.line(overlay, (cx - 6, cy), (cx + 6, cy), color, 2)
                        cv2.line(overlay, (cx, cy - 6), (cx, cy + 6), color, 2)

                    # Draw removed detections with strikethrough
                    for det in inside:
                        if det.get("id") in removals_set:
                            bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
                            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (100, 100, 100), 1)
                            cv2.line(overlay, (bx1, by1), (bx2, by2), (100, 100, 100), 1)

                    _, img_encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    zf.writestr(f"annotated/{image_id}.jpg", img_encoded.tobytes())

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{_sanitize_filename(session.experiment_name) or "export"}.zip"'
        },
    )


# ── Grid Detection ──────────────────────────────────────────────────────

@app.post("/api/grid/detect")
async def detect_grid(image_id: str):
    """Run grid detection only (no inference). Useful for calibration and debugging."""
    filepath = _find_image(image_id)
    pipeline: AnalysisPipeline = app.state.pipeline
    grid_result = pipeline.detect_grid_only(str(filepath))
    return grid_result.to_dict()


@app.post("/api/grid/manual")
async def manual_grid(request: Request):
    """Accept user-defined grid boundary via 3 corner points.

    Body JSON: {"points": [[x1,y1],[x2,y2],[x3,y3]]}
    The 3 points are consecutive corners of the 1mm x 1mm hemocytometer square.
    The 4th corner is computed as P4 = P1 + P3 - P2.
    """
    import math
    from grid_detection import GridResult

    body = await request.json()
    points = body.get("points", [])
    if len(points) != 3:
        raise HTTPException(400, "Exactly 3 points required")

    p1 = points[0]  # [x, y]
    p2 = points[1]
    p3 = points[2]
    # P4 = P1 + P3 - P2 (completes the parallelogram)
    p4 = [p1[0] + p3[0] - p2[0], p1[1] + p3[1] - p2[1]]

    # Compute rotation from edge P1->P2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rotation_deg = math.degrees(math.atan2(dy, dx))

    # Compute edge lengths (both edges of the square)
    edge1 = math.sqrt(dx * dx + dy * dy)
    dx2 = p3[0] - p2[0]
    dy2 = p3[1] - p2[1]
    edge2 = math.sqrt(dx2 * dx2 + dy2 * dy2)

    # pixels_per_mm from average edge length (1mm square)
    avg_edge = (edge1 + edge2) / 2
    pixels_per_mm = avg_edge / 1.0

    # Center is midpoint of diagonal P1-P3 (or P2-P4)
    cx = (p1[0] + p3[0]) / 2
    cy = (p1[1] + p3[1]) / 2

    # Axis-aligned bounding box of all 4 corners
    all_x = [p1[0], p2[0], p3[0], p4[0]]
    all_y = [p1[1], p2[1], p3[1], p4[1]]
    bbox = (int(min(all_x)), int(min(all_y)), int(max(all_x)), int(max(all_y)))

    grid_result = GridResult(
        detected=True,
        boundary=bbox,
        pixels_per_mm=pixels_per_mm,
        horizontal_lines=[],
        vertical_lines=[],
        rotation_deg=rotation_deg,
        confidence=1.0,  # user-defined = trusted
        grid_center=(cx, cy),
        grid_size=(edge1, edge2),
    )

    # Update session calibration
    sessions: SessionStore = app.state.sessions
    sessions.update_calibration({
        "pixels_per_mm": round(pixels_per_mm, 2),
        "pixels_per_mm_source": "manual_selection",
    })

    return {
        "grid": grid_result.to_dict(),
        "calibration": sessions.current.calibration,
    }


@app.post("/api/grid/overlay/{image_id}")
async def grid_overlay(image_id: str):
    """Return image with detected grid lines drawn for visual verification."""
    filepath = _find_image(image_id)
    image = cv2.imread(str(filepath))
    if image is None:
        raise HTTPException(500, "Failed to read image")

    grid_detector: GridDetector = app.state.grid_detector
    grid_result = grid_detector.detect(image)
    overlay = grid_detector.render_grid_overlay(image, grid_result)

    _, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return StreamingResponse(
        iter([jpeg.tobytes()]),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"attachment; filename=grid_{image_id}.jpg"},
    )


# ── Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
