# CellCount - Hemocytometer Cell Counter

## What This Is
A web app for automated hemocytometer cell counting using YOLO object detection. Captures microscope images (live camera or uploaded), detects viable and non-viable cells, computes concentrations using standard hemocytometer math, and exports results.

## Architecture
- **Backend**: FastAPI (Python) — `webapp/backend/`
- **Frontend**: Vanilla JS single-page app — `webapp/frontend/`
- **Detection model**: Ultralytics YOLOv11 (nano/medium/large checkpoints)
- **Camera**: USB microscope via OpenCV, streamed over WebSocket

## Running
```bash
# From project root:
start.bat        # Windows
./start.sh       # Linux/macOS
# Opens at http://localhost:8000
```
The launch scripts create a `.venv`, install deps, and start uvicorn.

## Backend Files (`webapp/backend/`)
| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — all REST/WebSocket endpoints, static file serving |
| `inference.py` | YOLO model loading and prediction (returns bounding boxes + classes) |
| `pipeline.py` | Full analysis pipeline: inference + grid detection + calibration |
| `grid_detection.py` | Detects hemocytometer grid lines via Hough transform, finds 4x4 square regions |
| `calibration.py` | Concentration math: cells/mL from counts, dilution factor, grid geometry |
| `camera.py` | OpenCV camera capture, JPEG streaming via WebSocket |
| `session.py` | Session persistence (JSON): samples, sample groups, experiment metadata |

## Frontend Files (`webapp/frontend/js/`)
| File | Purpose |
|------|---------|
| `app.js` | Main controller — state management, UI updates, API calls, 3-point manual grid selection, capture→stop→save→resume workflow |
| `annotator.js` | Canvas overlay for drawing/editing cell detections (click to add, shift+click to remove) |
| `camera.js` | WebSocket client for live camera feed with live tracking command support |
| `session.js` | Client-side session management UI |
| `export.js` | CSV/ZIP export with sample group support |
| `charts.js` | Viability donut chart |
| `shortcuts.js` | Keyboard shortcuts (Space=capture, A=annotate, S=save, etc.) |

## Key Endpoints
- `POST /api/analyze` — Upload image, run YOLO + grid detection, return detections & counts
- `POST /api/grid/manual` — Set manual 3-point grid (top-left, top-right, bottom-left corners)
- `GET /ws/camera` — WebSocket: live JPEG frames, supports `{"live_tracking": true/false}` text commands for real-time cell overlay
- `GET /api/camera/devices` — List available camera devices on the system
- `POST /api/camera/start?device_id=N` — Start specific camera device
- `POST /api/camera/stop` — Stop camera
- `POST /api/calibration` — Set pixels_per_mm, dilution_factor, squares_counted
- `GET/POST /api/session/*` — Session CRUD, sample groups
- `GET /api/session/export`, `/api/session/export/zip` — Export results

## Detection Classes
- Class 0 = **viable** (live) cells
- Class 1 = **non-viable** (dead) cells

## Model Weights
Trained checkpoints live in `checkpoints/yolo/{nano,medium,large}/weights/best.pt`. The app defaults to nano but can switch at runtime. Pre-trained base weights (`yolo*.pt` in root) are for training only.

## Security Notes
- Path traversal protection via `_safe_path()` in `main.py`
- Upload size limits: 50 MB images, 200 MB models
- CORS restricted to localhost origins
- Filenames sanitized in Content-Disposition headers and model uploads
- Calibration inputs validated (positive values, dilution >= 1)
- Frontend uses `textContent` (not `innerHTML`) for server-sourced data

## Packaging
```bash
python build_package.py                    # creates CellCount_package.zip
python build_package.py -o my_build.zip    # custom output name
```
Bundles webapp code, frontend, model weights, and launch scripts (~93 MB). Excludes training data, notebooks, .md files, and dev tooling.

## Camera & Capture Workflow
- User selects camera from device dropdown, clicks Start Camera
- **Live tracking** toggle runs nano inference every 5th frame (~3 fps) and overlays boxes on stream
- **Capture** freezes the frame, stops the camera, saves the image, and auto-runs analysis
- After annotation edits and **Save**, the camera auto-restarts if it was previously live
- **Stop** (manual) shows a black screen; the placeholder only shows before first camera use

## Repository Structure
- `webapp/` — Production app (backend + frontend)
- `docs/` — Training guides, annotation guidelines, Label Studio config, training scripts
- `checkpoints/` — Trained model weights (git-ignored, distributed via `build_package.py`)
- `data/` — Training data and annotations (git-ignored)

## Dev Notes
- Python 3.10+ required
- `requirements.txt` is at `webapp/backend/requirements.txt`
- Sessions are stored as JSON in `webapp/backend/sessions/`
- Uploaded images go to `webapp/uploads/images/`; uploaded models to `webapp/uploads/models/`
- Grid detection can be auto (Hough lines) or manual (3-point selection)
- The confidence threshold is adjustable at runtime (default 0.25)
- CSS has responsive breakpoints at 800px height and 1200px/900px width for laptop screens
- Training scripts and phase documentation are in `docs/`
