# CellSightr

Counting cells with a hemocytometer is tedious, operator dependent, and takes valuable time away from real research! Automated approaches are gated behind a paywall and the plastic waste generated builds up quickly. CellSightr is my foray into machine learning to create a fast, repeatable app for cell counting using YOLO inference to count cells in real time, adapting to a common academia/industry workflow. CellSightr also allows exports of images and CSV files for the cell counts, giving an audit trail for GMP environments.

ML-powered hemocytometer cell counter for automated viability analysis. Upload a microscope image or connect a live camera — CellSightr detects viable and non-viable cells using YOLO object detection, counts them within an automatically detected hemocytometer grid, and calculates concentration in cells/mL.

## Features

- **Live camera feed** with real-time cell detection overlay
- **Automatic grid detection** identifies the hemocytometer counting square via Hough transform
- **Dual-model inference** — fast nano model for live preview, accurate large model for final counts
- **Manual annotation** — add, remove, or reclassify cells with click controls
- **Session management** — organize samples into groups, track across experiments
- **Export** — CSV and ZIP with annotated overlay images
- **Camera selection** — choose between multiple connected cameras (webcam, microscope)
- **Keyboard-driven workflow** — Space to capture, S to save, A to annotate, Tab to cycle samples

## Quick Start

**Requirements:** Python 3.10+, a modern browser

```bash
# Clone and run
git clone https://github.com/raytransci-ux/CellSightr.git
cd CellSightr

# Windows
start.bat

# Linux/macOS
chmod +x start.sh && ./start.sh
```

The launch script creates a virtual environment, installs dependencies, and starts the server at **http://localhost:8000**.

> **Note:** Trained model weights (`checkpoints/`) are not included in the git repo due to size. Use `build_package.py` to create a deployable ZIP with weights included, or train your own with the scripts in `docs/scripts/`.

## How It Works

1. **Capture** an image from a connected microscope camera or upload one
2. **YOLO detection** identifies cells and classifies them as viable (green) or non-viable (red)
3. **Grid detection** finds the 1mm hemocytometer square and filters cells inside the counting region
4. **Concentration** is calculated using standard hemocytometer math: `cells/mL = count x dilution x 10,000`
5. **Review & annotate** — manually correct any missed or misclassified cells
6. **Save & export** — results persist in the session and export as CSV

## Architecture

```
webapp/
  backend/          FastAPI server (Python)
    main.py           REST + WebSocket endpoints
    inference.py      YOLO model wrapper (nano/medium/large)
    pipeline.py       Analysis orchestrator
    grid_detection.py Hough transform grid finder
    calibration.py    Hemocytometer math
    camera.py         Multi-backend camera abstraction
    session.py        JSON session persistence

  frontend/         Vanilla JS single-page app
    index.html        App layout
    js/app.js         Main controller & state
    js/camera.js      WebSocket camera client
    js/annotator.js   Click-to-annotate overlay
    css/styles.css    Dark theme, responsive layout
```

## Detection Classes

| Class | Label | Color | Description |
|-------|-------|-------|-------------|
| 0 | Viable | Green | Live cells (unstained or lightly stained) |
| 1 | Non-viable | Red | Dead cells (trypan blue stained) |

## Training Your Own Model

Training scripts and detailed phase guides are in `docs/`:

```bash
# 1. Annotate images with Label Studio
python docs/scripts/setup_labelstudio.py

# 2. Convert annotations to YOLO format
python docs/scripts/prepare_yolo_dataset.py

# 3. Train nano + medium models
python docs/scripts/train_yolo.py
```

See [docs/PHASE_2_MODEL_TRAINING.md](docs/PHASE_2_MODEL_TRAINING.md) for training configuration details.

## Deployment

```bash
# Build a portable ZIP with code + model weights
python build_package.py

# Creates CellCount_package.zip (~85 MB)
# Extract on target machine, run start.bat
```

## Tech Stack

- **Backend:** FastAPI, Uvicorn, OpenCV, NumPy
- **ML:** Ultralytics YOLOv11, PyTorch
- **Frontend:** Vanilla JavaScript, Chart.js, HTML Canvas
- **Camera:** OpenCV (DirectShow/UVC), Micro-Manager (optional)

## License

MIT
