"""
Start the YOLOv8 ML Backend for Label Studio
=============================================
Run this in a SECOND terminal (keep Label Studio running in its own terminal):

    .venv\\Scripts\\activate
    python scripts/start_ml_backend.py

Then in Label Studio:
    Settings → Model → Add Model
    URL: http://localhost:9090
    Enable: "Use for interactive preannotations"
    Click Save, then go back to annotating.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR  = PROJECT_ROOT / "ml_backend"
MODEL_PATH   = PROJECT_ROOT / "checkpoints" / "yolo" / "run" / "weights" / "best.pt"
PORT         = 9090

if not MODEL_PATH.exists():
    print("ERROR: No trained model found.")
    print(f"Expected: {MODEL_PATH}")
    print("\nRun these first:")
    print("  1. python scripts/prepare_yolo_dataset.py")
    print("  2. python scripts/train_yolo.py")
    sys.exit(1)

print(f"Starting ML backend on http://localhost:{PORT}")
print(f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
print("\nIn Label Studio: Settings → Model → Add Model → http://localhost:9090")
print("Press Ctrl+C to stop.\n")

wsgi = BACKEND_DIR / "_wsgi.py"

subprocess.run([sys.executable, str(wsgi)], check=True)
