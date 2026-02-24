"""
Label Studio Startup Script
============================
Starts Label Studio with local file serving enabled so you can reference
images directly from disk without uploading them.

Usage (from project root, with venv active):
    python scripts/start_labelstudio.py
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT    = PROJECT_ROOT / "data" / "raw"

# Enable Label Studio to serve local files from DATA_ROOT
os.environ["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"]   = str(DATA_ROOT)

print(f"Local file serving: ENABLED")
print(f"Serving files from: {DATA_ROOT}")
print(f"Opening:            http://localhost:8080\n")

label_studio_exe = Path(sys.executable).parent / "label-studio.exe"

subprocess.run(
    [str(label_studio_exe), "start"],
    check=True
)
