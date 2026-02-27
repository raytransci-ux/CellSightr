"""
Build a distributable CellSightr package.

Creates a ZIP file containing everything needed to run the app on another
machine — webapp code, frontend, model weights, and launch scripts.

Excludes: training data, raw images, .md prompt files, .venv, runs/, scripts/,
label_studio/, ml_backend/, notebooks/, training/, models/, inference/,
pre-trained base weights (yolo*.pt in root), and other dev-only files.

Usage:
    python build_package.py
    python build_package.py --output cellsightr_v1.zip
"""

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Files/dirs to include (relative to PROJECT_ROOT)
INCLUDE = [
    # Backend
    "webapp/backend/main.py",
    "webapp/backend/inference.py",
    "webapp/backend/pipeline.py",
    "webapp/backend/grid_detection.py",
    "webapp/backend/calibration.py",
    "webapp/backend/camera.py",
    "webapp/backend/session.py",
    "webapp/backend/requirements.txt",
    # Frontend (entire directory)
    "webapp/frontend/",
    # Launch scripts & docs
    "start.bat",
    "start.ps1",
    "start.sh",
    "CLAUDE.md",
    # Model weights (trained checkpoints only — best.pt for each size)
    "checkpoints/yolo/nano/weights/best.pt",
    "checkpoints/yolo/medium/weights/best.pt",
    "checkpoints/yolo/large/weights/best.pt",
]

# Default output filename
DEFAULT_OUTPUT = "CellSightr_package.zip"


def collect_files():
    """Resolve INCLUDE entries to a list of (src_path, archive_name) tuples."""
    files = []
    for entry in INCLUDE:
        src = PROJECT_ROOT / entry
        if src.is_file():
            files.append((src, entry))
        elif src.is_dir():
            for f in src.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(PROJECT_ROOT)
                    files.append((f, str(rel).replace("\\", "/")))
        else:
            print(f"  SKIP (not found): {entry}")
    return files


def build(output_path: str):
    print(f"Building CellSightr package...")
    print(f"  Source: {PROJECT_ROOT}")

    files = collect_files()

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files:
            # Nest under CellSightr/ inside the zip
            zf.write(str(src), f"CellSightr/{arcname}")
            size_kb = src.stat().st_size / 1024
            label = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
            print(f"  + {arcname}  ({label})")

    pkg_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nPackage created: {output_path} ({pkg_size:.1f} MB)")
    print(f"  Contains {len(files)} files")
    print(f"\nTo deploy:")
    print(f"  1. Extract the ZIP on the target machine")
    print(f"  2. Install Python 3.10+")
    print(f"  3. Run start.bat (Windows) or ./start.sh (Linux/macOS)")
    print(f"  4. Open http://localhost:8000 in a browser")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CellSightr deployment package")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output ZIP path")
    args = parser.parse_args()
    build(args.output)
