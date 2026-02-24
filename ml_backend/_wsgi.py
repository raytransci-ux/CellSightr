"""
ML Backend entry point — run via scripts/start_ml_backend.py
"""
import os
import sys

# Ensure the ml_backend directory is on the path so model.py can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from label_studio_ml.api import init_app
from model import CellDetectorBackend

app = init_app(model_class=CellDetectorBackend)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090, debug=False)
