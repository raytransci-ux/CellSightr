# Phase 4: Web Application Development

**Timeline**: Weeks 9-10 (14 days)  
**Priority**: Medium-High - User interface for model deployment  
**Estimated Effort**: 30-35 hours

## Objectives
1. Build FastAPI backend for model serving
2. Create simple, intuitive web interface
3. Implement image upload and processing
4. Display results with visualizations
5. Add export functionality (CSV, annotated images)
6. Deploy locally and create containerized version

## Deliverables
- [ ] FastAPI REST API with model endpoints
- [ ] Web interface (HTML/CSS/JS)
- [ ] Image upload and batch processing
- [ ] Results visualization and export
- [ ] Dockerized application
- [ ] User documentation

---

## Architecture Overview

```
┌─────────────────┐
│   Frontend      │  HTML + Vanilla JS + Tailwind CSS
│   (Browser)     │  - Upload interface
│                 │  - Results display
│                 │  - Image annotation viewer
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│   Backend       │  FastAPI (Python)
│   (Server)      │  - Model inference
│                 │  - Image processing
│                 │  - Result formatting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Models     │  PyTorch
│   (Inference)   │  - Cell detector
│                 │  - Viability classifier
│                 │  - Grid detector
└─────────────────┘
```

---

## Task Breakdown

### Task 4.1: Backend Setup (Day 1-2)
**Goal**: Create FastAPI server with model loading

**Project Structure**:
```
webapp/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── models_loader.py     # Load PyTorch models
│   ├── inference.py         # Inference logic
│   ├── utils.py             # Helper functions
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── uploads/                 # Temporary upload storage
├── results/                 # Processed results
└── Dockerfile
```

**Install Dependencies**:
```bash
# backend/requirements.txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
Pillow==10.2.0
opencv-python-headless==4.9.0
torch==2.1.2
torchvision==0.16.2
numpy==1.26.3
pandas==2.2.0
```

**FastAPI Application**: `backend/main.py`
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import uuid
from PIL import Image
import torch

from models_loader import load_models
from inference import CellCountingPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Hemocytometer Cell Counter API",
    description="Automated cell counting and viability assessment",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global model instances (loaded once at startup)
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Load models when server starts"""
    global pipeline
    print("Loading models...")
    
    pipeline = load_models(
        detector_path='../../checkpoints/detector/detector_best.pth',
        classifier_path='../../checkpoints/classifier/classifier_best.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("Models loaded successfully!")

@app.get("/")
async def root():
    """Serve frontend"""
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": pipeline is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    save_annotated: bool = True
):
    """
    Analyze hemocytometer image
    
    Args:
        file: Uploaded image file
        confidence_threshold: Minimum detection confidence (0-1)
        save_annotated: Whether to save annotated result image
    
    Returns:
        JSON with cell counts, viability, and optional image URL
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    # Generate unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    upload_path = os.path.join(upload_dir, f"{analysis_id}_{file.filename}")
    
    with open(upload_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # Run inference
        results = pipeline.process_image(
            upload_path,
            confidence_threshold=confidence_threshold
        )
        
        # Save annotated image if requested
        annotated_path = None
        if save_annotated:
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            annotated_path = os.path.join(results_dir, f"{analysis_id}_annotated.jpg")
            
            pipeline.visualize_results(
                upload_path,
                results,
                save_path=annotated_path
            )
        
        # Format response
        response = {
            "analysis_id": analysis_id,
            "filename": file.filename,
            "results": {
                "total_cells": int(results['total_cells']),
                "viable_cells": int(results['viable_cells']),
                "non_viable_cells": int(results['non_viable_cells']),
                "viability_percent": float(results['viability_percent']),
                "concentration_per_ml": float(results['concentration_per_ml']) if results['concentration_per_ml'] else None,
            },
            "detection_details": {
                "num_detections": len(results['boxes']),
                "confidence_threshold": confidence_threshold,
                "grid_detected": results['grid_bbox'] is not None,
            },
            "annotated_image_url": f"/api/results/{analysis_id}_annotated.jpg" if save_annotated else None
        }
        
        return JSONResponse(response)
    
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    
    finally:
        # Optional: cleanup uploaded file
        # os.remove(upload_path)
        pass

@app.post("/api/batch-analyze")
async def batch_analyze(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = 0.5
):
    """
    Analyze multiple images in batch
    """
    results = []
    
    for file in files:
        try:
            result = await analyze_image(file, confidence_threshold, save_annotated=False)
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"batch_results": results}

@app.get("/api/results/{filename}")
async def get_result_image(filename: str):
    """Serve annotated result images"""
    file_path = os.path.join("results", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, "Result not found")
    
    return FileResponse(file_path)

@app.post("/api/export-csv")
async def export_results_csv(analysis_ids: List[str]):
    """
    Export multiple analysis results to CSV
    """
    import pandas as pd
    import io
    from fastapi.responses import StreamingResponse
    
    # Load results (would need to cache these in production)
    # For now, return example
    data = []
    for aid in analysis_ids:
        # Load from database or cache
        data.append({
            'analysis_id': aid,
            'filename': f'{aid}.jpg',
            'total_cells': 25,
            'viable_cells': 20,
            'non_viable_cells': 5,
            'viability_percent': 80.0
        })
    
    df = pd.DataFrame(data)
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cell_counts.csv"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Model Loader**: `backend/models_loader.py`
```python
import torch
import sys
sys.path.append('../..')

from models.detector import CellDetector
from models.classifier import ViabilityClassifier
from models.grid_detector import GridDetector
from inference.pipeline import CellCountingPipeline

def load_models(detector_path, classifier_path, device='cuda'):
    """
    Load all models and create pipeline
    """
    # Load detector
    detector = CellDetector(num_classes=3)
    detector.load(detector_path)
    detector.to(device)
    detector.eval_mode()
    
    # Load classifier
    classifier = ViabilityClassifier()
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    
    # Initialize grid detector
    grid_detector = GridDetector()
    
    # Create pipeline
    pipeline = CellCountingPipeline(
        detector=detector,
        classifier=classifier,
        grid_detector=grid_detector,
        device=device
    )
    
    return pipeline
```

**Run Server**:
```bash
cd webapp/backend
python main.py

# Server will start at http://localhost:8000
```

**Deliverable**: Working FastAPI backend with model inference

---

### Task 4.2: Frontend Interface (Day 3-5)
**Goal**: Create user-friendly web interface

**HTML Structure**: `frontend/index.html`
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hemocytometer Cell Counter</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="styles.css">
</head>
<body class="bg-gray-50">
    <!-- Header -->
    <header class="bg-blue-600 text-white p-6 shadow-lg">
        <div class="container mx-auto">
            <h1 class="text-3xl font-bold">🔬 Cell Counter</h1>
            <p class="text-blue-100 mt-2">Automated hemocytometer analysis</p>
        </div>
    </header>

    <!-- Main Container -->
    <main class="container mx-auto p-6 max-w-6xl">
        
        <!-- Upload Section -->
        <section class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-2xl font-semibold mb-4">Upload Image</h2>
            
            <!-- Drop Zone -->
            <div id="dropzone" class="border-4 border-dashed border-gray-300 rounded-lg p-12 text-center cursor-pointer hover:border-blue-500 transition">
                <svg class="mx-auto h-16 w-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
                <p class="text-lg text-gray-600 mb-2">Drop image here or click to select</p>
                <p class="text-sm text-gray-500">Supports JPG, PNG (max 10MB)</p>
                <input type="file" id="fileInput" accept="image/*" class="hidden">
            </div>

            <!-- Settings -->
            <div class="mt-4 grid grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-2">
                        Confidence Threshold
                    </label>
                    <input type="range" id="confidenceSlider" min="0" max="1" step="0.05" value="0.5" 
                           class="w-full">
                    <span id="confidenceValue" class="text-sm text-gray-600">0.50</span>
                </div>
                
                <div class="flex items-end">
                    <button id="analyzeBtn" disabled
                            class="w-full bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition">
                        Analyze Image
                    </button>
                </div>
            </div>

            <!-- Preview -->
            <div id="imagePreview" class="mt-6 hidden">
                <img id="previewImg" class="max-w-full rounded-lg shadow-md mx-auto">
            </div>
        </section>

        <!-- Loading Indicator -->
        <div id="loading" class="hidden bg-white rounded-lg shadow-md p-8 text-center">
            <div class="inline-block animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
            <p class="mt-4 text-gray-600">Analyzing image...</p>
        </div>

        <!-- Results Section -->
        <section id="results" class="hidden bg-white rounded-lg shadow-md p-6">
            <h2 class="text-2xl font-semibold mb-6">Analysis Results</h2>
            
            <!-- Summary Cards -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div class="bg-blue-50 p-4 rounded-lg">
                    <p class="text-sm text-gray-600 mb-1">Total Cells</p>
                    <p id="totalCells" class="text-3xl font-bold text-blue-600">-</p>
                </div>
                <div class="bg-green-50 p-4 rounded-lg">
                    <p class="text-sm text-gray-600 mb-1">Viable Cells</p>
                    <p id="viableCells" class="text-3xl font-bold text-green-600">-</p>
                </div>
                <div class="bg-red-50 p-4 rounded-lg">
                    <p class="text-sm text-gray-600 mb-1">Non-viable</p>
                    <p id="nonViableCells" class="text-3xl font-bold text-red-600">-</p>
                </div>
                <div class="bg-purple-50 p-4 rounded-lg">
                    <p class="text-sm text-gray-600 mb-1">Viability</p>
                    <p id="viability" class="text-3xl font-bold text-purple-600">-</p>
                </div>
            </div>

            <!-- Concentration -->
            <div class="bg-gray-50 p-4 rounded-lg mb-6">
                <p class="text-sm text-gray-600 mb-1">Concentration</p>
                <p id="concentration" class="text-2xl font-semibold text-gray-800">-</p>
            </div>

            <!-- Annotated Image -->
            <div class="mb-6">
                <h3 class="text-lg font-semibold mb-3">Annotated Image</h3>
                <img id="annotatedImg" class="max-w-full rounded-lg shadow-md mx-auto">
            </div>

            <!-- Actions -->
            <div class="flex gap-4">
                <button id="downloadImage" class="flex-1 bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition">
                    📥 Download Image
                </button>
                <button id="exportCSV" class="flex-1 bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition">
                    📊 Export CSV
                </button>
                <button id="newAnalysis" class="flex-1 bg-gray-600 text-white px-6 py-3 rounded-lg hover:bg-gray-700 transition">
                    🔄 New Analysis
                </button>
            </div>
        </section>

    </main>

    <script src="script.js"></script>
</body>
</html>
```

**JavaScript**: `frontend/script.js`
```javascript
// DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

let selectedFile = null;
let currentAnalysisId = null;

// File selection handlers
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('border-blue-500');
});
dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('border-blue-500');
});
dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('border-blue-500');
    handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        imagePreview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
    
    // Enable analyze button
    analyzeBtn.disabled = false;
}

// Confidence slider
confidenceSlider.addEventListener('input', (e) => {
    confidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Show loading
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    
    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('confidence_threshold', confidenceSlider.value);
    formData.append('save_annotated', 'true');
    
    try {
        // Call API
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Analysis failed');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        loading.classList.add('hidden');
    }
});

function displayResults(data) {
    currentAnalysisId = data.analysis_id;
    
    // Update result cards
    document.getElementById('totalCells').textContent = data.results.total_cells;
    document.getElementById('viableCells').textContent = data.results.viable_cells;
    document.getElementById('nonViableCells').textContent = data.results.non_viable_cells;
    document.getElementById('viability').textContent = 
        data.results.viability_percent.toFixed(1) + '%';
    
    // Update concentration
    const concText = data.results.concentration_per_ml 
        ? `${(data.results.concentration_per_ml / 1e6).toFixed(2)} × 10⁶ cells/mL`
        : 'Grid not detected';
    document.getElementById('concentration').textContent = concText;
    
    // Show annotated image
    if (data.annotated_image_url) {
        document.getElementById('annotatedImg').src = data.annotated_image_url;
    }
    
    // Show results section
    results.classList.remove('hidden');
}

// Download annotated image
document.getElementById('downloadImage').addEventListener('click', () => {
    if (!currentAnalysisId) return;
    
    const link = document.createElement('a');
    link.href = `/api/results/${currentAnalysisId}_annotated.jpg`;
    link.download = `cell_count_${currentAnalysisId}.jpg`;
    link.click();
});

// Export CSV (placeholder - would need backend implementation)
document.getElementById('exportCSV').addEventListener('click', async () => {
    if (!currentAnalysisId) return;
    
    // For now, create simple CSV client-side
    const data = {
        analysis_id: currentAnalysisId,
        total_cells: document.getElementById('totalCells').textContent,
        viable_cells: document.getElementById('viableCells').textContent,
        non_viable_cells: document.getElementById('nonViableCells').textContent,
        viability_percent: document.getElementById('viability').textContent
    };
    
    const csv = [
        Object.keys(data).join(','),
        Object.values(data).join(',')
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `cell_count_${currentAnalysisId}.csv`;
    link.click();
});

// New analysis
document.getElementById('newAnalysis').addEventListener('click', () => {
    selectedFile = null;
    currentAnalysisId = null;
    fileInput.value = '';
    imagePreview.classList.add('hidden');
    results.classList.add('hidden');
    analyzeBtn.disabled = true;
    confidenceSlider.value = 0.5;
    confidenceValue.textContent = '0.50';
});
```

**Deliverable**: Working web interface

---

### Task 4.3: Testing & Refinement (Day 6-7)
**Goal**: Test with real users and refine UX

**Test Cases**:
1. Upload single image → verify results
2. Try different confidence thresholds
3. Download annotated image
4. Export CSV
5. Error handling (invalid files, server errors)
6. Mobile responsiveness
7. Multiple browsers (Chrome, Firefox, Safari)

**User Feedback Points**:
- Is the interface intuitive?
- Are results clear and useful?
- Any confusing aspects?
- Performance acceptable?

**Deliverable**: Tested, polished web app

---

### Task 4.4: Dockerization (Day 8-9)
**Goal**: Containerize application for easy deployment

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY checkpoints/ ./checkpoints/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  cellcounter:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./results:/app/results
    environment:
      - DEVICE=cpu  # Change to cuda if GPU available
    restart: unless-stopped
```

**Build and Run**:
```bash
# Build image
docker build -t cell-counter .

# Run container
docker run -p 8000:8000 cell-counter

# Or use docker-compose
docker-compose up -d
```

**Deliverable**: Dockerized application

---

### Task 4.5: Documentation (Day 10)
**Goal**: Create user guide and API docs

**User Guide**: `docs/USER_GUIDE.md`
```markdown
# Cell Counter User Guide

## Getting Started

1. **Access the Application**
   - Open web browser
   - Navigate to http://localhost:8000

2. **Upload Image**
   - Drag and drop hemocytometer image
   - Or click to select file from computer

3. **Adjust Settings** (Optional)
   - Set confidence threshold (0-1)
   - Default 0.5 works well for most images

4. **Analyze**
   - Click "Analyze Image" button
   - Wait for processing (typically 5-10 seconds)

5. **View Results**
   - Total cell count
   - Viable vs. non-viable breakdown
   - Viability percentage
   - Concentration (cells/mL)
   - Annotated image showing detections

6. **Export**
   - Download annotated image
   - Export results to CSV

## Tips for Best Results

- Use well-focused, well-lit images
- Ensure hemocytometer grid is visible
- 4x magnification recommended
- Avoid excessive cell overlap
- Clean camera lens and hemocytometer

## Troubleshooting

**No cells detected**
- Try lowering confidence threshold
- Check image quality and focus
- Ensure cells are visible to human eye

**Inaccurate counts**
- Very high density may cause undercounting
- Manual verification recommended for critical applications

**Grid not detected**
- Ensure grid lines are visible
- Concentration calculation will not be available
```

**API Documentation**: Auto-generated by FastAPI at `/docs`

**Deliverable**: Complete documentation

---

## Phase 4 Success Criteria
- [ ] Backend API functioning correctly
- [ ] Frontend interface intuitive and responsive
- [ ] Image upload and processing working
- [ ] Results display accurately
- [ ] Export functionality operational
- [ ] Application containerized
- [ ] Documentation complete

## Next Phase Preview
Phase 5 will integrate microscope cameras:
- Camera driver integration
- Real-time image capture
- Live analysis mode
- Automatic focus and lighting adjustment
