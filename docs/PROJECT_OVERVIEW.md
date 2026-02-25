# Hemocytometer Cell Counter - AI Project Plan

## Project Overview
Build an automated cell counting and viability classification system for hemocytometer images using computer vision and deep learning.

## Core Features
1. **Cell Detection & Counting** - Identify and count individual cells in hemocytometer images
2. **Viability Classification** - Distinguish live vs. dead cells (e.g., trypan blue staining)
3. **Grid Detection** - Automatically identify hemocytometer grid regions and calculate scale
4. **Concentration Calculation** - Convert counts to cells/mL based on grid geometry
5. **Web Interface** - User-friendly upload and analysis interface
6. **Real-time Integration** - Connect to microscope cameras for live counting

## Technology Stack
- **ML Framework**: PyTorch (vision models, custom architectures)
- **Object Detection**: Faster R-CNN or YOLO for cell localization
- **Classification**: ResNet/EfficientNet for viability assessment
- **Backend**: FastAPI (async, great for ML serving)
- **Frontend**: HTML/JS with Tailwind CSS (simple, no framework overhead)
- **Image Processing**: OpenCV, PIL, torchvision
- **Data Annotation**: Label Studio or CVAT
- **Deployment**: Docker containerization

## Project Timeline (3 months)

### Month 1: Data & Model Foundation
- **Weeks 1-2**: Phase 1 - Data Preparation & Annotation
- **Weeks 3-4**: Phase 2A - Initial model development

### Month 2: Model Refinement
- **Weeks 5-6**: Phase 2B - Model training & optimization
- **Weeks 7-8**: Phase 3 - Evaluation & iteration

### Month 3: Application Development
- **Weeks 9-10**: Phase 4 - Web application
- **Weeks 11-12**: Phase 5 - Microscope integration (foundation)

## Success Criteria
- **Cell Detection**: >90% precision/recall on test set
- **Viability Classification**: >85% accuracy
- **Grid Detection**: Successful identification in >95% of images
- **Processing Speed**: <5 seconds per image on CPU
- **Usability**: Non-technical users can upload and get results

## Key Risks & Mitigation
1. **Insufficient training data** → Start with data augmentation, synthetic data generation
2. **Overlapping cells** → Use watershed algorithm, instance segmentation
3. **Varied lighting conditions** → Normalize images, train on diverse conditions
4. **Grid detection fails** → Template matching fallback, manual region selection
5. **Model overfitting** → Cross-validation, regularization, diverse test set

## Phase Dependencies
```
Phase 1 (Data) → Phase 2 (Model Dev) → Phase 3 (Evaluation)
                                              ↓
                        Phase 4 (Web App) ← ─┘
                              ↓
                        Phase 5 (Microscope)
```

## Repository Structure (Planned)
```
cell-counter/
├── data/
│   ├── raw/                 # Original images
│   ├── annotated/           # Labeled datasets
│   └── processed/           # Training-ready data
├── models/
│   ├── detection/           # Cell detection models
│   ├── classification/      # Viability classifier
│   └── grid/                # Grid detection
├── training/
│   ├── train_detector.py
│   ├── train_classifier.py
│   └── augmentation.py
├── inference/
│   ├── pipeline.py          # End-to-end inference
│   └── postprocessing.py
├── webapp/
│   ├── backend/             # FastAPI server
│   └── frontend/            # HTML/JS interface
├── microscope/
│   └── camera_interface.py  # Driver integration
├── notebooks/               # Exploratory analysis
├── tests/
└── docs/
```

## Getting Started
1. Review Phase 1 plan: `PHASE_1_DATA_PREP.md`
2. Set up development environment
3. Organize existing hemocytometer images
4. Begin annotation workflow

## Notes
- Focus on model accuracy before UI polish
- Document data preprocessing decisions for reproducibility
- Keep microscope integration modular for different hardware
- Consider releasing as open-source tool for biology community
