# Phase 1: Data Preparation & Annotation

**Timeline**: Weeks 1-2 (14 days)  
**Priority**: Critical - Everything depends on quality training data  
**Estimated Effort**: 20-25 hours

## Objectives
1. Organize and catalog existing hemocytometer images
2. Set up annotation infrastructure
3. Annotate cells with bounding boxes and viability labels
4. Create train/validation/test splits
5. Establish data quality standards
6. Document grid specifications and scale calculations

## Deliverables
- [ ] Organized image dataset with metadata
- [ ] Annotation tool setup (Label Studio or CVAT)
- [ ] At least 500 annotated cell instances (start small, iterate)
- [ ] Annotation guidelines document
- [ ] Data loading pipeline in PyTorch
- [ ] Dataset statistics report

---

## Task Breakdown

### Task 1.1: Data Inventory & Organization (Day 1-2)
**Goal**: Understand what data you have and establish structure

**Actions**:
1. Create data directory structure:
```bash
data/
├── raw/
│   ├── empty_grid/        # Hemocytometer with no cells
│   ├── with_cells/        # Cells present
│   └── metadata.csv       # Image metadata
├── annotated/
│   └── labels/            # COCO format annotations
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
└── README.md
```

2. Catalog existing images:
   - Count total images
   - Note resolution, file format
   - Document microscope settings (4x magnification)
   - Identify which have cells vs empty grids
   - Check for trypan blue staining (viability indicator)

3. Create `metadata.csv` with columns:
   - `image_id`, `filename`, `cell_density` (none/low/medium/high)
   - `has_staining` (yes/no), `grid_visible` (yes/no)
   - `magnification`, `acquisition_date`, `notes`

**Deliverable**: Organized data folder + metadata spreadsheet

---

### Task 1.2: Annotation Tool Setup (Day 2-3)
**Goal**: Get annotation infrastructure running

**Recommended Tool**: Label Studio (easier setup than CVAT)

**Installation**:
```bash
pip install label-studio
label-studio start
```

**Configuration**:
1. Create project: "Hemocytometer Cell Counter"
2. Set up labeling interface with:
   - **Object detection**: Rectangle tool for cell bounding boxes
   - **Classification per box**: "viable" / "non-viable"
   - **Grid markers**: Polygons for corner grid intersections (optional)

3. Import images to Label Studio

**Deliverable**: Working annotation tool with project configured

---

### Task 1.3: Annotation Guidelines (Day 3)
**Goal**: Establish consistent labeling standards

**Create Guidelines Document** covering:

**Cell Annotation Rules**:
- Draw tight bounding box around each cell
- Include partial cells at grid edges
- Ignore debris/artifacts <10 pixels
- Mark overlapping cells separately when possible
- For cell clusters, draw individual boxes where boundaries visible

**Viability Classification**:
- **Viable (Live)**: Clear cytoplasm, no blue staining, intact membrane
- **Non-viable (Dead)**: Blue-stained (trypan blue), membrane compromised
- **Uncertain**: Skip or mark as "ambiguous" (filter out later)

**Grid Reference Points** (for scale detection):
- Mark 4 corner intersections of counting region
- Note grid square size: 0.1mm × 0.1mm (standard hemocytometer)
- Document depth: 0.1mm (for volume calculation)

**Quality Control**:
- Review annotations after every 50 images
- Calculate inter-annotator agreement if multiple people help
- Discard poor quality/blurry images

**Deliverable**: `ANNOTATION_GUIDELINES.md` document

---

### Task 1.4: Initial Annotation (Day 4-8)
**Goal**: Create training dataset

**Strategy**: Start small, iterate fast
1. **Pilot Phase** (Day 4-5):
   - Annotate 20 diverse images thoroughly
   - Test annotation guidelines
   - Refine rules based on edge cases
   - Estimate time per image (target: <5 min/image)

2. **Full Annotation** (Day 6-8):
   - Target: 100-150 images minimum (expand later)
   - Prioritize diversity:
     - Empty grids: 10-15 images
     - Low cell density: 30-40 images
     - Medium density: 40-50 images
     - High density: 20-30 images
   - Mix of viable/non-viable cells
   - Different lighting conditions

**Efficiency Tips**:
- Use keyboard shortcuts
- Annotate in batches by density
- Take breaks to maintain quality
- Consider recruiting help (with guidelines)

**Deliverable**: 100+ annotated images in Label Studio

---

### Task 1.5: Data Augmentation Strategy (Day 9)
**Goal**: Plan to expand effective dataset size

**Since you have limited data**, plan augmentation pipeline:

**Geometric Transforms** (keep cell appearance realistic):
- Horizontal/vertical flips
- Small rotations (±15°)
- Random crops (90-100% of image)
- Slight scaling (0.9-1.1x)

**Color/Intensity Transforms**:
- Brightness adjustment (±15%)
- Contrast adjustment (0.9-1.1x)
- Gaussian noise addition
- Blur (simulate focus variation)

**Implementation Plan**:
```python
# Use torchvision transforms + albumentations library
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
], bbox_params=A.BboxParams(format='coco'))
```

**Deliverable**: Augmentation pipeline code ready for training

---

### Task 1.6: Export & Dataset Splits (Day 10)
**Goal**: Create train/val/test sets in standard format

**Export from Label Studio**:
- Format: COCO JSON (standard for object detection)
- Includes: bounding boxes, class labels, image metadata

**Dataset Splits**:
- **Train**: 70% of images
- **Validation**: 15% of images
- **Test**: 15% of images (hold out completely)

**Stratification**: Ensure each split has similar distribution of:
- Cell densities (empty, low, medium, high)
- Viability ratios
- Image quality

**Implementation**:
```python
# data_prep/create_splits.py
from sklearn.model_selection import train_test_split
import json
import shutil

# Load COCO annotations
with open('annotations.json') as f:
    coco = json.load(f)

# Stratified split by cell density
# ... split logic ...

# Create separate annotation files
# train_annotations.json, val_annotations.json, test_annotations.json
```

**Deliverable**: Three annotation files + organized image folders

---

### Task 1.7: PyTorch Dataset Class (Day 11-12)
**Goal**: Create data loader for model training

**Implementation**: `data/dataset.py`

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class HemocytometerDataset(Dataset):
    """
    PyTorch dataset for hemocytometer cell detection
    """
    def __init__(self, 
                 coco_json_path,
                 img_dir,
                 transforms=None,
                 grid_detection=False):
        """
        Args:
            coco_json_path: Path to COCO format annotations
            img_dir: Directory containing images
            transforms: Albumentations transforms
            grid_detection: If True, include grid corner labels
        """
        self.img_dir = img_dir
        self.transforms = transforms
        
        # Load COCO annotations
        with open(coco_json_path) as f:
            self.coco = json.load(f)
        
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = self._group_annotations()
        
    def _group_annotations(self):
        """Group annotations by image_id"""
        grouped = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in grouped:
                grouped[img_id] = []
            grouped[img_id].append(ann)
        return grouped
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image info
        img_id = list(self.images.keys())[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.annotations.get(img_id, [])
        
        # Extract boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=np.array(image),
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Create target dict (Faster R-CNN format)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        return image, target

# Usage example
dataset = HemocytometerDataset(
    coco_json_path='data/annotated/train_annotations.json',
    img_dir='data/raw/with_cells',
    transforms=transform
)
```

**Testing**:
```python
# Test data loading
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset, 
    batch_size=2,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))  # Custom collate
)

# Verify data
for images, targets in dataloader:
    print(f"Batch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Boxes: {targets[0]['boxes'].shape}")
    break
```

**Deliverable**: Working PyTorch dataset class

---

### Task 1.8: Data Analysis & Statistics (Day 13-14)
**Goal**: Understand dataset characteristics

**Create Analysis Notebook**: `notebooks/data_exploration.ipynb`

**Analyze**:
1. **Dataset statistics**:
   - Total images, total cell annotations
   - Cells per image distribution (histogram)
   - Class balance (viable vs. non-viable)
   - Box size distribution (are cells consistent size?)

2. **Image properties**:
   - Resolution distribution
   - Brightness/contrast statistics
   - Grid visibility quality

3. **Visualizations**:
   - Sample images with annotations overlaid
   - Bounding box size heatmap
   - Spatial distribution of cells in grid

4. **Quality checks**:
   - Detect potential annotation errors (e.g., boxes too small/large)
   - Identify outlier images
   - Check for data leakage between splits

**Statistical Report**:
```markdown
# Dataset Statistics Report

## Overview
- Total Images: 150
- Total Cell Annotations: 2,847
- Training Set: 105 images (70%)
- Validation Set: 23 images (15%)
- Test Set: 22 images (15%)

## Cell Distribution
- Empty grids: 12 images (8%)
- Low density (<10 cells): 45 images (30%)
- Medium density (10-30 cells): 68 images (45%)
- High density (>30 cells): 25 images (17%)

## Class Balance
- Viable cells: 1,823 (64%)
- Non-viable cells: 1,024 (36%)

## Image Properties
- Resolution: 1920x1080 (consistent)
- Average cells per image: 19.0 ± 12.3
- Average box size: 45x48 pixels

## Recommendations
- Consider collecting more high-density images
- Class balance is acceptable for binary classification
- Grid detection should be robust given consistent resolution
```

**Deliverable**: Analysis notebook + statistics report

---

## Phase 1 Success Criteria
- [ ] At least 100 annotated images with 1,000+ cell instances
- [ ] Clear annotation guidelines documented
- [ ] Data properly split into train/val/test
- [ ] PyTorch dataset class loads data correctly
- [ ] Dataset statistics documented and reviewed
- [ ] No obvious quality issues or annotation errors

## Potential Issues & Solutions

**Issue**: Annotation is too slow
- **Solution**: Start with subset, use pre-trained model to suggest boxes (active learning)

**Issue**: Too few images for training
- **Solution**: Aggressive data augmentation, consider synthetic data generation

**Issue**: Inconsistent grid detection
- **Solution**: Collect more diverse grid images, consider template matching approach

**Issue**: Ambiguous viability cases
- **Solution**: Add "uncertain" category, focus on clear examples first

## Next Phase Preview
Phase 2 will use this annotated data to train:
1. Cell detection model (Faster R-CNN)
2. Viability classifier (ResNet)
3. Grid detection system (template matching + CNN)

Ensure annotations are high quality - model performance is limited by data quality!
