# Phase 2: Model Development & Training

**Timeline**: Weeks 3-6 (28 days)  
**Priority**: Critical - Core ML functionality  
**Estimated Effort**: 50-60 hours

## Objectives
1. Develop cell detection model (bounding box prediction)
2. Build viability classification model
3. Create grid detection system
4. Train models on annotated dataset
5. Implement inference pipeline
6. Establish evaluation metrics and baseline performance

## Deliverables
- [ ] Trained cell detection model (Faster R-CNN or YOLO)
- [ ] Trained viability classifier
- [ ] Grid detection algorithm
- [ ] Inference pipeline that processes images end-to-end
- [ ] Training scripts with logging (TensorBoard)
- [ ] Model checkpoints and configuration files
- [ ] Initial performance metrics report

---

## Task Breakdown

### Task 2.1: Development Environment Setup (Day 1)
**Goal**: Set up ML training infrastructure

**Install Dependencies**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python-headless
pip install albumentations
pip install pillow
pip install scikit-learn
pip install matplotlib seaborn
pip install tensorboard
pip install pycocotools
pip install tqdm

# Optional but recommended
pip install timm  # PyTorch Image Models
pip install wandb  # Experiment tracking
```

**Project Structure**:
```bash
cell-counter/
├── models/
│   ├── __init__.py
│   ├── detector.py          # Faster R-CNN implementation
│   ├── classifier.py        # Viability classifier
│   └── grid_detector.py     # Grid localization
├── training/
│   ├── train_detector.py
│   ├── train_classifier.py
│   ├── config.py            # Hyperparameters
│   └── utils.py
├── inference/
│   ├── pipeline.py          # End-to-end processing
│   └── visualize.py
└── checkpoints/             # Saved models
```

**GPU Setup**:
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**Deliverable**: Environment ready, project structure created

---

### Task 2.2: Cell Detection Model Architecture (Day 2-3)
**Goal**: Implement object detection model for cell localization

**Approach**: Faster R-CNN with ResNet50 backbone (good balance of accuracy/speed)

**Why Faster R-CNN**:
- Better for dense, small objects (cells)
- More accurate than single-shot detectors
- Pre-trained on COCO dataset (transfer learning)
- Well-supported in PyTorch

**Implementation**: `models/detector.py`

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CellDetector:
    """
    Faster R-CNN based cell detector
    """
    def __init__(self, num_classes=3, pretrained=True):
        """
        Args:
            num_classes: Number of classes (background + viable + non-viable)
            pretrained: Use pretrained COCO weights
        """
        # Load pre-trained model
        self.model = fasterrcnn_resnet50_fpn(
            weights='DEFAULT' if pretrained else None
        )
        
        # Replace classification head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, 
            num_classes
        )
        
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def train_mode(self):
        self.model.train()
        
    def eval_mode(self):
        self.model.eval()
        
    def forward(self, images, targets=None):
        """
        Args:
            images: List of tensors [C, H, W]
            targets: List of dicts with 'boxes' and 'labels'
        Returns:
            losses (if training) or predictions (if eval)
        """
        return self.model(images, targets)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

# Usage
detector = CellDetector(num_classes=3, pretrained=True)
detector.to('cuda')
```

**Alternative: YOLO** (consider if speed is critical)
```python
# Using ultralytics YOLOv8
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Nano model, fastest
# Train on custom data
model.train(data='data.yaml', epochs=100)
```

**Deliverable**: Cell detection model class implemented

---

### Task 2.3: Training Script for Detector (Day 4-6)
**Goal**: Train cell detection model

**Configuration**: `training/config.py`
```python
class DetectorConfig:
    # Data
    TRAIN_ANNOTATION = 'data/annotated/train_annotations.json'
    VAL_ANNOTATION = 'data/annotated/val_annotations.json'
    IMAGE_DIR = 'data/raw/with_cells'
    
    # Training
    BATCH_SIZE = 4  # Adjust based on GPU memory
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # Augmentation
    USE_AUGMENTATION = True
    
    # Model
    NUM_CLASSES = 3  # background, viable, non-viable
    BACKBONE = 'resnet50'
    PRETRAINED = True
    
    # Checkpointing
    CHECKPOINT_DIR = 'checkpoints/detector'
    SAVE_EVERY = 5  # epochs
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**Training Script**: `training/train_detector.py`
```python
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from models.detector import CellDetector
from data.dataset import HemocytometerDataset
from training.config import DetectorConfig as Config

def collate_fn(batch):
    """Custom collate for object detection"""
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer):
    model.train_mode()
    epoch_loss = 0
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for i, (images, targets) in enumerate(pbar):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model.forward(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Logging
        epoch_loss += losses.item()
        pbar.set_postfix({'loss': losses.item()})
        
        # TensorBoard
        step = epoch * len(data_loader) + i
        writer.add_scalar('Loss/train_step', losses.item(), step)
        for k, v in loss_dict.items():
            writer.add_scalar(f'Loss/{k}', v.item(), step)
    
    avg_loss = epoch_loss / len(data_loader)
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    return avg_loss

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval_mode()
    total_loss = 0
    
    for images, targets in tqdm(data_loader, desc='Evaluating'):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model.forward(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def main():
    # Setup
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(f'runs/detector_{time.time()}')
    device = torch.device(Config.DEVICE)
    
    # Data
    train_dataset = HemocytometerDataset(
        Config.TRAIN_ANNOTATION,
        Config.IMAGE_DIR,
        transforms=get_train_transforms() if Config.USE_AUGMENTATION else None
    )
    val_dataset = HemocytometerDataset(
        Config.VAL_ANNOTATION,
        Config.IMAGE_DIR,
        transforms=None  # No augmentation for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Model
    model = CellDetector(
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED
    ).to(device)
    
    # Optimizer
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.1
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, writer
        )
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR,
                f'detector_epoch_{epoch+1}.pth'
            )
            model.save(checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(Config.CHECKPOINT_DIR, 'detector_best.pth')
            model.save(best_path)
            print(f'Saved best model (val_loss={val_loss:.4f})')
        
        lr_scheduler.step()
    
    writer.close()
    print('Training complete!')

if __name__ == '__main__':
    main()
```

**Training Execution**:
```bash
# Start training
python training/train_detector.py

# Monitor with TensorBoard
tensorboard --logdir=runs/
```

**Expected Training Time**: 
- 50 epochs × 100 images ÷ 4 batch size = ~1,250 iterations
- ~5-10 hours on GPU (depends on hardware)

**Deliverable**: Trained cell detector model

---

### Task 2.4: Viability Classifier (Day 7-9)
**Goal**: Build model to classify viable vs. non-viable cells

**Approach**: Use ResNet18 classifier on cropped cell regions

**Why Separate Classifier**:
- Can use higher resolution crops of individual cells
- Easier to debug and improve than end-to-end
- Can train on more data (augment individual cells)

**Implementation**: `models/classifier.py`
```python
import torch
import torch.nn as nn
import torchvision.models as models

class ViabilityClassifier(nn.Module):
    """
    ResNet-based binary classifier for cell viability
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(
            weights='IMAGENET1K_V1' if pretrained else None
        )
        
        # Replace final layer for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 2)  # viable, non-viable
        
    def forward(self, x):
        return self.backbone(x)

# Usage
classifier = ViabilityClassifier(pretrained=True)
classifier.to('cuda')
```

**Dataset for Classifier**: Extract cell crops from annotated images
```python
# data/cell_crops_dataset.py
class CellCropsDataset(Dataset):
    """
    Dataset of individual cell crops with viability labels
    """
    def __init__(self, coco_json, img_dir, crop_size=64, transforms=None):
        self.img_dir = img_dir
        self.crop_size = crop_size
        self.transforms = transforms
        
        # Load annotations
        with open(coco_json) as f:
            self.coco = json.load(f)
        
        # Extract all cell instances
        self.cells = []
        for ann in self.coco['annotations']:
            img_info = self._get_image_info(ann['image_id'])
            self.cells.append({
                'image_file': img_info['file_name'],
                'bbox': ann['bbox'],
                'label': ann['category_id']  # 1=viable, 2=non-viable
            })
    
    def __getitem__(self, idx):
        cell = self.cells[idx]
        
        # Load image
        img = Image.open(os.path.join(self.img_dir, cell['image_file']))
        
        # Crop cell region
        x, y, w, h = cell['bbox']
        crop = img.crop((x, y, x+w, y+h))
        
        # Resize to fixed size
        crop = crop.resize((self.crop_size, self.crop_size))
        
        # Apply transforms
        if self.transforms:
            crop = self.transforms(crop)
        else:
            crop = transforms.ToTensor()(crop)
        
        # Convert label (1,2) to (0,1)
        label = cell['label'] - 1
        
        return crop, label
```

**Training Script**: `training/train_classifier.py` (similar structure to detector)

**Key Differences**:
- Simpler loss: CrossEntropyLoss instead of object detection losses
- Smaller batch size possible: 32-64
- Faster training: ~2-3 hours
- Add class balancing if needed: `WeightedRandomSampler`

**Deliverable**: Trained viability classifier

---

### Task 2.5: Grid Detection System (Day 10-12)
**Goal**: Automatically locate hemocytometer grid for scale calculation

**Approach**: Hybrid method combining template matching + CNN

**Method 1: Template Matching (Fast, Rule-based)**
```python
# models/grid_detector.py
import cv2
import numpy as np

class GridDetector:
    """
    Detect hemocytometer grid using computer vision
    """
    def __init__(self, grid_template_path=None):
        self.template = cv2.imread(grid_template_path, 0) if grid_template_path else None
        
    def detect_grid_template(self, image):
        """
        Use template matching to find grid
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Template matching
        result = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.7:  # Confidence threshold
            h, w = self.template.shape
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            return top_left, bottom_right
        
        return None
    
    def detect_grid_lines(self, image):
        """
        Find grid lines using Hough transform
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        # Filter for horizontal/vertical lines
        h_lines = []
        v_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            
            if abs(angle) < 10:  # Horizontal
                h_lines.append(line)
            elif abs(abs(angle) - 90) < 10:  # Vertical
                v_lines.append(line)
        
        return h_lines, v_lines
    
    def calculate_scale(self, grid_bbox, grid_size_mm=1.0):
        """
        Calculate pixels per mm based on grid
        
        Args:
            grid_bbox: (x1, y1, x2, y2) of counting grid
            grid_size_mm: Physical size of grid in mm
        """
        x1, y1, x2, y2 = grid_bbox
        pixels = max(x2 - x1, y2 - y1)
        scale = pixels / grid_size_mm  # pixels per mm
        
        return scale
```

**Method 2: CNN-based Corner Detection** (More robust)
```python
# Train small CNN to detect 4 corner points of grid
# Use simple architecture: ResNet18 with 8 output nodes (4 points × 2 coords)

class GridCornerDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, 8)  # 4 corners, (x,y) each
    
    def forward(self, x):
        coords = self.backbone(x)  # Output: [batch, 8]
        return coords.view(-1, 4, 2)  # Reshape to [batch, 4 corners, 2 coords]
```

**For MVP**: Start with template matching, add CNN if needed

**Deliverable**: Grid detection implementation

---

### Task 2.6: Inference Pipeline (Day 13-15)
**Goal**: Combine all models into end-to-end processing

**Implementation**: `inference/pipeline.py`
```python
import torch
import cv2
import numpy as np
from PIL import Image

class CellCountingPipeline:
    """
    End-to-end pipeline for automated cell counting
    """
    def __init__(self, 
                 detector_path,
                 classifier_path,
                 grid_detector,
                 device='cuda'):
        
        self.device = device
        
        # Load models
        self.detector = CellDetector(num_classes=3)
        self.detector.load(detector_path)
        self.detector.to(device)
        self.detector.eval_mode()
        
        self.classifier = ViabilityClassifier()
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.to(device)
        self.classifier.eval()
        
        self.grid_detector = grid_detector
        
    @torch.no_grad()
    def process_image(self, image_path, confidence_threshold=0.5):
        """
        Process single hemocytometer image
        
        Returns:
            results: dict with counts, viability, concentration
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        img_tensor = transforms.ToTensor()(image).to(self.device)
        
        # Step 1: Detect grid
        grid_bbox, scale = self._detect_grid(image)
        
        # Step 2: Detect cells
        predictions = self.detector.forward([img_tensor])
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence
        keep = scores > confidence_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        # Step 3: Classify viability (refine detection labels)
        viable_count = 0
        non_viable_count = 0
        
        for box, label in zip(boxes, labels):
            # Crop cell region
            x1, y1, x2, y2 = box.astype(int)
            cell_crop = image.crop((x1, y1, x2, y2))
            cell_crop = cell_crop.resize((64, 64))
            cell_tensor = transforms.ToTensor()(cell_crop).unsqueeze(0).to(self.device)
            
            # Classify
            viability = self.classifier(cell_tensor).argmax().item()
            
            if viability == 0:  # Viable
                viable_count += 1
            else:
                non_viable_count += 1
        
        # Step 4: Calculate concentration
        total_cells = viable_count + non_viable_count
        viability_percent = (viable_count / total_cells * 100) if total_cells > 0 else 0
        
        # Standard hemocytometer: 0.1mm depth, 1mm² counting area
        volume_ml = 1e-4  # 0.1mm × 1mm² = 0.0001 mL
        concentration = total_cells / volume_ml if grid_bbox else None
        
        results = {
            'total_cells': total_cells,
            'viable_cells': viable_count,
            'non_viable_cells': non_viable_count,
            'viability_percent': viability_percent,
            'concentration_per_ml': concentration,
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'grid_bbox': grid_bbox,
            'scale_pixels_per_mm': scale
        }
        
        return results
    
    def _detect_grid(self, image):
        """Detect grid and calculate scale"""
        grid_bbox = self.grid_detector.detect_grid_template(np.array(image))
        
        if grid_bbox:
            scale = self.grid_detector.calculate_scale(grid_bbox)
            return grid_bbox, scale
        
        return None, None
    
    def visualize_results(self, image_path, results, save_path=None):
        """Draw boxes and counts on image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw boxes
        for box, label in zip(results['boxes'], results['labels']):
            x1, y1, x2, y2 = box.astype(int)
            color = (0, 255, 0) if label == 0 else (255, 0, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw grid
        if results['grid_bbox']:
            x1, y1, x2, y2 = results['grid_bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Add text
        text = f"Total: {results['total_cells']} | Viable: {results['viable_cells']} | Viability: {results['viability_percent']:.1f}%"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return image

# Usage
pipeline = CellCountingPipeline(
    detector_path='checkpoints/detector/detector_best.pth',
    classifier_path='checkpoints/classifier/classifier_best.pth',
    grid_detector=GridDetector()
)

results = pipeline.process_image('test_image.jpg')
pipeline.visualize_results('test_image.jpg', results, 'output.jpg')
```

**Deliverable**: Working end-to-end pipeline

---

### Task 2.7: Initial Evaluation (Day 16-18)
**Goal**: Measure baseline model performance

**Metrics to Track**:

**Detection Metrics** (on test set):
- **Precision**: TP / (TP + FP) - how many detections are correct
- **Recall**: TP / (TP + FN) - how many cells were found
- **F1 Score**: Harmonic mean of precision/recall
- **mAP** (mean Average Precision): Standard object detection metric

**Classification Metrics**:
- **Accuracy**: Correct viability classifications
- **Confusion Matrix**: True viable vs. classified viable
- **Precision/Recall** per class

**End-to-End Metrics**:
- **Cell Count Error**: |predicted - actual| cells
- **Concentration Error**: Accuracy of cells/mL calculation
- **Processing Time**: Seconds per image

**Evaluation Script**: `evaluation/evaluate.py`
```python
def evaluate_detector(model, test_loader, iou_threshold=0.5):
    """
    Calculate detection metrics
    """
    all_predictions = []
    all_targets = []
    
    for images, targets in test_loader:
        predictions = model(images)
        # Match predictions to ground truth using IoU
        # Calculate TP, FP, FN
        # ...
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def evaluate_classifier(model, test_loader):
    """
    Calculate classification metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    all_preds = []
    all_labels = []
    
    for images, labels in test_loader:
        preds = model(images).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(all_labels, all_preds, 
                                    target_names=['viable', 'non-viable'])
    cm = confusion_matrix(all_labels, all_preds)
    
    return report, cm
```

**Target Baseline Performance**:
- Detection F1 > 0.85
- Viability accuracy > 0.80
- Cell count error < 15%

**Deliverable**: Evaluation metrics report

---

## Phase 2 Success Criteria
- [ ] Cell detector achieves >85% F1 score on test set
- [ ] Viability classifier achieves >80% accuracy
- [ ] Grid detection works on >90% of images
- [ ] End-to-end pipeline processes images successfully
- [ ] Inference time < 10 seconds per image
- [ ] All models saved and versioned

## Potential Issues & Solutions

**Issue**: Model underfitting (low training accuracy)
- **Solution**: Train longer, reduce regularization, check learning rate

**Issue**: Model overfitting (train accuracy >> test accuracy)
- **Solution**: More data augmentation, dropout, reduce model complexity

**Issue**: Poor detection of small/dense cells
- **Solution**: Increase anchor sizes, use FPN, adjust NMS threshold

**Issue**: Viability classification confusion
- **Solution**: Collect more clear examples, balance classes, use focal loss

**Issue**: Slow inference
- **Solution**: Use smaller model (MobileNet), optimize with TorchScript, batch processing

## Next Phase Preview
Phase 3 will focus on:
- Analyzing failure cases
- Improving model performance through iteration
- Hyperparameter tuning
- Ensemble methods if needed
- Preparing for deployment
