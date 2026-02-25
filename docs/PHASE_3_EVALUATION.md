# Phase 3: Model Evaluation & Iteration

**Timeline**: Weeks 7-8 (14 days)  
**Priority**: High - Improve accuracy to production-ready levels  
**Estimated Effort**: 25-30 hours

## Objectives
1. Comprehensive error analysis on test set
2. Identify failure modes and edge cases
3. Iterate on model architecture and training
4. Optimize hyperparameters
5. Implement advanced techniques (ensemble, post-processing)
6. Achieve target performance metrics
7. Create model documentation

## Deliverables
- [ ] Detailed error analysis report with visualizations
- [ ] Improved model versions with >90% F1 score
- [ ] Hyperparameter tuning results
- [ ] Final model benchmarks
- [ ] Model cards documenting architecture, training, performance
- [ ] Inference optimization (TorchScript or ONNX export)

---

## Task Breakdown

### Task 3.1: Comprehensive Error Analysis (Day 1-3)
**Goal**: Understand exactly where and why models fail

**Analysis Framework**:

**1. Detection Errors**
```python
# evaluation/error_analysis.py
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_detection_errors(predictions, ground_truth, images):
    """
    Categorize detection errors
    """
    errors = {
        'false_positives': [],  # Detected non-cell objects
        'false_negatives': [],  # Missed cells
        'localization_errors': [],  # Poor bounding boxes
        'classification_errors': []  # Wrong viability label
    }
    
    for pred, gt, img_path in zip(predictions, ground_truth, images):
        # Match predictions to GT using IoU
        matches, fps, fns = match_predictions(pred, gt, iou_threshold=0.5)
        
        # Categorize errors
        for fp_box in fps:
            errors['false_positives'].append({
                'image': img_path,
                'box': fp_box,
                'score': pred['scores'][fp_box['idx']]
            })
        
        for fn_box in fns:
            errors['false_negatives'].append({
                'image': img_path,
                'box': fn_box,
                'difficulty': assess_difficulty(fn_box, img)
            })
        
        for match in matches:
            if match['iou'] < 0.7:
                errors['localization_errors'].append(match)
            if match['label_mismatch']:
                errors['classification_errors'].append(match)
    
    return errors

def visualize_failure_cases(errors, output_dir='error_analysis/'):
    """
    Create visualizations of different error types
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot error distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # False positives by image region
    plot_fp_heatmap(errors['false_positives'], axes[0, 0])
    
    # False negatives by cell size
    plot_fn_by_size(errors['false_negatives'], axes[0, 1])
    
    # Localization error distribution
    plot_iou_distribution(errors['localization_errors'], axes[1, 0])
    
    # Classification confusion
    plot_class_confusion(errors['classification_errors'], axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_summary.png', dpi=300)
    
    # Save worst-case examples
    save_worst_cases(errors, output_dir)
```

**Error Categories to Investigate**:

**False Positives** (detecting non-cells):
- Debris or artifacts
- Grid lines or reflections
- Image edge effects
- What confidence scores? (maybe lower threshold)

**False Negatives** (missing cells):
- Very small cells (< X pixels)
- Overlapping/clustered cells
- Out-of-focus cells
- Edge of image cells
- Low contrast cells

**Localization Errors** (poor boxes):
- Boxes too large/small
- Off-center
- Including multiple cells
- Partial cell capture

**Classification Errors**:
- Viable misclassified as non-viable
- Non-viable misclassified as viable
- Ambiguous intermediate states

**Stratify errors by**:
- Cell density (low/medium/high)
- Cell size
- Image quality (blur, contrast)
- Location in image

**2. Viability Classification Analysis**
```python
def analyze_classification_errors(classifier, test_dataset):
    """
    Deep dive into viability classification mistakes
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    all_preds = []
    all_labels = []
    all_confidences = []
    error_cases = []
    
    for i in range(len(test_dataset)):
        img, label = test_dataset[i]
        with torch.no_grad():
            output = classifier(img.unsqueeze(0).to('cuda'))
            probs = torch.softmax(output, dim=1)
            pred = output.argmax().item()
            confidence = probs.max().item()
        
        all_preds.append(pred)
        all_labels.append(label)
        all_confidences.append(confidence)
        
        # Record errors
        if pred != label:
            error_cases.append({
                'index': i,
                'true_label': label,
                'predicted': pred,
                'confidence': confidence,
                'image': test_dataset.get_image_path(i)
            })
    
    # Metrics
    print(classification_report(all_labels, all_preds, 
                                target_names=['viable', 'non-viable']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, ['viable', 'non-viable'])
    
    # Confidence analysis
    analyze_confidence_distribution(all_confidences, all_preds, all_labels)
    
    # Most confident wrong predictions
    error_cases.sort(key=lambda x: x['confidence'], reverse=True)
    print(f"\nTop 10 most confident errors:")
    for case in error_cases[:10]:
        print(f"  {case}")
    
    return error_cases
```

**3. End-to-End Error Propagation**
```python
def analyze_pipeline_errors(pipeline, test_images):
    """
    Measure how detection errors affect final count
    """
    results = []
    
    for img_path, ground_truth in test_images:
        pred = pipeline.process_image(img_path)
        
        # Compare to manual count
        count_error = abs(pred['total_cells'] - ground_truth['total_cells'])
        viability_error = abs(pred['viability_percent'] - ground_truth['viability_percent'])
        
        results.append({
            'image': img_path,
            'count_error': count_error,
            'viability_error': viability_error,
            'cell_density': ground_truth['total_cells'],
            'detection_precision': calculate_precision(pred, ground_truth),
            'detection_recall': calculate_recall(pred, ground_truth)
        })
    
    # Analyze error correlation
    df = pd.DataFrame(results)
    
    print(f"Mean count error: {df['count_error'].mean():.2f} cells")
    print(f"Mean viability error: {df['viability_error'].mean():.2f}%")
    
    # Plot error vs density
    plt.figure(figsize=(10, 6))
    plt.scatter(df['cell_density'], df['count_error'])
    plt.xlabel('True Cell Count')
    plt.ylabel('Count Error')
    plt.title('Counting Error vs Cell Density')
    plt.savefig('error_analysis/count_vs_density.png')
    
    return df
```

**Deliverable**: Comprehensive error analysis report with visualizations

---

### Task 3.2: Targeted Improvements (Day 4-6)
**Goal**: Address specific failure modes identified

**Strategy**: Fix issues in priority order (highest impact first)

**Common Issues & Solutions**:

**Issue 1: Missing Small/Distant Cells**
- **Problem**: Small cells < 30 pixels not detected
- **Solutions**:
  - Adjust anchor boxes in Faster R-CNN
  - Use Feature Pyramid Network (FPN) more effectively
  - Train at higher resolution
  - Add scale augmentation during training

```python
# Adjust RPN anchor sizes
from torchvision.models.detection.rpn import AnchorGenerator

anchor_generator = AnchorGenerator(
    sizes=((16, 32, 64, 128),),  # Smaller anchors for small cells
    aspect_ratios=((0.5, 1.0, 2.0),) * 4
)

model.rpn.anchor_generator = anchor_generator
```

**Issue 2: Overlapping Cells Not Separated**
- **Problem**: Clustered cells detected as single object
- **Solutions**:
  - Lower NMS threshold (allow more overlapping boxes)
  - Post-processing with watershed algorithm
  - Use instance segmentation (Mask R-CNN) instead

```python
# Watershed post-processing
def separate_clusters(boxes, image):
    """
    Split clustered cell detections using watershed
    """
    import cv2
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    
    for box in boxes:
        if box_area(box) > 2 * avg_cell_area:  # Likely cluster
            # Extract region
            x1, y1, x2, y2 = box.astype(int)
            roi = image[y1:y2, x1:x2]
            
            # Distance transform
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            distance = ndi.distance_transform_edt(binary)
            
            # Find peaks (cell centers)
            local_max = peak_local_max(distance, min_distance=20)
            
            # Watershed segmentation
            markers = ndi.label(local_max)[0]
            labels = watershed(-distance, markers, mask=binary)
            
            # Create individual boxes from each segment
            new_boxes = extract_boxes_from_labels(labels)
            boxes = replace_box_with_new_boxes(boxes, box, new_boxes)
    
    return boxes
```

**Issue 3: False Positives (Grid Lines, Debris)**
- **Problem**: Non-cell objects being detected
- **Solutions**:
  - Hard negative mining (explicitly train on debris)
  - Increase confidence threshold
  - Add shape/size filters
  - Train with more diverse negative examples

```python
# Post-processing filters
def filter_detections(boxes, scores, labels, image):
    """
    Remove likely false positives
    """
    keep = []
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        # Size filter
        area = box_area(box)
        if area < 100 or area > 10000:  # Too small or too large
            continue
        
        # Aspect ratio filter
        w = box[2] - box[0]
        h = box[3] - box[1]
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 2.5:  # Too elongated (likely line)
            continue
        
        # Confidence filter
        if score < 0.6:
            continue
        
        # Edge filter (cells at edge often partial)
        if is_at_image_edge(box, image, margin=20):
            continue
        
        keep.append(i)
    
    return boxes[keep], scores[keep], labels[keep]
```

**Issue 4: Viability Misclassification**
- **Problem**: Confusion between viable and non-viable
- **Solutions**:
  - Collect more training examples of ambiguous cases
  - Use focal loss to handle class imbalance
  - Increase crop resolution (64→96 pixels)
  - Add attention mechanism to focus on membrane

```python
# Focal loss for hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Use in training
criterion = FocalLoss(alpha=0.25, gamma=2)
```

**Issue 5: Grid Detection Failures**
- **Problem**: Can't find grid in some images
- **Solutions**:
  - Add manual fallback mode
  - Train CNN corner detector
  - Use multiple template sizes
  - Implement user-assisted grid selection

**Deliverable**: Improved models with targeted fixes

---

### Task 3.3: Hyperparameter Optimization (Day 7-9)
**Goal**: Find optimal training configuration

**Key Hyperparameters to Tune**:

**Detector**:
- Learning rate: [1e-4, 5e-4, 1e-3, 5e-3]
- Batch size: [2, 4, 8]
- Anchor sizes: [(16,32,64), (32,64,128), (16,32,64,128)]
- NMS threshold: [0.3, 0.5, 0.7]
- Score threshold: [0.3, 0.5, 0.7]
- Training epochs: [50, 100, 150]

**Classifier**:
- Learning rate: [1e-4, 5e-4, 1e-3]
- Batch size: [32, 64, 128]
- Crop size: [64, 96, 128]
- Dropout: [0, 0.2, 0.5]
- Weight decay: [0, 1e-4, 1e-3]

**Systematic Approach**:

**1. Grid Search (for most important params)**
```python
# training/hyperparameter_search.py
from itertools import product

# Define search space
learning_rates = [1e-4, 5e-4, 1e-3]
batch_sizes = [2, 4, 8]
anchor_configs = [
    ((16, 32, 64, 128),),
    ((32, 64, 128),)
]

results = []

for lr, batch_size, anchors in product(learning_rates, batch_sizes, anchor_configs):
    config = {
        'lr': lr,
        'batch_size': batch_size,
        'anchors': anchors
    }
    
    # Train model
    model, metrics = train_detector_with_config(config)
    
    # Evaluate
    val_metrics = evaluate_detector(model, val_loader)
    
    results.append({
        'config': config,
        'train_metrics': metrics,
        'val_metrics': val_metrics
    })
    
    print(f"Config: {config}")
    print(f"Val F1: {val_metrics['f1']:.4f}")

# Find best configuration
best = max(results, key=lambda x: x['val_metrics']['f1'])
print(f"\nBest config: {best['config']}")
print(f"Best F1: {best['val_metrics']['f1']:.4f}")
```

**2. Random Search (for broader exploration)**
```python
import random

def sample_hyperparameters():
    return {
        'lr': 10 ** random.uniform(-5, -2),
        'batch_size': random.choice([2, 4, 8]),
        'weight_decay': 10 ** random.uniform(-6, -3),
        'dropout': random.uniform(0, 0.5)
    }

# Try 20 random configurations
for _ in range(20):
    config = sample_hyperparameters()
    # Train and evaluate...
```

**3. Bayesian Optimization (if time permits)**
```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

# Define search space
space = [
    Real(1e-5, 1e-2, name='learning_rate', prior='log-uniform'),
    Integer(2, 8, name='batch_size'),
    Real(0.0, 0.5, name='dropout'),
]

def objective(params):
    lr, batch_size, dropout = params
    
    # Train model
    val_f1 = train_and_evaluate(lr, batch_size, dropout)
    
    return -val_f1  # Minimize negative F1 = maximize F1

# Run optimization
result = gp_minimize(objective, space, n_calls=30, random_state=42)
```

**Track Experiments**:
```python
# Use Weights & Biases for experiment tracking
import wandb

wandb.init(project='hemocytometer-cell-counter', config=config)

# Log metrics during training
for epoch in range(num_epochs):
    train_loss = train_one_epoch(...)
    val_metrics = evaluate(...)
    
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_f1': val_metrics['f1'],
        'val_precision': val_metrics['precision'],
        'val_recall': val_metrics['recall']
    })

wandb.finish()
```

**Deliverable**: Optimized hyperparameters and training curves

---

### Task 3.4: Advanced Techniques (Day 10-11)
**Goal**: Squeeze out extra performance

**Technique 1: Test-Time Augmentation (TTA)**
```python
def predict_with_tta(model, image, n_augmentations=5):
    """
    Apply multiple augmentations and average predictions
    """
    predictions = []
    
    transforms = [
        lambda x: x,  # Original
        lambda x: torch.flip(x, dims=[2]),  # Horizontal flip
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # Rotate 90
        lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # Rotate 270
        lambda x: adjust_brightness(x, 1.1),  # Brighter
    ]
    
    for transform in transforms[:n_augmentations]:
        aug_image = transform(image)
        pred = model(aug_image)
        predictions.append(pred)
    
    # Average predictions
    avg_pred = ensemble_predictions(predictions)
    return avg_pred
```

**Technique 2: Model Ensemble**
```python
class EnsembleDetector:
    """
    Combine multiple detector models
    """
    def __init__(self, model_paths):
        self.models = []
        for path in model_paths:
            model = CellDetector(num_classes=3)
            model.load(path)
            model.eval()
            self.models.append(model)
    
    def predict(self, image):
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # Get predictions from each model
        for model in self.models:
            pred = model([image])
            all_boxes.append(pred[0]['boxes'])
            all_scores.append(pred[0]['scores'])
            all_labels.append(pred[0]['labels'])
        
        # Merge predictions using Non-Maximum Suppression
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=0.5,
            skip_box_thr=0.3
        )
        
        return merged_boxes, merged_scores, merged_labels
```

**Technique 3: Hard Negative Mining**
```python
def mine_hard_negatives(model, dataset, threshold=0.3):
    """
    Find false positives to add to training
    """
    hard_negatives = []
    
    for image, target in dataset:
        pred = model([image])
        
        # Find predictions that don't match any ground truth
        for box, score in zip(pred[0]['boxes'], pred[0]['scores']):
            if score > threshold:
                max_iou = compute_max_iou(box, target['boxes'])
                if max_iou < 0.3:  # Low overlap with GT = false positive
                    hard_negatives.append({
                        'image': image,
                        'box': box,
                        'score': score
                    })
    
    # Add these to training as background class
    return hard_negatives
```

**Technique 4: Self-Training / Pseudo-Labeling**
```python
def pseudo_label_unlabeled_data(model, unlabeled_images, confidence=0.8):
    """
    Use model to label new data (if you have unlabeled images)
    """
    pseudo_labeled = []
    
    for image_path in unlabeled_images:
        image = load_image(image_path)
        predictions = model([image])
        
        # Only keep high-confidence predictions
        high_conf = predictions[0]['scores'] > confidence
        boxes = predictions[0]['boxes'][high_conf]
        labels = predictions[0]['labels'][high_conf]
        
        if len(boxes) > 0:
            pseudo_labeled.append({
                'image': image_path,
                'boxes': boxes,
                'labels': labels
            })
    
    # Add to training set (with lower learning rate)
    return pseudo_labeled
```

**Deliverable**: Ensemble models and advanced technique implementations

---

### Task 3.5: Inference Optimization (Day 12-13)
**Goal**: Make models faster for deployment

**Optimization 1: TorchScript Export**
```python
# export_models.py
import torch

# Load trained model
detector = CellDetector(num_classes=3)
detector.load('checkpoints/detector/detector_best.pth')
detector.eval()

# Export to TorchScript
example_input = torch.randn(1, 3, 1080, 1920)
traced_model = torch.jit.trace(detector.model, example_input)
traced_model.save('models/detector_traced.pt')

# Load and use
loaded = torch.jit.load('models/detector_traced.pt')
output = loaded(example_input)
```

**Optimization 2: ONNX Export**
```python
# For cross-platform deployment
import torch.onnx

torch.onnx.export(
    detector.model,
    example_input,
    'models/detector.onnx',
    input_names=['image'],
    output_names=['boxes', 'labels', 'scores'],
    dynamic_axes={
        'image': {0: 'batch', 2: 'height', 3: 'width'}
    }
)
```

**Optimization 3: Model Quantization**
```python
# Reduce model size, faster inference
from torch.quantization import quantize_dynamic

# Dynamic quantization (easy, post-training)
quantized_model = quantize_dynamic(
    detector.model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 'models/detector_quantized.pth')
```

**Optimization 4: Batch Inference**
```python
def batch_inference(model, image_paths, batch_size=8):
    """
    Process multiple images efficiently
    """
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [load_and_preprocess(p) for p in batch_paths]
        
        with torch.no_grad():
            predictions = model(images)
        
        results.extend(predictions)
    
    return results
```

**Performance Benchmarking**:
```python
import time

def benchmark_inference(model, test_images, n_runs=100):
    """
    Measure inference speed
    """
    times = []
    
    # Warmup
    for _ in range(10):
        _ = model(test_images[0])
    
    # Benchmark
    for i in range(n_runs):
        start = time.time()
        _ = model(test_images[i % len(test_images)])
        end = time.time()
        times.append(end - start)
    
    print(f"Mean inference time: {np.mean(times)*1000:.2f} ms")
    print(f"Std: {np.std(times)*1000:.2f} ms")
    print(f"FPS: {1/np.mean(times):.2f}")
```

**Deliverable**: Optimized models with benchmarks

---

### Task 3.6: Final Evaluation & Documentation (Day 14)
**Goal**: Comprehensive final assessment and model cards

**Final Test Set Evaluation**:
```python
# Run on completely held-out test set
final_results = {
    'detector': evaluate_detector(detector, test_loader),
    'classifier': evaluate_classifier(classifier, test_loader),
    'end_to_end': evaluate_pipeline(pipeline, test_images)
}

# Create final report
print("=" * 50)
print("FINAL MODEL PERFORMANCE")
print("=" * 50)
print(f"\nDetection Metrics:")
print(f"  Precision: {final_results['detector']['precision']:.3f}")
print(f"  Recall: {final_results['detector']['recall']:.3f}")
print(f"  F1 Score: {final_results['detector']['f1']:.3f}")
print(f"  mAP@0.5: {final_results['detector']['map']:.3f}")

print(f"\nClassification Metrics:")
print(f"  Accuracy: {final_results['classifier']['accuracy']:.3f}")
print(f"  Precision (viable): {final_results['classifier']['precision_viable']:.3f}")
print(f"  Recall (viable): {final_results['classifier']['recall_viable']:.3f}")

print(f"\nEnd-to-End Performance:")
print(f"  Mean Count Error: {final_results['end_to_end']['mean_count_error']:.2f} cells")
print(f"  Mean Viability Error: {final_results['end_to_end']['mean_viability_error']:.2f}%")
print(f"  Inference Time: {final_results['end_to_end']['inference_time']:.3f}s")
```

**Create Model Cards**:
```markdown
# Model Card: Cell Detector v1.0

## Model Details
- **Architecture**: Faster R-CNN with ResNet50-FPN backbone
- **Framework**: PyTorch 2.0
- **Input**: RGB images (variable size, typically 1920x1080)
- **Output**: Bounding boxes, class labels, confidence scores
- **Classes**: Background, viable cell, non-viable cell
- **Parameters**: 41.3M
- **Training Date**: 2026-02-15

## Intended Use
- Automated counting of cells in hemocytometer images
- Research and laboratory applications
- Supports 4x magnification brightfield microscopy

## Training Data
- **Dataset Size**: 150 annotated images, 2,847 cell instances
- **Data Sources**: Laboratory hemocytometer images
- **Annotation**: Manual bounding boxes with viability labels
- **Split**: 70% train, 15% validation, 15% test
- **Augmentation**: Flips, rotations, brightness/contrast

## Performance
- **Detection F1**: 0.92
- **Mean Count Error**: 2.3 cells per image
- **Inference Time**: 4.2s per image (GPU)
- **Cell Size Range**: 30-200 pixels
- **Density Range**: 0-50 cells per image

## Limitations
- Trained only on 4x magnification
- May struggle with very high cell densities (>50 cells)
- Requires good lighting and focus
- Grid must be visible for concentration calculation

## Ethical Considerations
- For research use only
- Should not replace manual verification for critical decisions
- Performance may vary on different microscope setups

## Maintenance
- **Version**: 1.0
- **Last Updated**: 2026-02-15
- **Contact**: [your email]
```

**Deliverable**: Final evaluation report + model cards

---

## Phase 3 Success Criteria
- [ ] Cell detector F1 score >90% on test set
- [ ] Viability classifier accuracy >85%
- [ ] Mean cell count error <10%
- [ ] Inference time <5 seconds per image
- [ ] Comprehensive error analysis completed
- [ ] Models optimized and exported
- [ ] Documentation complete

## Common Pitfalls to Avoid
- Overfitting to validation set through excessive tuning
- Testing on data similar to training (data leakage)
- Ignoring rare/edge cases
- Premature optimization before understanding errors
- Not documenting changes and experiments

## Next Phase Preview
Phase 4 will create the web application interface:
- FastAPI backend for model serving
- Simple HTML/JS frontend for uploads
- Result visualization and export
- User-friendly interface for non-technical users
