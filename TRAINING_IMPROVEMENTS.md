# Rail YOLO Model - Training Improvements & Optimization Guide

## Overview
This document outlines all the improvements made to increase model accuracy and debugged issues in the Rail YOLO detection system.

---

## 🔧 Key Improvements Implemented

### 1. **Training Configuration Enhancements**

#### Before:
- ❌ Only 1 epoch training (minimal convergence)
- ❌ YOLOv8 nano (yolov8n) - smallest model with limited capacity
- ❌ CPU-only training (slow, limits batch sizes)
- ❌ No data augmentation parameters
- ❌ No early stopping (risk of overfitting)
- ❌ Hardcoded confidence threshold of 0.001 (too permissive)

#### After:
- ✅ **100 epochs** (configurable, can increase to 200+)
- ✅ **YOLOv8 medium (yolov8m)** for better accuracy (can use 'l' or 'x' for highest accuracy)
- ✅ **Auto GPU detection** - uses GPU if available, falls back to CPU
- ✅ **Enhanced data augmentation**:
  - HSV transformations (hue, saturation, value)
  - Rotation (±10°)
  - Translation (10%)
  - Scaling (50%)
  - Flips (horizontal, upside-down)
  - Mosaic augmentation for object context
- ✅ **Early stopping** - stops if no improvement for 20 epochs
- ✅ **Proper confidence thresholds** (0.5 for trained model, 0.25 for untrained)

### 2. **Model Inference Improvements**

| Aspect | Before | After |
|--------|--------|-------|
| Confidence Threshold | 0.001 (too low) | 0.5 for trained, 0.25 for untrained |
| IoU Threshold | 0.3 | 0.45 (standard NMS) |
| Model Size | nano (yolov8n) | medium (yolov8m) |
| Device Auto-detection | ❌ For CPU only | ✅ Detects GPU/CPU |
| Crop Padding | No padding | ±10px padding for context |

### 3. **Verdict Computation Logic**

#### Improved Risk Scoring:
```
Risk Score = (Defect_Proportion × 50) + (Avg_Defect_Confidence × 50)
Range: 0-100
```

#### Verdict Thresholds:
- **DEFECTIVE**: risk_score ≥ 70 (high confidence required)
- **CAUTION**: 40 ≤ risk_score < 70 (uncertain cases)
- **SAFE**: risk_score < 40 (low defect probability)

#### Previous Issues Fixed:
- ⚠️ Old formula: `(proportion × conf × 100) × 1.5` (up to 150, clamped to 100)
- 🔧 New formula: More balanced scoring favoring both quantity and confidence

### 4. **Dataset Validation Added**

New `validate_dataset()` function provides:
- ✅ Total image count
- ✅ Class distribution across train/val/test splits
- ✅ Class imbalance ratio detection
- ✅ Warnings for imbalanced datasets

Run: `python utils/prepare_dataset.py`

---

## 📈 Training Recommendations

### For Maximum Accuracy:
```python
train_custom_model(
    epochs=200,          # More iterations for convergence
    model_size="l",      # Larger model (yolov8l)
    batch_size=32,       # Larger batch size if GPU available
    imgsz=640           # Standard size
)
```

### For Balanced Performance:
```python
train_custom_model(
    epochs=100,          # Default recommendation
    model_size="m",      # Medium model
    batch_size=16,
    imgsz=640
)
```

### For Fast Training (Testing):
```python
train_custom_model(
    epochs=50,
    model_size="s",      # Smaller model
    batch_size=16,
    imgsz=416           # Smaller images for speed
)
```

---

## 🚀 How to Train the Model

### Option 1: Use Default Settings (Recommended)
```bash
python utils/train_yolo.py
```
- Trains for 100 epochs with yolov8m
- Auto-detects GPU/CPU
- Saves best weights to `models/best.pt`

### Option 2: Custom Configuration
```python
from utils.train_yolo import train_custom_model

results = train_custom_model(
    epochs=150,
    model_size="l",
    batch_size=32
)
```

### Option 3: Quick Resume from Existing Training
```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/last.pt")
results = model.train(resume=True, epochs=200)
```

---

## 📊 Expected Performance Improvements

### Metrics to Monitor During Training:
1. **mAP50**: Mean Average Precision @ IoU=0.5
2. **Precision**: True Positives / (True Positives + False Positives)
3. **Recall**: True Positives / (True Positives + False Negatives)
4. **Loss**: Should decrease with training

### Typical Improvement Timeline:
- **Epoch 1-10**: Rapid improvement
- **Epoch 10-50**: Steady gains
- **Epoch 50-100**: Marginal gains, approaching convergence
- **Epoch 100+**: Fine-tuning, diminishing returns

---

## 🐛 Bugs Fixed

### 1. **Hardcoded Ultra-Low Confidence Threshold**
- **Issue**: `conf=0.001` caused excessive false positives
- **Cause**: Model was undertrained (1 epoch)
- **Fix**: Dynamic threshold based on model training state (0.5 for trained, 0.25 for untrained)

### 2. **Nano Model Insufficient Capacity**
- **Issue**: yolov8n (nano) lacks capacity for accurate rail defect detection
- **Fix**: Upgraded to yolov8m (medium) by default, with option for yolov8l/x

### 3. **Device Detection Not Working**
- **Issue**: Always trained on CPU regardless of GPU availability
- **Fix**: Added `detect_device()` function with GPU auto-detection and CUDA support

### 4. **No Early Stopping**
- **Issue**: Model could overfit after convergence
- **Fix**: Added `patience=20` parameter (stops if no improvement for 20 epochs)

### 5. **Missing Data Augmentation Parameters**
- **Issue**: Default augmentation may be insufficient
- **Fix**: Explicitly defined comprehensive augmentation parameters:
  - HSV: h=0.015, s=0.7, v=0.4
  - Rotation: ±10°
  - Translation: 10%
  - Scale: 50%
  - Flips and Mosaic enabled

### 6. **Poor Verdict Logic**
- **Issue**: Risk score formula was unintuitive and thresholds were arbitrary
- **Fix**: Simplified scoring to explicitly balance defect proportion and confidence

### 7. **Missing Dataset Diagnostics**
- **Issue**: No visibility into dataset composition or imbalance
- **Fix**: Added `validate_dataset()` with comprehensive statistics

---

## 🔍 Debugging Checklist

### Before Training:
- [ ] Run `python utils/prepare_dataset.py` to validate dataset
- [ ] Check dataset imbalance ratio (should be < 2:1 for best results)
- [ ] Verify GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Ensure sufficient disk space (each training run: ~500MB-2GB)

### During Training:
- [ ] Monitor plots in `runs/detect/train*/` directory
- [ ] Check for exploding losses (indicates learning rate too high)
- [ ] Verify validation mAP increases monotonically (sign of good training)
- [ ] Watch for overfitting (train loss << val loss)

### After Training:
- [ ] Compare mAP50 metrics between model sizes
- [ ] Test inference with `python -m utils.yolo_module`
- [ ] Verify best.pt was copied to `models/best.pt`
- [ ] Run inference on known defective/non-defective samples

---

## 📋 File Changes Summary

### Modified Files:
1. **`utils/train_yolo.py`**
   - Added GPU detection
   - Increased default epochs: 1 → 100
   - Upgraded model: yolov8n → yolov8m
   - Added comprehensive augmentation params
   - Added early stopping and checkpoint saving

2. **`utils/yolo_module.py`**
   - Fixed hardcoded confidence threshold
   - Added dynamic threshold selection
   - Added GPU/CPU device selection
   - Improved crop extraction with padding
   - Better error handling for edge cases

3. **`utils/prepare_dataset.py`**
   - Added `validate_dataset()` function
   - Added class distribution reporting
   - Added imbalance detection warnings
   - Better logging and diagnostics

4. **`backend/main.py`**
   - Improved `compute_verdict()` function
   - Better risk score calculation
   - Updated verdict thresholds
   - Added confidence threshold tracking

---

## 🎯 Next Steps for Further Improvement

### 1. **Data Quality**
- Verify all images have proper labels
- Check for mislabeled images
- Consider manual review of borderline cases
- Add more diverse defect examples if available

### 2. **Class Balancing**
- If imbalance ratio > 2:1, consider:
  - Data augmentation for minority class
  - Class weights in loss function
  - Undersampling majority class

### 3. **Hyperparameter Tuning**
- Use YOLOv8's built-in hyperparameter tuning
- Test different learning rates (1e-4 to 1e-1)
- Experiment with different optimizers (SGD vs Adam)
- Try different image sizes (416, 512, 640, 768)

### 4. **Model Ensembling**
- Train multiple models with different seeds
- Average predictions from ensemble
- Typically improves performance by 1-2%

### 5. **Fine-tuning Strategy**
- Start with ImageNet pretrained weights
- Use lower learning rates for fine-tuning
- Freeze backbone layers initially, then unfreeze

---

## 📞 Troubleshooting

### **Problem**: Training is very slow
- **Solution**: Use GPU acceleration (CUDA 11.8+)
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- If False: Install CUDA-enabled PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### **Problem**: Model accuracy not improving
- **Solution**: 
  - Check dataset labels for errors
  - Ensure sufficient training data (ideally 100+ per class)
  - Increase augmentation aggressiveness
  - Train for more epochs

### **Problem**: Model overfitting (train loss << val loss)
- **Solution**:
  - Increase augmentation
  - Add more training data
  - Reduce model size
  - Use dropout/regularization

### **Problem**: Out of memory errors
- **Solution**:
  - Reduce batch size: `batch_size=8`
  - Reduce image size: `imgsz=416`
  - Use smaller model: `model_size="s"`

---

## 📚 Resources

- YOLOv8 Documentation: https://docs.ultralytics.com/
- Object Detection Best Practices: https://github.com/ultralytics/yolov5/wiki
- YOLO Hyperparameter Tuning: https://docs.ultralytics.com/tasks/detect/#train
- PyTorch CUDA Setup: https://pytorch.org/get-started/locally/

---

**Last Updated**: March 24, 2026
**Version**: 2.0 (Improved Training & Debugging)
