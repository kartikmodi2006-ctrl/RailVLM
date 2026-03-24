# Quick Start Guide - Rail YOLO Model Training

## 📋 Before You Start

1. **Check System Setup** (Required!)
   ```bash
   python diagnose.py
   ```
   This will verify all dependencies, GPU availability, and dataset configuration.

2. **Prepare Dataset** (First time only)
   ```bash
   python train.py --prepare --validate
   ```
   This creates the YOLO format dataset and shows statistics.

---

## 🚀 Training the Model

### Option A: Default Training (Recommended)
```bash
python train.py
```
- 100 epochs
- YOLOv8 Medium model
- Auto GPU/CPU detection
- Expected time: 2-8 hours depending on hardware

### Option B: Quick Test Training
```bash
python train.py --epochs 30 --model s
```
- Fast training for testing
- Smaller model
- Good for debugging
- Expected time: 30 minutes

### Option C: Maximum Accuracy Training
```bash
python train.py --epochs 200 --model l --batch 32
```
- More epochs for convergence
- Larger model for better accuracy
- Requires good GPU (24GB+ VRAM)
- Expected time: 8-24 hours

### Option D: Custom Configuration
```bash
python train.py --epochs 150 --model m --batch 16 --device 0
```
- Customize all parameters
- Run `python train.py --help` for all options

---

## 🧪 Testing & Inference

### Test Inference on Sample Images
```bash
python test_inference.py
```
Tests on default test directories (rail/test/)

### Test Specific Image
```bash
python test_inference.py --image path/to/image.jpg
```

### Test Directory of Images
```bash
python test_inference.py --dir path/to/images/
```

### Adjust Confidence Threshold
```bash
python test_inference.py --conf 0.6
```
Higher confidence = fewer but more confident detections

---

## 🌐 Run Web Application

In a **new terminal**:
```bash
python -m uvicorn backend.main:app --reload
```

Then open: http://localhost:8000

The app will automatically use the trained model from `models/best.pt`

---

## 📊 Monitoring Training Progress

During training, check the `runs/detect/train/` directory:
- **results.csv**: Training and validation metrics
- **plots/**: Confusion matrix, precision-recall, losses
- **weights/best.pt**: Best model checkpoint (auto-copied to `models/best.pt`)

### View Results
Open `runs/detect/train/results.csv` with Excel or Python:
```python
import pandas as pd
df = pd.read_csv('runs/detect/train/results.csv')
print(df[['epoch', 'train/loss', 'val/loss', 'metrics/mAP50']])
```

---

## 🔥 Important Training Parameters

| Parameter | Values | Impact |
|-----------|--------|--------|
| `--epochs` | 50-200 | More epochs = better accuracy (but slower) |
| `--model` | n,s,m,l,x | Larger model = higher accuracy (but slower) |
| `--batch` | 8,16,32 | Larger batch = faster (but needs more memory) |
| `--imgsz` | 416,512,640 | Larger size = more detail (but slower) |

### Recommended Combinations:

**CPU Training:**
- `--epochs 50 --model s --batch 8 --imgsz 416`

**GPU with 8GB VRAM:**
- `--epochs 100 --model s --batch 16 --imgsz 640`

**GPU with 16GB+ VRAM:**
- `--epochs 150 --model m --batch 32 --imgsz 640`

**GPU with 24GB+ VRAM:**
- `--epochs 200 --model l --batch 48 --imgsz 768`

---

## ✅ Expected Outcomes

### After First Training (100 epochs):
- mAP50: 70-85% (depending on data quality)
- Training time: 2-8 hours
- Model size: ~50 MB (medium model)

### Signs of Good Training:
- ✓ Loss decreases over time
- ✓ Validation mAP increases
- ✓ Overfitting occurs after epoch 100+ (expected for small datasets)
- ✓ No NaN or infinity values in loss

### Common Issues & Fixes:

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA out of memory | Batch too large | Reduce `--batch` to 8 |
| No improvement | Learning rate issues | Train for more epochs |
| Model not loading | Wrong path | Verify `models/best.pt` exists |
| Slow training | CPU only | Check `python diagnose.py` for GPU |

---

## 📁 Output Files Structure

```
models/
├── best.pt              ← Use this for inference
└── last.pt             ← Latest checkpoint

runs/detect/train/
├── weights/
│   ├── best.pt        ← Best model
│   └── last.pt        ← Last epoch
├── results.csv        ← Training metrics
├── plots/
│   ├── confusion_matrix.png
│   ├── results.png
│   └── F1_curve.png
└── args.yaml          ← Training configuration
```

---

## 🐛 Debugging

### Full Diagnostics:
```bash
python diagnose.py
```

### Test Model Loading:
```python
from utils.yolo_module import YOLOModel
model = YOLOModel()
print("✓ Model loaded successfully")
```

### Check Dataset:
```bash
python utils/prepare_dataset.py
```

### View Training Logs:
```bash
cat runs/detect/train/results.csv | head -20
```

---

## 🚀 Next Steps After Training

1. **Deploy Model:**
   - Start web app and test on real images
   - Verify accuracy on test set
   
2. **Fine-tune if Needed:**
   - If accuracy < 70%, retrain with:
     - More epochs (increase to 200)
     - Larger model (use --model l)
     - Better augmentation

3. **Production Deployment:**
   - Package model for deployment
   - Set up inference API
   - Monitor model performance

---

## 📚 Useful Commands Reference

```bash
# Diagnostic check
python diagnose.py

# Prepare dataset
python utils/prepare_dataset.py

# Train model
python train.py --help                          # Show all options
python train.py                                 # Default training
python train.py --epochs 200 --model l          # Best accuracy
python train.py --epochs 50 --model s           # Fast training

# Test inference
python test_inference.py                        # Test on default images
python test_inference.py --image path/to/img.jpg
python test_inference.py --dir path/to/images/
python test_inference.py --conf 0.6             # Custom threshold

# Run web app
python -m uvicorn backend.main:app --reload     # http://localhost:8000

# Install dependencies
pip install -r requirements.txt --upgrade

# Update YOLOv8
pip install ultralytics --upgrade
```

---

## ⚠️ Important Notes

- **GPU Recommended**: Training on CPU takes 10x longer
- **Data Quality**: Model accuracy depends heavily on label quality
- **Class Balance**: If one class has >2x more samples, consider balancing
- **Validation**: Always test on held-out test set
- **Monitor Memory**: Watch GPU memory during training

---

## 📞 Troubleshooting

### Problem: "No module named 'utils'"
→ Run the script from the project root directory

### Problem: CUDA out of memory
→ Reduce batch size: `python train.py --batch 8`

### Problem: Training loss increases
→ Learning rate might be too high. Let training continue for more epochs.

### Problem: Model accuracy is low
→ Check dataset labels are correct, increase epochs/model size

---

**Last Updated**: March 24, 2026
**Version**: 2.0 (Improved Training Guide)

For detailed technical improvements, see: [TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md)
