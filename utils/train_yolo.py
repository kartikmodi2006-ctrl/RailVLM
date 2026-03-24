from ultralytics import YOLO
import os
import shutil
import glob
import torch

def detect_device():
    """
    Auto-detects available device: GPU if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = "0"  # Use first GPU
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("GPU not available, using CPU for training")
    return device

def train_custom_model(
    data_yaml="rail_yolo/data.yaml", 
    epochs=100,  # Increased from 1 to 100 epochs for better convergence
    model_size="m",  # Changed from 'n' (nano) to 'm' (medium) for better capacity
    imgsz=640,
    batch_size=16,  # Adaptive batch size (will be adjusted based on device)
    device=None
):
    """
    Improved training script for YOLOv8 on railway defect detection dataset.
    
    Args:
        data_yaml: Path to data.yaml config
        epochs: Number of training epochs (default 100)
        model_size: Model size - 'n'(nano), 's'(small), 'm'(medium), 'l'(large), 'x'(xlarge)
        imgsz: Input image size
        batch_size: Batch size for training
        device: Device to use (auto-detected if None)
    """
    
    if device is None:
        device = detect_device()
    
    # Adjust batch size for CPU training
    if device == "cpu":
        batch_size = 8
        print("Running on CPU - batch size reduced to 8")
    else:
        print(f"Running on GPU - batch size set to {batch_size}")
    
    model_name = f"yolov8{model_size}.pt"
    print(f"\nLoading base pre-trained {model_name} model...")
    model = YOLO(model_name)
    
    print(f"Starting training for {epochs} epochs with improved parameters...")
    print(f"Model size: {model_size}, Image size: {imgsz}, Batch size: {batch_size}")
    print("=" * 60)
    
    # Enhanced training parameters for better accuracy
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        
        # Improved hyperparameters
        patience=20,  # Early stopping: stop if no improvement for 20 epochs
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        
        # Data augmentation parameters for better generalization
        hsv_h=0.015,  # HSV hue augmentation
        hsv_s=0.7,    # HSV saturation augmentation
        hsv_v=0.4,    # HSV value augmentation
        degrees=10.0,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,     # Scale augmentation
        flipud=0.5,    # Flip upside down
        fliplr=0.5,    # Flip left-right
        mosaic=1.0,    # Mosaic augmentation
        mixup=0.0,     # MixUp augmentation (disabled for this task)
        
        # Optimization parameters
        optimizer="SGD",  # SGD or Adam
        lr0=0.01,        # Initial learning rate
        lrf=0.01,        # Final learning rate
        momentum=0.937,  # Momentum
        weight_decay=0.0005,  # Weight decay for regularization
        
        # Validation and monitoring
        val=True,
        split=0.1,  # 10% validation from training data
        overlap_mask=True,
        
        # Confidence and IoU thresholds
        conf=0.5,  # Confidence threshold during training
        iou=0.6,   # IoU threshold during training
        
        # Logging and verbosity
        verbose=True,
        plots=True,  # Generate training plots
        
        # Device-specific optimizations
        half=True if device != "cpu" else False,  # Half precision (FP16) on GPU
        cache=True,  # Cache images for faster loading
    )
    
    print("\n" + "=" * 60)
    print("Training finished!")
    print("=" * 60)
    
    # Print training results summary
    if hasattr(results, 'results_dict'):
        print("\nTraining Results Summary:")
        for key, value in results.results_dict.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    # Copy best weights to models directory
    best_weights = "runs/detect/train/weights/best.pt"
    if os.path.exists(best_weights):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_weights, "models/best.pt")
        print(f"\n✓ Copied best.pt to models/best.pt")
        print(f"  File size: {os.path.getsize('models/best.pt') / 1024 / 1024:.2f} MB")
    else:
        # Check if it created a new run folder like train2, train3
        runs = sorted(glob.glob("runs/detect/train*/weights/best.pt"))
        if runs:
            latest = runs[-1]
            os.makedirs("models", exist_ok=True)
            shutil.copy(latest, "models/best.pt")
            print(f"\n✓ Copied {latest} to models/best.pt")
            print(f"  File size: {os.path.getsize('models/best.pt') / 1024 / 1024:.2f} MB")
    
    return results

if __name__ == "__main__":
    # IMPROVED SETTINGS - Change these parameters to customize training:
    # - epochs: increase for better accuracy (50-200 recommended)
    # - model_size: 'n' (fast), 's', 'm' (balanced), 'l', 'x' (most accurate)
    # - batch_size: larger batches improve stability but need more memory
    
    results = train_custom_model(
        epochs=100,           # Increased from 1 to 100
        model_size="m",       # Changed from 'n' to 'm' for better capacity
        batch_size=16,
        imgsz=640
    )
