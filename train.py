#!/usr/bin/env python3
"""
Rail YOLO Model Training Launcher
Provides an easy interface for training with different configurations
"""

import sys
import argparse
from utils.train_yolo import train_custom_model
from utils.prepare_dataset import setup_yolo_dataset, validate_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for rail defect detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training (default settings)
  python train.py
  
  # High accuracy training
  python train.py --epochs 200 --model l --batch 32
  
  # Fast training for testing
  python train.py --epochs 50 --model s --batch 8
  
  # GPU training with medium model
  python train.py --epochs 150 --model m --batch 24
        """
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100, recommended: 50-200)"
    )
    
    parser.add_argument(
        "--model",
        choices=["n", "s", "m", "l", "x"],
        default="m",
        help="""Model size:
        n=nano (fastest, lowest accuracy)
        s=small (fast, lower accuracy)
        m=medium (balanced, default)
        l=large (slow, higher accuracy)
        x=xlarge (slowest, highest accuracy)"""
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16, use 8 for CPU, 32+ for GPU)"
    )
    
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640, use 416 for speed)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "0"],
        help="Training device (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare dataset before training"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset and show statistics"
    )
    
    args = parser.parse_args()
    
    # Prepare dataset if requested
    if args.prepare:
        print("\n" + "="*60)
        print("PREPARING DATASET")
        print("="*60)
        setup_yolo_dataset()
        print("\n")
    
    # Validate dataset
    if args.validate or args.prepare:
        print("\n" + "="*60)
        print("VALIDATING DATASET")
        print("="*60)
        validate_dataset()
    
    # Show training configuration
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: yolov8{args.model} (size: {args.model})")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device: {args.device or 'auto-detected'}")
    print("="*60)
    
    input("\nPress Enter to start training...")
    
    # Train model
    try:
        results = train_custom_model(
            epochs=args.epochs,
            model_size=args.model,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device
        )
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Best weights saved to: models/best.pt")
        print("\nTo use the trained model:")
        print("  - It will be automatically loaded in the web app")
        print("  - Restart the FastAPI backend to use new weights")
        print("\nTo continue training:")
        print("  - Run: python train.py --epochs 50 (additional epochs)")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
