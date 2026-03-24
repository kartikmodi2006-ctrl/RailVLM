#!/usr/bin/env python3
"""
Quick Inference Testing Script
Test the trained YOLO model on local images for debugging and validation
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

def test_inference():
    """Test YOLOv8 model inference on images"""
    
    parser = argparse.ArgumentParser(
        description="Test YOLO inference on images"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image"
    )
    
    parser.add_argument(
        "--dir",
        type=str,
        help="Directory with test images"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Path to model weights (default: models/best.pt)"
    )
    
    args = parser.parse_args()
    
    # Import model
    try:
        from utils.yolo_module import YOLOModel
        from backend.main import compute_verdict
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = YOLOModel(weights_path=args.model)
    
    # Get test images
    images_to_test = []
    
    if args.image:
        if os.path.exists(args.image):
            images_to_test.append(args.image)
        else:
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
    
    elif args.dir:
        if os.path.isdir(args.dir):
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
                images_to_test.extend(Path(args.dir).glob(ext))
        else:
            print(f"❌ Directory not found: {args.dir}")
            sys.exit(1)
    
    else:
        # Default test directories
        test_dirs = [
            "rail/test/Defective",
            "rail/test/Non defective",
            "rail_yolo/images/test"
        ]
        
        for test_dir in test_dirs:
            if os.path.isdir(test_dir):
                for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
                    images_to_test.extend(Path(test_dir).glob(ext))
    
    if not images_to_test:
        print("❌ No test images found!")
        print("\nUsage:")
        print("  python test_inference.py --image path/to/image.jpg")
        print("  python test_inference.py --dir path/to/images/")
        print("  python test_inference.py (uses default test directories)")
        sys.exit(1)
    
    print(f"Found {len(images_to_test)} images to test")
    print("=" * 70)
    
    # Test each image
    for i, image_path in enumerate(sorted(images_to_test)[:10], 1):  # Limit to 10 for quick test
        print(f"\n[{i}] Testing: {image_path}")
        print("-" * 70)
        
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  ⚠️  Could not read image")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run inference
            print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
            print(f"  Confidence threshold: {args.conf}")
            
            result = model.detect(image_rgb, conf_threshold=args.conf)
            detections = model.get_crops_and_info(result, image_rgb)
            
            # Compute verdict
            verdict = compute_verdict(detections)
            
            # Print results
            print(f"\n  Results:")
            print(f"    Detections found: {len(detections)}")
            print(f"    - Defective: {verdict['defect_count']}")
            print(f"    - Non-defective: {verdict['non_defect_count']}")
            print(f"    Risk score: {verdict['risk_score']}/100")
            print(f"    Verdict: {verdict['verdict']}")
            
            if detections:
                print(f"\n  Detection details:")
                for j, det in enumerate(detections[:5], 1):  # Show top 5
                    print(f"    [{j}] {det['class_name']}")
                    print(f"        Confidence: {det['confidence']:.3f}")
                    print(f"        BBox: {det['bbox']}")
                
                if len(detections) > 5:
                    print(f"    ... and {len(detections) - 5} more detections")
            else:
                print(f"  ℹ️  No detections")
        
        except Exception as e:
            print(f"  ❌ Error processing image: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Inference testing completed")
    print("\nNotes:")
    print("- Confidence threshold can be adjusted with --conf flag")
    print("- Model path can be changed with --model flag")
    print("- For more detailed analysis, integrate with the web API")

if __name__ == "__main__":
    test_inference()
