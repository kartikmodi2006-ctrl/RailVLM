#!/usr/bin/env python3
"""
Rail YOLO System Diagnostics & Health Check
Identifies configuration issues, missing dependencies, and model problems
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_python_env():
    """Check Python environment"""
    print_header("1. Python Environment")
    
    print(f"✓ Python version: {sys.version}")
    print(f"✓ Python executable: {sys.executable}")
    print(f"✓ Working directory: {os.getcwd()}")
    
    return True

def check_dependencies():
    """Check required packages"""
    print_header("2. Dependencies")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'ultralytics': 'YOLOv8',
        'transformers': 'Transformers'
    }
    
    all_ok = True
    
    for package_name, display_name in required_packages.items():
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name:<20} ({package_name}): {version}")
        except ImportError:
            print(f"✗ {display_name:<20} ({package_name}): NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Check project folder structure"""
    print_header("3. Project Structure")
    
    required_dirs = {
        'backend': 'Backend modules',
        'frontend': 'Frontend files',
        'utils': 'Utility scripts',
        'models': 'Model weights (will be created if needed)',
        'rail_yolo': 'YOLO dataset',
        'runs': 'Training outputs (will be created)',
    }
    
    all_ok = True
    
    for dir_name, description in required_dirs.items():
        exists = os.path.isdir(dir_name)
        status = "✓" if exists else "✗"
        print(f"{status} {dir_name:<15} - {description}")
        if not exists and dir_name not in ['models', 'runs']:
            all_ok = False
    
    # Create missing directories
    for dir_name in ['models', 'runs']:
        os.makedirs(dir_name, exist_ok=True)
    
    return all_ok

def check_files():
    """Check required files"""
    print_header("4. Required Files")
    
    required_files = {
        'backend/main.py': 'FastAPI backend',
        'frontend/index.html': 'Web interface',
        'utils/train_yolo.py': 'Training script',
        'utils/yolo_module.py': 'YOLO inference module',
        'utils/prepare_dataset.py': 'Dataset preparation',
        'rail_yolo/data.yaml': 'Dataset configuration',
    }
    
    all_ok = True
    
    for file_path, description in required_files.items():
        exists = os.path.isfile(file_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file_path:<30} - {description}")
        if not exists:
            all_ok = False
    
    return all_ok

def check_model_weights():
    """Check model weights availability"""
    print_header("5. Model Weights")
    
    weights_to_check = {
        'models/best.pt': 'Custom trained model',
        'yolov8n.pt': 'YOLOv8 Nano (fallback)',
        'yolov8m.pt': 'YOLOv8 Medium',
        'yolov8s.pt': 'YOLOv8 Small',
    }
    
    for weights_path, description in weights_to_check.items():
        exists = os.path.isfile(weights_path)
        status = "✓" if exists else "✗"
        if exists:
            size_mb = os.path.getsize(weights_path) / (1024 * 1024)
            print(f"{status} {weights_path:<25} - {description:<30} ({size_mb:.1f} MB)")
        else:
            print(f"{status} {weights_path:<25} - {description:<30} (not found)")
    
    has_custom = os.path.isfile('models/best.pt')
    has_fallback = os.path.isfile('yolov8m.pt') or os.path.isfile('yolov8n.pt')
    
    return has_custom or has_fallback

def check_dataset():
    """Check dataset structure and statistics"""
    print_header("6. Dataset Information")
    
    dataset_path = 'rail_yolo'
    
    if not os.path.isdir(dataset_path):
        print(f"✗ Dataset directory not found: {dataset_path}")
        return False
    
    print(f"✓ Dataset path: {dataset_path}")
    
    splits = ['train', 'valid', 'test']
    total_images = 0
    
    print("\nDataset contents:")
    for split in splits:
        images_dir = os.path.join(dataset_path, 'images', split)
        labels_dir = os.path.join(dataset_path, 'labels', split)
        
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            image_count = len([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            label_count = len([f for f in os.listdir(labels_dir) 
                             if f.endswith('.txt')])
            total_images += image_count
            
            match = "✓" if image_count == label_count else "⚠"
            print(f"  {match} {split}: {image_count} images, {label_count} labels")
    
    print(f"\n✓ Total images: {total_images}")
    
    return total_images > 0

def check_gpu():
    """Check GPU availability"""
    print_header("7. GPU/CUDA Support")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            
            # Memory check
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name} with {total_memory:.1f} GB memory")
            
            return True
        else:
            print("⚠ CUDA not available - will use CPU (training will be slow)")
            print("\nTo enable GPU support:")
            print("1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
            print("2. Reinstall PyTorch with CUDA support:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
    
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_imports():
    """Test critical module imports"""
    print_header("8. Module Imports")
    
    modules_to_test = [
        ('utils.train_yolo', 'Training module'),
        ('utils.yolo_module', 'YOLO module'),
        ('utils.prepare_dataset', 'Dataset preparation'),
        ('backend.main', 'Backend API'),
        ('utils.vlm_module', 'VLM explanation module'),
        ('utils.xai_module', 'XAI visualization module'),
    ]
    
    all_ok = True
    
    for module_path, description in modules_to_test:
        try:
            __import__(module_path)
            print(f"✓ {module_path:<30} - {description}")
        except Exception as e:
            print(f"✗ {module_path:<30} - {description}")
            print(f"  Error: {str(e)[:60]}")
            all_ok = False
    
    return all_ok

def print_summary(results):
    """Print diagnostic summary"""
    print_header("Summary & Recommendations")
    
    checks = [
        ('Python Environment', results.get('python', True)),
        ('Dependencies', results.get('deps', True)),
        ('Project Structure', results.get('structure', True)),
        ('Required Files', results.get('files', True)),
        ('Model Weights', results.get('weights', True)),
        ('Dataset', results.get('dataset', True)),
        ('GPU Support', results.get('gpu', True)),
        ('Module Imports', results.get('imports', True)),
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    print(f"\nStatus: {passed}/{total} checks passed\n")
    
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:<8} - {check_name}")
    
    if passed == total:
        print("\n✅ All systems operational! You can start training:")
        print("\n  1. Prepare dataset (if not done):")
        print("     python train.py --prepare --validate")
        print("\n  2. Start training with default settings:")
        print("     python train.py")
        print("\n  3. Or use custom settings:")
        print("     python train.py --epochs 200 --model l --batch 32")
        print("\n  4. Test inference on images:")
        print("     python test_inference.py --dir rail/test/")
        print("\n  5. Start web app (in another terminal):")
        print("     python -m uvicorn backend.main:app --reload")
    else:
        print("\n⚠️  Some issues detected. Please fix them before training.")
        print("\nCommon fixes:")
        print("1. Install missing packages:")
        print("   pip install -r requirements.txt --upgrade")
        print("\n2. Check file paths are correct")
        print("\n3. For GPU support, install CUDA and reinstall PyTorch")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")

def main():
    """Run all diagnostics"""
    print("\n" + "="*70)
    print("  Rail YOLO System Diagnostics & Health Check")
    print("="*70)
    
    results = {
        'python': check_python_env(),
        'deps': check_dependencies(),
        'structure': check_project_structure(),
        'files': check_files(),
        'weights': check_model_weights(),
        'dataset': check_dataset(),
        'gpu': check_gpu(),
        'imports': check_imports(),
    }
    
    print_summary(results)
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
