import os
import shutil
import glob
from pathlib import Path
from collections import defaultdict

def validate_dataset(target_dir="rail_yolo"):
    """
    Validates the YOLO dataset and prints diagnostic information.
    """
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    
    class_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(target_dir, "labels", split)
        if not os.path.exists(split_path):
            continue
            
        label_files = glob.glob(os.path.join(split_path, "*.txt"))
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id][split] += 1
                        total_images += 1
    
    # Print statistics
    class_names = {0: "Defective", 1: "Non defective"}
    
    print(f"\nTotal Images: {total_images}")
    print("\nClass Distribution:")
    print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 55)
    
    for class_id in sorted(class_counts.keys()):
        counts = class_counts[class_id]
        total = counts['train'] + counts['val'] + counts['test']
        print(f"{class_names.get(class_id, f'Class {class_id}'):<20} "
              f"{counts['train']:<10} {counts['val']:<10} {counts['test']:<10} {total:<10}")
    
    # Check for class imbalance
    all_counts = [sum(counts.values()) for counts in class_counts.values()]
    if all_counts:
        max_count = max(all_counts)
        min_count = min(all_counts)
        if max_count > 0:
            ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"\nClass Imbalance Ratio: {ratio:.2f}:1")
            if ratio > 2:
                print("⚠ Warning: Significant class imbalance detected!")
                print("  Consider using class weights in training or data balancing.")
    
    print("\n" + "=" * 60 + "\n")

def setup_yolo_dataset(source_dir="rail", target_dir="rail_yolo"):
    """
    Converts a classification dataset (folders: train/Defective, train/Non defective)
    into a YOLOv8 Object Detection dataset by creating full-image bounding boxes.
    
    Args:
        source_dir: Source directory with classification-style structure
        target_dir: Target directory for YOLO detection format
    """
    classes = ["Defective", "Non defective"]
    
    print(f"\nSetting up YOLO dataset from '{source_dir}' to '{target_dir}'...")
    
    # Create target directories
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(target_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "labels", split), exist_ok=True)
        
        # In the source dir 'valid' is 'valid' but sometimes 'test'
        source_split = split
        if not os.path.exists(os.path.join(source_dir, source_split)):
            continue

        for cls_idx, cls_name in enumerate(classes):
            source_cls_dir = os.path.join(source_dir, source_split, cls_name)
            if not os.path.exists(source_cls_dir):
                continue
                
            img_files = glob.glob(os.path.join(source_cls_dir, "*.*"))
            
            for img_path in img_files:
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')):
                    continue
                    
                filename = os.path.basename(img_path)
                # Copy image
                target_img_path = os.path.join(target_dir, "images", split, filename)
                shutil.copy(img_path, target_img_path)
                
                # Create label (Full image bounding box: class_id x_center y_center width height)
                label_filename = os.path.splitext(filename)[0] + ".txt"
                target_label_path = os.path.join(target_dir, "labels", split, label_filename)
                
                with open(target_label_path, "w") as f:
                    # YOLO format: class_id center_x center_y width height (normalized 0-1)
                    f.write(f"{cls_idx} 0.5 0.5 1.0 1.0\n")
                    
    print("✓ Dataset folder structure created and images copied.")

    # Generate data.yaml with improved configuration
    yaml_content = f"""path: ../{target_dir}
train: images/train
val: images/valid
test: images/test

nc: 2
names:
  0: Defective
  1: Non defective
"""
    yaml_path = os.path.join(target_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())
        
    print(f"✓ data.yaml generated at {yaml_path}")
    
    # Validate and report
    validate_dataset(target_dir)

if __name__ == "__main__":
    setup_yolo_dataset()
