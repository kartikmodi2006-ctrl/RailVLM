import os
import shutil
import glob
from pathlib import Path

def setup_yolo_dataset(source_dir="rail", target_dir="rail_yolo"):
    """
    Converts a classification dataset (folders: train/Defective, train/Non defective)
    into a YOLOv8 Object Detection dataset by creating full-image bounding boxes.
    """
    classes = ["Defective", "Non defective"]
    
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
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    continue
                    
                filename = os.path.basename(img_path)
                # Copy image
                target_img_path = os.path.join(target_dir, "images", split, filename)
                shutil.copy(img_path, target_img_path)
                
                # Create label (Full image bounding box: class_id x_center y_center width height)
                label_filename = os.path.splitext(filename)[0] + ".txt"
                target_label_path = os.path.join(target_dir, "labels", split, label_filename)
                
                with open(target_label_path, "w") as f:
                    f.write(f"{cls_idx} 0.5 0.5 1.0 1.0\n")
                    
    print("Dataset converted successfully!")

    # Generate data.yaml
    yaml_content = f"""
path: ../{target_dir}
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
        
    print(f"data.yaml generated at {yaml_path}")

if __name__ == "__main__":
    setup_yolo_dataset()
