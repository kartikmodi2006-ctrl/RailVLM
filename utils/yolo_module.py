import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, weights_path="models/best.pt", fallback_model="yolov8n.pt"):
        """
        Loads the YOLOv8 model. Uses custom weights if available, else standard fallback.
        """
        if os.path.exists(weights_path):
            self.model = YOLO(weights_path)
            print(f"Loaded YOLOv8 with custom weights from {weights_path}")
        else:
            self.model = YOLO(fallback_model)
            print(f"Loaded placeholder YOLOv8 model ({fallback_model})")

    def detect(self, image: np.ndarray):
        """
        Runs YOLOv8 inference.
        Returns the Ultralytics Results object.
        """
        # Run inference with extremely low confidence threshold (0.01) 
        # because the custom model was only trained for 1 epoch and produces low confidence scores.
        results = self.model.predict(source=image, conf=0.001, iou=0.3, max_det=200, save=False)
        return results[0]

    def get_crops_and_info(self, result, original_image_rgb: np.ndarray):
        """
        Extracts bounding box data and cropped PIL Images for each detection.
        """
        boxes = result.boxes
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            
            # Extract crop for VLM processing
            crop = original_image_rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            crop_pil = Image.fromarray(crop)
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_name": class_name,
                "crop_pil": crop_pil
            })
        return detections
