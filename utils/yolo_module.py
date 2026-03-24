import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, weights_path="models/best.pt", fallback_model="yolov8m.pt"):
        """
        Loads the YOLOv8 model. Uses custom weights if available, else standard fallback.
        
        Args:
            weights_path: Path to custom trained weights
            fallback_model: Default model to use if custom weights not found
        """
        self.weights_path = weights_path
        self.fallback_model = fallback_model
        
        if os.path.exists(weights_path):
            self.model = YOLO(weights_path)
            print(f"✓ Loaded YOLOv8 with custom weights from {weights_path}")
            self.is_custom_model = True
        else:
            self.model = YOLO(fallback_model)
            print(f"⚠ Custom weights not found, loaded fallback model ({fallback_model})")
            self.is_custom_model = False
        
        # Set device
        import torch
        self.device = "0" if torch.cuda.is_available() else "cpu"

    def detect(self, image: np.ndarray, conf_threshold=None, iou_threshold=0.45):
        """
        Runs YOLOv8 inference with improved confidence handling.
        
        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold (0.25-0.5 recommended).
                           Auto-adjusted based on model type.
            iou_threshold: IoU threshold for NMS (default 0.45)
        
        Returns:
            Ultralytics Results object
        """
        # Auto-select confidence threshold based on whether using trained model
        if conf_threshold is None:
            # Use higher threshold for custom trained model (lower for untrained)
            conf_threshold = 0.5 if self.is_custom_model else 0.25
        
        # Run inference with improved parameters
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=200,  # Allow up to 200 detections
            save=False,
            device=self.device
        )
        return results[0]

    def get_crops_and_info(self, result, original_image_rgb: np.ndarray):
        """
        Extracts bounding box data and cropped PIL Images for each detection.
        
        Args:
            result: YOLOv8 results object
            original_image_rgb: Original image as numpy RGB array
        
        Returns:
            List of detection dictionaries with bbox, confidence, class_name, and crop_pil
        """
        boxes = result.boxes
        detections = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            
            # Extract crop for VLM processing with padding to avoid edge artifacts
            crop_h, crop_w = original_image_rgb.shape[:2]
            pad = 10  # Small padding around detection
            y1_padded = max(0, y1 - pad)
            x1_padded = max(0, x1 - pad)
            y2_padded = min(crop_h, y2 + pad)
            x2_padded = min(crop_w, x2 + pad)
            
            crop = original_image_rgb[y1_padded:y2_padded, x1_padded:x2_padded]
            
            # Handle edge cases for very small crops
            if crop.size > 0:
                crop_pil = Image.fromarray(crop)
            else:
                crop_pil = Image.fromarray(original_image_rgb)
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_name": class_name,
                "crop_pil": crop_pil
            })
        
        return detections
