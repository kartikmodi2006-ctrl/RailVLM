import cv2
import numpy as np
from PIL import Image


class XAIOverlay:
    """
    Generates a visually meaningful AI attention heatmap.

    Strategy:
    - If YOLO detections are provided → draw smooth Gaussian blobs at
      each detected bounding box, scaled by confidence score.
    - If no detections → use edge-gradient saliency (Laplacian) to show
      which areas of the track have the most complex texture/features.

    This completely avoids EigenCAM/pytorch-grad-cam compatibility issues
    with newer YOLOv8 versions and always produces a colorized heatmap.
    """

    def __init__(self, detection_model=None):
        """
        detection_model is accepted but not used — kept for API compatibility
        with main.py so no changes are needed there.
        """
        print("Loaded XAI Overlay (detection-aware heatmap engine)")

    def _gaussian_blob(self, heatmap: np.ndarray, cx: int, cy: int,
                       radius: int, intensity: float) -> np.ndarray:
        """Adds a smooth 2D Gaussian blob to a heatmap at (cx, cy)."""
        h, w = heatmap.shape
        y_idx, x_idx = np.ogrid[:h, :w]
        dist_sq = (x_idx - cx) ** 2 + (y_idx - cy) ** 2
        sigma_sq = (radius ** 2) / (2 * np.log(max(intensity + 1e-6, 1e-6) + 2))
        blob = intensity * np.exp(-dist_sq / (2 * max(sigma_sq, 1e-3)))
        heatmap = np.maximum(heatmap, blob.astype(np.float32))
        return heatmap

    def _gradient_saliency(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Fallback when no detections: builds a saliency map from edge gradients.
        Shows areas of high texture complexity — where the AI would focus most.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Sobel gradients (horizontal + vertical)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Laplacian for high-frequency detail (cracks show up here)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

        # Combine
        saliency = magnitude * 0.6 + laplacian * 0.4

        # Smooth and normalize
        saliency = cv2.GaussianBlur(saliency, (21, 21), 0)
        max_val = saliency.max()
        if max_val > 0:
            saliency = saliency / max_val
        return saliency

    def generate_heatmap(self, image_rgb: np.ndarray,
                         detections: list = None) -> np.ndarray:
        """
        Generates and returns an RGB heatmap overlay of the same size as image_rgb.

        Args:
            image_rgb: Original image as uint8 RGB numpy array.
            detections: List of detection dicts with keys 'bbox', 'confidence'.
                        If None or empty, uses gradient saliency fallback.
        """
        h, w = image_rgb.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        if detections:
            # Build Gaussian blobs at each detection box
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = float(det.get("confidence", 0.5))
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = max(abs(x2 - x1), abs(y2 - y1)) // 2
                radius = max(radius, 20)  # minimum blob size
                heatmap = self._gaussian_blob(heatmap, cx, cy, radius,
                                              intensity=max(conf, 0.3))

            # Normalize
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
        else:
            # No detections — fall back to gradient saliency
            heatmap = self._gradient_saliency(image_rgb)

        # Apply colormap (JET: blue=low, green=mid, red=high)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend: 55% original + 45% heatmap
        overlay = cv2.addWeighted(image_rgb.astype(np.uint8), 0.55,
                                  heatmap_colored_rgb, 0.45, 0)
        return overlay
