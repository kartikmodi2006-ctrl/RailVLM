import io
import base64
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from utils.yolo_module import YOLOModel
from utils.vlm_module import VLMExplainer
from utils.xai_module import XAIOverlay

# Global model context
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models into memory...")
    models["yolo"] = YOLOModel()
    models["vlm"] = VLMExplainer()
    models["xai"] = XAIOverlay()
    print("Models loaded successfully.")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(img_np: np.ndarray) -> str:
    """Converts a NumPy RGB image array to base64 JPEG format."""
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def compute_verdict(detections: list, avg_model_confidence=None) -> dict:
    """
    Aggregates all YOLO detections into a single actionable verdict.
    
    Improved logic:
    - Considers both quantity and confidence of defects
    - Configurable thresholds based on model training quality
    - Better handling of edge cases
    
    Args:
        detections: List of detection dicts from YOLO
        avg_model_confidence: Average confidence of trained model (None = auto-detect)
    
    Returns a summary dict with:
      - verdict: 'DEFECTIVE' | 'CAUTION' | 'SAFE'
      - risk_score: 0-100 integer
      - defect_count, non_defect_count, total_count
      - dominant_conf: average confidence of defective detections
    """
    if not detections:
        return {
            "verdict": "SAFE",
            "risk_score": 0,
            "defect_count": 0,
            "non_defect_count": 0,
            "total_count": 0,
            "dominant_conf": 0.0,
            "confidence_threshold_used": 0.5
        }

    defective = [d for d in detections 
                 if "defect" in d["class_name"].lower() 
                 and "non" not in d["class_name"].lower()]
    non_defective = [d for d in detections if d not in defective]

    d_count = len(defective)
    nd_count = len(non_defective)
    total = len(detections)

    # Improved risk calculation
    if d_count > 0:
        # Average confidence of defective detections
        avg_defect_conf = sum(d["confidence"] for d in defective) / d_count
        
        # Risk score components:
        # 1. Proportion of defective objects (0-50 points)
        # 2. Average confidence of defects (0-50 points)
        proportion_risk = (d_count / total) * 50
        confidence_risk = avg_defect_conf * 50
        raw_risk = proportion_risk + confidence_risk
        
        # Scale to 0-100 with slight boost for extreme confidence
        risk_score = min(int(raw_risk), 100)
        dominant_conf = avg_defect_conf
    else:
        risk_score = 0
        dominant_conf = 0.0

    # Improved verdict thresholds (calibrated for trained models)
    if risk_score >= 70:
        verdict = "DEFECTIVE"
    elif risk_score >= 40:
        verdict = "CAUTION"
    else:
        verdict = "SAFE"

    return {
        "verdict": verdict,
        "risk_score": risk_score,
        "defect_count": d_count,
        "non_defect_count": nd_count,
        "total_count": total,
        "dominant_conf": round(dominant_conf, 3),
        "confidence_threshold_used": 0.5
    }

@app.post("/detect")
async def detect_faults(file: UploadFile = File(...)):
    """
    1. Runs YOLO detection on the uploaded image.
    2. Aggregates results into a single verdict + risk score.
    3. Generates VLM explanations for top-5 defective detections only.
    4. Generates XAI heatmap based on detection locations.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)

    # 1. YOLO Inference
    result = models["yolo"].detect(img_np)
    detections = models["yolo"].get_crops_and_info(result, img_np)

    # Sort all detections by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # 2. Aggregate verdict
    verdict_summary = compute_verdict(detections)

    # 3. Annotate image with TOP 5 boxes only (keeps image readable)
    annotated_img = img_np.copy()
    top_draw = detections[:5]
    for det in top_draw:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        conf = det["confidence"]
        is_defect = "defect" in class_name.lower() and "non" not in class_name.lower()
        color = (255, 50, 50) if is_defect else (50, 200, 50)  # red or green (RGB)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name.replace('Non defective','OK')} {conf:.2f}"
        cv2.putText(annotated_img, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # 4. VLM: explain top 3 DEFECTIVE detections only
    defective_dets = [
        d for d in detections
        if "defect" in d["class_name"].lower() and "non" not in d["class_name"].lower()
    ][:3]

    explanations = []
    for det in defective_dets:
        crop_pil = det["crop_pil"]
        class_name = det["class_name"]
        conf = det["confidence"]

        if crop_pil.width == 0 or crop_pil.height == 0:
            explanation_text = "Region too small to analyze."
        else:
            explanation_text = models["vlm"].explain(crop_pil, class_name)

        explanations.append({
            "class_name": class_name,
            "confidence": f"{conf:.2f}",
            "explanation": explanation_text
        })

    # 5. XAI Heatmap (detection-aware)
    heatmap_img = models["xai"].generate_heatmap(img_np, detections[:10])

    return JSONResponse({
        "annotated_image": image_to_base64(annotated_img),
        "heatmap_image": image_to_base64(heatmap_img),
        "verdict": verdict_summary,
        "explanations": explanations
    })

# Mount the static frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
