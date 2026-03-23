from ultralytics import YOLO
import os
import shutil

def train_custom_model(data_yaml="rail_yolo/data.yaml", epochs=1):
    """
    A simple script to train YOLOv8 on the converted railway dataset.
    """
    print("Loading base pre-trained YOLOv8n model...")
    model = YOLO("yolov8n.pt")  

    print("Starting training...")
    # Using 5 epochs for speed. Users can increase this for better accuracy.
    results = model.train(data=data_yaml, epochs=epochs, imgsz=640, device="cpu")
    
    print("Training finished.")
    
    # The trained weights will be saved typically in runs/detect/train/weights/best.pt
    # Let's copy it to models/best.pt
    best_weights = "runs/detect/train/weights/best.pt"
    if os.path.exists(best_weights):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_weights, "models/best.pt")
        print("Copied best.pt to models directory.")
    else:
        # Check if it created a new run folder like train2, train3 
        import glob
        runs = sorted(glob.glob("runs/detect/train*/weights/best.pt"))
        if runs:
            latest = runs[-1]
            os.makedirs("models", exist_ok=True)
            shutil.copy(latest, "models/best.pt")
            print(f"Copied {latest} to models directory.")

if __name__ == "__main__":
    train_custom_model()
