import os
from pathlib import Path
from ultralytics import YOLO
import cv2

# User inputs
YOLO_MODEL_PATH = "yolov8n.pt"  # replace with your actual model path
ROOT_FOLDER = "output"       # replace with your actual folder path
CONFIDENCE_THRESHOLD = 0.9

# Load model
model = YOLO(YOLO_MODEL_PATH)

# Iterate over all images in all subfolders
image_extensions = (".jpg", ".jpeg", ".png")

for img_path in Path(ROOT_FOLDER).rglob("*"):
    if not img_path.is_file() or img_path.suffix.lower() not in image_extensions:
        continue

    # Run detection
    results = model(img_path, conf=CONFIDENCE_THRESHOLD)

    # Filter detections for 'car' class (you must confirm class ID or name, e.g., COCO 'car' = class 2)
    # Here assuming 'car' class index is 2 in COCO
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 2 and conf >= CONFIDENCE_THRESHOLD:
                    detections.append((conf, box.xyxy[0].cpu().numpy()))

    if not detections:
        # No detection, delete image
        img_path.unlink()
        continue

    # Select highest-confidence detection
    best_conf, best_box = max(detections, key=lambda x: x[0])

    # Crop and overwrite
    img = cv2.imread(str(img_path))
    x1, y1, x2, y2 = map(int, best_box)
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite(str(img_path), cropped)
