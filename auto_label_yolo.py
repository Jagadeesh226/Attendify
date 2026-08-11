from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

IMAGE_DIR = "Attendify data/images/val"
LABEL_DIR = "Attendify data/labels/val"

os.makedirs(LABEL_DIR, exist_ok=True)

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w, _ = img.shape
    results = model(img, conf=0.4)

    label_path = os.path.join(
        LABEL_DIR, os.path.splitext(img_name)[0] + ".txt"
    )

    with open(label_path, "w") as f:
        for box in results[0].boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
