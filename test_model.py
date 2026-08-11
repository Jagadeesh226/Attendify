from ultralytics import YOLO

model = YOLO("runs/detect/runs_attendify/yolo_person_v1/weights/best.pt")

model.predict(
    source="Attendify data/images/val",
    conf=0.4,
    save=True,
    project="runs_attendify",
    name="predictions_v1"
)
