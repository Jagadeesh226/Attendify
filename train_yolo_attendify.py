from ultralytics import YOLO

def train():
    # Choose base pretrained model
    # n = nano (fastest), s = small (better), m/l/x (heavier)
    model = YOLO("yolov8n.pt")

    model.train(
        data="Attendify data/data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        device="mps",      # use "mps" on Mac (Apple Silicon) OR 0 for NVIDIA GPU
        workers=4,
        project="runs_attendify",
        name="yolo_person_v1"
    )

if __name__ == "__main__":
    train()
