# mark_attendance_video_to_excel.py

import cv2
import numpy as np
import pickle
import torch
import pandas as pd
from datetime import datetime
from facenet_pytorch import InceptionResnetV1, MTCNN

# ---------------- CONFIG ----------------
FACE_DB_FILE = "face_db.pkl"
VIDEO_PATH   = "/Users/jagadeesh/Downloads/WhatsApp Video 2026-01-27 at 19.37.25.mp4"     # your 5-sec classroom snippet
EXCEL_PATH   = "attendance.xlsx"

CONF_THRESHOLD = 0.75                # cosine similarity threshold (tune if needed)
FRAME_SKIP     = 3                   # process every 3rd frame (speed)
# ---------------------------------------


# Use MPS for embeddings if available (Apple Silicon), else CPU
device_embed = "mps" if torch.backends.mps.is_available() else "cpu"

# IMPORTANT: run MTCNN on CPU to avoid MPS adaptive pool error
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device="cpu")
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device_embed)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding(face_tensor):
    """
    face_tensor: torch tensor [3,160,160] from MTCNN
    returns: normalized embedding numpy array (512,)
    """
    face_tensor = face_tensor.unsqueeze(0).to(device_embed)
    with torch.no_grad():
        emb = facenet(face_tensor)[0].cpu().numpy()
    emb = emb / np.linalg.norm(emb)
    return emb


# -------- Load Face DB ----------
with open(FACE_DB_FILE, "rb") as f:
    db = pickle.load(f)

student_names = list(db.keys())
print("✅ Loaded students from face DB:", student_names)


# -------- Process Video ----------
present_students = set()

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = mtcnn(rgb)  # returns list/tensor of faces
    if faces is None:
        continue

    # faces is a tensor [N, 3, 160, 160] when keep_all=True
    for face_tensor in faces:
        emb = get_embedding(face_tensor)

        best_name = "Unknown"
        best_score = -1

        # Match against all known students
        for name in student_names:
            for db_emb in db[name]:
                score = cosine_sim(emb, db_emb)
                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score >= CONF_THRESHOLD:
            present_students.add(best_name)

cap.release()

print("\n✅ Present students detected:", sorted(list(present_students)))


# -------- Update Excel ----------
df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

# Validate required columns
required_cols = {"Name", "RollNo", "ParentPhone"}
missing = required_cols - set(df.columns)

if missing:
    raise ValueError(f"❌ Excel sheet missing required columns: {missing}")

# New column name for this lecture
lecture_col = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Create new lecture column if not exists
if lecture_col not in df.columns:
    df[lecture_col] = ""

# Mark P/A based on detected names
df[lecture_col] = df["Name"].astype(str).apply(lambda x: "P" if x in present_students else "A")

# Save back
df.to_excel(EXCEL_PATH, index=False)

print(f"\n✅ Attendance updated successfully in: {EXCEL_PATH}")
print(f"✅ Lecture column added: {lecture_col}")
