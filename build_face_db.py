import os
import cv2
import numpy as np
import torch
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN


RAW_DB_DIR = "student_db_aug"    # your augmented folder
OUTPUT_FILE = "face_db.pkl"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Face detector + face embedder
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device="cpu")
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_embedding(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)

    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = facenet(face)

    emb = emb[0].cpu().numpy()
    emb = emb / np.linalg.norm(emb)  # normalize
    return emb

db = {}  # {student_name: [embeddings...]}

students = [d for d in os.listdir(RAW_DB_DIR) if os.path.isdir(os.path.join(RAW_DB_DIR, d))]

for student in students:
    student_path = os.path.join(RAW_DB_DIR, student)
    embeddings = []

    for file in os.listdir(student_path):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(student_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        emb = get_embedding(img)
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) > 0:
        db[student] = embeddings
        print(f"✅ {student}: {len(embeddings)} face embeddings stored")
    else:
        print(f"❌ {student}: no face detected in images")

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(db, f)

print(f"\n✅ Face database created: {OUTPUT_FILE}")
print(f"Total students stored: {len(db)}")
