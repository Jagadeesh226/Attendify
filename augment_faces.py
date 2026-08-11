import os
import cv2
from glob import glob
import albumentations as A

input_dir = "student_db_raw"
output_dir = "student_db_aug"
os.makedirs(output_dir, exist_ok=True)

transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.2, p=0.9),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.Rotate(limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=0, p=0.5),
])

AUG_PER_IMAGE = 8  # creates 8 new faces from 1

students = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

for student in students:
    student_in = os.path.join(input_dir, student)
    student_out = os.path.join(output_dir, student)
    os.makedirs(student_out, exist_ok=True)

    for img_path in glob(os.path.join(student_in, "*.*")):
        img = cv2.imread(img_path)
        if img is None:
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]

        # Save original too
        cv2.imwrite(os.path.join(student_out, f"{base}_orig.jpg"), img)

        # Augmented versions
        for i in range(AUG_PER_IMAGE):
            aug_img = transform(image=img)["image"]
            out_path = os.path.join(student_out, f"{base}_aug{i}.jpg")
            cv2.imwrite(out_path, aug_img)

print("✅ Face augmentation completed.")
