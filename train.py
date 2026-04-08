"""
train.py — Učitava slike iz dataset/ foldera, trenira OpenCV LBPH prepoznavač
i sprema model u trained_model.yml te encodings.pkl s imenima i labelama.
NAPOMENA: Koristi OpenCV LBPH — ne treba dlib ni Visual Studio Build Tools!
"""

import cv2
import os
import pickle
import numpy as np

DATASET_PATH = "dataset"
MODEL_FILE = "trained_model.yml"
LABELS_FILE = "labels.pkl"
IMG_SIZE = (150, 150)  # Veličina na koju se normalizuju slike

# --- Haar Cascade za detekciju lica ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

def get_faces(gray, min_size=(30, 30)):
    faces = list(face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size))
    if len(faces) == 0:
        faces = list(profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size))
    if len(faces) == 0:
        flipped_gray = cv2.flip(gray, 1)
        flipped_faces = profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
        faces = []
        w_img = gray.shape[1]
        for (x, y, w, h) in flipped_faces:
            faces.append((w_img - x - w, y, w, h))
    return faces

face_data = []
labels = []
label_map = {}  # int → ime osobe
current_label = 0

print("[INFO] Učitavanje slika i detekcija lica...")

for person in sorted(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person
    print(f"  → Osoba: '{person}' (label={current_label})")
    count = 0

    for image_name in sorted(os.listdir(person_path)):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path)
        if img is None:
            print(f"    ✗ {image_name} — nije moguće učitati")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = get_faces(gray, min_size=(30, 30))

        for (x, y, w, h) in faces:
            roi = gray[y : y + h, x : x + w]
            roi_resized = cv2.resize(roi, IMG_SIZE)
            face_data.append(roi_resized)
            labels.append(current_label)
            count += 1
            break  # uzmi samo prvi pronađeni ROI po slici

    print(f"    ✓ {count} lica pronađena i dodana")
    current_label += 1

if len(face_data) == 0:
    print("\n[ERROR] Nije pronađeno nijedno lice u datasetu!")
    print("       Provjeri da slike sadrže jasno vidljiva lica.")
    exit(1)

# --- Treniranje LBPH modela ---
print(f"\n[INFO] Treniranje LBPH modela na {len(face_data)} uzoraka...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(face_data, np.array(labels))
recognizer.save(MODEL_FILE)

# --- Spašavanje mape labela ---
with open(LABELS_FILE, "wb") as f:
    pickle.dump(label_map, f)

print(f"\n[DONE] Training završen!")
print(f"[DONE] Model sačuvan u: {MODEL_FILE}")
print(f"[DONE] Labele sačuvane u: {LABELS_FILE}")
print(f"[DONE] Osobe: {list(label_map.values())}")
