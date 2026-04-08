"""
recognize.py — Real-time prepoznavanje lica putem webcam-a koristeći OpenCV LBPH.
Prikazuje ime i vjerovatnoću (confidence) za svako prepoznato lice.
NAPOMENA: Koristi OpenCV LBPH — ne treba dlib ni Visual Studio Build Tools!
"""

import cv2
import pickle
import numpy as np

MODEL_FILE = "trained_model.yml"
LABELS_FILE = "labels.pkl"

# Confidence threshold za LBPH:
# Niža vrijednost = veće pouzdanje (0 = savršen match)
# Povećana granica na 100 kako bi te bolje prepoznavao (npr. otvorenih usta)
CONFIDENCE_THRESHOLD = 100

# --- Učitavanje modela i labela ---
print("[INFO] Učitavanje modela...")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

with open(LABELS_FILE, "rb") as f:
    label_map = pickle.load(f)  # {0: "nedim", 1: "drugi", ...}

print(f"[INFO] Prepoznate osobe: {list(label_map.values())}")

# --- Haar Cascade za detekciju lica ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

def get_faces(gray, min_size=(60, 60)):
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

print("[INFO] Pokrećem kameru... Pritisni ESC za izlaz.")

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("[ERROR] Nije moguće otvoriti kameru!")
    exit(1)

IMG_SIZE = (150, 150)

while True:
    ret, frame = video.read()
    if not ret:
        print("[ERROR] Nije moguće čitati frame s kamere.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcija lica
    faces = get_faces(gray, min_size=(60, 60))

    for (x, y, w, h) in faces:
        roi = gray[y : y + h, x : x + w]
        roi_resized = cv2.resize(roi, IMG_SIZE)

        # Prepoznavanje
        label_id, confidence = recognizer.predict(roi_resized)

        if confidence < CONFIDENCE_THRESHOLD:
            name = label_map.get(label_id, "Nepoznat")
            color = (0, 220, 0)  # zelena
        else:
            name = "Nepoznat"
            color = (0, 60, 230)  # crvena

        # Crtaj okvir
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Priprema teksta (samo ime, bez postotka)
        label_text = f"{name}"

        # Pozadina za tekst
        (tw, th), bl = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame,
            (x, y - th - bl - 10),
            (x + tw + 6, y),
            color,
            cv2.FILLED,
        )

        # Tekst
        cv2.putText(
            frame,
            label_text,
            (x + 3, y - bl - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # bijeli tekst
            2,
        )

    cv2.imshow("AIza Face Recognition — ESC za izlaz", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

video.release()
cv2.destroyAllWindows()
print("[INFO] Kamera zatvorena. Doviđenja!")
