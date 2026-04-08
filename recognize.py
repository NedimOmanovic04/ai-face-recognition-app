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
# Preporučena granica za markiranje kao "poznato": ispod 80
CONFIDENCE_THRESHOLD = 80

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
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        roi = gray[y : y + h, x : x + w]
        roi_resized = cv2.resize(roi, IMG_SIZE)

        # Prepoznavanje
        label_id, confidence = recognizer.predict(roi_resized)

        # LBPH: confidence = udaljenost (manje = bolje)
        # Konvertiraj u postotak (0–100%) gdje 100% = savršen match
        confidence_pct = max(0.0, 100 - confidence)

        if confidence < CONFIDENCE_THRESHOLD:
            name = label_map.get(label_id, "Nepoznat")
            color = (0, 220, 0)  # zelena
        else:
            name = "Nepoznat"
            color = (0, 60, 230)  # crvena

        # Crtaj okvir
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Priprema teksta
        label_text = f"{name}  {confidence_pct:.1f}%"

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
