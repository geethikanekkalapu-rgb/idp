# -- coding: utf-8 --
"""
Face Recognition Attendance System
✔ Multi Face
✔ Proxy if detected in phone
✔ Present if stable for few seconds
✔ Confusion Matrix Displayed
✔ MTCNN Used for Face Detection
"""

import numpy as np
import cv2
import os
import pickle
import pandas as pd
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from keras_facenet import FaceNet
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PATHS ----------------
ENCODE_FILE = "encodings.pkl"
EXCEL_FILE = "Attendance.xlsx"

# ---------------- SETTINGS ----------------
THRESHOLD = 0.80
DETECTION_SECONDS = 4

# ---------------- LOAD FACENET ----------------
print("🔹 Loading FaceNet...")
embedder = FaceNet()

# ---------------- LOAD ENCODINGS ----------------
if os.path.exists(ENCODE_FILE):
    with open(ENCODE_FILE, "rb") as f:
        known_encodings, known_names = pickle.load(f)
else:
    known_encodings = []
    known_names = []

known_encodings = np.array(known_encodings)

# ---------------- MTCNN INITIALIZATION ----------------
detector = MTCNN()

# ---------------- MEMORY ----------------
attendance_records = []
marked_today = set()
detection_timer = {}

# ---------------- PROXY DETECTION ----------------
def detect_phone_proxy(face_region):

    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])

    if brightness > 170 and edge_density > 0.10:
        return True
    return False

# ---------------- START CAMERA ----------------
cap = cv2.VideoCapture(0)

y_true = []
y_pred = []

print("✅ System Started (Press Q to Quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------- MTCNN FACE DETECTION --------
    results = detector.detect_faces(rgb)

    for result in results:

        x, y, width, height = result['box']

        # Ensure positive coordinates
        x, y = abs(x), abs(y)
        x_max = x + width
        y_max = y + height

        face = frame[y:y_max, x:x_max]
        if face.size == 0:
            continue

        # Resize for FaceNet
        face_resized = cv2.resize(face, (160,160))
        embedding = embedder.embeddings([face_resized])[0]

        name = "Unknown"
        predicted_known = 0

        # ---------- FACE MATCH ----------
        if len(known_encodings) > 0:
            distances = np.linalg.norm(known_encodings - embedding, axis=1)
            min_dist = np.min(distances)
            index = np.argmin(distances)

            if min_dist < THRESHOLD:
                name = known_names[index]
                predicted_known = 1

        # ---------- PROXY CHECK ----------
        proxy_detected = detect_phone_proxy(face)

        current_time = time.time()
        today = datetime.now().strftime('%Y-%m-%d')
        unique_key = f"{name}_{today}"

        # ---------- ATTENDANCE LOGIC ----------
        if predicted_known == 1:

            if proxy_detected:

                if unique_key not in marked_today:
                    marked_today.add(unique_key)
                    attendance_records.append([
                        name,
                        today,
                        datetime.now().strftime('%H:%M:%S'),
                        "Proxy"
                    ])
                    print(f"⚠ Proxy Detected for {name}")

                display_name = "PROXY"
                color = (0,0,255)

            else:
                if name not in detection_timer:
                    detection_timer[name] = current_time

                elif current_time - detection_timer[name] >= DETECTION_SECONDS:

                    if unique_key not in marked_today:
                        marked_today.add(unique_key)
                        attendance_records.append([
                            name,
                            today,
                            datetime.now().strftime('%H:%M:%S'),
                            "Present"
                        ])
                        print(f"✅ Attendance Confirmed for {name}")

                display_name = name
                color = (0,255,0)

        else:
            display_name = "Unknown"
            color = (0,0,255)

        # ---------- METRICS ----------
        actual = 1 if predicted_known == 1 else 0
        predicted = 1 if predicted_known == 1 and not proxy_detected else 0

        y_true.append(actual)
        y_pred.append(predicted)

        cv2.rectangle(frame,(x,y),(x_max,y_max),color,2)
        cv2.putText(frame,display_name,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- METRICS ----------------
if len(y_true) > 0:

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\nAccuracy :", round(acc*100,2), "%")
    print("Precision:", round(prec*100,2), "%")
    print("Recall   :", round(rec*100,2), "%")
    print("F1 Score :", round(f1*100,2), "%")

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Pred Unknown","Pred Known"],
                yticklabels=["Actual Unknown","Actual Known"],
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.show()

# ---------------- EXCEL SAVE ----------------
if len(attendance_records) > 0:
    df = pd.DataFrame(
        attendance_records,
        columns=["Name","Date","Time","Status"]
    )
    df.to_excel(EXCEL_FILE, index=False)
    print("\n📂 Excel File Created Successfully!")
    os.startfile(EXCEL_FILE)
else:
    print("No attendance recorded.")