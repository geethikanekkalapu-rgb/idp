import os
import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet
import dlib

DATASET_PATH = "dataset"
ENCODINGS_FILE = "encodings.pkl"

print("🔹 Loading FaceNet...")
embedder = FaceNet()
print("✅ FaceNet loaded")

face_detector = dlib.get_frontal_face_detector()

known_embeddings = []
known_names = []

print("🔹 Generating encodings...")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = face_detector(gray)

        for rect in rects:
            x1 = rect.left()
            y1 = rect.top()
            x2 = rect.right()
            y2 = rect.bottom()

            face = rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.resize(face, (160,160))
            embedding = embedder.embeddings([face])[0]

            known_embeddings.append(np.array(embedding, dtype=np.float32))
            known_names.append(person)

print("Saving encodings...")
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((known_embeddings, known_names), f)

print("✅ Encodings saved successfully!")