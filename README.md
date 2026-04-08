# Face Recognition Attendance System

## Overview

This project is a **Face Recognition–Based Attendance System** that automatically detects and recognizes faces using deep learning models and marks attendance in real-time.
It eliminates manual attendance and improves accuracy, efficiency, and security.
---

## Features

* Real-time face detection using MTCNN
* Face recognition using FaceNet
* Automatic attendance marking
* Stores attendance in Excel file
* Easy-to-use interface
* High accuracy and fast processing

---

## Technologies Used

* Python
* OpenCV
* FaceNet
* MTCNN
* NumPy
* Pandas

---

## Project Structure

```
idp/
│── dataset/                     # Images for training
│── face_landmarker.task        # Face landmark model
│── shape_predictor_68_face_landmarks.dat  # Dlib model
│── generate_encodings.py       # Encoding generation script
│── face_recog_app.py           # Main application
│── encodings.pkl               # Stored face encodings
│── attendance.xlsx             # Attendance record
```

---

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/your-repo-name.git
```

2. Navigate to the folder:

```
cd your-repo-name
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

1. Add images to the `dataset/` folder
2. Run encoding generation:

```
python generate_encodings.py
```

3. Run the application:

```
python face_recog_app.py
```

4. Attendance will be saved automatically

---

## Important Note

Large files like:

* `.dat` model file
* `dataset/` folder

are not included in this repository.
Download required models from:

* Dlib Face Landmark Model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

---

## Output

* Detects faces in real-time
* Marks attendance in Excel sheet

---
