# 😃 Face Detection with Haar Cascades and DNN

This project demonstrates multiple face detection techniques using OpenCV:
1. Detecting faces in static images using Haar Cascades.
2. Detecting faces using OpenCV's Deep Neural Network (DNN) module.
3. Real-time face detection from webcam video using Haar Cascades.

---

## 📁 Contents

- `main.ipynb`: Complete implementation of all 3 face detection modes
- `test_1.jpg` and `test_2.jpg`: Sample input images (add to folder)
- `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`: DNN model files (required for DNN detection)

---

## 🧰 Technologies Used

- Python
- OpenCV
- Haar Cascade XML
- OpenCV DNN face detection
- Matplotlib

---

## 🧪 Features Implemented

### 📸 1. **Face Detection in Static Images (Haar)**
- Reads an image
- Detects faces using `haarcascade_frontalface_default.xml`
- Draws bounding boxes around detected faces

### 🧠 2. **Face Detection with DNN (Deep Learning Model)**
- Uses `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`
- Detects faces with higher accuracy and confidence threshold
- Shows bounding boxes with confidence score

### 🎥 3. **Live Face Detection from Webcam**
- Captures frames in real-time using `cv2.VideoCapture`
- Applies Haar face detection on each frame
- Draws rectangles on detected faces
- Quits on pressing `q`

---

## ▶️ How to Run

1. **Install dependencies:**

```bash
pip install opencv-python matplotlib
