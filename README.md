# Face Mask Detection using Computer Vision

## 📌 Problem Statement

Monitoring mask compliance manually is difficult in public environments. This project automates detection using computer vision.

## 🎯 Objective

To build a real-time system that detects whether a person is wearing a mask or not using a webcam.

## 🧠 Methodology

* Face detection using OpenCV Haar Cascade
* Image preprocessing (resize, normalization)
* CNN model for classification
* Real-time prediction using webcam

## 🛠️ Technologies Used

* Python
* OpenCV
* TensorFlow/Keras
* NumPy

## 📂 Project Structure

```
dataset/
models/
src/
outputs/
train_model.py
main.py
```

## ▶️ How to Run

1. Train model:

```
python train_model.py
```

2. Run detection:

```
python main.py
```

## 📊 Output

* Detects face
* Shows "Mask" or "No Mask"
* Real-time webcam output

## 🚀 Future Improvements

* Deploy as web/mobile app
* Use YOLO for better detection
* Add alert system

## 👨‍💻 Author

Your Name
