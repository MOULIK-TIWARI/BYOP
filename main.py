import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/mask_model.h5")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Preprocess (same as training!)
        face = cv2.resize(face, (100, 100))
        face = face / 255.0
        face = np.reshape(face, (1, 100, 100, 3))

        # Predict
        pred = model.predict(face)

        label = "Mask" if pred[0][0] > pred[0][1] else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Display
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow("Face Mask Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()