import cv2
import numpy as np
import os
from keras.models import load_model

# Project paths
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_PATH, "isl_model.h5")
DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")

# Load the trained model
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded!")

# Load class labels from dataset folders
CLASSES = sorted([f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))])
print(f"📂 Classes: {CLASSES}")

# Settings
IMG_SIZE = 224  # Must match the training image size

# Start webcam
cap = cv2.VideoCapture(0)
print("🎥 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Define ROI box
    x, y, w, h = 100, 100, 300, 300
    roi = frame[y:y+h, x:x+w]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)

    # Prediction
    predictions = model.predict(roi_expanded)
    class_index = np.argmax(predictions)
    class_label = CLASSES[class_index]
    confidence = predictions[0][class_index]

    # Display ROI and prediction
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, f"{class_label} ({confidence:.2f})",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ISL Prediction", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
