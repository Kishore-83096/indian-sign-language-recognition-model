import cv2
import os
import shutil

# Base project folder (where this script is located)
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

# Dataset folder inside project
DATASET_PATH = os.path.join(PROJECT_PATH, "dataset")

IMG_SIZE = 128
NUM_SAMPLES = 1000  # max images per class

# Ask which alphabet to capture
target_class = input("Enter the alphabet to capture (e.g., A, B, C): ").upper()

# Remove old folder if exists (overwrite mode)
class_path = os.path.join(DATASET_PATH, target_class)
if os.path.exists(class_path):
    print(f"⚠️ Deleting existing data for '{target_class}'...")
    shutil.rmtree(class_path)  # delete old images
os.makedirs(class_path, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

print(f"📸 Ready to capture images for '{target_class}'")
print("➡️ Press 'c' to capture an image, 'q' to quit")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Define ROI (blue box)
    x, y, w, h = 100, 100, 300, 300
    roi = frame[y:y+h, x:x+w]

    # Draw ROI rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display capture info
    cv2.putText(frame, f"Class: {target_class} | {count}/{NUM_SAMPLES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'c' to capture, 'q' to quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Capture Dataset", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        if count < NUM_SAMPLES:
            img_path = os.path.join(class_path, f"{target_class}_{count}.jpg")
            roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(img_path, roi_resized)
            print(f"✅ Saved {img_path}")
            count += 1
        else:
            print(f"✅ Done! Collected {NUM_SAMPLES} images for '{target_class}'.")
            break
    elif key == ord('q'):
        print("👋 Exiting capture.")
        break

cap.release()
cv2.destroyAllWindows()
