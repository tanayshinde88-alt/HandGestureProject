print("Program started...")

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ======================
# DATASET PATH
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "leapGestRecog", "00")

print("Expected path:", DATASET_PATH)

if not os.path.exists(DATASET_PATH):
    print("Dataset folder not found!")
    exit()

IMAGE_SIZE = 64
MAX_IMAGES_PER_CLASS = 200

X, y = [], []
label = 0

gesture_folders = [
    f for f in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, f))
]

print("Detected gesture classes:", gesture_folders)

if len(gesture_folders) < 2:
    print("Need at least 2 gesture classes")
    exit()

for gesture in gesture_folders:
    gesture_path = os.path.join(DATASET_PATH, gesture)
    images = os.listdir(gesture_path)[:MAX_IMAGES_PER_CLASS]

    for img_name in images:
        img_path = os.path.join(gesture_path, img_name)
        try:
            img = Image.open(img_path).convert("L")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = np.array(img).flatten()
            X.append(img)
            y.append(label)
        except:
            pass

    print("Loaded:", gesture)
    label += 1

print("Images loaded successfully")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("Program finished successfully")
