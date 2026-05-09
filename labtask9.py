import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset paths
paths = {
    "cats": r"C:\Users\HB LAPTOP\Ai_LabTasks\dataset\cats",
    "dogs": r"C:\Users\HB LAPTOP\Ai_LabTasks\dataset\dogs",
    "cars": r"C:\Users\HB LAPTOP\Ai_LabTasks\dataset\cars",
    "cycles": r"C:\Users\HB LAPTOP\Ai_LabTasks\dataset\cycles",
    "motorcycles": r"C:\Users\HB LAPTOP\Ai_LabTasks\dataset\motorcycles"
}

# Function to load images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            images.append(img.flatten())
            labels.append(label)
    return images, labels

# Load all classes
X, y = [], []
label_map = {}
for idx, (cls, path) in enumerate(paths.items()):
    imgs, labels = load_images_from_folder(path, idx)
    X.extend(imgs)
    y.extend(labels)
    label_map[idx] = cls

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc*100:.2f}%\n")

# Classification Report
print("📊 Detailed Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=list(label_map.values())))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(label_map.values()), yticklabels=list(label_map.values()))
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
