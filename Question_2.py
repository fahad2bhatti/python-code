import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

IMG_SIZE = 64
dataset_path = "dataset"
categories = ["cars", "cycles", "motorcycles"]

X = []
y = []

print("Loading images...")

for label, category in enumerate(categories):
    folder = os.path.join(dataset_path, category)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.flatten()
            X.append(img)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Total images loaded: {len(X)}")
print(f"Classes: {categories}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("\nModel trained successfully!")

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=categories))

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten().reshape(1, -1)
    img = scaler.transform(img)
    prediction = knn.predict(img)
    return categories[prediction[0]]

test_img = os.path.join(dataset_path, "cars", os.listdir(f"{dataset_path}/cars")[0])
result = predict_image(test_img)
print(f"\nTest Image Prediction: {result}")