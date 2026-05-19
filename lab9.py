import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────
# STEP 1: Load / Create Dataset
# ─────────────────────────────────────────────────────────────
# UCI Car Evaluation Dataset
# Features: buying, maint, doors, persons, lug_boot, safety
# Target  : class (unacc, acc, good, vgood)

data = {
    'buying':   ['vhigh','vhigh','vhigh','vhigh','high','high','med','med','low','low',
                 'vhigh','high','med','low','vhigh','high','med','low','vhigh','high',
                 'med','low','vhigh','high','med','low','vhigh','high','med','low'],
    'maint':    ['vhigh','vhigh','high','med','vhigh','high','high','med','med','low',
                 'vhigh','high','med','low','high','med','low','low','med','low',
                 'low','vhigh','high','med','low','vhigh','high','med','low','low'],
    'doors':    ['2','2','2','4','4','4','3','3','3','5more','2','4','3','5more','2','4','3','5more','4','3',
                 '5more','2','4','3','5more','2','4','3','5more','4'],
    'persons':  ['2','2','4','4','2','4','4','4','4','more','2','4','4','more','2','4','4','more','more','more',
                 'more','2','4','4','more','2','4','4','more','more'],
    'lug_boot': ['small','small','med','big','small','med','big','big','med','big','small','med','big','big',
                 'small','big','big','big','big','big','big','small','big','big','big','small','big','big','big','big'],
    'safety':   ['low','med','high','high','med','high','med','high','high','high','low','high','high','high',
                 'med','high','high','high','high','high','high','low','high','high','high','low','high','high','high','high'],
    'class':    ['unacc','unacc','unacc','acc','unacc','acc','acc','good','good','vgood','unacc','acc','good','vgood',
                 'unacc','good','good','vgood','good','vgood','vgood','unacc','vgood','good','vgood','unacc','vgood','good','vgood','vgood']
}

df = pd.DataFrame(data)

print("=" * 55)
print("   KNN - CAR EVALUATION DATASET")
print("=" * 55)
print(f"\nDataset Shape  : {df.shape}")
print(f"Features       : {list(df.columns[:-1])}")
print(f"Target Classes : {df['class'].unique().tolist()}")
print(f"\nClass Distribution:")
print(df['class'].value_counts().to_string())

# ─────────────────────────────────────────────────────────────
# STEP 2: Preprocessing — Label Encoding
# ─────────────────────────────────────────────────────────────
print("\n" + "-" * 55)
print("STEP 2: Label Encoding (Categorical → Numeric)")
print("-" * 55)

le = LabelEncoder()
df_encoded = df.copy()
for col in df.columns:
    df_encoded[col] = le.fit_transform(df[col])

print(df_encoded.head())

X = df_encoded.drop('class', axis=1)
y = df_encoded['class']

# ─────────────────────────────────────────────────────────────
# STEP 3: Feature Scaling (StandardScaler)
# ─────────────────────────────────────────────────────────────
print("\n" + "-" * 55)
print("STEP 3: Feature Scaling (StandardScaler)")
print("-" * 55)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled to mean=0, std=1 ✓")

# ─────────────────────────────────────────────────────────────
# STEP 4: Train/Test Split (80% / 20%)
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# STEP 5: Try Different K Values
# ─────────────────────────────────────────────────────────────
print("\n" + "-" * 55)
print("STEP 5: Trying Different K Values")
print("-" * 55)
print(f"{'K':>5}  {'Accuracy':>10}")
print("-" * 20)

best_k, best_acc = 1, 0
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    marker = " ← BEST" if acc > best_acc else ""
    if acc > best_acc:
        best_k, best_acc = k, acc
    print(f"K = {k:>2}   {acc*100:>8.2f}%{marker}")

# ─────────────────────────────────────────────────────────────
# STEP 6: Train Best Model
# ─────────────────────────────────────────────────────────────
print("\n" + "-" * 55)
print(f"STEP 6: Training Best Model (K = {best_k})")
print("-" * 55)

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)

print(f"\nFinal Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# STEP 7: Evaluation
# ─────────────────────────────────────────────────────────────
print("\n" + "-" * 55)
print("STEP 7: Classification Report")
print("-" * 55)
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────────────────────
# STEP 8: Predict a New Car
# ─────────────────────────────────────────────────────────────
print("\n" + "-" * 55)
print("STEP 8: Predict New Car Sample")
print("-" * 55)

# New sample: high buying, high maint, 4 doors, more persons, big boot, high safety
new_car = pd.DataFrame([[2, 2, 2, 2, 0, 2]], columns=X.columns)  # encoded values
new_car_scaled = scaler.transform(new_car)
prediction_encoded = knn_best.predict(new_car_scaled)[0]

class_map = {0: 'acc', 1: 'good', 2: 'unacc', 3: 'vgood'}
print(f"Input   : high price, high maint, 4 doors, more persons, big boot, high safety")
print(f"Predicted class: {class_map.get(prediction_encoded, prediction_encoded)}")
print("\n✓ KNN model trained and evaluated successfully!")