from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# Large training dataset
house_size = np.array([
    [500], [600], [700], [800], [900],
    [1000], [1100], [1200], [1300], [1400],   # ✅ Fixed: removed stray 1600
    [1500], [1600], [1700], [1800], [1900],
    [2000], [2200], [2400], [2600], [2800]
])

house_price = np.array([
    120, 150, 175, 200, 225,
    250, 275, 300, 325, 350,
    375, 400, 430, 460, 490,
    520, 580, 640, 700, 760
])

# Split into train/test sets for accuracy evaluation
X_train, X_test, y_train, y_test = train_test_split(
    house_size, house_price, test_size=0.2, random_state=42
)

# Create and train model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)

# ✅ Accuracy Calculation
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)   

print("===== Model Accuracy =====")
print(f"R² Score       : {r2:.4f}  ({r2*100:.2f}% accuracy)")
print(f"Mean Abs Error : ${mae:.2f}k")

# Prediction loop
print("\n===== House Price Prediction System =====")
while True:
    size = float(input("\nEnter house size in sq ft (0 to exit): "))
    if size == 0:
        print("Program Closed")
        break
    predicted_price = model.predict([[size]])
    print(f"Estimated House Price: ${predicted_price[0]:.2f}k")