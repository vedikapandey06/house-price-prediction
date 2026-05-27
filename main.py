#Linear Regression – House Price Prediction


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()

X = housing.data
y = housing.target

feature_names = housing.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['Price'] = y

print("First 5 rows of dataset:")
print(df.head())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

print("\nFeature Coefficients:")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef}")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")


plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

plt.show()
