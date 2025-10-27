

import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

print("===================")
print("\n Day 8: Logistic Regression – Will Customer Buy?\n")


data = {
    "Age": [22, 25, 28, 30, 35, 40, 45, 50, 60, 65],
    "Income": [20000, 25000, 30000, 32000, 40000, 45000, 50000, 55000, 60000, 65000],
    "Bought": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0 = No, 1 = Yes
}

df = pd.DataFrame(data)
print("Dataset:\n", df, "\n")


X = df[["Age", "Income"]]
y = df["Bought"]

model = LogisticRegression()
model.fit(X, y)
print(" Model trained successfully!\n")


test_customer = [[33, 35000]]  # Age=33, Income=35k
probability = model.predict_proba(test_customer)[0][1]  # Probability of "Buy"
prediction = model.predict(test_customer)[0]
result = "Will Buy" if prediction == 1 else "Will Not Buy"

print(f"Customer Age: {test_customer[0][0]}, Income: {test_customer[0][1]}")
print(f"Predicted Probability: {probability*100:.2f}%")
print(f"Final Decision: {result}")

plt.scatter(df["Age"], df["Bought"], color='blue', label='Actual Data')
plt.xlabel("Age")
plt.ylabel("Buy Decision (0=No, 1=Yes)")
plt.title("Customer Purchase Prediction (Logistic Regression)")
plt.grid(True)
plt.legend()
plt.show()
