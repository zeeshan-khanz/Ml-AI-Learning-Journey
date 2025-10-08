import numpy as npm 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
print("===================")
print("\n Day 4: Linear Regression\n")

Data={
    "House_Size":[1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    "price":[300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}

df=pd.DataFrame(Data)
print("Data is Loading ....\n")
print(df)
print("\nData Loaded Successfully\n")

model=LinearRegression()
model.fit(df[["House_Size"]], df[["price"]])
print("Model Trained Successfully\n")
print("Predicting the price of a house with size 2500 sq ft")

test_size=[[2500]]
predicted_price=model.predict(test_size)
print(f"The predicted price for a house with size {test_size[0][0]} sq ft is ${predicted_price[0][0]:,.2f}\n")


plt.scatter(df["House_Size"], df["price"], color='blue', label='Data Points')
plt.plot(df["House_Size"], model.predict(df[["House_Size"]]), color='red', label='Regression Line')
plt.scatter(test_size, predicted_price, color='green', label='Predicted Point', marker='x', s=100)
plt.title('House Size vs Price')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()