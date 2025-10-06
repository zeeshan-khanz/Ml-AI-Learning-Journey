import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
print("\n Day One of AI/Ml Learninig")
print("My First Program in AI/Ml Predication of House Price")
house_Data={
    "Size":[1000,1500,2000,2500,3000],
    "Price":[200000,250000,300000,350000,400000]

}

df=pd.DataFrame(house_Data)
print("Data is Loading.....................")
print("\n Wait until Data is load Suceessfully")
print("Data is Load")
print(df)

ai=LinearRegression()
ai.fit(df[["Size"]],df["Price"])

print("AI are to train to Predict House Price")


new_House=[4500]
prdict=ai.predict([new_House])
print(f"The New House Price According to the size is :{prdict[0]:,.0f}")

