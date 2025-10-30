import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

print("*************************")
print("\n Customer buying Decision Prediction using Logistic Regression\n")

data={
    "timespents":[4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14],
    "purchase":[0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1]

}
df=pd.DataFrame(data)
print("Data is Loading ....\n")
print(df)
X=df["timespents"]
y=df["purchase"]

model=LogisticRegression()
model.fit(X.values.reshape(-1,1), y)

test_time=[[11.5]]
predication=model.predict(test_time)
print(f"The predicted buying decision for a customer spending {test_time[0][0]} hours on the website is: {'Buy' if predication[0]==1 else 'Not Buy'}\n")