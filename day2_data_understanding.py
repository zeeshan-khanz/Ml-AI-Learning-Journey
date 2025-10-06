import pandas as pd
import matplotlib.pyplot as plt

data={
        "Hours_Studied": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Sleep_Hours": [9, 8, 7, 7, 6, 6, 5, 5, 4],
    "Grade": [55, 60, 65, 70, 75, 80, 85, 90, 95]

}
print("Data is loading")
df=pd.DataFrame(data)
print(df.head())

print("\n Dataset Info")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())


print("\n Now Create Bar chart")
plt.bar(df["Hours_Studied"], df["Grade"], color='blue')
plt.title("Hours Studied vs Grade")
plt.xlabel("Hours Studied")
plt.ylabel("Grade")
plt.show()