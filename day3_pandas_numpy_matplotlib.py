# DAY3 - NumPy, Pandas, Matplotlib practical
# Run: pip install numpy pandas matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Part A: NumPy basics
# -----------------------
print("=== NumPy Demo ===")
arr = np.array([1, 2, 3, 4, 5])
print("arr:", arr)
print("mean:", arr.mean(), "sum:", arr.sum(), "std:", arr.std())

# Broadcasting example
a = np.array([1, 2, 3])
b = 2
print("a * 2:", a * b)

# Reshape example
mat = np.arange(1, 13).reshape((3, 4))
print("matrix shape:", mat.shape)
print(mat)

# -----------------------
# Part B: Create sample dataset (like CSV)
# -----------------------
print("\n=== Pandas Demo ===")
data = {
    "StudentID": range(1, 11),
    "Hours_Studied": [1.5, 2.0, 2.5, 3.5, 4.0, 5.0, 6.0, 7.5, 8.0, np.nan],
    "Sleep_Hours": [8, 7.5, 8, 7, 6.5, 6, 5.5, 5, 4.5, 6],
    "Grade": [50, 60, 62, 70, 74, 78, 85, 88, 92, 90],
    "Major": ["CS", "EE", "CS", "ME", "CS", "EE", "ME", "CS", "EE", "CS"]
}
df = pd.DataFrame(data)

# Quick inspection
print("\n-- head() --")
print(df.head())
print("\n-- info() --")
print(df.info())
print("\n-- describe() --")
print(df.describe())

# -----------------------
# Part C: Clean & transform
# -----------------------
# 1. Detect missing
print("\nMissing values per column:\n", df.isnull().sum())

# 2. Fill missing Hours_Studied with mean (example)
mean_hours = df['Hours_Studied'].mean()
df['Hours_Studied'] = df['Hours_Studied'].fillna(mean_hours)
print(f"\nFilled missing Hours_Studied with mean: {mean_hours:.2f}")

# 3. Create a new feature: Study Efficiency = Grade / Hours_Studied
#    Avoid division by zero (add small epsilon)
eps = 1e-6
df['Study_Efficiency'] = df['Grade'] / (df['Hours_Studied'] + eps)

# 4. Type conversion example (StudentID -> string)
df['StudentID'] = df['StudentID'].astype(str)

print("\n-- After cleaning & new feature --")
print(df.head())

# -----------------------
# Part D: Selection & groupby
# -----------------------
# Select CS majors and sort by efficiency
cs_students = df[df['Major'] == 'CS'].sort_values('Study_Efficiency', ascending=False)
print("\nTop CS students by efficiency:\n", cs_students[['StudentID','Hours_Studied','Grade','Study_Efficiency']])

# Group by Major and get average grade
avg_grade_by_major = df.groupby('Major')['Grade'].mean().reset_index().sort_values('Grade', ascending=False)
print("\nAverage grade by major:\n", avg_grade_by_major)

# -----------------------
# Part E: Merge example (create another small df)
# -----------------------
advisor = pd.DataFrame({
    "Major": ["CS", "EE", "ME"],
    "Advisor": ["Dr. A", "Dr. B", "Dr. C"]
})
df_merged = pd.merge(df, advisor, on='Major', how='left')
print("\nMerged DataFrame with advisors (first 5 rows):\n", df_merged.head())

# -----------------------
# Part F: Visualization
# -----------------------
plt.figure(figsize=(8,5))
plt.scatter(df['Hours_Studied'], df['Grade'])
plt.title("Hours Studied vs Grade")
plt.xlabel("Hours Studied")
plt.ylabel("Grade")
plt.grid(True)
plt.tight_layout()
plt.show()

# Histogram of Study Efficiency
plt.figure(figsize=(7,4))
plt.hist(df['Study_Efficiency'], bins=8)
plt.title("Distribution of Study Efficiency")
plt.xlabel("Efficiency (Grade / Hours)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Boxplot of Grade by Major
plt.figure(figsize=(7,4))
df.boxplot(column='Grade', by='Major')
plt.title("Grade distribution by Major")
plt.suptitle("")  # remove automatic subtitle
plt.xlabel("Major")
plt.ylabel("Grade")
plt.tight_layout()
plt.show()

# -----------------------
# Part G: Save cleaned dataset
# -----------------------
df.to_csv("day3_cleaned_students.csv", index=False)
print("\nCleaned dataset saved to day3_cleaned_students.csv")

# -----------------------
# Practice exercises (try these)
# -----------------------
print("""
PRACTICE:
1) Replace fillna(mean) with median — compare results.
2) Create a binary column 'Passed' where Grade >= 60.
3) Use groupby to compute average Study_Efficiency per Major.
4) Create a pivot table: index=Major, columns=Passed, values=Grade (agg='mean').
5) Load a real CSV (your own) with pd.read_csv and perform these steps.
""")
