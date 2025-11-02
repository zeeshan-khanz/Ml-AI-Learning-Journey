# 🎯 Day 11 - Model Evaluation
# Run once: pip install scikit-learn pandas matplotlib

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

print("Model Evaluation Lab")
print("=" * 45)
Data={
    "Email":[50, 30, 20, 40, 60, 80, 90, 10, 70, 55],
    "Spam":[0, 0, 0, 1, 1, 1, 0, 0, 1, 0]

}
df=pd.DataFrame(Data)
X = df[["Email"]]
y = df["Spam"]
model=LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
