
from sklearn.tree import DecisionTreeClassifier

# Training data
temperature = [[30], [40], [50], [60], [70], [80]]
rain = [0, 1, 0, 1, 0, 1]



model = DecisionTreeClassifier(random_state=42)
model.fit(temperature, rain)

prediction = model.predict([[25]])
print("Rain dont go Outside" if prediction[0] == 1 else "No Rain")
