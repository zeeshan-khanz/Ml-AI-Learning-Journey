from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
print(" Model Evalution ")
print("\n===============")
X=[[1000],[2000],[3000],[4000]]
y=[50,100,150,200]

model=LinearRegression()
model.fit(X,y)
print('\n Model is Train')
test_size=[[5000]]
actual_price=250
model_predicaton=model.predict(test_size)[0]
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("\n===============")

print(f"\nMean Absolute Error: {mae:.2f}")
print("\n..............")
print(f"R² Score: {r2:.2f}")
print("\n..............")


print(f"\n Actual Price :{actual_price}")
print("\n..............")
print(f"\n Model Predicated Price Of the House Size is 5000:{model_predicaton}")
