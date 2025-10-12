from sklearn.linear_model import LogisticRegression

print(" Student Pass/Fail Predictor")

study_hours=[[4],[6],[9],[10]]
p_f=[0,0,1,1]

mode1=LogisticRegression()
mode1.fit(study_hours,p_f)
print("\n Model is Train")
print('\n Now We Test')
test_data=[[5]]
model_predicat=mode1.predict(test_data)
result="Pass" if model_predicat==1 else "Fail"
print("\n model 1 Predication")
print(f"\n model predicatt:{result}")