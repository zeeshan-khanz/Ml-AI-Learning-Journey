from sklearn.tree import DecisionTreeClassifier
import pandas as pd

print("Zeeshan's Baby Product Decision AI")
print("=" * 45)

# Step 1: Create parental decision training data
print(" Loading parental decision examples...")
baby_products = {
    'good_for_baby': [1, 1, 0, 1, 1, 0, 1, 0],      # 1=Yes, 0=No
    'is_safe': [1, 1, 1, 0, 1, 1, 1, 0],            # 1=Safe, 0=Not safe
    'price_affordable': [1, 0, 1, 1, 1, 0, 0, 1],   # 1=Affordable, 0=Expensive
    'buy_decision': [1, 0, 0, 0, 1, 0, 0, 0]        # 1=Buy, 0=Don't buy
}

df = pd.DataFrame(baby_products)
print("Parental decision data loaded!")
print(df)

# Step 2: Train Decision Tree AI
print("\nTraining AI to make parental decisions...")
X = df[['good_for_baby', 'is_safe', 'price_affordable']].values
y = df['buy_decision'].values

ai_parent = DecisionTreeClassifier(random_state=42)
ai_parent.fit(X, y)
print("AI learned parental decision-making!")

# Step 3: Test AI on new baby products
print("\nAI Parental Decision Tests:")
test_products = [
    [1, 1, 1],    # Good=YES, Safe=YES, Affordable=YES
    [1, 1, 0],    # Good=YES, Safe=YES, Affordable=NO  
    [0, 1, 1],    # Good=NO, Safe=YES, Affordable=YES
    [1, 0, 1],    # Good=YES, Safe=NO, Affordable=YES
    [0, 0, 0],    # Good=NO, Safe=NO, Affordable=NO
]

product_names = ["Premium Baby Food", "Organic Baby Clothes", "Cheap Plastic Toy", "Unsafe Baby Walker", "Unknown Brand Formula"]

for i, product in enumerate(test_products):
    prediction = ai_parent.predict([product])[0]
    result = "BUY IT!" if prediction == 1 else "DON'T BUY"
    
    good = "YES" if product[0] == 1 else "NO"
    safe = "YES" if product[1] == 1 else "NO"
    affordable = "YES" if product[2] == 1 else "NO"
    
    print(f"\nProduct: {product_names[i]}")
    print(f"   Good for baby: {good}")
    print(f"   Is safe: {safe}")  
    print(f"   Affordable: {affordable}")
    print(f"   AI Decision: {result}")