# ============================================
# Decision Tree using ID3 (Entropy)
# Play Tennis Prediction
# Clean & Readable Visualization
# ============================================

# Step 1: Import required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 2: Create the Play Tennis dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain',
                'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast',
                'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Mild',
                    'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High',
                 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong',
             'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No',
                   'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes',
                   'Yes', 'No']
}

df = pd.DataFrame(data)
print("Original Dataset:\n")
print(df)

# Step 3: Manual encoding (exam-safe)
mapping = {
    'Outlook': {'Sunny': 0, 'Overcast': 1, 'Rain': 2},
    'Temperature': {'Cool': 0, 'Mild': 1, 'Hot': 2},
    'Humidity': {'Normal': 0, 'High': 1},
    'Wind': {'Weak': 0, 'Strong': 1},
    'PlayTennis': {'No': 0, 'Yes': 1}
}

df_encoded = df.copy()
for col in mapping:
    df_encoded[col] = df_encoded[col].map(mapping[col])

print("\nEncoded Dataset:\n")
print(df_encoded)

# Step 4: Split features and target
X = df_encoded.drop('PlayTennis', axis=1)
y = df_encoded['PlayTennis']

# Step 5: Train Decision Tree using ID3 (Entropy)
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,       # 🔑 keeps tree readable
    random_state=42
)

model.fit(X, y)

# Step 6: Predict for a new instance
# Example: Sunny, Mild, High humidity, Weak wind
new_instance = pd.DataFrame(
    [[0, 1, 1, 0]],
    columns=X.columns
)

prediction = model.predict(new_instance)

print("\nPrediction for new instance:")
if prediction[0] == 1:
    print("Play Tennis: YES 🎾")
else:
    print("Play Tennis: NO ❌")

# Step 7: Visualize the Decision Tree (CLEAN VIEW)
plt.figure(figsize=(22, 12), dpi=120)
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    impurity=False,   # removes entropy text
    proportion=True, # shows proportions instead of raw counts
    fontsize=11
)

plt.title("Decision Tree – Play Tennis (ID3 Algorithm)", fontsize=16)
plt.show()
