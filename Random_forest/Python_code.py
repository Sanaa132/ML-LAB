import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
data = load_breast_cancer()

X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

print("Dataset Loaded Successfully")
print("Classes:", class_names)

# --------------------------------------------------
# SAVE DATASET AS CSV
# --------------------------------------------------
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
df["target"] = df["target"].map({0: "malignant", 1: "benign"})

df.to_csv("breast_cancer_dataset.csv", index=False)
print("Dataset saved as breast_cancer_dataset.csv")

# --------------------------------------------------
# BUILD RANDOM FOREST MODEL (USING ENTROPY)
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=5,
    criterion="entropy",
    max_depth=3,
    random_state=42
)

model.fit(X, y)
print("\nRandom Forest built using ENTROPY")

# --------------------------------------------------
# SAVE ALL DECISION TREES INTO PDF
# --------------------------------------------------
with PdfPages("RandomForest_Trees.pdf") as pdf:
    for i, tree in enumerate(model.estimators_):
        plt.figure(figsize=(16, 9))
        plot_tree(
            tree,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True
        )
        plt.title(f"Decision Tree {i + 1}")
        pdf.savefig()
        plt.close()

print("All trees saved in RandomForest_Trees.pdf")

# --------------------------------------------------
# PREDICTION USING MAJORITY VOTING
# --------------------------------------------------
new_sample = X[0].reshape(1, -1)

print("\nTree-wise Predictions:")
tree_predictions = []

for i, tree in enumerate(model.estimators_):
    pred = tree.predict(new_sample)[0]
    decoded = class_names[pred]
    tree_predictions.append(decoded)
    print(f"Tree {i + 1} Prediction: {decoded}")

final_prediction = max(set(tree_predictions), key=tree_predictions.count)

print("\nFinal Prediction (Majority Voting):", final_prediction)
