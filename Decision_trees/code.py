import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --------------------------------------------------
# LOAD IRIS DATASET
# --------------------------------------------------
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["target"] = df["target"].map({
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
})

df.to_csv("iris_dataset.csv", index=False)
print("Dataset saved as iris_dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

# --------------------------------------------------
# TRAIN DECISION TREE MODEL (ENTROPY)
# --------------------------------------------------
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

print("Sakthi Priya - 2303717710422044")

# --------------------------------------------------
# INPUT HANDLING FUNCTIONS
# --------------------------------------------------
def parse_input_values(raw):
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != X.shape[1]:
        raise ValueError(f"Expected {X.shape[1]} values, got {len(parts)}")
    return [float(p) for p in parts]

def get_user_input(args):
    if args.input:
        return parse_input_values(args.input)

    if args.interactive:
        vals = []
        print("Enter feature values:")
        for col in X.columns:
            while True:
                try:
                    v = float(input(f"{col}: "))
                    vals.append(v)
                    break
                except ValueError:
                    print("Enter a valid number.")
        return vals

    # Default sample
    return [5.1, 3.5, 1.4, 0.2]

# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Decision Tree on Iris Dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Comma-separated values (e.g. 5.1,3.5,1.4,0.2)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter values interactively"
    )

    args = parser.parse_args()
    sample = get_user_input(args)

    prediction = model.predict(
        pd.DataFrame([sample], columns=X.columns)
    )

    print("Predicted Class:", prediction[0])

# --------------------------------------------------
# EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    main()

# --------------------------------------------------
# PLOT DECISION TREE
# --------------------------------------------------
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    fontsize=12
)

plt.savefig("iris_decision_tree.pdf", bbox_inches="tight")
plt.show()
