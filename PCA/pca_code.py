
# PCA Step-by-Step in Python

import numpy as np
import pandas as pd

# Step 1: Dataset
X = np.array([
    [78, 85, 80, 82],
    [65, 78, 68, 72],
    [90, 92, 88, 91],
    [72, 75, 70, 74],
    [85, 88, 84, 86]
])

# Step 2: Convert to DataFrame for clarity
df = pd.DataFrame(X, columns=['Var1', 'Var2', 'Var3', 'Var4'])
print("Original Dataset:\n", df, "\n")

# Step 3: Standardization
mean = np.mean(X, axis=0)
std = np.std(X, axis=0, ddof=1)  # sample std
Z = (X - mean) / std
print("Standardized Data (Z):\n", np.round(Z, 4), "\n")

# Step 4: Covariance Matrix
C = np.cov(Z.T)  # covariance matrix of features
print("Covariance Matrix:\n", np.round(C, 4), "\n")

# Step 5: Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(C)
print("Eigenvalues:\n", np.round(eigenvalues, 4))
print("Eigenvectors:\n", np.round(eigenvectors, 4), "\n")

# Step 6: Sorting Eigenvalues and Eigenvectors
idx = np.argsort(eigenvalues)[::-1]  # descending order
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Sorted Eigenvalues:\n", np.round(eigenvalues, 4))
print("Sorted Eigenvectors:\n", np.round(eigenvectors, 4), "\n")

# Step 7: Explained Variance Ratio
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("Explained Variance Ratio (%):\n", np.round(explained_variance_ratio*100, 2), "\n")

# Step 8: Project Data onto Principal Components
# Let's use only the first principal component
PC1 = eigenvectors[:, 0].reshape(-1, 1)  # first eigenvector
P = Z.dot(PC1)
print("Principal Component Matrix (PC1):\n", np.round(PC1, 4))
print("Reduced Dataset (After PCA):\n", np.round(P, 4))
