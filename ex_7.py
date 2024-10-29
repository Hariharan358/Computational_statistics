import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the Wine dataset
wine = load_wine()

# Get the features and target
X = wine.data
y = wine.target

# Calculate the number of classes and features
n_classes = len(np.unique(y))
n_features = X.shape[1]

print("Number of classes and Features:")
print("Number of classes:", n_classes)
print("Number of features:", n_features)

# Perform LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Calculate the variance ratio
variance_ratio = lda.explained_variance_ratio_
print("Variance Ratio:")
print(variance_ratio)

# Plot the LDA scatter plot
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("LDA Scatter Plot")
plt.show()
