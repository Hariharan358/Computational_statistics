import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(X)

# Create a PCA object with 2 components
pca = PCA(n_components=2)

# Fit and transform the standardized data
X_pca = pca.fit_transform(X_scaled)

# Print the shape of the transformed data
print(X_pca.shape)  # output: (150, 2)

# Plot the 2-component PCA
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2-Component PCA of Iris Dataset')
plt.show()
