import numpy as np
from sklearn.decomposition import PCA

# Create random dataset
X = np.random.rand(10, 5)   # 10 rows (samples), 5 columns (features)

print(X)
print("Original Shape:", X.shape)

# Create PCA object
# n_components=2 means reduce data to 2 dimensions
pca = PCA(n_components=2)

# Fit PCA on data and transform it
# fit() -> learns important directions
# transform() -> converts data into reduced form
X_reduced = pca.fit_transform(X)

# Display reduced dataset
print("\nReduced Data:")
print(X_reduced)

# Display new shape
print("\nReduced Shape:", X_reduced.shape)

# Display variance captured by each component
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)