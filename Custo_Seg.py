# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
np.random.seed(42)  # Generate same random data every run
n_customers = 200   # Total customers

# Create dataset
data = {'CustomerID': np.arange(1, n_customers + 1),  # IDs from 1 to 200
    # Generate annual income data
    # mean = 60, std deviation = 20
    'Annual Income (k$)': np.random.normal(60, 20, n_customers).astype(int),
    # Random spending score between 1 and 100
    'Spending Score (1-100)': np.random.randint(1, 101, n_customers)}
df = pd.DataFrame(data)
df.shape

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()   # Create scaler object
X_scaled = scaler.fit_transform(X)  # Normalize data
print(X_scaled) # Display scaled data


# Create KMeans model with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)    # Train model and assign cluster labels
# Save clustered dataset
df.to_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(6) KMeans/Custom_Seg.csv",index=False)

plt.figure(figsize=(8, 6))
plt.scatter(                    # Plot customer points
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],            # Color based on cluster
    cmap='viridis',
    s=50)

# Plot centroids
plt.scatter(
    scaler.inverse_transform(kmeans.cluster_centers_)[:, 0], # X-axis values
    scaler.inverse_transform(kmeans.cluster_centers_)[:, 1], # Y-axis values
    s=200,
    c='red',
    marker='X',
    label='Centroids')

# Graph title and labels
plt.title('Customer Segmentation Based on Spending Habits')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')

# Show legend and grid
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout
plt.show()

# kmeans.cluster_centers_ :  Returns centroid coordinates in scaled form 
# scaler.inverse_transform(...) : Converts scaled centroids back to original values
# [:,0] = All rows, first column → Income values
# [:,1] = All rows, second column → Spending Score values