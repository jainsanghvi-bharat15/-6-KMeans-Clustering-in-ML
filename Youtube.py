# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)  # Generate same random data every run
n_videos = 150  # Total number of videos

# Create synthetic YouTube dataset
data = {
    'VideoID': np.arange(1, n_videos + 1),  # IDs from 1 to 150

    # Random views between 1,000 and 10,00,000
    'Views': np.random.randint(1000, 1000000, n_videos),

    # Random likes between 100 and 50,000
    'Likes': np.random.randint(100, 50000, n_videos),

    # Mean = 5000 mins, Std deviation = 1500
    'Watch Time (mins)': np.random.normal(5000, 1500, n_videos).astype(int) }
df = pd.DataFrame(data)
df.head()

# Select features for clustering
features = ['Views', 'Likes', 'Watch Time (mins)']
X = df[features]
scaler = StandardScaler()           # Create scaler object
X_scaled = scaler.fit_transform(X)  # Normalize data

# Create KMeans model with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)    # Train model and assign cluster labels

# Save clustered dataset
df.to_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(6) KMeans/Youtube.csv", index=False)

plt.figure(figsize=(8, 6))
# Scatter plot:
# X-axis -> Views
# Y-axis -> Watch Time
# Color represents clusters
plt.scatter(
    df['Views'],
    df['Watch Time (mins)'],
    c=df['Cluster'],
    cmap='viridis',
    s=60)

# Plot cluster centroids
plt.scatter(
    scaler.inverse_transform(kmeans.cluster_centers_)[:, 0],  # Views
    scaler.inverse_transform(kmeans.cluster_centers_)[:, 2],  # Watch Time
    s=200,
    c='red',
    marker='X',
    label='Centroids')

# Graph title and labels
plt.title('YouTube Video Clustering by Views and Watch Time')
plt.xlabel('Views')
plt.ylabel('Watch Time (mins)')

# Show legend and grid
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout
plt.show()          # Display graph

# kmeans.cluster_centers_ = Returns centroid coordinates in scaled form
# scaler.inverse_transform(...) = Converts centroids back to original values