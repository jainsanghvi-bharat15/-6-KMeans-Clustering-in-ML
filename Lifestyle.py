import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)  # Generate same random data every run
n_people = 200      # Total people

# Create synthetic lifestyle dataset
data = {
    'PersonID': np.arange(1, n_people + 1),  # IDs from 1 to 200

    # Mean = 3 hrs, Std deviation = 1.5
    # clip(0,10) limits values between 0 and 10
    'Exercise (hrs/week)': np.random.normal(3, 1.5, n_people).clip(0, 10),

    # Mean = 7 hrs, Std deviation = 1
    'Sleep (hrs/day)': np.random.normal(7, 1.0, n_people).clip(3, 10),

    # Random junk food frequency between 0 and 7
    'Junk Food (times/week)': np.random.randint(0, 8, n_people),

    # Mean = 6 hrs, Std deviation = 2
    'Screen Time (hrs/day)': np.random.normal(6, 2.0, n_people).clip(2, 14)}

df = pd.DataFrame(data)
print(df)

# Select features for clustering
features = [
    'Exercise (hrs/week)',
    'Sleep (hrs/day)',
    'Junk Food (times/week)',
    'Screen Time (hrs/day)']
X = df[features]
scaler = StandardScaler()   # Create scaler object
X_scaled = scaler.fit_transform(X)  # Normalize feature values
kmeans = KMeans(n_clusters=3, random_state=42)  # Create KMeans model with 3 clusters
df['Health Risk Category'] = kmeans.fit_predict(X_scaled)   # Train model 
# Save clustered dataset into CSV file
df.to_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(6) KMeans/LifeStyle.csv", index=False)

plt.figure(figsize=(8, 6))

# Scatter plot:
# X-axis -> Sleep hours
# Y-axis -> Junk food frequency
# Color represents cluster/category
plt.scatter(
    df['Sleep (hrs/day)'],
    df['Junk Food (times/week)'],
    c=df['Health Risk Category'],
    cmap='coolwarm',
    s=60)

# Plot cluster centroids
plt.scatter(
    scaler.inverse_transform(kmeans.cluster_centers_)[:, 1],  # Sleep values
    scaler.inverse_transform(kmeans.cluster_centers_)[:, 2],  # Junk food values
    s=200,
    c='black',
    marker='X',
    label='Centroids')

# Graph title and labels
plt.title('Lifestyle Clustering: Sleep vs. Junk Food')
plt.xlabel('Sleep (hrs/day)')
plt.ylabel('Junk Food (times/week)')

# Show legend and grid
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout
plt.show()          # Display graph

# kmeans.cluster_centers_ = Gives centroid coordinates in scaled form
# scaler.inverse_transform(...) = Converts centroids back to original values