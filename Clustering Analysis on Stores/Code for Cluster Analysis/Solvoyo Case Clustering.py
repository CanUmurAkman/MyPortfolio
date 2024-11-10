#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
file_path = '/Users/macbookpro/Desktop/Data for K-Means Clustering.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset
data.head()


# In[2]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[3]:


# Extract the relevant columns for clustering
X = data[['Average Quarter', 'Average Excess', 'Average Revenue']]


# In[4]:


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[5]:


from sklearn.metrics import silhouette_score

# Calculate silhouette scores for different values of K
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot the silhouette scores to find the optimal K
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Analysis')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()


# In[6]:


# Apply K-Means with 5 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)


# In[7]:


import os


# In[8]:


# Save the Excel File with the added column indicating which cluster a store is assigned to & the plot below
export_dataset_path = os.path.join('/Users/macbookpro/Desktop', 'Clustered_Store_Data.xlsx')
export_plot_path = os.path.join('/Users/macbookpro/Desktop', 'Store_Clusters.pdf')

# Save the updated dataset with cluster assignments
data.to_excel(export_dataset_path, index=False)


# In[9]:


# Visualize the clusters (Note: Does not give full information on the Average Quarter of the orders)
plt.figure(figsize=(10, 6))
plt.scatter(data['Average Excess'], data['Average Revenue'], c=data['Cluster'], cmap='viridis', marker='o')
plt.xlabel('Average Excess')
plt.ylabel('Average Revenue')
plt.title('Store Clusters')
plt.colorbar(label='Cluster')
plt.savefig(export_plot_path, format='pdf')
plt.show()


# In[10]:


# Display the first few rows with cluster assignments
data.head()

