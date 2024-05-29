#K-Means clustering to group mall customers based on their spending score.

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv('Clustering/Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Using dendogram method to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian distances')
plt.show()


#Training the Hierarchical Clustering Model on the dataset
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=5,metric='euclidean',linkage='ward')
y_hc = model.fit_predict(X)

print(y_hc)

#Visualization
plt.scatter(X[y_hc == 0, 0],X[y_hc == 0, 1],s=100,c='red',label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0],X[y_hc == 1, 1],s=100,c='blue',label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0],X[y_hc == 2, 1],s=100,c='green',label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0],X[y_hc == 3, 1],s=100,c='cyan',label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0],X[y_hc == 4, 1],s=100,c='magenta',label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()