import csv
import pandas as pd 
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.figure_factory as ff

df=pd.read_csv("stars.csv")

X = df.iloc[:, [0, 1]].values

print (X)

wcss = []

for i in range(1, 11):
  kmeans=KMeans(n_clusters=i, init="k-means++", random_state = 42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1')
sns. scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2')
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label='Cluster 3')
sns.scatterplot(kmeans.cluster_centers [:, 0], kmeans.cluster_centers [:, 1], color = 'red', label = 'Centroids',s=100, marker=',')
plt.grid(False)
plt.title('Clusters of Interstellar Objects')
plt.xlabel('Size')
plt.ylabel('Light')
plt.legend()
plt.show()
