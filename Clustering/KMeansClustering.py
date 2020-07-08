import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,3:5].values

#use elbow method to find the no of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters= i,init = 'k-means++',max_iter=300,n_init = 10,
                    random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(list(range(1,11)),wcss)
plt.title("Elbow Method")
plt.xlabel("Cluseters")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters = 5,init ='k-means++',max_iter=300,n_init=10,random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#visualize
plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0 , 1],s = 100, c = 'red', label = 'Cluster1')
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1 , 1],s = 100, c = 'blue', label = 'Cluster2')
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2 , 1],s = 100, c = 'green', label = 'Cluster3')
plt.scatter(x[y_kmeans == 3,0],x[y_kmeans == 3 , 1],s = 100, c = 'cyan', label = 'Cluster4')
plt.scatter(x[y_kmeans == 4,0],x[y_kmeans == 4 , 1],s = 100, c = 'magenta', label = 'Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300, c = 'yellow', label = 'Centroid')