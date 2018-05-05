#importing libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')

x = dataset.iloc[:,[3,4]].values

#using the elpow method th get the optimal clutering number 
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11) :
    kmeans = KMeans(n_clusters = i,init = 'k-means++',max_iter = 300 ,n_init=10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11,1),wcss)
plt.title('the elbow method')
plt.xlabel('numer of clusters')
plt.ylabel('wcss')
plt.show()
    
#Appling KMeans with the dataset 
kmeans = KMeans(n_clusters = 5 ,init = 'k-means++',max_iter = 300 ,n_init = 10, random_state = 0)
grouping = kmeans.fit_predict(x)
#VISUALIZING THE CLUSTERS
plt.scatter(x[grouping ==0,0],x[grouping == 0 , 1],s = 100, c = 'red', label = 'carefull')
plt.scatter(x[grouping == 1 , 0],x[grouping == 1 , 1], s = 100 , c = 'green' , label = 'standard')
plt.scatter(x[grouping == 2 , 0],x[grouping == 2 , 1], s = 100 , c = 'blue' , label = 'Target')
plt.scatter(x[grouping == 3 , 0],x[grouping == 3 , 1], s = 100 , c = 'cyan' , label = 'careless')
plt.scatter(x[grouping == 4 , 0],x[grouping == 4 , 1], s = 100 , c = 'magenta' , label = 'sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'centroids')
plt.title('clients segmentation')
plt.xlabel('anual income(K$)')
plt.ylabel('spending score[1-100]')
plt.legend()
plt.show()