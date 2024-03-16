# Importing necessary libraries
from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import numpy as np
import random
import wkpp
import matplotlib as mpl
import matplotlib.pyplot as plt

# Real data input
dataset = fetch_kddcup99()                                                              
data = dataset.data                                                                               
data = np.delete(data,[0,1,2,3],1)                                                
data = data.astype(float)                                                               
data = StandardScaler().fit_transform(data)                               

n = np.size(data,0)                                                                                
d = np.size(data,1)                                                                                
k = 17                                                                                                       
Sample_size = 100                                                                                 

# Function for D2 sampling
def D2(X, k):                                                                                        
    def d2_sampling(X, k, mu_X):
        B = []
        # Sample the first point randomly
        x = np.random.choice(len(X), p=mu_X)
        B.append(X[x])

        for _ in range(2, k+1):
            # Compute distance to the closest point in B for each point in X
            distances = np.array([min([np.linalg.norm(x - b) for b in B]) ** 2 for x in X])
            # Compute the sampling probabilities
            probabilities = mu_X * distances / np.sum(mu_X * distances)
            # Sample the next point
            x = np.random.choice(len(X), p=probabilities)
            B.append(X[x])

        return B

    centers = d2_sampling(X, k, np.ones(len(X))/len(X))
    return centers

# Function for coreset sampling
def Sampling(X, k, centers, Sample_size):
    def coreset_construction(X, k, B, m):
        alpha = 16 * (np.log(k) + 2)
        c_phi = np.mean([min([np.linalg.norm(x - b) for b in B]) for x in X])

        # Compute s(x) for each point in X
        s = {}
        for x in X:
            bi = np.argmin([np.linalg.norm(x - b) for b in B])
            Bi = [x for x in X if np.linalg.norm(x - B[bi]) == min([np.linalg.norm(x - b) for b in B])]
            s[x] = (alpha * np.linalg.norm(x - B[bi]) / (c_phi + 2 * alpha * sum([np.linalg.norm(x - b) for b in Bi]) / len(Bi) + 4 * len(X) / (len(Bi) * c_phi)))

        # Normalize s(x)
        s_sum = sum(s.values())
        p = {x: s_x / s_sum for x, s_x in s.items()}

        # Sample m weighted points
        C_indices = np.random.choice(len(X), size=m, p=list(p.values()))
        C = X[C_indices]

        return C

    coreset = coreset_construction(X, k, centers, Sample_size)
    weight = np.ones(Sample_size) / Sample_size  # Equal weights for simplicity
    return coreset, weight

# Sampling centers using D2 sampling
centers = D2(data, k)                                                                        
coreset, weight = Sampling(data, k, centers, Sample_size)        

#---Running KMeans Clustering---#
fkmeans = KMeans(n_clusters=k, init='k-means++')
fkmeans.fit(data)

#----Practical Coresets Performance----#         
Coreset_centers, _ = wkpp.kmeans_plusplus_w(coreset, k, w=weight, n_local_trials=100)
wt_kmeansclus = KMeans(n_clusters=k, init=Coreset_centers, max_iter=10).fit(coreset, sample_weight=weight)
Coreset_centers = wt_kmeansclus.cluster_centers_
coreset_cost = np.sum(np.min(cdist(data, Coreset_centers) ** 2, axis=1))
relative_error_practicalCoreset = abs(coreset_cost - fkmeans.inertia_) / fkmeans.inertia_

#-----Uniform Sampling based Coreset-----#
tmp = np.random.choice(range(n), size=Sample_size, replace=False)                
sample = data[tmp][:]
sweight = n * np.ones(Sample_size) / Sample_size                                                                                                 
sweight = sweight / np.sum(sweight)                                                                                                

#-----Uniform Sampling based Coreset Performance-----#         
wt_kmeansclus = KMeans(n_clusters=k, init='k-means++', max_iter=10).fit(sample, sample_weight=sweight)
Uniform_centers = wt_kmeansclus.cluster_centers_
uniform_cost = np.sum(np.min(cdist(data, Uniform_centers) ** 2, axis=1))
relative_error_uniformCoreset = abs(uniform_cost - fkmeans.inertia_) / fkmeans.inertia_

print("Relative error from Practical Coreset is", relative_error_practicalCoreset)
print("Relative error from Uniformly random Coreset is", relative_error_uniformCoreset)

# Additional code for image segmentation
image = mpl.image.imread("Downloads/final.JPG")
plt.imshow(image)
image.shape
x = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=4, n_init=10)
kmeans.fit(x)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)
plt.imshow(segmented_img/255)
plt.show()

