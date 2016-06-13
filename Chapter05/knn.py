import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Input data
X = np.array([[1, 1], [1, 3], [2, 2], [2.5, 5], [3, 1], 
        [4, 2], [2, 3.5], [3, 3], [3.5, 4]])

# Number of neighbors we want to find
num_neighbors = 3

# Input point
input_point = [2.6, 1.7]

# Plot datapoints
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='o', s=25, color='k')

# Build nearest neighbors model
knn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)
distances, indices = knn.kneighbors(input_point)

# Print the 'k' nearest neighbors
print "\nk nearest neighbors"
for rank, index in enumerate(indices[0][:num_neighbors]):
    print str(rank+1) + " -->", X[index]

# Plot the nearest neighbors 
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='o', s=25, color='k')
plt.scatter(X[indices][0][:][:,0], X[indices][0][:][:,1], 
        marker='o', s=150, color='k', facecolors='none')
plt.scatter(input_point[0], input_point[1],
        marker='x', s=150, color='k', facecolors='none')

plt.show()
