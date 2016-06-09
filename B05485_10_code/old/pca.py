import numpy as np
from sklearn import decomposition

# Define individual features
x1 = np.random.normal(size=250)
x2 = np.random.normal(size=250)
x3 = 2*x1 + 3*x2
x4 = 4*x1 - x2
x5 = x3 + 2*x4

# Create dataset with the above features
X = np.c_[x1, x3, x2, x5, x4]

# Perform Principal Components Analysis
pca = decomposition.PCA()
pca.fit(X)

# Print variances
variances = pca.explained_variance_
print '\nVariances in decreasing order:\n', variances

# Find the number of useful dimensions
thresh_variance = 0.8
num_useful_dims = len(np.where(variances > thresh_variance)[0])
print '\nNumber of useful dimensions:', num_useful_dims

# As we can see, only the 2 first components are useful
pca.n_components = num_useful_dims
X_new = pca.fit_transform(X)
print '\nShape before:', X.shape
print 'Shape after:', X_new.shape

