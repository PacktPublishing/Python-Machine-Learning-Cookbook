import datetime

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

from convert_to_timeseries import convert_data_to_timeseries

# Load data from input file
input_file = 'data_hmm.txt'
data = np.loadtxt(input_file, delimiter=',')

# Arrange data for training 
X = np.column_stack([data[:,2]])

# Create and train Gaussian HMM 
print "\nTraining HMM...."
num_components = 4
model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000)
model.fit(X)

# Predict the hidden states of HMM 
hidden_states = model.predict(X)

print "\nMeans and variances of hidden states:"
for i in range(model.n_components):
    print "\nHidden state", i+1
    print "Mean =", round(model.means_[i][0], 3)
    print "Variance =", round(np.diag(model.covars_[i])[0], 3)

# Generate data using model
num_samples = 1000
samples, _ = model.sample(num_samples) 
plt.plot(np.arange(num_samples), samples[:,0], c='black')
plt.title('Number of components = ' + str(num_components))

plt.show()

