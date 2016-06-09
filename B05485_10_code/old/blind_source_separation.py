import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import PCA, FastICA

# Load data
input_file = 'mixture_of_signals.txt'
X = np.loadtxt(input_file)

# Compute ICA
ica = FastICA(n_components=4)

# Reconstruct the signals
signals_ica = ica.fit_transform(X)

# Get estimated mixing matrix
mixing_mat = ica.mixing_  

# Perform PCA 
pca = PCA(n_components=4)
signals_pca = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# Specify parameters for output plots 
models = [X, signals_ica, signals_pca]
colors = ['blue', 'red', 'black', 'green']

# Plotting input signal
plt.figure()
plt.title('Input signal (mixture)')
for i, (sig, color) in enumerate(zip(X.T, colors), 1):
    plt.plot(sig, color=color)

# Plotting ICA signals 
plt.figure()
plt.title('ICA separated signals')
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.94, 
        top=0.94, wspace=0.25, hspace=0.45)
for i, (sig, color) in enumerate(zip(signals_ica.T, colors), 1):
    plt.subplot(4, 1, i)
    plt.title('Signal ' + str(i))
    plt.plot(sig, color=color)

# Plotting PCA signals  
plt.figure()
plt.title('PCA separated signals')
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.94, 
        top=0.94, wspace=0.25, hspace=0.45)
for i, (sig, color) in enumerate(zip(signals_pca.T, colors), 1):
    plt.subplot(4, 1, i)
    plt.title('Signal ' + str(i))
    plt.plot(sig, color=color)

plt.show()

