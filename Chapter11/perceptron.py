import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

# Define input data
data = np.array([[0.3, 0.2], [0.1, 0.4], [0.4, 0.6], [0.9, 0.5]])
labels = np.array([[0], [0], [0], [1]])

# Plot input data
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Input data')

# Define a perceptron with 2 inputs;
# Each element of the list in the first argument 
# specifies the min and max values of the inputs
perceptron = nl.net.newp([[0, 1],[0, 1]], 1)

# Train the perceptron
error = perceptron.train(data, labels, epochs=50, show=15, lr=0.01)

# plot results
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()
plt.title('Training error progress')

plt.show()