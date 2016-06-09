import numpy as np
import neurolab as nl

# Input file
input_file = 'letter.data'

# Number of datapoints to load from the input file
num_datapoints = 20

# Distinct characters
orig_labels = 'omandig'

# Number of distinct characters
num_output = len(orig_labels)

# Training and testing parameters
num_train = int(0.9 * num_datapoints)
num_test = num_datapoints - num_train

# Define dataset extraction parameters 
start_index = 6
end_index = -1

# Creating the dataset
data = []
labels = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        # Split the line tabwise
        list_vals = line.split('\t')

        # If the label is not in our ground truth labels, skip it
        if list_vals[1] not in orig_labels:
            continue

        # Extract the label and append it to the main list
        label = np.zeros((num_output, 1))
        label[orig_labels.index(list_vals[1])] = 1
        labels.append(label)

        # Extract the character vector and append it to the main list
        cur_char = np.array([float(x) for x in list_vals[start_index:end_index]])
        data.append(cur_char)

        # Exit the loop once the required dataset has been loaded
        if len(data) >= num_datapoints:
            break

# Convert data and labels to numpy arrays
data = np.asfarray(data)
labels = np.array(labels).reshape(num_datapoints, num_output)

# Extract number of dimensions
num_dims = len(data[0])

# Create and train neural network
net = nl.net.newff([[0, 1] for _ in range(len(data[0]))], [128, 16, num_output])
net.trainf = nl.train.train_gd
error = net.train(data[:num_train,:], labels[:num_train,:], epochs=10000, 
        show=100, goal=0.01)

# Predict the output for test inputs 
predicted_output = net.sim(data[num_train:, :])
print "\nTesting on unknown data:"
for i in range(num_test):
    print "\nOriginal:", orig_labels[np.argmax(labels[i])]
    print "Predicted:", orig_labels[np.argmax(predicted_output[i])]

