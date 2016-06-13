import numpy as np
import matplotlib.pyplot as plt

# Define the two groups 
group1 = ['France', 'Italy', 'Spain', 'Portugal', 'Germany'] 
group2 = ['Japan', 'China', 'Brazil', 'Russia', 'Australia']

# Generate some random values
data = np.random.rand(5, 5)

# Create a figure
fig, ax = plt.subplots()

# Create the heat map
heatmap = ax.pcolor(data, cmap=plt.cm.gray)

# Add major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

# Make it look like a table 
ax.invert_yaxis()
ax.xaxis.tick_top()

# Add tick labels
ax.set_xticklabels(group2, minor=False)
ax.set_yticklabels(group1, minor=False)

plt.show()
