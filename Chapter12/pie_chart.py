import numpy as np
import matplotlib.pyplot as plt

# Labels and corresponding values in counter clockwise direction
data = {'Apple': 26, 
        'Mango': 17,
        'Pineapple': 21, 
        'Banana': 29, 
        'Strawberry': 11}

# List of corresponding colors
colors = ['orange', 'lightgreen', 'lightblue', 'gold', 'cyan']

# Needed if we want to highlight a section
explode = (0, 0, 0, 0, 0)  

# Plot the pie chart
plt.pie(data.values(), explode=explode, labels=data.keys(), 
        colors=colors, autopct='%1.1f%%', shadow=False, startangle=90)

# Aspect ratio of the pie chart, 'equal' indicates tht we 
# want it to be a circle
plt.axis('equal')

plt.show()
