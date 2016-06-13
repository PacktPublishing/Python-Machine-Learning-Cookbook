import os
import sys

import cv2
import numpy as np

# Load input data 
input_file = 'letter.data' 

# Define visualization parameters 
scaling_factor = 10
start_index = 6
end_index = -1
h, w = 16, 8

# Loop until you encounter the Esc key
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = np.array([255*float(x) for x in line.split('\t')[start_index:end_index]])
        img = np.reshape(data, (h,w))
        img_scaled = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)
        cv2.imshow('Image', img_scaled)
        c = cv2.waitKey()
        if c == 27:
            break
