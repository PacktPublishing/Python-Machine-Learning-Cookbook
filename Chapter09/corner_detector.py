import sys

import cv2
import numpy as np

# Load input image -- 'box.png'
input_file = sys.argv[1]
img = cv2.imread(input_file)
cv2.imshow('Input image', img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = np.float32(img_gray)

# Harris corner detector 
img_harris = cv2.cornerHarris(img_gray, 7, 5, 0.04)

# Resultant image is dilated to mark the corners
img_harris = cv2.dilate(img_harris, None)

# Threshold the image 
img[img_harris > 0.01 * img_harris.max()] = [0, 0, 0]

cv2.imshow('Harris Corners', img)
cv2.waitKey()