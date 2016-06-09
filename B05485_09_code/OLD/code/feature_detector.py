import sys

import cv2
import numpy as np

# Load input image -- 'table.jpg'
input_file = sys.argv[1]
img = cv2.imread(input_file)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
keypoints = sift.detect(img_gray, None)

img_sift = cv2.drawKeypoints(img, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Input image', img)
cv2.imshow('SIFT features', img_sift)
cv2.waitKey()