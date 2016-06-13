import sys

import cv2
import numpy as np

class StarFeatureDetector(object):
    def __init__(self):
        self.detector = cv2.xfeatures2d.StarDetector_create()

    def detect(self, img):
        return self.detector.detect(img)

if __name__=='__main__':
    # Load input image -- 'table.jpg'
    input_file = sys.argv[1]
    input_img = cv2.imread(input_file)

    # Convert to grayscale
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Detect features using Star feature detector
    keypoints = StarFeatureDetector().detect(input_img)

    # Draw keypoints and display the image
    cv2.drawKeypoints(input_img, keypoints, input_img, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Star features', input_img)

    cv2.waitKey()
