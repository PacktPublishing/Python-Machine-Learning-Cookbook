import sys

import cv2
import numpy as np

class DenseDetector(object):
    def __init__(self, step_size=25, feature_scale=50, img_bound=25):
        self.detector = cv2.FeatureDetector_create("Dense")
        self.detector.setInt("initXyStep", step_size)
        self.detector.setInt("initFeatureScale", feature_scale)
        self.detector.setInt("initImgBound", img_bound)

    def detect(self, img):
        return self.detector.detect(img)

if __name__=='__main__':
    # Load input image -- 'table.jpg'
    input_file = sys.argv[1]
    input_img = cv2.imread(input_file)

    # Convert to grayscale
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Detect features using dense feature detector
    keypoints = DenseDetector(20, 20, 5).detect(input_img)

    # Draw keypoints and display the image
    input_img = cv2.drawKeypoints(input_img, keypoints, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Dense features', input_img)

    cv2.waitKey()