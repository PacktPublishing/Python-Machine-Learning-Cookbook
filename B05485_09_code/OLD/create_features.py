import os
import sys
import argparse
import cPickle as pickle
import json

import cv2
import numpy as np
from sklearn.cluster import KMeans

from dense_detector import DenseDetector

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Extract features from a given \
            set of images')

    parser.add_argument("--data", dest="training_data", nargs="+", action="append", 
            required=True, help="Folders containing the training images. \
            The first element should be the class label.")
    parser.add_argument("--codebook-file", dest='codebook_file', required=True,
            help="Output file where the codebook will be stored")
    parser.add_argument("--feature-map-file", dest='feature_map_file', required=True,
            help="Output file where the feature map will be stored")
    parser.add_argument("--scaling-size", dest="scaling_size", type=int, 
            default=200, help="Scales the longer dimension of the image down \
                    to this size.")

    return parser

def load_input_map(label, input_folder):
    combined_data = []

    if not os.path.isdir(input_folder):
        print "The folder " + input_folder + " doesn't exist"
        raise IOError
        
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if x.endswith('.jpg')):
            combined_data.append({'label': label, 'image': os.path.join(root, filename)})
                    
    return combined_data

class FeatureExtractor(object):
    def extract_image_features(self, img):
        keypoints = DenseDetector().detect(img)
        keypoints, feature_vectors = SIFTExtractor().compute(img, keypoints)
        return feature_vectors

    def get_centroids(self, input_map, scaling_size, num_samples_to_fit=10):
        keypoints_all = []
        
        count = 0
        cur_label = ''
        for item in input_map:
            if count >= num_samples_to_fit:
                if cur_label != item['label']:
                    count = 0
                else:
                    continue

            count += 1

            if count == num_samples_to_fit:
                print "Built centroids for", item['label']

            cur_label = item['label']
            img = cv2.imread(item['image'])
            img = resize_image(img, scaling_size)

            num_dims = 128
            feature_vectors = self.extract_image_features(img)
            keypoints_all.extend(feature_vectors) 

        kmeans, centroids = Quantizer().quantize(keypoints_all)
        return kmeans, centroids

    def get_feature_vector(self, img, kmeans, centroids):
        return Quantizer().get_feature_vector(img, kmeans, centroids)

def extract_feature_map(input_map, kmeans, centroids, scaling_size):
    feature_map = []
     
    for item in input_map:
        temp_dict = {}
        temp_dict['label'] = item['label']
    
        print "Extracting features for", item['image']
        img = cv2.imread(item['image'])
        img = resize_image(img, scaling_size)

        temp_dict['feature_vector'] = FeatureExtractor().get_feature_vector(
                    img, kmeans, centroids)

        if temp_dict['feature_vector'] is not None:
            feature_map.append(temp_dict)

    return feature_map

class Quantizer(object):
    def __init__(self, num_clusters=32):
        self.num_dims = 128
        self.extractor = SIFTExtractor()
        self.num_clusters = num_clusters
        self.num_retries = 10

    def quantize(self, datapoints):
        kmeans = KMeans(self.num_clusters, 
                        n_init=max(self.num_retries, 1),
                        max_iter=10, tol=1.0)

        res = kmeans.fit(datapoints)
        centroids = res.cluster_centers_
        return kmeans, centroids

    def normalize(self, input_data):
        sum_input = np.sum(input_data)

        if sum_input > 0:
            return input_data / sum_input
        else:
            return input_data

    def get_feature_vector(self, img, kmeans, centroids):
        keypoints = DenseDetector().detect(img)
        keypoints, feature_vectors = self.extractor.compute(img, keypoints)
        labels = kmeans.predict(feature_vectors)
        feature_vector = np.zeros(self.num_clusters)

        for i, item in enumerate(feature_vectors):
            feature_vector[labels[i]] += 1

        feature_vector_img = np.reshape(feature_vector, 
                ((1, feature_vector.shape[0])))
        return self.normalize(feature_vector_img)

# Class to extract SIFT features
class SIFTExtractor(object):
    def compute(self, img, keypoints):
        if img is None:
            raise TypeError('Invalid input image')

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = cv2.SIFT().compute(img_gray, keypoints)
        return keypoints, descriptors

# Resize the shorter dimension to 'new_size' 
# while maintaining the aspect ratio
def resize_image(input_img, new_size):
    h, w = input_img.shape[:2]
    scaling_factor = new_size / float(h)

    if w < h:
        scaling_factor = new_size / float(w)

    new_shape = (int(w * scaling_factor), int(h * scaling_factor))
    return cv2.resize(input_img, new_shape) 

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    scaling_size = args.scaling_size
    
    input_map = []
    for training_data in args.training_data:
        if len(training_data) != 2: 
            raise IOError("Invalid format; should be '<label-name> <folder>'")

        label = training_data[0]
        input_map += load_input_map(label, training_data[1])

    # Building the visual codebook
    print "====== Building visual codebook ======"
    kmeans, centroids = FeatureExtractor().get_centroids(input_map, scaling_size)
    if args.codebook_file:
        with open(args.codebook_file, 'w') as f:
            pickle.dump((kmeans, centroids), f)
    
    # Input data and labels
    print "\n====== Building the feature map ======"
    feature_map = extract_feature_map(input_map, kmeans, centroids, scaling_size)
    if args.feature_map_file:
        with open(args.feature_map_file, 'w') as f:
            pickle.dump(feature_map, f)

