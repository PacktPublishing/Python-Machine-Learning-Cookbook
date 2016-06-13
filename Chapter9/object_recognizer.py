import argparse 
import cPickle as pickle 

import cv2
import numpy as np

import build_features as bf
from trainer import ERFTrainer

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Extracts features \
            from each line and classifies the data')
    parser.add_argument("--input-image", dest="input_image", required=True,
            help="Input image to be classified")
    parser.add_argument("--model-file", dest="model_file", required=True,
            help="Input file containing the trained model")
    parser.add_argument("--codebook-file", dest="codebook_file", 
            required=True, help="Input file containing the codebook")
    return parser

class ImageTagExtractor(object):
    def __init__(self, model_file, codebook_file):
        with open(model_file, 'r') as f:
            self.erf = pickle.load(f)

        with open(codebook_file, 'r') as f:
            self.kmeans, self.centroids = pickle.load(f)

    def predict(self, img, scaling_size):
        img = bf.resize_image(img, scaling_size)
        feature_vector = bf.BagOfWords().construct_feature(
                img, self.kmeans, self.centroids)
        image_tag = self.erf.classify(feature_vector)[0]
        return image_tag

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    model_file = args.model_file
    codebook_file = args.codebook_file
    input_image = cv2.imread(args.input_image)

    scaling_size = 200

    print "\nOutput:", ImageTagExtractor(model_file, 
            codebook_file).predict(input_image, scaling_size)
