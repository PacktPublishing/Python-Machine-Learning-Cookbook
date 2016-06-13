import os
import argparse 
import cPickle as pickle 

import numpy as np
import matplotlib.pyplot as plt
from pystruct.datasets import load_letters
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the CRF classifier')
    parser.add_argument("--c-value", dest="c_value", required=False, type=float,
            default=1.0, help="The C value that will be used for training")
    return parser

class CRFTrainer(object):
    def __init__(self, c_value, classifier_name='ChainCRF'):
        self.c_value = c_value
        self.classifier_name = classifier_name

        if self.classifier_name == 'ChainCRF':
            model = ChainCRF()
            self.clf = FrankWolfeSSVM(model=model, C=self.c_value, max_iter=50) 
        else:
            raise TypeError('Invalid classifier type')

    def load_data(self):
        letters = load_letters()
        X, y, folds = letters['data'], letters['labels'], letters['folds']
        X, y = np.array(X), np.array(y)
        return X, y, folds

    # X is a numpy array of samples where each sample
    # has the shape (n_letters, n_features) 
    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        return self.clf.score(X_test, y_test)

    # Run the classifier on input data
    def classify(self, input_data):
        return self.clf.predict(input_data)[0]

def decoder(arr):
    alphabets = 'abcdefghijklmnopqrstuvwxyz'
    output = ''
    for i in arr:
        output += alphabets[i] 

    return output

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    c_value = args.c_value

    crf = CRFTrainer(c_value)
    X, y, folds = crf.load_data()
    X_train, X_test = X[folds == 1], X[folds != 1]
    y_train, y_test = y[folds == 1], y[folds != 1]

    print "\nTraining the CRF model..."
    crf.train(X_train, y_train)

    score = crf.evaluate(X_test, y_test)
    print "\nAccuracy score =", str(round(score*100, 2)) + '%'

    print "\nTrue label =", decoder(y_test[0])
    predicted_output = crf.classify([X_test[0]])
    print "Predicted output =", decoder(predicted_output)

