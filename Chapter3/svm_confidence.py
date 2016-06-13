import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import utilities 

# Load input data
input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)

###############################################
# Train test split
from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)

params = {'kernel': 'rbf'}
classifier = SVC(**params)
classifier.fit(X_train, y_train)

###############################################
# Measure distance from the boundary

input_datapoints = np.array([[2, 1.5], [8, 9], [4.8, 5.2], [4, 4], [2.5, 7], [7.6, 2], [5.4, 5.9]])
print "\nDistance from the boundary:"
for i in input_datapoints:
    print i, '-->', classifier.decision_function(i)[0]

# Confidence measure
params = {'kernel': 'rbf', 'probability': True}
classifier = SVC(**params)
classifier.fit(X_train, y_train)
print "\nConfidence measure:"
for i in input_datapoints:
    print i, '-->', classifier.predict_proba(i)[0]

utilities.plot_classifier(classifier, input_datapoints, [0]*len(input_datapoints), 'Input datapoints', 'True')
plt.show()
