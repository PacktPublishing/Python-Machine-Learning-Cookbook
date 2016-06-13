import os

import cv2
import numpy as np
from sklearn import preprocessing

# Class to handle tasks related to label encoding
class LabelEncoder(object):
    # Method to encode labels from words to numbers
    def encode_labels(self, label_words):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label_words)

    # Convert input label from word to number
    def word_to_num(self, label_word):
        return int(self.le.transform([label_word])[0])

    # Convert input label from number to word
    def num_to_word(self, label_num):
        return self.le.inverse_transform([label_num])[0]

# Extract images and labels from input path
def get_images_and_labels(input_path):
    label_words = []

    # Iterate through the input path and append files
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            label_words.append(filepath.split('/')[-2]) 
            
    # Initialize variables
    images = []
    le = LabelEncoder()
    le.encode_labels(label_words)
    labels = []

    # Parse the input directory
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)

            # Read the image in grayscale format
            image = cv2.imread(filepath, 0) 

            # Extract the label
            name = filepath.split('/')[-2]
                
            # Perform face detection
            faces = faceCascade.detectMultiScale(image, 1.1, 2, minSize=(100,100))

            # Iterate through face rectangles
            for (x, y, w, h) in faces:
                images.append(image[y:y+h, x:x+w])
                labels.append(le.word_to_num(name))

    return images, labels, le

if __name__=='__main__':
    cascade_path = "cascade_files/haarcascade_frontalface_alt.xml"
    path_train = 'faces_dataset/train'
    path_test = 'faces_dataset/test'

    # Load face cascade file
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # Initialize Local Binary Patterns Histogram face recognizer
    recognizer = cv2.face.createLBPHFaceRecognizer()

    # Extract images, labels, and label encoder from training dataset
    images, labels, le = get_images_and_labels(path_train)

    # Train the face recognizer 
    print "\nTraining..."
    recognizer.train(images, np.array(labels))

    # Test the recognizer on unknown images
    print '\nPerforming prediction on test images...'
    stop_flag = False
    for root, dirs, files in os.walk(path_test):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)

            # Read the image
            predict_image = cv2.imread(filepath, 0)

            # Detect faces
            faces = faceCascade.detectMultiScale(predict_image, 1.1, 
                    2, minSize=(100,100))

            # Iterate through face rectangles
            for (x, y, w, h) in faces:
                # Predict the output
                predicted_index, conf = recognizer.predict(
                        predict_image[y:y+h, x:x+w])

                # Convert to word label
                predicted_person = le.num_to_word(predicted_index)

                # Overlay text on the output image and display it
                cv2.putText(predict_image, 'Prediction: ' + predicted_person, 
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)
                cv2.imshow("Recognizing face", predict_image)

            c = cv2.waitKey(0)
            if c == 27:
                stop_flag = True
                break

        if stop_flag:
            break

