import cv2
import numpy as np

# Load face, eye, and nose cascade files
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_nose.xml')

# Check if face cascade file has been loaded
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

# Check if eye cascade file has been loaded
if eye_cascade.empty():
	raise IOError('Unable to load the eye cascade classifier xml file')

# Check if nose cascade file has been loaded
if nose_cascade.empty():
    raise IOError('Unable to load the nose cascade classifier xml file')

# Initialize video capture object and define scaling factor
cap = cv2.VideoCapture(0)
scaling_factor = 0.5

while True:
    # Read current frame, resize it, and convert it to grayscale
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, 
            interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run face detector on the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Run eye and nose detectors within each face rectangle
    for (x,y,w,h) in faces:
        # Grab the current ROI in both color and grayscale images
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Run eye detector in the grayscale ROI
        eye_rects = eye_cascade.detectMultiScale(roi_gray)

        # Run nose detector in the grayscale ROI
        nose_rects = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

        # Draw green circles around the eyes
        for (x_eye, y_eye, w_eye, h_eye) in eye_rects:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)

        for (x_nose, y_nose, w_nose, h_nose) in nose_rects:
            cv2.rectangle(roi_color, (x_nose, y_nose), (x_nose+w_nose, 
                y_nose+h_nose), (0,255,0), 3)
            break
    
    # Display the image
    cv2.imshow('Eye and nose detector', frame)

    # Check if Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

