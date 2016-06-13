import cv2

# Initialize video capture object
cap = cv2.VideoCapture(0)

# Define the image size scaling factor
scaling_factor = 0.5

# Loop until you hit the Esc key
while True:
    # Capture the current frame
    ret, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, 
            interpolation=cv2.INTER_AREA)

    # Display the image
    cv2.imshow('Webcam', frame)

    # Detect if the Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break

# Release the video capture object
cap.release()

# Close all active windows
cv2.destroyAllWindows()
