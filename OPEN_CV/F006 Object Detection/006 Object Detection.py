"""

Object detection is a crucial task in computer vision that involves identifying and locating objects
of interest within an image or video. OpenCV provides various methods for object detection,
including pre-trained deep learning models and traditional computer vision techniques. 
Here's a Python code sample demonstrating how to perform object detection using OpenCV with the Haar cascade classifier:

In this code:


The Haar cascade classifier for face detection is loaded using cv2.CascadeClassifier().

The image is loaded using cv2.imread() from a file named 'example.jpg'.

The image is converted to grayscale using cv2.cvtColor().

Faces are detected in the grayscale image using face_cascade.detectMultiScale().

Rectangles are drawn around the detected faces using cv2.rectangle().

The image with detected faces is displayed using cv2.imshow().

Press any key to close the window.

Make sure to replace 'example.jpg' with the path to your image file. This code demonstrates face detection using the Haar cascade classifier, but
 you can use similar techniques for detecting other objects by using different pre-trained classifiers or deep learning models.

"""

import cv2

# File path to the image
file_path = 'K:\PYTHON\OPEN_CV\F001 Image Loading and Display\example_several_faces.jpg'



# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image from file
image = cv2.imread(file_path)

# Check if the image was successfully loaded
if image is not None:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the image with detected faces
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load the image.")
