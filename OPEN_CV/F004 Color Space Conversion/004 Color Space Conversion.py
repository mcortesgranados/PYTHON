
"""
In this code:

The image is loaded using cv2.imread() from a file named 'example.jpg'.
Various color space conversions are performed using cv2.cvtColor():
BGR to RGB
BGR to HSV (Hue, Saturation, Value)
BGR to HLS (Hue, Lightness, Saturation)
BGR to Lab (CIELAB)
BGR to YUV (YCbCr)
Each converted image is displayed in a separate window using cv2.imshow().
Press any key to close each window and proceed to the next conversion.
Make sure to replace 'example.jpg' with the path to your image file. 
This code demonstrates some common color space conversions, but OpenCV supports many more conversions and color spaces.

"""

import cv2

# File path to the image
file_path = 'K:\PYTHON\OPEN_CV\F001 Image Loading and Display\example.jpg'

# Load an image from file
image = cv2.imread(file_path)

# Check if the image was successfully loaded
if image is not None:
    # Display the original image
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    # Convert the image from BGR to RGB color space
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('RGB Image', rgb_image)
    cv2.waitKey(0)

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV Image', hsv_image)
    cv2.waitKey(0)

    # Convert the image from BGR to HLS color space
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    cv2.imshow('HLS Image', hls_image)
    cv2.waitKey(0)

    # Convert the image from BGR to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    cv2.imshow('Lab Image', lab_image)
    cv2.waitKey(0)

    # Convert the image from BGR to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    cv2.imshow('YUV Image', yuv_image)
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load the image.")
