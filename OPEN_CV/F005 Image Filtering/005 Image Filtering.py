
"""
Image filtering is a fundamental operation in image processing used to enhance or modify an image by applying various filters
such as blurring, sharpening, edge detection, and noise reduction. OpenCV provides functions to apply
these filters efficiently. Here's a Python code sample demonstrating how to perform image filtering using OpenCV:

In this code:


The image is loaded using cv2.imread() from a file named 'example.jpg'.

Various image filtering operations are performed using OpenCV functions:

Gaussian blur (cv2.GaussianBlur()): Smoothes the image using a Gaussian filter.

Median blur (cv2.medianBlur()): Computes the median of all the pixels under the kernel window and replaces the central pixel with this median value.

Bilateral filter (cv2.bilateralFilter()): Applies a bilateral filter to preserve edges while reducing noise.

Sobel edge detection (cv2.Sobel()): Applies Sobel edge detection along the x and y directions.

Each filtered image is displayed in a separate window using cv2.imshow().

Press any key to close each window and proceed to the next filter.

Make sure to replace 'example.jpg' with the path to your image file. This code demonstrates some common image filtering techniques, 
but OpenCV provides many more functions for advanced filtering operations.
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

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # Kernel size: (5, 5)
    cv2.imshow('Blurred Image', blurred_image)
    cv2.waitKey(0)

    # Apply median blur to the image
    median_blurred_image = cv2.medianBlur(image, 5)  # Kernel size: 5
    cv2.imshow('Median Blurred Image', median_blurred_image)
    cv2.waitKey(0)

    # Apply bilateral filter to the image
    bilateral_filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    cv2.imshow('Bilateral Filtered Image', bilateral_filtered_image)
    cv2.waitKey(0)

    # Apply Sobel edge detection to the image
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Gradient along x-axis
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Gradient along y-axis
    sobel_edges_image = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges_image = cv2.convertScaleAbs(sobel_edges_image)
    cv2.imshow('Sobel Edges Image', sobel_edges_image)
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load the image.")
