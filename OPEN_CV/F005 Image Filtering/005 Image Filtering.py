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
