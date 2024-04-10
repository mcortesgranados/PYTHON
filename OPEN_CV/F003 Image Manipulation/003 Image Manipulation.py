"""
In this code:

The image is loaded using cv2.imread() from a file named 'example.jpg'.
Various image manipulation operations are performed:
Cropping a region of interest (ROI) from the image.
Resizing the image to a specified width and height.
Rotating the image by a specified angle.
Flipping the image horizontally.
Warping the image using a perspective transformation.
Each manipulated image is displayed in a separate window using cv2.imshow().
Make sure to replace 'example.jpg' with the path to your image file. This code demonstrates some common image manipulation techniques, 
but OpenCV provides many more functions for advanced image processing and manipulation tasks.
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

    # Crop a region of interest (ROI) from the image
    roi = image[100:300, 200:400]  # Example: Extracting a 200x200 region starting from (100, 200)
    cv2.imshow('ROI', roi)
    cv2.waitKey(0)

    # Resize the image
    resized_image = cv2.resize(image, (400, 300))  # Example: Resize to 400x300
    cv2.imshow('Resized Image', resized_image)
    cv2.waitKey(0)

    # Rotate the image
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)  # Rotate by 45 degrees
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)

    # Flip the image horizontally
    flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
    cv2.imshow('Flipped Image', flipped_image)
    cv2.waitKey(0)

    # Warp the image (perspective transformation)
    # Example: Define source and destination points for a perspective transformation
    src_points = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    dst_points = [[50, 50], [width - 100, 100], [width - 50, height - 50], [100, height - 100]]
    perspective_matrix = cv2.getPerspectiveTransform(
        src=np.float32(src_points),
        dst=np.float32(dst_points)
    )
    warped_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
    cv2.imshow('Warped Image', warped_image)
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()
else:
    print("Error: Unable to load the image.")
