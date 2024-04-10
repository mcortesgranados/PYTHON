import cv2

# File path to the image
file_path = 'K:\PYTHON\OPEN_CV\F001 Image Loading and Display\example.jpg'

# Load an image from file
image = cv2.imread(file_path)

# Check if the image was successfully loaded
if image is not None:
    # Display the image in a window
    cv2.imshow('Image', image)
    
    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Error: Unable to load the image '{file_path}'. Please check the file path and try again.")
