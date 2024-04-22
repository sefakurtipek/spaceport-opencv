import cv2
import numpy as np
import os

def nothing(x):
    pass

# Setup window and trackbars
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 300, 300)
cv2.createTrackbar('Lower-H', "Trackbars", 35, 179, nothing)  # Start from green hue
cv2.createTrackbar('Lower-S', "Trackbars", 50, 255, nothing)  # Avoid very low saturation
cv2.createTrackbar('Lower-V', "Trackbars", 50, 255, nothing)  # Avoid very low value
cv2.createTrackbar('Upper-H', "Trackbars", 85, 179, nothing)  # End at green hue
cv2.createTrackbar('Upper-S', "Trackbars", 255, 255, nothing)
cv2.createTrackbar('Upper-V', "Trackbars", 255, 255, nothing)

# Path to the directory containing images
path_to_images = 'samples'
output_directory = 'outputs'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
image_files = [f for f in os.listdir(path_to_images) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Define the structuring element for morphological operation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Process each image in the folder
for file in image_files:
    image_path = os.path.join(path_to_images, file)
    image = cv2.imread(image_path)
    if image is None:
        continue

    # Read trackbar positions for all
    lh = cv2.getTrackbarPos('Lower-H', 'Trackbars')
    ls = cv2.getTrackbarPos('Lower-S', 'Trackbars')
    lv = cv2.getTrackbarPos('Lower-V', 'Trackbars')
    uh = cv2.getTrackbarPos('Upper-H', 'Trackbars')
    us = cv2.getTrackbarPos('Upper-S', 'Trackbars')
    uv = cv2.getTrackbarPos('Upper-V', 'Trackbars')

    # Convert to HSV and create a mask
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([lh, ls, lv])
    upper_green = np.array([uh, us, uv])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)  # Invert the mask
    result = cv2.bitwise_and(image, image, mask=mask_inv)

    # Apply opening to clean up small noise
    opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    eroded = cv2.morphologyEx(opening, cv2.MORPH_ERODE, kernel)

    hsv_image_2 = cv2.cvtColor(eroded, cv2.COLOR_BGR2HSV)
    lower_green_2 = np.array([lh, ls, lv])
    upper_green_2 = np.array([uh, us, uv])
    mask_2 = cv2.inRange(hsv_image_2, lower_green_2, upper_green_2)
    mask_inv_2 = cv2.bitwise_not(mask_2)  # Invert the mask
    result_2 = cv2.bitwise_and(eroded, eroded, mask=mask_inv_2)

    # Find contours on the opening output
    contours, _ = cv2.findContours(cv2.cvtColor(result_2, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the largest contour
        contour_mask = np.zeros_like(mask_inv_2)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Segment the largest contour from the original image
        segmented_image = cv2.bitwise_and(eroded, eroded, mask=contour_mask)

        # Save the segmented image
        output_path = os.path.join(output_directory, f"segmented_{file}")
        cv2.imwrite(output_path, segmented_image)

        # Display the result
        cv2.imshow('Original', image)
        cv2.imshow('Segmented Image', segmented_image)
        key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        if key == 27:  # Press 'ESC' to exit
            break

cv2.destroyAllWindows()