import cv2
import numpy as np

# Load images
object_image = cv2.imread('15_14.png')
background_image = cv2.imread('background-10.png')

# Resize background to match the object image
background_image = cv2.resize(background_image, (object_image.shape[1], object_image.shape[0]))

# Convert object image to grayscale
gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and smooth edges
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

cv2.imshow('blurred', blurred)
cv2.waitKey(0)

# Thresholding operation on the blurred image
_, thresh = cv2.threshold(blurred, 235, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('thresh', thresh)
cv2.waitKey(0)

# Find contours on the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    rect = (x, y, w, h)
else:
    rect = (0, 0, 1, 1)

# Initialize masks for GrabCut
mask = np.zeros(object_image.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Apply GrabCut algorithm
cv2.grabCut(object_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

cv2.imshow('mask', mask)
cv2.waitKey(0)

# Refine mask with where condition
bin_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

cv2.imshow('bin_mask', bin_mask)
cv2.waitKey(0)

# Morphological operations for smoother edges
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask_eroded = cv2.morphologyEx(bin_mask, cv2.MORPH_ERODE, kernel)

cv2.imshow('mask_eroded', mask_eroded)
cv2.waitKey(0)

mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=1)

cv2.imshow('mask_dilated', mask_dilated)
cv2.waitKey(0)

final_mask = cv2.erode(mask_dilated, kernel, iterations=1)

cv2.imshow('final_mask', final_mask)
cv2.waitKey(0)

# Final mask thresholding to ensure it's binary
final_mask = cv2.threshold(final_mask, 0.5, 255, cv2.THRESH_BINARY)[1].astype('uint8')

cv2.imshow('final_mask 2', final_mask)
cv2.waitKey(0)

# Use final mask to segment object from image
segmented_image = cv2.bitwise_and(object_image, object_image, mask=final_mask)

# Prepare the background, clear area where the object will be placed
background_ready = cv2.bitwise_and(background_image, background_image, mask=cv2.bitwise_not(final_mask))

cv2.imshow('background_ready', background_ready)
cv2.waitKey(0)

# Combine the segmented object with the new background
final_image = cv2.add(segmented_image, background_ready)

# Save and display the final image
cv2.imwrite('final_output.png', final_image)
cv2.imshow('Segmented Image', segmented_image)
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
