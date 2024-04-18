import cv2
import numpy as np

# Load the image and downscale for faster processing
object_image = cv2.imread('15_14.png')
background_image = cv2.imread('backgroud-10.png')

#scale_percent = 50  # percentage of original size
#width = int(object_image.shape[1] * scale_percent / 100)
#height = int(object_image.shape[0] * scale_percent / 100)
#dim = (width, height)
#resized = cv2.resize(object_image, dim, interpolation=cv2.INTER_AREA)

# Convert to grayscale
gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

# Threshold to isolate the object from a white background
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Use the bounding rect of the largest contour
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    rect = (x, y, w, h)
else:
    rect = (0, 0, 1, 1)

# Initialize GrabCut mask and models
mask = np.zeros(object_image.shape[:2], dtype=np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Apply GrabCut with the found rectangle
cv2.grabCut(object_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Convert the GrabCut mask to a binary mask
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Erode the mask to reduce boundary artifacts
kernel = np.ones((3, 3), np.uint8)
mask_eroded = cv2.erode(mask2, kernel, iterations=1)

# Use the eroded mask to create the final segmented image
segmented_image = object_image * mask_eroded[:, :, np.newaxis]

cv2.imshow('Segmented Image', segmented_image)
#cv2.imshow('combined Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()