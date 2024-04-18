import cv2
import numpy as np

object_image = cv2.imread('15_14.png')
background_image = cv2.imread('background-10.png')

background_image = cv2.resize(background_image, (object_image.shape[1], object_image.shape[0]))

gray = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    rect = (x, y, w, h)
else:
    rect = (0, 0, 1, 1)

mask = np.zeros(object_image.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv2.grabCut(object_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

bin_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

kernel = np.ones((3, 3), np.uint8)
mask_eroded = cv2.erode(bin_mask, kernel, iterations=1)

final_mask = cv2.threshold(mask_eroded, 0.5, 255, cv2.THRESH_BINARY)[1].astype('uint8')

segmented_image = cv2.bitwise_and(object_image, object_image, mask=final_mask)

background_ready = cv2.bitwise_and(background_image, background_image, mask=cv2.bitwise_not(final_mask))

final_image = cv2.add(segmented_image, background_ready)

cv2.imwrite('final_output.png', final_image)

cv2.imshow('Segmented Image', segmented_image)
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
