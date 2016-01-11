import cv2
import numpy as np
import glob

image_files = sorted(glob.glob("hsv_*.jpg"))

blue_lower = np.array([60, 50, 50])
blue_upper = np.array([200, 255, 255])

green_lower = np.array([20, 180, 20])
green_upper = np.array([45, 255, 255])

hsv1 = cv2.imread(image_files[0])
for f_name in image_files[1:4]:
    hsv1 = np.hstack((hsv1, cv2.imread(f_name)))

hsv2 = cv2.imread(image_files[4])
for f_name in image_files[5:]:
    hsv2 = np.hstack((hsv2, cv2.imread(f_name)))

hsv = np.vstack((hsv1, hsv2))

green = cv2.inRange(hsv, green_lower, green_upper)
blue = cv2.inRange(hsv, blue_lower, blue_upper)
red = np.zeros(green.shape, np.uint8)

result = np.dstack([blue, green, red])

cv2.imshow("combination", result)
cv2.waitKey()
cv2.destroyAllWindows()
