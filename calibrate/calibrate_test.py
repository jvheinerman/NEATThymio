import cv2
import numpy as np
import glob

image_files = sorted(glob.glob("hsv_*.jpg"))

blue_lower = np.array([67, 50, 50])
blue_upper = np.array([200, 255, 255])

green_lower = np.array([15, 165, 30])
green_upper = np.array([55, 255, 155])

# already blurred (5, 5)!
blur = (15, 15)
# so final blur should be (19, 19)
# ... approximately

hsv1 = cv2.imread(image_files[0])
hsv1 = cv2.GaussianBlur(hsv1, blur, 0)
for f_name in image_files[1:4]:
    hsv1_next = cv2.imread(f_name)
    hsv1_next = cv2.GaussianBlur(hsv1_next, blur, 0)
    hsv1 = np.hstack((hsv1, hsv1_next))

hsv2 = cv2.imread(image_files[4])
hsv2 = cv2.GaussianBlur(hsv2, blur, 0)
for f_name in image_files[5:]:
    hsv2_next = cv2.imread(f_name)
    hsv2_next = cv2.GaussianBlur(hsv2_next, blur, 0)
    hsv2 = np.hstack((hsv2, hsv2_next))

hsv = np.vstack((hsv1, hsv2))

green = cv2.inRange(hsv, green_lower, green_upper)
blue = cv2.inRange(hsv, blue_lower, blue_upper)
red = np.zeros(green.shape, np.uint8)

result = np.dstack([blue, green, red])

cv2.imshow("combination", result)
cv2.waitKey()
cv2.destroyAllWindows()
