import cv2
import pickle
import numpy as np

def generate_dilations():
    shape = (240, 320)
    point = np.zeros(shape)
    mid_point = point.shape[0] - 1, point.shape[1] / 2 - 1
    point[mid_point] = 1
    kernel = np.array([[0, 0, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 0, 0]], dtype=np.uint8)

    points = []
    counter = 0
    while 0 in point:
        counter += 1
        print('Doing ' + str(counter))
        points.append(point)
        point = cv2.dilate(point, kernel)

    print('Pickling...')
    pickle.dump(points, open('pickled/dilated_points.p', 'wb'))


generate_dilations()
