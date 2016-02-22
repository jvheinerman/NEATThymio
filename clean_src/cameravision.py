# -*- coding: utf-8 -*-
import sys
import math
import traceback
import threading
import numpy as np
import io
import pickle
import time

import cv2

import picamera

from parameters import MIN_FPS, MIN_GOAL_DIST

# Recognize color using the camera
class CameraVision(threading.Thread):
    def __init__(self, camera, simulationLogger):
        threading.Thread.__init__(self)
        self.CAMERA_WIDTH = 320
        self.CAMERA_HEIGHT = 240
        self.scale_down = 1
        self.camera = camera
        self.__isCameraAlive = threading.Condition()
        self.__isStopped = threading.Event()
        self.__simLogger = simulationLogger
        self.__imageAreaThreshold = 1000

        loaded_dist_angles = pickle.load(open("distances.p"))
        self.distances = loaded_dist_angles['distances']
        self.angles = loaded_dist_angles['angles']
        self.MAX_DISTANCE = np.max(self.distances) + 1

        # self.presence = [self.MAX_DISTANCE, 0]
        # self.presenceGoal = [self.MAX_DISTANCE, 0]
        self.presence = None
        self.presenceGoal = None

    def stop(self):
        self.__isStopped.set()
        with self.__isCameraAlive:
            self.__isCameraAlive.notify()

    def _stopped(self):
        return self.__isStopped.isSet()

    def readPuckPresence(self):
        return self.presence

    def readGoalPresence(self):
        return self.presenceGoal

    #  return contours with largest area in the image
    def retMaxArea(self, contours):
        max_area = 0
        largest_contour = None
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour
        return largest_contour

    # return area of the largest contour
    def retLargestContour(self, contour, image2, name):
        if not contour is None:
            moment = cv2.moments(contour)
            # m00 is the area
            if moment["m00"] > self.__imageAreaThreshold / self.scale_down:
                # rect_bl = cv2.minAreaRect(contour)
                # rect_bl = ((rect_bl[0][0] * self.scale_down, rect_bl[0][1] * self.scale_down),
                #            (rect_bl[1][0] * self.scale_down, rect_bl[1][1] * self.scale_down), rect_bl[2])
                # box_bl = cv2.cv.BoxPoints(rect_bl)
                # box_bl = np.int0(box_bl)
                # cv2.drawContours(image2, [box_bl], 0, (255, 255, 0), 2)
                # cv2.imshow(name, image2)
                return moment["m00"]
        return 0

    # return sum of the area of all the contours in the image
    def retAllContours(self, contours):
        presence = 0
        for idx, contour in enumerate(contours):
            moment = cv2.moments(contour)

            # m00 is the area
            if moment["m00"] > self.__imageAreaThreshold / self.scale_down:
                presence += moment["m00"]
        return presence

    def retContours(self, lower_color, upper_color, image_total, selector):
        presence = [0, 0, 0, 0]
        binary = cv2.inRange(image_total["bottom"], lower_color, upper_color)
        binary_left = cv2.inRange(image_total["left"], lower_color, upper_color)
        binary_central = cv2.inRange(image_total["central"], lower_color, upper_color)
        binary_right = cv2.inRange(image_total["right"], lower_color, upper_color)

        dilation = np.ones((15, 15), "uint8")

        color_binary = cv2.dilate(binary, dilation)
        color_binary_left = cv2.dilate(binary_left, dilation)
        color_binary_central = cv2.dilate(binary_central, dilation)
        color_binary_right = cv2.dilate(binary_right, dilation)

        binary_total = [color_binary_left, color_binary_central, color_binary_right, color_binary]

        for i in range(len(binary_total)):
            binary_total[i] = cv2.GaussianBlur(binary_total[i], (5, 5), 0)

        contours_total = []
        for el in binary_total:
            contours, hierarchy = cv2.findContours(el, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_total.append(contours)

        # selector == 0 check only largest area
        if selector == 0:
            largest_contour_total = []
            # Checking the largest area
            for el in contours_total:
                largest_contour_total.append(self.retMaxArea(el))


            # returning the value of the largest contour
            for i in range(len(largest_contour_total)):
                name = image_total.keys()[i]
                presence[i] = self.retLargestContour(largest_contour_total[i], image_total[name], name)

        else:
            # selector == 1 check all the area
            for i in range(len(contours_total)):
                presence[i] = self.retAllContours(contours_total[i])

        return presence

    def run(self):
        try:
            with picamera.PiCamera() as camera:
                camera.resolution = (self.CAMERA_WIDTH, self.CAMERA_HEIGHT)
                camera.framerate = 30

                # capture into stream
                stream = io.BytesIO()
                for foo in camera.capture_continuous(stream, 'jpeg'):
                    last_time = time.time()

                    data = np.fromstring(stream.getvalue(), dtype=np.uint8)
                    # "Decode" the image from the array, preserving colour
                    image = cv2.imdecode(data, 1)

                    # Convert BGR to HSV
                    image = cv2.GaussianBlur(image, (5, 5), 0)
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    # Resize image
                    hsv = cv2.resize(hsv, (len(image[0]) / self.scale_down, len(image) / self.scale_down))

                    # Calculate value to divide the image into three/four different part
                    valueDivision = math.floor((self.CAMERA_WIDTH / 3) / self.scale_down)
                    valueDivisionVertical = math.floor((self.CAMERA_HEIGHT / 4) / self.scale_down)

                    # Divide image in three pieces
                    sub_image_left = hsv[0:valueDivisionVertical * 3, 0:0 + valueDivision]
                    sub_image_central = hsv[0:valueDivisionVertical * 3,
                                        valueDivision:valueDivision + valueDivision]
                    sub_image_right = hsv[0:valueDivisionVertical * 3,
                                      valueDivision + valueDivision: (self.CAMERA_WIDTH / self.scale_down)]
                    sub_image_bottom = hsv[valueDivisionVertical * 3:]

                    image_total = {"left": sub_image_left, "central": sub_image_central,
                                   "right": sub_image_right, "bottom": sub_image_bottom}

                    # define range of blue color in HSV
                    lower_blue = np.array([80, 60, 50])
                    upper_blue = np.array([120, 255, 255])

                    # define range of red color in HSV
                    # My value
                    red_lower = np.array([120, 80, 0])
                    red_upper = np.array([180, 255, 255])

                    # define range of green color in HSV
                    green_lower = np.array([30, 75, 75])
                    green_upper = np.array([60, 255, 255])

                    # define range of white color in HSV
                    test = 110
                    lower_white = np.array([0, 0, 255 - test])
                    upper_white = np.array([360, test, 255])

                    # define range of black color in HSV
                    black_lower = np.array([0, 0, 0])
                    black_upper = np.array([180, 255, 30])

                    self.presence = self.retContours(red_lower, red_upper, image_total, 0)

                    # black color changed into blu color (thymio doesn't have blu part. Only goal is blu)
                    self.presenceGoal = self.retContours(lower_blue, upper_blue, image_total, 1)

                    # print("presenceRed {}".format(self.presence))
                    # print("presenceBlack {}".format(self.presenceGoal))
                    # print("presenceBlack {}".format(self.presenceGoal))

                    if self.camera:
                        cv2.imshow("ColourTrackerWindow", image)
                    # cv2.imshow("sub_image_left", sub_image_left)
                    # cv2.imshow("sub_image_central", sub_image_central)
                    # cv2.imshow("sub_image_right", sub_image_right)
                    # cv2.imshow("sub_image_bottom", sub_image_bottom)

                    stream.truncate()
                    stream.seek(0)

                    # stop thread
                    if self._stopped():
                        self.__simLogger.debug("Stopping camera thread")
                        break
                    print "time left:", (time.time() - last_time)
                    time.sleep(MIN_FPS - (time.time() - last_time))
            cv2.destroyAllWindows()
        except Exception as e:
            self.__simLogger.critical("Camera exception: " + str(e) + str(
                sys.exc_info()[0]) + ' - ' + traceback.format_exc())


class CameraVisionVectors(CameraVision):
    def __init__(self, camera, logger):
        CameraVision.__init__(self, camera, logger)
        # Divide image in three pieces
        # define range of blue color in HSV
        self.blue_lower = np.array([67, 50, 50])
        self.blue_upper = np.array([200, 255, 255])

        # define range of red color in HSV
        # My value
        self.red_lower = np.array([120, 80, 0])
        self.red_upper = np.array([180, 255, 255])

        # define range of green color in HSV
        self.green_lower = np.array([15, 165, 30])
        self.green_upper = np.array([55, 255, 155])

        # define range of white color in HSV
        test = 110
        self.white_lower = np.array([0, 0, 255 - test])
        self.white_upper = np.array([360, test, 255])

        # define range of black color in HSV
        self.black_lower = np.array([0, 0, 0])
        self.black_upper = np.array([180, 255, 30])

        self.blur = (19, 19)
        self.binary_channels = None
        self.img_ready = False

    def find_shortest(self, binary, check_puck=False):
        # if object is not found
        if not binary.any():
            return -np.inf, 0

        central_index = binary.shape[1] / 2
        if check_puck and np.all(binary[-1, (central_index - 1):(central_index + 1)]):
            return 0, 0

        distances_copy = self.distances.copy()
        distances_copy[binary == 0] = np.finfo(distances_copy.dtype).max

        shortest_dist_index = np.argmin(distances_copy)
        shortest_dist = self.distances.reshape(-1)[shortest_dist_index]
        angle = self.angles.reshape(-1)[shortest_dist_index]

        return shortest_dist, angle

    def get_binary_img(self, check_puck=False):
        if check_puck:
            lower_color = self.green_lower
            upper_color = self.green_upper
        else:
            lower_color = self.blue_lower
            upper_color = self.blue_upper

        return cv2.inRange(self.hsv, lower_color, upper_color)

    def img_to_vector(self, binary, check_puck=False):
        # if not check_puck:
        #    pickle.dump(binary, open('binary.p', 'wb'))
        #    import sys
        #    print 'Exiting...'
        #    sys.exit(0)

        dist, angle = self.find_shortest(binary, check_puck=check_puck)

        # print('Found distance: ' + str(dist) + ' and angle: ' + str(angle))
        return dist, angle

    def goal_reached(self, box_dist, goal_dist, MIN_GOAL_DIST=100):
        return goal_dist <= MIN_GOAL_DIST and box_dist == 0

    def run(self):
        try:
            with picamera.PiCamera() as camera:
                camera.resolution = (self.CAMERA_WIDTH, self.CAMERA_HEIGHT)
                camera.framerate = 30

                time.sleep(2)

                # capture into stream
                stream = io.BytesIO()
                for foo in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                    data = np.fromstring(stream.getvalue(), dtype=np.uint8)
                    # "Decode" the image from the array, preserving colour
                    image = cv2.imdecode(data, 1)

                    # Convert BGR to HSV
                    image = cv2.GaussianBlur(image, self.blur, 0)
                    self.hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    # Resize image
                    self.hsv = cv2.resize(self.hsv, (len(image[0]) / self.scale_down, len(image) / self.scale_down))

                    # cv2.imwrite('hsv.jpg', self.hsv)
                    # sys.exit(0)

                    # print 'Bri: %d' % camera.brightness
                    # print 'Con: %d' % camera.contrast
                    # print 'Sat: %d' % camera.saturation
                    # print 'Sha: %d' % camera.sharpness
                    # print ''

                    self.puck_binary = self.get_binary_img(check_puck=True)
                    self.presence = self.img_to_vector(self.puck_binary, check_puck=True)

                    self.goal_binary = self.get_binary_img()
                    self.presenceGoal = self.img_to_vector(self.goal_binary)

                    self.binary_channels = [self.goal_binary, self.puck_binary]

                    self.img_ready = True

                    stream.truncate()
                    stream.seek(0)

                    # stop thread
                    if self._stopped():
                        print("Stopping camera thread")
                        break
            cv2.destroyAllWindows()
        except Exception as e:
            print("Camera exception: " + str(e) + str(
                sys.exc_info()[0]) + ' - ' + traceback.format_exc())
