import math

__author__ = 'alessandro'

import io
import logging
import picamera
import time
import cv2
import numpy as np
# from ThymioFunctions import *


# to speed things up, lower the resolution of the camera
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
scale_down = 1

LOG_FORMAT = "%(asctime)-15s:%(levelname)-8s:%(threadName)s:%(filename)s:%(funcName)s: %(message)s"
LOG_LEVEL = logging.DEBUG
targetCoordinates = [0, 0, 0]


def __init__():
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    logging.info("starting application")


def retMaxArea(contours):
    max_area = 0
    largest_contour = None
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    return largest_contour


def retLargestContour(contour, image2):
    if not contour is None:
        moment = cv2.moments(contour)
        # m00 is the area
        if moment["m00"] > 1000 / scale_down:

            rect_bl = cv2.minAreaRect(contour)
            rect_bl = ((rect_bl[0][0] * scale_down, rect_bl[0][1] * scale_down),
                       (rect_bl[1][0] * scale_down, rect_bl[1][1] * scale_down), rect_bl[2])
            box_bl = cv2.cv.BoxPoints(rect_bl)
            box_bl = np.int0(box_bl)
            cv2.drawContours(image2, [box_bl], 0, (255, 255, 0), 2)
            cv2.imshow("image2", image2)


if __name__ == '__main__':
    __init__()
    with picamera.PiCamera() as camera:
        camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)
        camera.framerate = 30
        camera.awb_mode = 'flash'

        # presence of something
        maximun = 0
        presence = 0
        stop = False

        # capture into stream
        stream = io.BytesIO()
        for foo in camera.capture_continuous(stream, 'jpeg'):

            data = np.fromstring(stream.getvalue(), dtype=np.uint8)
            # "Decode" the image from the array, preserving colour
            image = cv2.imdecode(data, 1)

            # Convert BGR to HSV
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image2 = cv2.GaussianBlur(image, (5, 5), 0)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = cv2.resize(hsv, (len(image[0]) / scale_down, len(image) / scale_down))

            # Calculate value to divide the image into three different part
            # Calculate value to divide the image into three different part
            valueDivision = math.floor((CAMERA_WIDTH / 3) / scale_down)
            valueDivisionVertical = math.floor((CAMERA_HEIGHT / 4) / scale_down)
            # Divide image in three pieces
            sub_image_left = hsv[0:valueDivisionVertical * 3, 0:0 + valueDivision]
            sub_image_central = hsv[0:valueDivisionVertical * 3, valueDivision:valueDivision + valueDivision]
            sub_image_right = hsv[0:valueDivisionVertical * 3,
                              valueDivision + valueDivision:(CAMERA_WIDTH / scale_down)]
            sub_image_bottom = hsv[valueDivisionVertical * 3:]

            # define range of blue color in HSV
            lower_blue = np.array([80, 60, 50])
            upper_blue = np.array([120, 255, 255])

            # define range of red color in HSV
            # My value
            red_lower = np.array([120, 130, 0])
            red_upper = np.array([180, 255, 255])

            # define range of green color in HSV
            green_lower = np.array([30, 75, 75])
            green_upper = np.array([60, 255, 255])

            # define range of white color in HSV
            test = 110
            lower_white = np.array([0, 0, 255 - test])
            upper_white = np.array([360, test, 255])

            # define range of black color in HSV
            # red_lower = np.array([0, 0, 0])
            # red_upper = np.array([180, 255, 50])

            # green_binary = cv2.inRange(hsv, green_lower, green_upper)
            # red_binary = cv2.inRange(hsv, lower_blue, upper_blue)
            red_binary_left = cv2.inRange(sub_image_left, red_lower, red_upper)
            red_binary_central = cv2.inRange(sub_image_central, red_lower, red_upper)
            red_binary_right = cv2.inRange(sub_image_right, red_lower, red_upper)

            # green_binary = cv2.inRange(hsv, green_lower, green_upper)
            # black_binary = cv2.inRange(hsv, black_lower, black_upper)
            # white_binary = cv2.inRange(hsv, lower_white, upper_white)
            # blue_binary = cv2.inRange(hsv2, lower_blue, upper_blue)
            dilation = np.ones((15, 15), "uint8")
            # color_binary1 = cv2.dilate(blue_binary, dilation)
            # color_binary_black = cv2.dilate(black_binary, dilation)
            # color_binary = cv2.dilate(red_binary, dilation)
            color_binary_left = cv2.dilate(red_binary_left, dilation)
            color_binary_central = cv2.dilate(red_binary_central, dilation)
            color_binary_right = cv2.dilate(red_binary_right, dilation)
            color_binary_left = cv2.GaussianBlur(red_binary_left, (5, 5), 0)
            color_binary_central = cv2.GaussianBlur(red_binary_central, (5, 5), 0)
            color_binary_right = cv2.GaussianBlur(red_binary_right, (5, 5), 0)
            # color_binary = cv2.GaussianBlur(color_binary, (5, 5), 0)
            # color_binary1 = cv2.GaussianBlur(color_binary1, (5, 5), 0)
            # color_binary_black = cv2.GaussianBlur(color_binary_black, (5, 5), 0)

            # find contours
            # contours, hierarchy = cv2.findContours(color_binary_two, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contours_black, hierarchy_black = cv2.findContours(color_binary_black, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_left, hierarchy = cv2.findContours(red_binary_left, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # contours1, hierarchy_left = cv2.findContours(color_binary1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_central, hierarchy_central = cv2.findContours(color_binary_central, cv2.RETR_LIST,
                                                                   cv2.CHAIN_APPROX_SIMPLE)
            contours_right, hierarchy_right = cv2.findContours(color_binary_right, cv2.RETR_LIST,
                                                               cv2.CHAIN_APPROX_SIMPLE)

            # largest_contour_total = [retMaxArea(contours_left), retMaxArea(contours_central),
            #                          retMaxArea(contours_right)]
            #
            # largest_contour_total = [retMaxArea(contours), retMaxArea(contours1)]

            centres = []

            # targetCoordinates[0] = -1
            # targetCoordinates[1] = -1
            # targetCoordinates[2] = 0

            # Now I use only the big one
            presence = 0
            for idx, contour in enumerate(contours_left):
                centres = []
                moment = cv2.moments(contour)
                if moment["m00"] > 400 / scale_down:
                    presence += moment["m00"]  # area
            logging.debug("presence left {}".format(presence))
            if presence > maximun:
                maximun = presence
                logging.debug("maximun left -----> {}".format(maximun))

            presence = 0
            for idx, contour in enumerate(contours_central):
                centres = []
                moment = cv2.moments(contour)
                if moment["m00"] > 400 / scale_down:
                    presence += moment["m00"]  # area
            logging.debug("presence central {}".format(presence))
            if presence > maximun:
                maximun = presence
                logging.debug("maximun central -----> {}".format(maximun))

            presence = 0
            for idx, contour in enumerate(contours_right):
                centres = []
                moment = cv2.moments(contour)
                if moment["m00"] > 400 / scale_down:
                    presence += moment["m00"]  # area
            logging.debug("presence right {}".format(presence))
            if presence > maximun:
                maximun = presence
                logging.debug("maximun right -----> {}".format(maximun))

            # value = 1  # red 0 #  blue 1
            # value1 = 0
            # if not largest_contour_total[value] is None:
            #     moment = cv2.moments(largest_contour_total[value])
            #     # m00 is the area
            #     if moment["m00"] > 400 / scale_down:
            #         presence = moment["m00"]  # area
            #
            #         # calculate centroid
            #         x = (int(moment['m10'] / moment['m00'])) * scale_down
            #         y = (int(moment['m01'] / moment['m00'])) * scale_down
            #
            #         # targetCoordinates[0] = x
            #         # targetCoordinates[1] = y
            #         # targetCoordinates[2] = presence  # area
            #         centres.append((x, y))
            #
            #         # show centroid into image
            #         cv2.circle(image, centres[0], 5, (255, 0, 0), 3)
            #         # logging.info("centre {}".format(centres))
            #
            #         rect = cv2.minAreaRect(largest_contour_total[value])
            #         rect = ((rect[0][0] * scale_down, rect[0][1] * scale_down),
            #                 (rect[1][0] * scale_down, rect[1][1] * scale_down), rect[2])
            #         box = cv2.cv.BoxPoints(rect)
            #         box = np.int0(box)
            #         cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
            # else:
            #     presence = 0
            #     # targetCoordinates[0] = -1
            #     # targetCoordinates[1] = -1
            #     # targetCoordinates[2] = 0
            #
            # # retLargestContour(largest_contour_total[value1], image2)
            #
            # logging.debug("presence {}".format(presence))
            # if presence > maximun:
            #     maximun = presence
            #     logging.debug("maximun -----> {}".format(maximun))
            # logging.debug("coordinate {}".format(targetCoordinates[0]))
            # logging.debug("coordinate {}".format(targetCoordinates[1]))
            # logging.debug("coordinate {}".format(targetCoordinates[2]))
            # logging.debug("temperature {}".format(getTemperatureValue()))
            # with open("Output2.txt", "a") as text_file:
            #     text_file.write("Temp: {}".format(getTemperatureValue()))
            #     text_file.write("\n")
            # logging.debug("left = {}".format(retLargestContour(largest_contour_total[1])))
            # logging.debug("central = {}".format(retLargestContour(largest_contour_total[2])))
            # logging.debug("right = {}".format(retLargestContour(largest_contour_total[3])))

            # Threshold the HSV image to get only blue colors
            mask1 = cv2.inRange(sub_image_left, lower_blue, upper_blue)
            mask2 = cv2.inRange(sub_image_central, lower_blue, upper_blue)
            mask3 = cv2.inRange(sub_image_right, lower_blue, upper_blue)
            mask4 = cv2.inRange(sub_image_bottom, lower_blue, upper_blue)
            mask5 = cv2.inRange(sub_image_left, red_lower, red_upper)
            mask6 = cv2.inRange(sub_image_central, red_lower, red_upper)
            mask7 = cv2.inRange(sub_image_right, red_lower, red_upper)
            mask8 = cv2.inRange(sub_image_bottom, red_lower, red_upper)


#            mask = cv2.inRange(hsv2, red_lower, red_upper)
            # # mask = cv2.inRange(hsv2, green_lower, green_upper)
            # # mask = cv2.inRange(hsv2, lower_white, upper_white)
            # # mask = cv2.inRange(hsv2, black_lower, black_upper)
            # # mask = mask1 + mask2
            #
            # # Bitwise-AND mask and original image
            # res = cv2.bitwise_and(image, image, mask=mask)

            print "Showing stream"
            cv2.imshow('frame', image)
            cv2.imshow('left blue', mask1)
            cv2.imshow('central blue', mask2)
            cv2.imshow("right blue", mask3)
            cv2.imshow("left red", mask5)
            cv2.imshow("central red", mask6)
            cv2.imshow("right red", mask7)
            cv2.imshow("bottom red", mask8)
            stream.truncate()
            stream.seek(0)

            # actualLeftSpeed, actualRightSpeed = getMotorSpeed()
            # print "Actual speed: ", actualLeftSpeed, ",", actualRightSpeed
            #
            # detected = False
            # if targetCoordinates[0] != -1:
            #     detected = True
            #
            # if not detected:
            #     print "do not move"
            #     setMotorSpeed(0, 0)
            #     stop = True
            # else:
            #     if targetCoordinates[0] > 212:
            #         print "Moving forward right"
            #         setMotorSpeed(20, 10)
            #         stop = False
            #     elif targetCoordinates[0] < 106:
            #         print "Moving forward left"
            #         setMotorSpeed(10, 20)
            #         stop = False
            #     else:
            #         print "Moving forward"
            #         setMotorSpeed(100, 100)
            #         stop = False
            #
            # # only front sensors
            # for i in xrange(5):
            #     proxSensors[i] = getSensorValue(i)
            #     # logging.debug("proxSensor {}".format(proxSensors[i]))
            #
            # for el in proxSensors:
            #     if el > 2000 and stop == True:
            #         print("sound")
            #         setSound(4)
            #         time.sleep(1)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

    cv2.destroyAllWindows()
