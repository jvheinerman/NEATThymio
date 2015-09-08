#!/usr/bin/env python
import json
import logging
import os
import classes
import pickle
import socket
import struct
import threading
import traceback
import cv2
import math
import numpy as np
import pipes
import sys

__author__ = 'vcgorka'
# modified by Alessandro Zonta
# improvement:
# - change hsv color value
# - add four color
# - add size image/resize custom size
# - add show map parameters
# - add socket communication (Massimiliano implementation)

LOG_FORMAT = "%(asctime)-15s:%(levelname)-8s:%(threadName)s:%(filename)s:%(funcName)s: %(message)s"
LOG_LEVEL = logging.INFO
CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(CURRENT_FILE_PATH, 'config.json')
show_map = True


def sendOneMessage(conn, data):
    packed_data = pickle.dumps(data)
    length = len(packed_data)
    conn.sendall(struct.pack('!I', length))
    conn.sendall(packed_data)


# Reads the values contained in the passed configuration file and stores them in this object
class ConfigParser(object):
    def __init__(self, filename):
        json_data = open(filename)
        data = json.load(json_data)
        json_data.close()
        self.__address = data["address"]
        self.__port = data["port"]
        self.__bots = data["bots"]

    @property
    def address(self):
        return self.__address

    @property
    def port(self):
        return self.__port

    @property
    def bots(self):
        return self.__bots


# Sends outgoing messages to the remote host
class MessageSender(threading.Thread):
    def __init__(self, ipAddress, port):
        threading.Thread.__init__(self)
        self.__ipAddress = ipAddress
        self.__port = port
        self.__outbox = list()
        self.__outboxNotEmpty = threading.Condition()
        self.__connectionSocket = None
        self.__isStopped = threading.Event()

    @property
    def ipAddress(self):
        return self.__ipAddress

    def __estabilishConnection(self):
        nAttempt = 0
        if self.__connectionSocket:
            logging.debug('Sender - ' + self.__ipAddress + ' - ALREADY CONNECTED')
            return True
        # Otherwise retry to connect unless stop signal is sent
        while not self.__stopped():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.__ipAddress, self.__port))
                self.__connectionSocket = sock
                logging.debug('Sender - ' + self.__ipAddress + ' - CONNECTED @ attempt' + str(nAttempt))
                return True
            except:
                # Error during connect, new attempt if not stopped
                nAttempt += 1
        return False

    def outboxAppend(self, item):
        with self.__outboxNotEmpty:
            self.__outbox.append(item)
            self.__outboxNotEmpty.notify()

    def __outboxPop(self):
        item = None
        with self.__outboxNotEmpty:
            if not self.__outbox:
                logging.debug('Sender - ' + self.__ipAddress + ' - EMPTY OUTBOX: WAIT')
                self.__outboxNotEmpty.wait()
            if not self.__stopped():
                logging.debug(
                    'Sender - ' + self.__ipAddress + ' - OUTBOX is' + str(self.__outbox) + ' - taking ' + str(
                        self.__outbox[0]))
                item = self.__outbox.pop(0)
        return item

    def run(self):
        try:
            logging.debug('Sender - ' + self.__ipAddress + ' - RUNNING')
            while not self.__stopped():
                item = self.__outboxPop()
                logging.debug('Sender - ' + self.__ipAddress + ' - OUTBOX popped ' + str(item))
                if item and self.__estabilishConnection():
                    # Not stopped and has an item to send and an estabilished connection
                    try:
                        sendOneMessage(self.__connectionSocket, item)
                        logging.debug('Sender - ' + self.__ipAddress + ' - SENT' + str(item))
                    except:
                        # Error while sending: put back item in the outbox
                        with self.__outboxNotEmpty:
                            self.__outbox.insert(0, item)
                        # Current socket is corrupted: closing it
                        self.__connectionSocket.close()
                        self.__connectionSocket = None
                        logging.warning(
                            'Sender - ' + self.__ipAddress + ' - Error while sending - CLOSED socket and restored OUTBOX:' + str(
                                self.__outbox))
            logging.debug('Sender - ' + self.__ipAddress + ' - STOPPED -> EXITING...')
        except:
            logging.critical(
                'Error in Sender ' + self.__ipAddress + ': ' + str(sys.exc_info()[0]) + ' - ' + traceback.format_exc())

    def stop(self):
        self.__isStopped.set()
        with self.__outboxNotEmpty:
            self.__outboxNotEmpty.notify()

    def __stopped(self):
        return self.__isStopped.isSet()


class Tracker:
    def __init__(self):
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
        logging.info("starting application")
        if show_map:
            cv2.namedWindow("Colour Tracker", cv2.CV_WINDOW_AUTOSIZE)
        self.capture = cv2.VideoCapture(0)
        # I need 16:9 to see the goal
        self.capture.set(3, 640)
        self.capture.set(4, 360)
        self.scale_down = 1
        self.x_red = 0
        self.y_red = 0
        self.x_yellow = 0
        self.y_yellow = 0
        self.x_green = 0
        self.y_green = 0
        self.x_blue = 0
        self.y_blue = 0
        # Precedent value vector
        self.precedent_position = [0, 0, 0, 0]
        # radius of circle in track map (BGR)
        self.circle = 5
        # Color of object track object_tracking_box
        self.color_yellow = (0, 255, 255)  # yellow
        self.color_red = (0, 0, 255)  # red
        self.color_green = (0, 255, 0)  # green
        self.color_blue = (255, 0, 0)  # blue
        # threshold area color tracked
        self.threshold_area = 500
        # send message
        config = ConfigParser(CONFIG_PATH)
        self.__msgSenders = dict()
        for bot in config.bots:
            address = bot["address"]
            self.__msgSenders[address] = MessageSender(address, bot["port"])
        # Start message senders
        for addr in self.__msgSenders:
            self.__msgSenders[addr].start()

    def run(self):
        # Counter for frame number
        counter = 0

        # create image_frame for track map, set to input resolution
        image_frame2 = np.zeros((235, 495, 3), np.uint8)

        while True:
            counter += 1
            # Import video frame from capture
            f, video_frame = self.capture.read()

            # preprocess the video frame # resize image cutting useless border
            video_frame = video_frame[50:285, 95:590]
            image_frame = cv2.GaussianBlur(video_frame, (5, 5), 0)
            image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2HSV)
            image_frame = cv2.resize(image_frame,
                                     (len(video_frame[0]) / self.scale_down, len(video_frame) / self.scale_down))

            # define the colors, need to be set before experiment
            # red tracking -> box
            self.red_tracking(image_frame, video_frame, image_frame2, counter)

            # yellow tracking -> robot
            self.yellow_tracking(image_frame, video_frame, image_frame2, counter)

            # green tracking
            self.green_tracking(image_frame, video_frame, image_frame2, counter)

            if cv2.waitKey(20) == 27:
                cv2.destroyWindow("Colour Tracker")
                self.capture.release()

            # black object tracking - search for biggest black object in frame
            # Problems with real time tracking. Fixed position. I know where it is
            x_black = 475
            y_black = 110
            # logging.debug("Black." + str(counter) + ", " + str(x_black) + ", " + str(y_black) + ";")

            # Global External (GE)
            # The distance between the box at its start position and the box at its end position minus half of the
            # distance between the robot and the box at the end position (to see if the robot is still pushing)
            # Bt -> position at time t of the box -> red_tracking (x_red, y_red)
            # Rt -> position at time t of the robot -> yellow_tracking (x_yellow, y_yellow)
            # B0 -> position of the box at its end position -> black fix position (x_black, y_yellow)
            # d() -> distance function
            # d(Bt, B0) - 1/2d(Bt, Rt)
            # first_distance = self.distance(self.x_red, self.y_red, x_black, y_black)
            # logging.debug("first_distance {}".format(first_distance))
            # second_distance = self.distance(self.x_red, self.y_red, self.x_yellow, self.y_yellow)
            # logging.debug("second_distance {}".format(second_distance))
            # global_external = first_distance - 0.5 * second_distance
            # logging.info("global_external {}".format(global_external))

            # Local External (LE)
            # after each step the distance change of the box minus half the change in distance between the robot and
            # the box is calculated as a local result. All local results are summed
            # Bt -> position at time t of the box -> red_tracking (x_red, y_red)
            # Bt-1 -> position at time t-1 of the box -> (self.precedent_position[0], self.precedent_position[1])
            # Rt -> position at time t of the robot -> yellow_tracking (x_yellow, y_yellow)
            # Rt-1 -> position at time t-1 of the robot -> (self.precedent_position[2], self.precedent_position[3])
            # first_distance = self.distance(self.x_red, self.y_red, self.precedent_position[0],
            #                                self.precedent_position[1])
            # second_distance = self.distance(self.precedent_position[0], self.precedent_position[1],
            #                                 self.precedent_position[2], self.precedent_position[3])
            # third_distance = self.distance(self.x_red, self.y_red, self.x_yellow, self.y_yellow)
            # local_external = first_distance + 0.5 * (second_distance - third_distance)
            # logging.info("local_external {}".format(local_external))

            # My external
            # distance from robot and box
            # Bt -> position at time t of the box -> red_tracking (x_red, y_red)
            # Rt -> position at time t of the robot -> yellow_tracking (x_yellow, y_yellow)
            # B0 -> position of the goal -> black fix position (x_black, y_black)
            distance_robot_box = self.distance(self.x_red, self.y_red, self.x_yellow, self.y_yellow)
            distance_robot_goal = self.distance(x_black, y_black, self.x_yellow, self.y_yellow)
            logging.info("distance_robot_box {}".format(distance_robot_box))
            logging.info("distance_robot_goal {}".format(distance_robot_goal))

            # distance from goal
            # red_distance = round(math.sqrt(pow(self.x_red - x_black, 2) + pow(self.y_red - y_black, 2)), 3)
            # yellow_distance = round(math.sqrt(pow(self.x_yellow - x_black, 2) + pow(self.y_yellow - y_black, 2)), 3)
            # green_distance = round(math.sqrt(pow(self.x_green - x_black, 2) + pow(self.y_green - y_black, 2)), 3)
            # blue_distance = round(math.sqrt(pow(self.x_blue - x_black, 2) + pow(self.y_blue - y_black, 2)), 3)
            # logging.info("Red distance from goal is {}".format(red_distance))
            # logging.info("Yellow distance from goal is {}".format(yellow_distance))
            # logging.info("Green distance from goal is {}".format(green_distance))
            # logging.info("Blue distance from goal is {}".format(blue_distance))

            # update precedent position
            self.precedent_position = [self.x_red, self.y_red, self.x_yellow, self.y_yellow]

            # Send messages: tell MessageSenders to do that
            # Decide what i need to send.
            # Position of what?
            logging.info("Sending messages...")
            for addr in self.__msgSenders:
                messageDist = classes.FitnessDataMessage(distance_robot_box, distance_robot_goal)
                logging.info("Broadcast message global external fitness function = " + str(messageDist.distance_robot_box) + " " + str(messageDist.distance_robot_goal))
                self.__msgSenders[addr].outboxAppend(messageDist)

            if show_map:
                cv2.imshow("Colour Tracker", video_frame)
            sys.stdout.flush()

    # Distance between two point (x_position_one, y_position_two) (x_position_two, y_position_two)
    def distance(self, x_position_one, y_position_one, x_position_two, y_position_two):
        # result xxx,xxx -> only three number after comma
        return round(math.sqrt(pow(x_position_one - x_position_two, 2) + pow(y_position_one - y_position_two, 2)), 3)

    # red -> box tracking
    def red_tracking(self, image_frame, video_frame, image_frame2, counter):
        # define the colors, need to be set before experiment
        # red_lower = np.array([0, 150, 100], np.uint8)
        # red_upper = np.array([5, 255, 255], np.uint8)
        red_lower = np.array([120, 80, 0], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_binary = cv2.inRange(image_frame, red_lower, red_upper)
        red_binary = cv2.dilate(red_binary, np.ones((15, 15), "uint8"))
        contours_red_objects, hierarchy_red = cv2.findContours(red_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        max_area_red = 0
        largest_contour_red = None

        for idx, contour in enumerate(contours_red_objects):
            area_red = cv2.contourArea(contour)
            if area_red > max_area_red:
                max_area_red = area_red
                largest_contour_red = contour

        if not largest_contour_red is None:
            moment = cv2.moments(largest_contour_red)
            if moment["m00"] > self.threshold_area / self.scale_down:
                object_tracking_rect = cv2.minAreaRect(largest_contour_red)
                object_tracking_box = np.int0(cv2.cv.BoxPoints(((object_tracking_rect[0][0] * self.scale_down,
                                                                 object_tracking_rect[0][1] * self.scale_down), (
                                                                    object_tracking_rect[1][0] * self.scale_down,
                                                                    object_tracking_rect[1][1] * self.scale_down),
                                                                object_tracking_rect[2])))
                cv2.drawContours(video_frame, [object_tracking_box], 0, self.color_red, 2)

                # calculate Coordinates
                self.x_red = ((object_tracking_box[0][0] + object_tracking_box[1][0] + object_tracking_box[2][0] +
                               object_tracking_box[3][0]) / 4)
                self.y_red = ((object_tracking_box[0][1] + object_tracking_box[1][1] + object_tracking_box[2][1] +
                               object_tracking_box[3][1]) / 4)

                # display coordinates in map
                if show_map:
                    cv2.circle(image_frame2, (self.x_red, self.y_red), self.circle, self.color_red, thickness=-1,
                               lineType=8, shift=0)
                    cv2.imshow("map", image_frame2)

                # print objectnr + frame number + x coordinate + y coordinate
                logging.debug("1." + str(counter) + ", " + str(self.x_red) + ", " + str(self.y_red) + ";")

                # chek the mask color -> only for debug
                # mask = cv2.inRange(image_frame, red_lower, red_upper)
                # cv2.imshow('maskRed', mask)

    # yellow -> robot tracking
    def yellow_tracking(self, image_frame, video_frame, image_frame2, counter):
        # yellow object tracking - search for biggest yellow object in frame
        # define the colors, need to be set before experiment
        yellow_lower = np.array([25, 100, 100], np.uint8)
        yellow_upper = np.array([35, 255, 255], np.uint8)
        yellow_binary = cv2.inRange(image_frame, yellow_lower, yellow_upper)
        yellow_binary = cv2.dilate(yellow_binary, np.ones((15, 15), "uint8"))
        contours_yellow_objects, hierarchy_blue = cv2.findContours(yellow_binary, cv2.RETR_LIST,
                                                                   cv2.CHAIN_APPROX_SIMPLE)

        max_yellow_area_object = 0
        largest_contour_yellow = None

        for idx, contour in enumerate(contours_yellow_objects):
            yellow_area_object = cv2.contourArea(contour)
            if yellow_area_object > max_yellow_area_object:
                max_yellow_area_object = yellow_area_object
                largest_contour_yellow = contour

        if not largest_contour_yellow is None:
            moment = cv2.moments(largest_contour_yellow)
            if moment["m00"] > self.threshold_area / self.scale_down:
                object_tracking_rect = cv2.minAreaRect(largest_contour_yellow)
                object_tracking_box = np.int0(cv2.cv.BoxPoints(((object_tracking_rect[0][0] * self.scale_down,
                                                                 object_tracking_rect[0][1] * self.scale_down), (
                                                                    object_tracking_rect[1][0] * self.scale_down,
                                                                    object_tracking_rect[1][1] * self.scale_down),
                                                                object_tracking_rect[2])))
                cv2.drawContours(video_frame, [object_tracking_box], 0, self.color_yellow, 2)

                # calculate Coordinates
                self.x_yellow = (
                    (object_tracking_box[0][0] + object_tracking_box[1][0] + object_tracking_box[2][0] +
                     object_tracking_box[3][0]) / 4)
                self.y_yellow = (
                    (object_tracking_box[0][1] + object_tracking_box[1][1] + object_tracking_box[2][1] +
                     object_tracking_box[3][1]) / 4)

                # display coordinates in map
                if show_map:
                    cv2.circle(image_frame2, (self.x_yellow, self.y_yellow), self.circle, self.color_yellow,
                               thickness=-1,
                               lineType=8,
                               shift=0)
                    cv2.imshow("map", image_frame2)

                # print objectnr + frame number + x coordinate + y coordinate
                logging.debug("2." + str(counter) + ", " + str(self.x_yellow) + ", " + str(self.y_yellow) + ";")

                # chek the mask color -> only for debug
                # mask = cv2.inRange(image_frame, yellow_lower, yellow_upper)
                # cv2.imshow('maskYellow', mask)

    # green
    def green_tracking(self, image_frame, video_frame, image_frame2, counter):
        # green object tracking - search for biggest green object in frame

        # define the colors, need to be set before experiment
        # [40,100,100]
        # [80, 150, 150]
        # these value fit with green box, but not the other green
        green_lower = np.array([45, 40, 110], np.uint8)
        green_upper = np.array([90, 90, 160], np.uint8)
        green_binary = cv2.inRange(image_frame, green_lower, green_upper)
        green_binary = cv2.dilate(green_binary, np.ones((15, 15), "uint8"))
        contours_green_objects, hierarchy_blue = cv2.findContours(green_binary, cv2.RETR_LIST,
                                                                  cv2.CHAIN_APPROX_SIMPLE)

        max_green_area_object = 0
        largest_contour_green = None

        for idx, contour in enumerate(contours_green_objects):
            green_area_object = cv2.contourArea(contour)
            if green_area_object > max_green_area_object:
                max_green_area_object = green_area_object
                largest_contour_green = contour

        if largest_contour_green is not None:
            moment = cv2.moments(largest_contour_green)
            if moment["m00"] > self.threshold_area / self.scale_down:
                object_tracking_rect = cv2.minAreaRect(largest_contour_green)
                object_tracking_box = np.int0(cv2.cv.BoxPoints(((object_tracking_rect[0][0] * self.scale_down,
                                                                 object_tracking_rect[0][1] * self.scale_down), (
                                                                    object_tracking_rect[1][0] * self.scale_down,
                                                                    object_tracking_rect[1][1] * self.scale_down),
                                                                object_tracking_rect[2])))
                cv2.drawContours(video_frame, [object_tracking_box], 0, self.color_green, 2)

                # calculate Coordinates
                self.x_green = (
                    (object_tracking_box[0][0] + object_tracking_box[1][0] + object_tracking_box[2][0] +
                     object_tracking_box[3][0]) / 4)
                self.y_green = (
                    (object_tracking_box[0][1] + object_tracking_box[1][1] + object_tracking_box[2][1] +
                     object_tracking_box[3][1]) / 4)

                # display coordinates in map
                if show_map:
                    cv2.circle(image_frame2, (self.x_green, self.y_green), self.circle, self.color_green, thickness=-1,
                               lineType=8,
                               shift=0)
                    cv2.imshow("map", image_frame2)

                # print objectnr + frame number + x coordinate + y coordinate
                logging.debug("3." + str(counter) + ", " + str(self.x_green) + ", " + str(self.y_green) + ";")

                # chek the mask color -> only for debug
                # mask = cv2.inRange(image_frame, green_lower, green_upper)
                # cv2.imshow('maskGreen', mask)

    # blue
    def blue_tracking(self, image_frame, video_frame, image_frame2, counter):
        # blue object tracking - search for biggest blue object in frame

        # define the colors, need to be set before experiment
        blue_lower = np.array([80, 20, 30], np.uint8)
        blue_upper = np.array([120, 255, 255], np.uint8)
        blue_binary = cv2.inRange(image_frame, blue_lower, blue_upper)
        blue_binary = cv2.dilate(blue_binary, np.ones((15, 15), "uint8"))
        contours_blue_objects, hierarchy_blue = cv2.findContours(blue_binary, cv2.RETR_LIST,
                                                                 cv2.CHAIN_APPROX_SIMPLE)

        max_blue_area_object = 0
        largest_contour_blue = None

        for idx, contour in enumerate(contours_blue_objects):
            blue_area_object = cv2.contourArea(contour)
            if blue_area_object > max_blue_area_object:
                max_blue_area_object = blue_area_object
                largest_contour_blue = contour

        if not largest_contour_blue is None:
            moment = cv2.moments(largest_contour_blue)
            if moment["m00"] > self.threshold_area / self.scale_down:
                object_tracking_rect = cv2.minAreaRect(largest_contour_blue)
                object_tracking_box = np.int0(cv2.cv.BoxPoints(((object_tracking_rect[0][0] * self.scale_down,
                                                                 object_tracking_rect[0][1] * self.scale_down), (
                                                                    object_tracking_rect[1][0] * self.scale_down,
                                                                    object_tracking_rect[1][1] * self.scale_down),
                                                                object_tracking_rect[2])))
                cv2.drawContours(video_frame, [object_tracking_box], 0, self.color_blue, 2)

                # calculate Coordinates
                self.x_blue = (
                    (object_tracking_box[0][0] + object_tracking_box[1][0] + object_tracking_box[2][0] +
                     object_tracking_box[3][0]) / 4)
                self.y_blue = (
                    (object_tracking_box[0][1] + object_tracking_box[1][1] + object_tracking_box[2][1] +
                     object_tracking_box[3][1]) / 4)

                # display coordinates in map
                if show_map:
                    cv2.circle(image_frame2, (self.x_blue, self.y_blue), self.circle, self.color_blue, thickness=-1,
                               lineType=8,
                               shift=0)
                    cv2.imshow("map", image_frame2)

                # print objectnr + frame number + x coordinate + y coordinate
                logging.debug("4." + str(counter) + ", " + str(self.x_blue) + ", " + str(self.y_blue) + ";")

                # chek the mask color -> only for debug
                # mask = cv2.inRange(image_frame, blue_lower, blue_upper)
                # cv2.imshow('maskBlue', mask)

    # black
    def black_tracking(self, image_frame, video_frame, image_frame2, counter):
        # black object tracking - search for biggest black object in frame

        # define the colors, need to be set before experiment
        blue_lower = np.array([0, 0, 0], np.uint8)
        blue_upper = np.array([180, 255, 50], np.uint8)
        blue_binary = cv2.inRange(image_frame, blue_lower, blue_upper)
        blue_binary = cv2.dilate(blue_binary, np.ones((15, 15), "uint8"))
        contours_blue_objects, hierarchy_blue = cv2.findContours(blue_binary, cv2.RETR_LIST,
                                                                 cv2.CHAIN_APPROX_SIMPLE)

        max_blue_area_object = 0
        largest_contour_blue = None

        for idx, contour in enumerate(contours_blue_objects):
            blue_area_object = cv2.contourArea(contour)
            if blue_area_object > max_blue_area_object:
                max_blue_area_object = blue_area_object
                largest_contour_blue = contour

        if not largest_contour_blue is None:
            moment = cv2.moments(largest_contour_blue)
            if moment["m00"] > self.threshold_area / self.scale_down:
                object_tracking_rect = cv2.minAreaRect(largest_contour_blue)
                object_tracking_box = np.int0(cv2.cv.BoxPoints(((object_tracking_rect[0][0] * self.scale_down,
                                                                 object_tracking_rect[0][1] * self.scale_down), (
                                                                    object_tracking_rect[1][0] * self.scale_down,
                                                                    object_tracking_rect[1][1] * self.scale_down),
                                                                object_tracking_rect[2])))
                cv2.drawContours(video_frame, [object_tracking_box], 0, self.color_red, 2)

                # calculate Coordinates
                self.x_blue = (
                    (object_tracking_box[0][0] + object_tracking_box[1][0] + object_tracking_box[2][0] +
                     object_tracking_box[3][0]) / 4)
                self.y_blue = (
                    (object_tracking_box[0][1] + object_tracking_box[1][1] + object_tracking_box[2][1] +
                     object_tracking_box[3][1]) / 4)

                # print objectnr + frame number + x coordinate + y coordinate
                logging.debug("BLACK" + str(counter) + ", " + str(self.x_blue) + ", " + str(self.y_blue) + ";")

                if cv2.waitKey(20) == 27:
                    cv2.destroyWindow("Colour Tracker")
                    self.capture.release()
                    return False

        # chek the mask color -> only for debug
        # mask = cv2.inRange(image_frame, blue_lower, blue_upper)
        # cv2.imshow('maskBlue', mask)
        return True


if __name__ == "__main__":
    tracker = Tracker()
    tracker.run()
