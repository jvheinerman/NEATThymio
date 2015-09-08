# algorithm.py created on March 12, 2015. Jacqueline Heinerman & Massimiliano Rango
# modified by Alessandro Zonta on June 25, 2015
import copy
import parameters as pr
import classes as cl
import dbus, dbus.mainloop.glib
import glib, gobject
import sys, os, errno
import random, math, time
import logging, traceback, json
import threading, socket, select, struct, pickle
import numpy as np
import cv2
import io
import picamera

RAND_MAX = sys.maxint
LEFT = 0
RIGHT = 1

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(CURRENT_FILE_PATH, 'config.json')
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')
MAIN_LOG_PATH = os.path.join(CURRENT_FILE_PATH, 'log_main')
OUTPUT_PATH = os.path.join(CURRENT_FILE_PATH, 'output')
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

LOCALHOST = '127.0.0.1'
PC_FIXED_IP = '192.168.1.100'
TRUSTED_CLIENTS = [LOCALHOST, PC_FIXED_IP]
COMMANDS_LISTENER_HOST = ''
COMMANDS_LISTENER_PORT = 54321
OUTPUT_FILE_RECEIVER_PORT = 23456  # 23456  # 24537
EOF_REACHED = 'EOF_REACHED'

# Uniform distribution (0..1]
def drand():
    return random.randint(0, RAND_MAX) / float(RAND_MAX + 1)


# Normal distribution, centered on 0, std dev 1
def random_normal():
    return -2 * math.log(drand())


# Used because else webots gives a strange segfault during cross compilation
def sqrt_rand_normal():
    return math.sqrt(random_normal())


def gaussrand():
    return sqrt_rand_normal() * math.cos(2 * math.pi * drand())


def recvall(conn, count):
    buf = b''
    while count:
        newbuf = conn.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def recvOneMessage(socket):
    lengthbuf = recvall(socket, 4)
    length, = struct.unpack('!I', lengthbuf)
    data = pickle.loads(recvall(socket, length))
    return data


def sendOneMessage(conn, data):
    packed_data = pickle.dumps(data)
    length = len(packed_data)
    conn.sendall(struct.pack('!I', length))
    conn.sendall(packed_data)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def getNextIDPath(path):
    nextID = 0
    filelist = sorted(os.listdir(path))
    if filelist and filelist[-1][0].isdigit():
        nextID = int(filelist[-1][0]) + 1
    return str(nextID)


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


# Represents a shared inbox object
class Inbox(object):
    def __init__(self, simulationLogger):
        self.__inbox = list()
        self.__inboxLock = threading.Lock()
        self.__simLogger = simulationLogger

    def append(self, data):
        with self.__inboxLock:
            self.__inbox.append(data)

    def popAll(self):
        itemsList = list()
        with self.__inboxLock:
            for i in self.__inbox:
                item = self.__inbox.pop(0)
                # self.__simLogger.debug("popAll - message fitness = " + str(item.fitness))
                itemsList.append(item)
        # self.__simLogger.debug("popAll - Popped " + str(itemsList))
        return itemsList

    def popExternalFitness(self):
        itemsList = list()
        itemposition = list()
        with self.__inboxLock:
            for mex in self.__inbox:
                if type(mex) is cl.FitnessDataMessage:
                    itemsList.append(mex)
            for element in itemsList:
                self.__inbox.remove(element)

                # self.__simLogger.debug("popAll - message fitness = " + str(item.fitness))
        # self.__simLogger.debug("popAll - Popped " + str(itemsList))
        return itemsList


# Listens to incoming connections from other agents and delivers them to the corresponding thread
class ConnectionsListener(threading.Thread):
    def __init__(self, address, port, msgReceivers, simulationLogger):
        threading.Thread.__init__(self)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((address, port))
        sock.listen(5)
        self.__address = address
        self.__port = port
        self.__socket = sock
        self.__msgReceivers = msgReceivers
        self.__simLogger = simulationLogger
        self.__isStopped = threading.Event()

    def run(self):
        try:
            self.__simLogger.debug('ConnectionsListener - RUNNING')

            nStopSockets = 0
            iterator = self.__msgReceivers.itervalues()
            while nStopSockets < len(self.__msgReceivers):
                conn, (addr, port) = self.__socket.accept()
                if addr == LOCALHOST:
                    iterator.next().setStopSocket(conn)
                    nStopSockets += 1

            while not self.__stopped():
                self.__simLogger.debug('ConnectionsListener - Waiting for accept')
                conn, (addr, port) = self.__socket.accept()
                if not self.__stopped():
                    try:
                        self.__simLogger.debug(
                            'ConnectionsListener - Received request from ' + addr + ' - FORWARDING to Receiver')
                        self.__msgReceivers[addr].setConnectionSocket(conn)
                    except:
                        # Received connection from unknown IP
                        self.__simLogger.warning(
                            "Exception: " + str(sys.exc_info()[0]) + ' - ' + traceback.format_exc())

            self.__socket.close()
            self.__simLogger.debug('ConnectionsListener STOPPED -> EXITING...')
        except:
            self.__simLogger.critical(
                'Error in ConnectionsListener: ' + str(sys.exc_info()[0]) + ' - ' + traceback.format_exc())

    def stop(self):
        self.__isStopped.set()
        # If blocked on accept() wakes it up with a fake connection
        fake = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        fake.connect((LOCALHOST, self.__port))
        self.__simLogger.debug('ConnectionsListener - Fake connection')
        fake.close()

    def __stopped(self):
        return self.__isStopped.isSet()


# Waits for incoming messages on one socket and stores them in the shared inbox
class MessageReceiver(threading.Thread):
    def __init__(self, ipAddress, inbox, simulationLogger):
        threading.Thread.__init__(self)
        self.__ipAddress = ipAddress
        self.__inbox = inbox
        self.__connectionSocket = None
        self.__stopSocket = None
        self.__isSocketAlive = threading.Condition()
        self.__isStopped = threading.Event()
        self.__simLogger = simulationLogger

    @property
    def ipAddress(self):
        return self.__ipAddress

    def setConnectionSocket(self, newSock):
        with self.__isSocketAlive:
            if not self.__connectionSocket and newSock:
                self.__connectionSocket = newSock
                self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - CONNECTED!!!')
                self.__isSocketAlive.notify()

    def setStopSocket(self, stopSock):
        self.__stopSocket = stopSock

    def run(self):
        try:
            self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - RUNNING')

            while not self.__stopped():

                # Waits while the connection is not set
                with self.__isSocketAlive:
                    if not self.__connectionSocket and not self.__stopped():
                        self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - Not connected: WAIT')
                        self.__isSocketAlive.wait()

                if not self.__stopped():
                    self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - Waiting on select')
                    readable, _, _ = select.select([self.__connectionSocket, self.__stopSocket], [], [])
                    if self.__stopSocket in readable:
                        # 	Received a message (stop) from localhost
                        self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - StopSocket is in readable')
                        data = recvOneMessage(self.__stopSocket)
                        self.__simLogger.debug('Received ' + data)
                    elif self.__connectionSocket in readable:
                        self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - ConnectionSocket is in readable')
                        # 	Received a message from remote host
                        try:
                            data = recvOneMessage(self.__connectionSocket)
                            self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - Received ' + str(data))
                            if data and not self.__stopped():
                                # self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - Appending ' + str(data))
                                self.__inbox.append(data)
                                self.__simLogger.debug(
                                    'Receiver - ' + self.__ipAddress + ' - Appended ' + str(data) + ' to inbox.')
                        except:
                            # Error while receiving: current socket is corrupted -> closing it
                            self.__simLogger.warning(
                                'Receiver - ' + self.__ipAddress + ' - Error while receiving - CLOSING socket!' + str(
                                    sys.exc_info()[0]) + ' - ' + traceback.format_exc())
                            self.__connectionSocket.close()
                            self.__connectionSocket = None
        except:
            self.__simLogger.critical('Error in Receiver ' + self.__ipAddress + ': ' + str(
                sys.exc_info()[0]) + ' - ' + traceback.format_exc())

        self.__simLogger.debug('Receiver - ' + self.__ipAddress + ' - STOPPED -> EXITING...')

    def stop(self):
        self.__isStopped.set()
        with self.__isSocketAlive:
            self.__isSocketAlive.notify()

    def __stopped(self):
        return self.__isStopped.isSet()


# Sends outgoing messages to the remote host
class MessageSender(threading.Thread):
    def __init__(self, ipAddress, port, simulationLogger):
        threading.Thread.__init__(self)
        self.__ipAddress = ipAddress
        self.__port = port
        self.__outbox = list()
        self.__outboxNotEmpty = threading.Condition()
        self.__connectionSocket = None
        self.__isStopped = threading.Event()
        self.__simLogger = simulationLogger

    @property
    def ipAddress(self):
        return self.__ipAddress

    def __estabilishConnection(self):
        nAttempt = 0
        if self.__connectionSocket:
            self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - ALREADY CONNECTED')
            return True
        # Otherwise retry to connect unless stop signal is sent
        while not self.__stopped():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.__ipAddress, self.__port))
                self.__connectionSocket = sock
                self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - CONNECTED @ attempt' + str(nAttempt))
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
                self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - EMPTY OUTBOX: WAIT')
                self.__outboxNotEmpty.wait()
            if not self.__stopped():
                self.__simLogger.debug(
                    'Sender - ' + self.__ipAddress + ' - OUTBOX is' + str(self.__outbox) + ' - taking ' + str(
                        self.__outbox[0]))
                item = self.__outbox.pop(0)
        return item

    def run(self):
        try:
            self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - RUNNING')
            while not self.__stopped():
                item = self.__outboxPop()
                self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - OUTBOX popped ' + str(item))
                if item and self.__estabilishConnection():
                    # Not stopped and has an item to send and an estabilished connection
                    try:
                        sendOneMessage(self.__connectionSocket, item)
                        self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - SENT' + str(item))
                    except:
                        # Error while sending: put back item in the outbox
                        with self.__outboxNotEmpty:
                            self.__outbox.insert(0, item)
                        # Current socket is corrupted: closing it
                        self.__connectionSocket.close()
                        self.__connectionSocket = None
                        self.__simLogger.warning(
                            'Sender - ' + self.__ipAddress + ' - Error while sending - CLOSED socket and restored OUTBOX:' + str(
                                self.__outbox))
            self.__simLogger.debug('Sender - ' + self.__ipAddress + ' - STOPPED -> EXITING...')
        except:
            self.__simLogger.critical(
                'Error in Sender ' + self.__ipAddress + ': ' + str(sys.exc_info()[0]) + ' - ' + traceback.format_exc())

    def stop(self):
        self.__isStopped.set()
        with self.__outboxNotEmpty:
            self.__outboxNotEmpty.notify()

    def __stopped(self):
        return self.__isStopped.isSet()


# Recognize color using the camera
class cameraVision(threading.Thread):
    def __init__(self, camera, simulationLogger):
        threading.Thread.__init__(self)
        self.CAMERA_WIDTH = 320
        self.CAMERA_HEIGHT = 240
        self.scale_down = 1
        self.presence = [0, 0, 0, 0]
        self.presenceGoal = [0, 0, 0, 0]
        self.camera = camera
        self.__isCameraAlive = threading.Condition()
        self.__isStopped = threading.Event()
        self.__simLogger = simulationLogger
        self.__imageAreaThreshold = 1000

    def stop(self):
        self.__isStopped.set()
        with self.__isCameraAlive:
            self.__isCameraAlive.notify()

    def __stopped(self):
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
                    if self.__stopped():
                        self.__simLogger.debug("Stopping camera thread")
                        break
            cv2.destroyAllWindows()
        except Exception as e:
            self.__simLogger.critical("Camera exception: " + str(e) + str(
                sys.exc_info()[0]) + ' - ' + traceback.format_exc())


# Simulation class -> neural network, fitness function, individual/social learning
class Simulation(threading.Thread):
    def __init__(self, thymioController, debug, experiment_name):
        threading.Thread.__init__(self)
        config = ConfigParser(CONFIG_PATH)
        self.__address = config.address
        self.__port = config.port
        self.__thymioController = thymioController
        self.__tcPerformedAction = threading.Condition()
        self.__tcPA = False
        self.__msgSenders = dict()
        self.__msgReceivers = dict()
        self.__stopSockets = list()
        self.__stopped = False
        self.__previous_motor_speed = [0, 0]
        # simulation logging file
        self.__simLogger = logging.getLogger('simulationLogger')
        logLevel = logging.INFO
        if debug:
            logLevel = logging.DEBUG
        self.__simLogger.setLevel(logLevel)
        self.__experiment_name = experiment_name
        outputDir = os.path.join(OUTPUT_PATH, self.__experiment_name)
        mkdir_p(outputDir)
        # self.__nextSimFilename = getNextIDPath(SIM_LOG_PATH) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
        # simHandler = logging.FileHandler(os.path.join(SIM_LOG_PATH, self.__nextSimFilename + '_sim_debug.log'))
        self.__simulationLogFile = os.path.join(outputDir, self.__experiment_name + '_sim_debug.log')
        self.__simulationOutputFile = os.path.join(outputDir, self.__experiment_name + '_out.txt')
        self.__simulationWeightOutputFile = os.path.join(outputDir, self.__experiment_name + '_weight_out.txt')
        self.__simulationTempFile = os.path.join(outputDir, self.__experiment_name + '_temp.txt')
        simHandler = logging.FileHandler(self.__simulationLogFile)
        simHandler.setFormatter(FORMATTER)
        self.__simLogger.addHandler(simHandler)
        self.__threadCamera = cameraVision(False, self.__simLogger)

        self.__inbox = Inbox(self.__simLogger)
        for bot in config.bots:
            address = bot["address"]
            self.__msgSenders[address] = MessageSender(address, bot["port"], self.__simLogger)
            self.__msgReceivers[address] = MessageReceiver(address, self.__inbox, self.__simLogger)
        self.__connListener = ConnectionsListener(self.__address, self.__port, self.__msgReceivers, self.__simLogger)

    def getLogger(self):
        return self.__simLogger

    def thymioControllerPerformedAction(self):
        with self.__tcPerformedAction:
            self.__tcPA = True
            self.__tcPerformedAction.notify()

    def __waitForControllerResponse(self):
        # Wait for ThymioController response
        with self.__tcPerformedAction:
            while not self.__tcPA:
                self.__tcPerformedAction.wait()
            self.__tcPA = False

    def __sendFiles(self, filepathOut, filepathLog, filepathTemp, filepathWeight):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((PC_FIXED_IP, OUTPUT_FILE_RECEIVER_PORT))

            # Send output file
            self.__simLogger.debug("Sending file " + str(filepathOut))
            sendOneMessage(s, self.__experiment_name)
            fO = open(filepathOut, 'rb')
            l = fO.read(1024)
            while (l):
                sendOneMessage(s, l)
                l = fO.read(1024)
            self.__simLogger.debug("End of output file")
            sendOneMessage(s, EOF_REACHED)
            fO.close()

            # Send log file
            self.__simLogger.debug("Sending file " + str(filepathLog))
            fL = open(filepathLog, 'rb')
            l = fL.read(1024)
            while (l):
                sendOneMessage(s, l)
                l = fL.read(1024)
            self.__simLogger.debug("End of output file")
            sendOneMessage(s, EOF_REACHED)
            fL.close()

            # Send temp file
            self.__simLogger.debug("Sending file " + str(filepathTemp))
            fT = open(filepathTemp, 'rb')
            l = fT.read(1024)
            while (l):
                sendOneMessage(s, l)
                l = fT.read(1024)
            self.__simLogger.debug("End of output file")
            sendOneMessage(s, EOF_REACHED)
            fT.close()

            # Send weight file
            self.__simLogger.debug("Sending file " + str(filepathWeight))
            fW = open(filepathWeight, 'rb')
            l = fW.read(1024)
            while (l):
                sendOneMessage(s, l)
                l = fW.read(1024)
            fW.close()

            s.shutdown(socket.SHUT_WR)
            s.close()
        except:
            self.__simLogger.critical(
                'Error while sending file: ' + str(sys.exc_info()[0]) + ' - ' + traceback.format_exc())

    def __mutateMemome(self, solution, weights):
        for i in range(weights):
            # mutate weight adding gaussrand() * sigma
            solution.memome[i] += gaussrand() * solution.sigma
            if solution.memome[i] > pr.range_weights:
                solution.memome[i] = pr.range_weights
            elif solution.memome[i] < -pr.range_weights:
                solution.memome[i] = -pr.range_weights

    def __runAndEvaluate(self, evaluee, change_tau):
        # Recovery period (tau timesteps)
        self.__simLogger.info("Recovery period")
        tau = pr.tau
        if change_tau:
            tau = pr.tau_goal
        for i in range(0, tau):
            self.__runAndEvaluateForOneTimeStep(evaluee)
        # self.__simLogger.info("Timestep " + str(i) + " done")

        # Evaluate (evaltime timesteps)
        self.__simLogger.info("Evaluation")

        fitness = [0, 0, 0, 0, 0, False]
        # fitness[0] = total obstacle avoidance
        # fitness[1] = total fitness looking for box
        # fitness[2] = total fitness pushing the box
        # fitness[3] = total fitness looking for goal
        # fitness[4] = total fitness bonus
        # fitness[5] = total found goal

        for i in range(0, pr.eval_time):
            fitness = self.__fitnessFunction(evaluee, fitness)

        print("single ff -> " + str(fitness[0]) + " " + str(fitness[1]) + " " + str(fitness[2]) + " " + str(fitness[3]) + " " + str(fitness[4]))

        total_fitness = sum(fitness[0:5])

        print(str(total_fitness) + "\n ")

        self.__simLogger.info("Fitness ->" + str(total_fitness))
        return total_fitness, fitness[5]


    # fitness function
    def __fitnessFunction(self, evaluee, fitness):
        # result_fitness[0] = obstacle avoidance / external fitness
        # result_fitness[1] = fitness looking for box
        # result_fitness[2] = fitness pushing the box
        # result_fitness[3] = fitness looking for goal
        # result_fitness[4] = found goal
        result_fitness = self.__runAndEvaluateForOneTimeStep(evaluee)

        fitness[0] += result_fitness[0]                             # total fitness obstacle avoidance
        fitness[1] += result_fitness[1]                             # total fitness looking for box
        fitness[2] += result_fitness[2] * 2                         # total fitness pushing the box
        fitness[3] += result_fitness[3]                             # total fitness looking for goal

        fitness[4] += result_fitness[2] * result_fitness[3]         # total fitness bonus
        fitness[5] = fitness[5] or result_fitness[4]                # total found goal

        return fitness

    def __runAndEvaluateForOneTimeStep(self, evaluee):
        # Read sensors: request to ThymioController
        self.__thymioController.readSensorsRequest()
        self.__waitForControllerResponse()
        psValues = self.__thymioController.getPSValues()

        psValues = [psValues[0], psValues[2], psValues[4], psValues[5], psValues[6]]

        # return presence value from camera
        presence_box = self.__threadCamera.readPuckPresence()
        presence_goal = self.__threadCamera.readGoalPresence()
        totalPresence = presence_box + presence_goal
        for i in range(len(totalPresence)):
            threshold = 1500 if i == 3 else 2000  # for bottom part better higher threshold
            if totalPresence[i] > threshold:
                totalPresence[i] = 1
            else:
                totalPresence[i] = 0

        motorspeed = [0, 0]

        # Neural networks
        if pr.hidden_layer == 1:
            # Version with one hidden layer with classes.HIDDEN_NEURONS hidden neurons
            hidden_layer = [0 for x in range(cl.HIDDEN_NEURONS)]
            for y in range(cl.HIDDEN_NEURONS):
                for i in range(0, cl.NB_DIST_SENS):  # Calculate weight only for normal sensor
                    normalizedSensor = min((psValues[i] - (cl.SENSOR_MAX[i] / 2)) / (cl.SENSOR_MAX[i] / 2), 1.0)
                    hidden_layer[y] += normalizedSensor * evaluee.memome[
                        i + ((cl.NN_WEIGHTS / cl.HIDDEN_NEURONS) * y)]
                for i in range(0, cl.NB_CAM_SENS):  # Calculate weight only for normal sensor
                    # normalizedSensor = min((totalPresence[i] - (cl.CAMERA_MAX[i] / 2)) / (cl.CAMERA_MAX[i] / 2),
                    #                        1.0)
                    hidden_layer[y] += totalPresence[i] * evaluee.memome[i + cl.NB_DIST_SENS + ((cl.NN_WEIGHTS / cl.HIDDEN_NEURONS) * y)]

                # Adding bias weight
                hidden_layer[y] += evaluee.memome[((cl.NN_WEIGHTS / cl.HIDDEN_NEURONS) * (y + 1)) - 1]
                # Apply hyberbolic tangent activation function -> left and right in [-1, 1]
                hidden_layer[y] = math.tanh(hidden_layer[y])


            left = 0
            right = 0
            for i in range(0, cl.HIDDEN_NEURONS):  # Calculate weight for hidden neurons
                left += hidden_layer[i] * evaluee.memome[i + cl.NN_WEIGHTS]
                right += hidden_layer[i] * evaluee.memome[i + cl.NN_WEIGHTS + (cl.NN_WEIGHTS_HIDDEN / 2)]
            # Add bias weights
            left += evaluee.memome[cl.NN_WEIGHTS + (cl.NN_WEIGHTS_HIDDEN / 2) - 1]
            right += evaluee.memome[cl.TOTAL_WEIGHTS - 1]
        else:
            # Version without hidden layer
            left = 0
            right = 0
            for i in range(0, cl.NB_DIST_SENS):  # Calculate weight only for normal sensor
                # NormalizedSensor in [-1,1]
                normalizedSensor = min((psValues[i] - (cl.SENSOR_MAX[i] / 2)) / (cl.SENSOR_MAX[i] / 2), 1.0)
                left += totalPresence[i] * evaluee.memome[i]
                right += totalPresence[i] * evaluee.memome[i + (cl.NN_WEIGHTS_NO_HIDDEN / 2)]
            for i in range(0, cl.NB_CAM_SENS):  # Calculate weight only for camera sensor
                # NormalizedSensor in [-1,1]
                # normalizedSensor = min((totalPresence[i] - (cl.CAMERA_MAX[i] / 2)) / (cl.CAMERA_MAX[i] / 2),
                #                        1.0)
                left += normalizedSensor * evaluee.memome[i + cl.NB_DIST_SENS]
                right += normalizedSensor * evaluee.memome[i + cl.NB_DIST_SENS + (cl.NN_WEIGHTS_NO_HIDDEN / 2)]
            # Add bias weights
            left += evaluee.memome[(cl.NN_WEIGHTS_NO_HIDDEN / 2) - 1]
            right += evaluee.memome[cl.NN_WEIGHTS_NO_HIDDEN - 1]

        # Apply hyberbolic tangent activation function -> left and right in [-1, 1]
        left = math.tanh(left)
        right = math.tanh(right)
        if left > 1 or left < -1 or right > 1 or right < -1:
            self.__simLogger.warning("WUT? left = " + str(left) + ", right = " + str(right))

        # Set motorspeed
        motorspeed[LEFT] = int(left * pr.real_max_speed)
        motorspeed[RIGHT] = int(right * pr.real_max_speed)

        if (motorspeed[LEFT] != self.__previous_motor_speed[LEFT]) or (
                    motorspeed[RIGHT] != self.__previous_motor_speed[RIGHT]):
            # Set motor speed: request to ThymioController only if the values are different from previous one
            self.__thymioController.writeMotorspeedRequest(motorspeed)
            self.__waitForControllerResponse()
            # self.__simLogger.info("Simulation - Set motorspeed " + str(motorspeed)[1:-1])

        # remember previous motor speed
        self.__previous_motor_speed = motorspeed


        # FITNESS FUNCTION SECTION -------------------------------------------------------------------------------------

        # Calculate normalized distance to the nearest object
        sensorpenalty = 0
        for i in range(0, cl.NB_DIST_SENS):
            if sensorpenalty < (psValues[i] / float(cl.SENSOR_MAX[i])):
                sensorpenalty = (psValues[i] / float(cl.SENSOR_MAX[i]))
        if sensorpenalty > 1:
            sensorpenalty = 1

        # Normalize all the part of the fitness from -1 to 1
        normalizedSensor = (abs(motorspeed[LEFT]) + abs(motorspeed[RIGHT])) / (pr.real_max_speed * 2)
        fitness_obs = float(normalizedSensor) * (1 - sensorpenalty)

        found = False

        # total ppresence is still 0 or 1
        fitness_looking_for_box = totalPresence[1]

        #     normalized_fitness_box_pushing = 1
        normalized_fitness_box_pushing = totalPresence[3]

        # normalize fitness_looking_for_goal 0 -> 1 (minimum is 0 so i can not write it )
        normalized_fitness_looking_goal = totalPresence[5]

        total_area_goal = sum(presence_goal[0:3])
        # Reach the goal
        if total_area_goal > 15000 and normalized_fitness_box_pushing == 1:
            found = True
            # Make sound
            self.__thymioController.soundRequest(
                        [cl.SOUND])  # system sound -> value from 0 to 7 (4 = free-fall (scary) sound)
            self.__waitForControllerResponse()
            # print goal
            self.__simLogger.info("GOAL REACHED\t")

        fitness_result = (
            fitness_obs, fitness_looking_for_box, normalized_fitness_box_pushing,
            normalized_fitness_looking_goal, found)

        return fitness_result

    def run(self):
        # Start ConnectionsListener
        self.__connListener.start()

        # Set connections for stopping the receivers
        for i in range(0, len(self.__msgReceivers)):
            stopSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            stopSocket.connect((LOCALHOST, self.__port))
            self.__stopSockets.append(stopSocket)

        # Start message receivers
        for addr in self.__msgReceivers:
            self.__msgReceivers[addr].start()

        # Start message senders
        for addr in self.__msgSenders:
            self.__msgSenders[addr].start()

        # Wait for the simulation to be set on the controller
        self.__waitForControllerResponse()

        # Set color: request to ThymioController
        self.__thymioController.writeColorRequest([0, 0, 0, 0])  # Switch off all the leds
        self.__waitForControllerResponse()

        # Starting CPU timer -> for temperature
        t_start = time.clock()

        # Start camera thread
        self.__threadCamera.start()

        # wait until camera starts
        time.sleep(5)


        # set correct weights
        if pr.hidden_layer == 1:
            weights = cl.TOTAL_WEIGHTS
        else:
            weights = cl.NN_WEIGHTS_NO_HIDDEN


        # Set parameters
        lifetime = pr.max_robot_lifetime if pr.evolution == 1 else pr.total_evals  # when no evolution, robot life
        # is whole experiment length
        collected_memomes = [cl.RobotMemomeDataMessage(0.0, [0.0 for i in range(weights)]) for i in
                             range(pr.collected_memomes_max)]
        evals = 0  # current evaluation
        generation = 0  # current generation
        champion = cl.Candidate([0.0 for i in range(weights)], 0.0, 0.0)
        actual_weights = cl.Candidate([0.0 for i in range(weights)], 0.0, 0.0)
        total_goal = 0  # count numbers of goals

        # start main loop
        while evals < pr.total_evals:
            generation += 1
            self.__simLogger.info("########## GENERATION " + str(generation) + " ##########")
            self.__simLogger.info("Current champion: \nMemome: " + str(champion.memome) + "\nSigma: " +
                                  str(champion.sigma) + "\nFitness: " + str(champion.fitness))

            pr.collected_memomes_total = 0

            # INIT NEW MEMOME
            tmp_weight = [random.uniform(-1 * pr.range_weights, pr.range_weights) for x in range(weights)]
            # Set random neural network weights
            for i in range(weights):
                champion.memome[i] = tmp_weight[i]

            champion.sigma = pr.sigmainitial
            try:
                change_tau = False  # Track if i found the goal
                # Evaluate the new individual

                champion.fitness, goalI = self.__runAndEvaluate(copy.deepcopy(champion), change_tau)
                self.__simLogger.info("New champion: \nMemome: " + str(champion.memome) + "\nSigma: " +
                                      str(champion.sigma) + "\nFitness: " + str(champion.fitness))

                # DURING INDIVIDUAL LIFE
                for l in range(lifetime):
                    self.__simLogger.info("@@@@@ EVALUATION  " + str(evals) + " @@@@@")

                    evals += 1
                    goal = False

                    # Choose one between: Reevaluation | Social learning | Individual learning
                    if random.random() <= 0.2:
                        # Reevaluation
                        self.__simLogger.info("----- REEVALUATION -----")
                        self.__simLogger.info("Old fitness = " + str(champion.fitness))
                        fitness, goal = self.__runAndEvaluate(champion, change_tau)
                        self.__simLogger.info("Reevaluated fitness = " + str(fitness))
                        champion.fitness = champion.fitness * pr.re_weight + fitness * (1 - pr.re_weight)
                        self.__simLogger.info("New fitness = " + str(champion.fitness))
                        # Save neural network weights
                        actual_weights = champion
                    else:
                        if pr.collected_memomes_total > 0 and pr.sociallearning == 1 and random.random() <= 0.3:
                            # Social learning
                            self.__simLogger.info("----- SOCIAL LEARNING -----")

                            socialChallenger = copy.deepcopy(
                                champion)  # Deep copy: we don't want to change the champion

                            # Memome Crossover with last memotype in collected_memomes (LIFO order)
                            lastCollectedMemome = collected_memomes[pr.collected_memomes_total - 1].memome
                            # 25% probability (One-Point crossover) average two memome
                            if random.random() <= 0.75:
                                for i in range(weights):
                                    # overwrite value on champion
                                    socialChallenger.memome[i] = lastCollectedMemome[i]
                            else:
                                # One-point crossover from 0 to cutting_value
                                # cutting_value = random.randrange(0, cl.NMBRWEIGHTS)
                                # for i in range(cl.NMBRWEIGHTS):
                                #     if (socialChallenger.memome[i] != 100.0 and lastCollectedMemome[i] != 100.0):
                                #         # If sensor is enabled on both champion and lastCollectedMemome -> overwrite
                                #         if i <= cutting_value:
                                #             # value on champion
                                #             socialChallenger.memome[i] = lastCollectedMemome[i]

                                # Average two memome
                                for i in range(weights):
                                    socialChallenger.memome[i] = (lastCollectedMemome[i] + socialChallenger.memome[i]) / 2

                            pr.collected_memomes_total -= 1

                            socialChallenger.fitness, goal = self.__runAndEvaluate(socialChallenger, change_tau)
                            self.__simLogger.info("Social challenger memome = " + str(socialChallenger.memome))
                            self.__simLogger.info("Social challenger fitness = " + str(socialChallenger.fitness))
                            # Save neural network weights
                            actual_weights = socialChallenger

                            if socialChallenger.fitness > champion.fitness:
                                self.__simLogger.info("Social challenger IS better -> Becomes champion")
                                champion = socialChallenger
                                champion.sigma = pr.sigma_min
                            else:
                                self.__simLogger.info("Social challenger NOT better -> Sigma is doubled")
                                champion.sigma = champion.sigma * pr.sigma_increase
                                if champion.sigma > pr.sigma_max:
                                    champion.sigma = pr.sigma_max
                            self.__simLogger.info("New sigma = " + str(champion.sigma))
                        else:
                            # Individual learning
                            self.__simLogger.info("----- INDIVIDUAL LEARNING -----")
                            if pr.lifetimelearning == 1:
                                challenger = copy.deepcopy(champion)  # Deep copy: we don't want to mutate the champion
                                self.__simLogger.info("Sigma = " + str(champion.sigma))
                                self.__simLogger.info("Challenger memome before mutation = " + str(challenger.memome))
                                self.__mutateMemome(challenger, weights)
                                self.__simLogger.info("Mutated challenger memome = " + str(challenger.memome))
                                challenger.fitness, goal = self.__runAndEvaluate(challenger, change_tau)
                                self.__simLogger.info("Mutated challenger fitness = " + str(challenger.fitness) +
                                                      " VS Champion fitness = " + str(champion.fitness))
                                # Save neural network weights
                                actual_weights = challenger

                                if challenger.fitness > champion.fitness:
                                    self.__simLogger.info("Challenger IS better -> Becomes champion")
                                    champion = challenger
                                    champion.sigma = pr.sigma_min
                                else:
                                    self.__simLogger.info("Challenger NOT better -> Sigma is doubled")
                                    champion.sigma = champion.sigma * pr.sigma_increase
                                    if champion.sigma > pr.sigma_max:  # boundary rule
                                        champion.sigma = pr.sigma_max
                                self.__simLogger.info("New sigma = " + str(champion.sigma))

                    self.__simLogger.info("Current champion: \nMemome: " + str(champion.memome) + "\nSigma: " +
                                          str(champion.sigma) + "\nFitness: " + str(champion.fitness))

                    # Make sound if I found the goal
                    if goal or goalI:
                        change_tau = True  # Longer recovery time
                        total_goal += 1  # Increase nnumbers of goals
                    else:
                        change_tau = False  # Make sure that tau is correct if I did't find the goal

                    # Write output: open file to append the values
                    with open(self.__simulationOutputFile, 'a') as outputFile:
                        outputFile.write(str(evals) + " \t " + str(generation) + " \t " + str(l) + " \t " + str(
                            champion.fitness))
                        if goal or goalI:
                            outputFile.write("\tGOAL n: " + str(total_goal))
                        outputFile.write("\n")

                    # Write weight output: open file to append the values
                    with open(self.__simulationWeightOutputFile, 'a') as outputFile:
                        outputFile.write(str(evals) + " \t " + str(generation) + " \t " + str(l) + " \t " + str(
                            actual_weights.fitness) + "\nMemome: " + str(actual_weights.memome) +
                                         "\nChampion: " + str(champion.memome))
                        outputFile.write("\n")

                    # Retrieve temperature value
                    # Read sensors: request to ThymioController
                    self.__thymioController.readTemperatureRequest()
                    self.__waitForControllerResponse()
                    temperature = self.__thymioController.getTemperature()
                    # second from start
                    t_from_start = time.clock() - t_start
                    # Write temp output: open file to append the values
                    with open(self.__simulationTempFile, 'a') as outputFile:
                        outputFile.write(str(evals) + " \t Temperature -> " + str(temperature[0]) + " \t after " + str(
                            t_from_start) + " \t seconds")
                        outputFile.write("\n")

                    # Send messages: tell MessageSenders to do that
                    self.__simLogger.info("Broadcasting messages...")
                    if pr.sociallearning == 1 and (champion.fitness / pr.max_fitness) > pr.threshold:
                        messageMem = cl.RobotMemomeDataMessage(champion.fitness, champion.memome)
                        self.__simLogger.info("Broadcast memome = " + str(messageMem.memome))
                        for addr in self.__msgSenders:
                                self.__msgSenders[addr].outboxAppend(messageMem)

                    # Read received messages from Inbox
                    receivedMsgs = self.__inbox.popAll()
                    self.__simLogger.info("Reading " + str(len(receivedMsgs)) + " received messages")
                    for rmsg in receivedMsgs:
                        if type(rmsg) is cl.RobotMemomeDataMessage:
                            if pr.collected_memomes_total < pr.collected_memomes_max:
                                self.__simLogger.info("Received memome = " + str(rmsg.memome))
                                collected_memomes[pr.collected_memomes_total] = rmsg
                                pr.collected_memomes_total += 1
                        else:
                            self.__simLogger.warning("WUT? Received stuff = " + str(rmsg))

                    # check if camera thread is still alive
                    # Camera could raise one exception (corrupted image). If this happens stop that thread and start
                    # one new
                    if not self.__threadCamera.isAlive():
                        self.__threadCamera.stop()
                        self.__threadCamera.join()

                        self.__threadCamera = cameraVision(False, self.__simLogger)

                        self.__threadCamera.start()
                        self.__simLogger.warning("Reanimating Camera Thread")

            except Exception as e:
                self.__simLogger.critical("Some exception: " + str(e) + str(
                    sys.exc_info()[0]) + ' - ' + traceback.format_exc())
                self.__thymioController.stopThymioRequest()
                break

            self.__simLogger.info("End of while loop: " + str(evals) + " >= " + str(pr.total_evals))

        self.stop()

        self.__simLogger.info("_____END OF SIMULATION_____\n")

    def isStopped(self):
        return self.__stopped

    def stop(self):
        if self.__stopped:
            self.__simLogger.info('Simulation already stopped.')
            return
        self.__stopped = True

        # Send log files to server.
        self.__simLogger.info("Sending Files...\n")
        self.__sendFiles(self.__simulationOutputFile, self.__simulationLogFile, self.__simulationTempFile,
                         self.__simulationWeightOutputFile)

        # Stop Thymio from moving
        self.__thymioController.stopThymioRequest()

        # Stopping all the message senders: no more outgoing messages
        for addr in self.__msgSenders:
            self.__simLogger.info('Killing Sender ' + addr)
            self.__msgSenders[addr].stop()
            self.__msgSenders[addr].join()
        self.__simLogger.info('All MessageSenders: KILLED')

        # Stopping connections listener: no more incoming connections
        self.__connListener.stop()
        self.__connListener.join()
        self.__simLogger.info('ConnectionsListener: KILLED')

        # Stopping camera thread
        self.__threadCamera.stop()
        self.__threadCamera.join()
        self.__simLogger.info('CameraThread: KILLED')

        # Stopping all the message receivers: no more incoming messages
        i = 0
        for addr in self.__msgReceivers:
            self.__simLogger.info('Killing Receiver ' + addr)
            self.__msgReceivers[addr].stop()
            sendOneMessage(self.__stopSockets[i], 'STOP')  # Send stop messages
            self.__msgReceivers[addr].join()
            self.__stopSockets[i].close()
            i += 1
        self.__simLogger.info('All MessageReceivers: KILLED')

# Thymio robot controller
class ThymioController(object):
    def __init__(self, mainLogger):
        self.__mainLogger = mainLogger
        self.__psValue = [0, 0, 0, 0, 0, 0, 0]
        self.__psGroundAmbiantSensors = [0, 0]
        self.__psGroundReflectedSensors = [0, 0]
        self.__psGroundDeltaSensors = [0, 0]
        self.__motorspeed = [0, 0]
        self.__realMotorSpeed = [0, 0]
        self.__color = [0, 0, 0, 0]
        self.__temperature = [0]
        self.__sound = [0]

        self.__performActionReq = threading.Condition()
        self.__rSensorsReq = False
        self.__temperatureReq = False
        self.__rGroundSensorsReq = False
        self.__wMotorspeedReq = False
        self.__rMotorspeedReq = False
        self.__wColorReq = False
        self.__stopThymioReq = False
        self.__killReq = False
        self.__soundReq = False

        self.__commandsListener = None
        self.__simulationStarted = threading.Event()
        self.__simulation = None
        self.__simLogger = None

        # Init the main loop
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        # Get stub of the Aseba network
        bus = dbus.SessionBus()
        # Create Aseba network
        asebaNetworkObject = bus.get_object('ch.epfl.mobots.Aseba', '/')
        self.__asebaNetwork = dbus.Interface(asebaNetworkObject, dbus_interface='ch.epfl.mobots.AsebaNetwork')
        # self.__mainLogger.debug('Aseba nodes: ' + str(self.__asebaNetwork.GetNodesList()))
        # Load the aesl file
        self.__asebaNetwork.LoadScripts(AESL_PATH, reply_handler=self.__dbusEventReply, error_handler=self.__dbusError)
        # Schedules first run of the controller
        glib.idle_add(self.__execute)

    def setSimulation(self, simulation):
        if not self.__simulation:
            self.__simulation = simulation
            self.__simLogger = simulation.getLogger()
            self.__simulationStarted.set()

    def __dbusError(self, e):
        # there was an error on D-Bus, stop loop
        self.__simLogger.critical('dbus error: %s' % str(e) + "\nNow sleeping for 1 second and retrying...")
        time.sleep(1)
        raise Exception("dbus error")


    def __dbusEventReply(self):
        # correct replay on D-Bus, ignore
        pass

    def __dbusSendEventName(self, eventName, params):
        ok = False
        while not ok:
            try:
                self.__asebaNetwork.SendEventName(eventName, params, reply_handler=self.__dbusEventReply,
                                                  error_handler=self.__dbusError)
                ok = True
            except:
                self.__simLogger.critical("Error during SEND EVENT NAME: " + eventName + " - " + str(params))
                ok = False

    def __dbusGetVariable(self, varName, replyHandler):
        ok = False
        while not ok:
            try:
                self.__asebaNetwork.GetVariable("thymio-II", varName, reply_handler=replyHandler,
                                                error_handler=self.__dbusError)
                ok = True
            except:
                self.__simLogger.critical("Error during GET VARIABLE: " + varName)
                ok = False

    def __dbusSetMotorspeed(self):
        self.__dbusSendEventName("SetSpeed", self.__motorspeed)

    def __dbusSetColor(self):
        self.__dbusSendEventName("SetColor", self.__color)

    def __dbusSetSound(self):
        self.__dbusSendEventName("PlaySound", self.__sound)

    def __dbusGetProxSensorsReply(self, r):
        self.__psValue = r

    def __dbusGetProxSensors(self):
        self.__dbusGetVariable("prox.horizontal", self.__dbusGetProxSensorsReply)

    def __dbusGetTemperatureReply(self, r):
        self.__temperature = r

    def __dbusGetTemperature(self):
        self.__dbusGetVariable("temperature", self.__dbusGetTemperatureReply)

    def __dbusGetMotorSpeed(self):
        self.__dbusGetVariable("motor.left.speed ", self.__dbusGetLeftSpeedReply)
        self.__dbusGetVariable("motor.right.speed ", self.__dbusGetRightSpeedReply)

    def __dbusGetLeftSpeedReply(self, r):
        self.__realMotorSpeed[0] = r

    def __dbusGetRightSpeedReply(self, r):
        self.__realMotorSpeed[1] = r

    def __dbusGetGroundAmbiantReply(self, r):
        self.__psGroundAmbiantSensors = r

    def __dbusGetGroundReflectedReply(self, r):
        self.__psGroundReflectedSensors = r

    def __dbusGetGroundDeltaReply(self, r):
        self.__psGroundDeltaSensors = r

    def __dbusGetGroundSensors(self):
        self.__dbusGetVariable("prox.ground.ambiant", self.__dbusGetGroundAmbiantReply)
        self.__dbusGetVariable("prox.ground.reflected", self.__dbusGetGroundReflectedReply)
        self.__dbusGetVariable("prox.ground.delta", self.__dbusGetGroundDeltaReply)

    def __stopThymio(self):
        # Red LEDs: Thymio stops moving
        self.__sound = [1]
        self.__dbusSetSound()
        self.__motorspeed = [0, 0]
        self.__dbusSetMotorspeed()
        self.__simulation = None
        self.__simulationStarted.clear()

    def __execute(self):
        # Wait for the simulation to be set
        while not self.__simulationStarted.isSet():
            self.__simulationStarted.wait()
        # Notifying that simulation has been set
        self.__simulation.thymioControllerPerformedAction()

        with self.__performActionReq:
            # Wait for requests:
            while not (self.__soundReq or self.__rSensorsReq or self.__rGroundSensorsReq or self.__temperatureReq or
                       self.__wMotorspeedReq or self.__rMotorspeedReq or self.__wColorReq or self.__stopThymioReq or
                       self.__killReq):
                self.__performActionReq.wait()
            if self.__rSensorsReq:
                # Read sensor values
                self.__dbusGetProxSensors()
                self.__rSensorsReq = False
            elif self.__rGroundSensorsReq:
                # Read ground sensor values
                self.__dbusGetGroundSensors()
                self.__rGroundSensorsReq = False
            elif self.__temperatureReq:
                # Read temperature values
                self.__dbusGetTemperature()
                self.__temperatureReq = False
            elif self.__soundReq:
                # emit sound
                self.__dbusSetSound()
                self.__soundReq = False
            elif self.__wMotorspeedReq:
                # Write motorspeed
                self.__dbusSetMotorspeed()  # IF COMMENTED: wheels don't move
                # Make sure that Thymio moved for 1 timestep
                time.sleep(
                    cl.TIME_STEP)  # TODO: more precise -> thymio should notify controller when it moved for 50 ms
                self.__wMotorspeedReq = False
            elif self.__rMotorspeedReq:
                self.__dbusGetMotorSpeed()
                self.__rMotorspeedReq = False
            elif self.__wColorReq:
                self.__dbusSetColor()
                self.__wColorReq = False
            elif self.__stopThymioReq:
                # Stop Thymio
                self.__stopThymio()
                self.__stopThymioReq = False
            elif self.__killReq:
                # Kill everything
                self.__stopThymio()
                self.__loop.quit()
                return False
        return True

    def readTemperatureRequest(self):
        with self.__performActionReq:
            self.__temperatureReq = True
            self.__performActionReq.notify()

    def readSensorsRequest(self):
        with self.__performActionReq:
            self.__rSensorsReq = True
            self.__performActionReq.notify()

    def readGroundSensorsRequest(self):
        with self.__performActionReq:
            self.__rGroundSensorsReq = True
            self.__performActionReq.notify()

    def writeMotorspeedRequest(self, motorspeed):
        with self.__performActionReq:
            self.__motorspeed = motorspeed
            self.__wMotorspeedReq = True
            self.__performActionReq.notify()

    def readMotorspeedRequest(self):
        with self.__performActionReq:
            self.__rMotorspeedReq = True
            self.__performActionReq.notify()

    def writeColorRequest(self, color):
        with self.__performActionReq:
            self.__color = color
            self.__wColorReq = True
            self.__performActionReq.notify()

    def soundRequest(self, sound):
        with self.__performActionReq:
            self.__sound = sound
            self.__soundReq = True
            self.__performActionReq.notify()

    def stopThymioRequest(self):
        with self.__performActionReq:
            self.__stopThymioReq = True
            self.__performActionReq.notify()

    def killRequest(self):
        with self.__performActionReq:
            self.__killReq = True
            self.__performActionReq.notify()

    def getTemperature(self):
        return self.__temperature

    def getMotorSpeed(self):
        return self.__realMotorSpeed

    def getPSValues(self):
        return self.__psValue

    def getDeltaValues(self):
        return self.__psGroundDeltaSensors

    def getGroundSensorsValues(self):
        return self.__psGroundAmbiantSensors, self.__psGroundReflectedSensors, self.__psGroundDeltaSensors

    def run(self):
        self.__mainLogger.debug('Controller - RUNNING')
        # Starts commands listener
        self.__cmdListener = CommandsListener(self, self.__mainLogger)
        self.__cmdListener.start()
        # Run gobject loop
        self.__loop = gobject.MainLoop()
        self.__loop.run()

# Control command listener
class CommandsListener(threading.Thread):
    def __init__(self, thymioController, mainLogger):
        threading.Thread.__init__(self)
        # Create socket for listening to simulation commands
        self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.__sock.bind((COMMANDS_LISTENER_HOST, COMMANDS_LISTENER_PORT))
        self.__sock.listen(5)
        self.__thymioController = thymioController
        self.__mainLogger = mainLogger
        self.__simulation = None
        self.__counter = pr.starter_number

    def run(self):
        while 1:
            try:
                # Waiting for client...
                self.__mainLogger.debug("CommandListener - Waiting on accept...")
                conn, (addr, port) = self.__sock.accept()
                self.__mainLogger.debug('CommandListener - Received command from (' + addr + ', ' + str(port) + ')')
                if addr not in TRUSTED_CLIENTS:
                    self.__mainLogger.error(
                        'CommandListener - Received connection request from untrusted client (' + addr + ', ' + str(
                            port) + ')')
                    continue

                # Receive one message
                self.__mainLogger.debug('CommandListener - Receiving command...')
                recvOptions = recvOneMessage(conn)
                self.__mainLogger.debug('CommandListener - Received ' + str(recvOptions))

                if recvOptions.kill:
                    # Received killing command -> Stop everything
                    self.__thymioController.killRequest()
                    if self.__simulation:
                        self.__simulation.stop()
                    break
                elif recvOptions.start and (not self.__simulation or self.__simulation.isStopped()):
                    # Adding experiment number to pr.EXPERIMENT_NAME
                    experiment_name = pr.EXPERIMENT_NAME + "_" + str(self.__counter)
                    self.__counter += 1
                    # Received start request AND simulation is not running -> Start a new simulation
                    self.__mainLogger.debug("CommandListener - Starting simulation...")
                    self.__simulation = Simulation(self.__thymioController, recvOptions.debug, experiment_name)
                    self.__thymioController.setSimulation(self.__simulation)
                    self.__simulation.start()
                elif recvOptions.stop and self.__simulation and not self.__simulation.isStopped():  # TODO: Stop properly
                    # Received stop request AND simulation is up and running -> Stop the simulation
                    self.__mainLogger.debug("CommandListener - Stopping simulation...")
                    self.__simulation.stop()
                    self.__simulation = None
                elif recvOptions.stopthymio:
                    self.__mainLogger.debug("CommandListener - Stopping Thymio...")
                    self.__thymioController.stopThymio()

            except:
                self.__mainLogger.critical(
                    'Error in CommandsListener: ' + str(sys.exc_info()[0]) + ' - ' + traceback.format_exc())

        self.__mainLogger.debug('CommandListener - KILLED -> Exiting...')


if __name__ == '__main__':
    # Main logger for ThymioController and CommandsListener
    mainLogger = logging.getLogger('mainLogger')
    mainLogger.setLevel(logging.DEBUG)
    mainLogFilename = getNextIDPath(MAIN_LOG_PATH) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '_main_debug.log'
    mainHandler = logging.FileHandler(os.path.join(MAIN_LOG_PATH, mainLogFilename))
    mainHandler.setFormatter(FORMATTER)
    mainLogger.addHandler(mainHandler)
    try:
        # To avoid conflicts between gobject and python threads
        gobject.threads_init()
        dbus.mainloop.glib.threads_init()

        tC = ThymioController(mainLogger)
        tC.run()
        # ThymioController is the main loop now (needed for communication with the Thymio).
        mainLogger.debug("ThymioController stopped -> main stops.")
    except:
        # mainLogger.critical('Error in main: ' + str(sys.exc_info()[0]) + ' - ' + traceback.forma_exc())
        mainLogger.critical('Error in main')


