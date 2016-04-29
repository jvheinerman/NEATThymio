import dbus
import dbus.mainloop.glib
import gobject
import os
import random
import numpy as np
import sys
import time
import csv
import itertools
from optparse import OptionParser
from cameravision import CameraVision

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')
MAX_MOTOR_SPEED = 250
MAX_PRESENCE = 3000
STOP_CONDITION = MAX_PRESENCE * 0.35/4
MAX_PRESENCE_TARGET = 18000
STOP_CONDITION_TARGET = MAX_PRESENCE_TARGET * 1
BASE_SPEED = 0.7
ADAPTIVITY = 0.4

PROX_SENSOR_MAX = 4500

class Robot():

    def __init__(self, thymio_controller, loop):
        self.thymioController = thymio_controller
        self.loop = loop

        self.hasPuck = False
        self.motorspeed = {'left': BASE_SPEED, 'right': BASE_SPEED}
        self.stopped = False
        self.psValues = []
        self.searching = False

    def get_prox_values(self):
        def main_lambda(task):
            getProxReadings(task.thymioController, task.ok_call, task.nok_call)
            self.loop.quit()
        gobject.idle_add(lambda: main_lambda(self))
        self.loop.run()

    def updatePath(self, presence, logger_obj, image):
        self.get_prox_values()

        if len(self.psValues) == 5 and image is not None:
            logger_obj.writeValues(self.psValues, image)

        if self.hasPuck:
            if sum(presence['target']) == 0:
                print "Search target"
                self.search()
            else:
                print "Collect target"
                self.searching = False
                self.driveToTarget(presence)
        elif not self.stopped:
            if sum(presence['puck']) == 0:
                print "Search puck"
                self.search()
            else:
                print "Collect puck"
                self.searching = False
                self.driveToPuck(presence)

    # rotate until a puck is found
    def search(self):
        self.avoidObstacle()
        # self.searching = True
        self.motorspeed['left'] = BASE_SPEED*0.4
        self.motorspeed['right'] = 0
        self.writeMotorSpeed(self.motorspeed)

    # assign random motorspeed based on previous speed combined with a random number between -1 and 1
    def searchRandom(self):
        self.avoidObstacle()
        self.motorspeed['left'] = (1 - ADAPTIVITY) * self.motorspeed['left'] + ADAPTIVITY * (random.random() * 1.5 - 0.5)
        self.motorspeed['right'] = (1 - ADAPTIVITY) * self.motorspeed['left'] + ADAPTIVITY * (random.random() * 1.5 - 0.5)
        self.writeMotorSpeed(self.motorspeed)

    def driveToPuck(self, presence):
        (left, middle, right, below) = presence['puck']
        if below >= STOP_CONDITION:
            print "Found puck!"
            self.hasPuck = True
        else:
            fraction_left = right / MAX_PRESENCE
            fraction_right = left / MAX_PRESENCE
            self.motorspeed = {'left': BASE_SPEED + fraction_left, 'right': BASE_SPEED + fraction_right}
            self.writeMotorSpeed(self.motorspeed)
            time.sleep(.5)

    def driveToTarget(self, presence):
        (left, middle, right, below) = presence['target']

        if sum(presence['target']) >= STOP_CONDITION_TARGET:
            print "Found target!"
            self.stop()
            time.sleep(10)
            # self.hasPuck = False
        else:
            fraction_left = (right / MAX_PRESENCE_TARGET) * 0.6
            fraction_right = (left / MAX_PRESENCE_TARGET) * 0.6
            self.motorspeed = {'left': BASE_SPEED + fraction_left, 'right': BASE_SPEED + fraction_right}

        self.writeMotorSpeed(self.motorspeed)

    def stop(self):
        self.stopped = True
        self.motorspeed = {'left': 0, 'right': 0}
        self.writeMotorSpeed(self.motorspeed)

    def writeMotorSpeed(self, motorspeed, max_speed=MAX_MOTOR_SPEED):
        self.thymioController.SetVariable("thymio-II", "motor.left.target", [motorspeed['left'] * max_speed])
        self.thymioController.SetVariable("thymio-II", "motor.right.target", [motorspeed['right'] * max_speed])

    def ok_call(self, new_values):
        # in order: left-front, middle, right-front, right-back, left-back
        self.psValues = np.array([new_values[0], new_values[2], new_values[4], new_values[5], new_values[6]], dtype='f')

    def nok_call(self, error):
            print " Error while reading proximity sensors: ", error

    def avoidObstacle(self):
        print "reason to avoid: ", self.psValues
        while abs(sum(self.psValues[0:3])) >= 100:
            pos = maxItemIndex(self.psValues[0:3])
            if pos == 0:
                self.motorspeed = {'left': -BASE_SPEED/4, 'right': -BASE_SPEED/2}
            elif pos == 1:
                print('ride back')
                self.motorspeed = {'left': -BASE_SPEED/2, 'right': -BASE_SPEED/2}
            else:
                self.motorspeed = {'left': -BASE_SPEED/2, 'right': -BASE_SPEED/4}
            self.writeMotorSpeed(self.motorspeed)
            self.psValues = []
            self.get_prox_values()
            time.sleep(.25)

        while abs(sum(self.psValues[3:5]) >= 100):
            if (self.psValues[3] - self.psValues[4]) > 100:
                self.motorspeed = {'left': BASE_SPEED/4, 'right': BASE_SPEED/2}
            elif (self.psValues[4] - self.psValues[3]) > 100:
                self.motorspeed = {'left': BASE_SPEED/2, 'right': BASE_SPEED/4}
            else:
                self.motorspeed = {'left': BASE_SPEED/2, 'right': BASE_SPEED/2}
            self.writeMotorSpeed(self.motorspeed)
            self.psValues = []
            self.get_prox_values()
            time.sleep(.25)


class Logger():

    def __init__(self):
        self.step = 1
        self.pucknr = 4
        self.rows = []
        self.csvfile = open("./log_"+str(self.pucknr)+".csv", 'wb')
        self.logger = csv.writer(self.csvfile, delimiter=',')
        self.writeHeader(["pucknr", "step", "ps left", 'ps middle', 'ps right', 'ps right back', 'ps left back', 'bgr values'])

    def writeHeader(self, values):
        self.logger.writerow(values)

    def writeValues(self, sensorvalues, cameravalues):
        sensorvalues = normalizeProxReadings(sensorvalues)
        row = list(itertools.chain.from_iterable([[self.pucknr, self.step], sensorvalues, [cameravalues.tolist()]]))
        self.rows += [row]
        self.step += 1

    def writeFile(self):
        self.logger.writerows(self.rows)
        self.csvfile.close()


def dbusReply():
    pass


def maxItemIndex(array):
    maxIndex = 0
    maxValue = -sys.maxint - 1
    for id, value in enumerate(array):
        if value > maxValue:
            maxValue = value
            maxIndex = id
    return maxIndex


def dbusError(e):
    print 'error %s' % str(e)


def getProxReadings(controller, ok_callback, nok_callback):
    controller.GetVariable("thymio-II", "prox.horizontal", reply_handler=ok_callback, error_handler=nok_callback)

def normalizeProxReadings(proxreadings):
    normalized = [(float(x) - float(PROX_SENSOR_MAX/2))/float(PROX_SENSOR_MAX/2) for x in proxreadings[0:5]]
    return normalized


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-s", "--system", action="store_true", dest="system", default=False,
                      help="use the system bus instead of the session bus")
    (options, args) = parser.parse_args()
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    if options.system:
        bus = dbus.SystemBus()
    else:
        bus = dbus.SessionBus()

    controller = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')
    controller.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)

    print controller.GetNodesList()

    loop = gobject.MainLoop()

    robot = Robot(controller, loop)
    logger = Logger()

    def cameracallback(presence, image=None):
        print "callback"
        robot.updatePath(presence, logger, image)

    def cameraErrorCallback():
        robot.stop()
        print "writing file"
        logger.writeFile()

    cameravision = CameraVision(False, None)
    cameravision.run(cameracallback, cameraErrorCallback, hsv=False)

