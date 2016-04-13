import dbus
import dbus.mainloop.glib
import os
import random

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')
MAX_MOTOR_SPEED = 250
MAX_PRESENCE = 16000
STOP_CONDITION = MAX_PRESENCE * 0.6
MAX_PRESENCE_TARGET = 9000
STOP_CONDITION_TARGET = 36000
BASE_SPEED = 0.5
ADAPTIVITY = 0.3

class Robot():

    def __init__(self):
        #init Thymio control
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()
        self.thymioController = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')
        self.thymioController.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)
        self.hasPuck = False
        self.motorspeed = {'left': BASE_SPEED, 'right': BASE_SPEED}
        self.stopped = False

    def updatePath(self, presence):
        # self.driveToTarget(presence)
        if self.hasPuck:
            if sum(presence['target']) == 0:
                print "Search target random"
                self.search()
            else:
                print "Collect target"
                self.driveToTarget(presence)
        elif not self.stopped:
            if sum(presence['puck']) == 0:
                print "Search puck random"
                self.search()
            else:
                print "Collect puck"
                self.driveToPuck(presence)

    # assign random motorspeed based on previous speed combined with a random number between -1 and 1
    def search(self):
        self.motorspeed['left'] = (1-ADAPTIVITY)*self.motorspeed['left'] + ADAPTIVITY * (random.random()*2-1)
        self.motorspeed['right'] = (1-ADAPTIVITY)*self.motorspeed['left'] + ADAPTIVITY * (random.random()*2-1)
        self.writeMotorSpeed(self.motorspeed)

    def driveToPuck(self, presence):
        (left, middle, right, below) = presence['puck']
        if below >= STOP_CONDITION:
            print "found Puck!"
            self.hasPuck = True
        else:
            fraction_left = right/MAX_PRESENCE
            fraction_right = left/MAX_PRESENCE
            self.motorspeed = {'left': BASE_SPEED+fraction_left, 'right': BASE_SPEED+fraction_right}
            # print self.motorspeed['left']
            self.writeMotorSpeed(self.motorspeed)

    def driveToTarget(self, presence):
        (left, middle, right, below) = presence['target']

        if sum(presence['target']) >= STOP_CONDITION_TARGET:
            print "Found target!"
            self.stop()
        else:
            fraction_left = right/MAX_PRESENCE_TARGET
            fraction_right = left/MAX_PRESENCE_TARGET
            self.motorspeed = {'left': BASE_SPEED+fraction_left, 'right': BASE_SPEED + fraction_right}

        self.writeMotorSpeed(self.motorspeed)

    def stop(self):
        self.stopped = True
        self.motorspeed = {'left': 0, 'right': 0}
        self.writeMotorSpeed(self.motorspeed)

    def writeMotorSpeed(self, motorspeed, max_speed=MAX_MOTOR_SPEED):
        self.thymioController.SetVariable("thymio-II", "motor.left.target", [motorspeed['left'] * max_speed])
        self.thymioController.SetVariable("thymio-II", "motor.right.target", [motorspeed['right'] * max_speed])

    def avoidObstacle(self):
        def ok_call(psValues):
            print "all ok!"
            psValues = np.array([psValues[0], psValues[2], psValues[4], psValues[5], psValues[6], 1],dtype='f')
            print str(psValues)
            self.writeMotorSpeed({'left': BASE_SPEED, 'right': BASE_SPEED})
            psValues[0:5] = [(float(x) - float(pr.SENSOR_MAX[0]/2))/float(pr.SENSOR_MAX[0]/2) for x in psValues[0:5]]

        def nok_call():
            print " Error while reading proximity sensors"

        getProxReadings(self.thymioController, ok_call, nok_call)

def dbusReply():
    pass

def dbusError(e):
    print 'error %s' % str(e)

def getProxReadings(controller, ok_callback, nok_callback):
    print "Get prox readings"
    controller.GetVariable("thymio-II", "prox.horizontal", reply_handler=ok_callback, error_handler=nok_callback)
