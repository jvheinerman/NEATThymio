import dbus
import dbus.mainloop.glib
import os

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')
MAX_MOTOR_SPEED = 250
MAX_PRESENCE = 18000
STOP_CONDITION = 16000
BASE_SPEED = 0.5

class Robot():

    def __init__(self):
        #init Thymio control
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()
        self.thymioController = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')
        self.thymioController.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)

    def driveToPuck(self, presence):
        (left, middle, right, below) = presence
        if below >= STOP_CONDITION:
            print "stop!"
            self.stop()
        else:
            fraction_left = right/MAX_PRESENCE
            fraction_right = left/MAX_PRESENCE
            motorspeed = {'left': BASE_SPEED+fraction_left, 'right': BASE_SPEED+fraction_right}
            print motorspeed['left']
            self.writeMotorSpeed(motorspeed)

    def stop(self):
        motorspeed = {'left':0, 'right':0}
        self.writeMotorSpeed(motorspeed)

    def writeMotorSpeed(self, motorspeed, max_speed=MAX_MOTOR_SPEED):
        self.thymioController.SetVariable("thymio-II", "motor.left.target", [motorspeed['left'] * max_speed])
        self.thymioController.SetVariable("thymio-II", "motor.right.target", [motorspeed['right'] * max_speed])

def dbusReply():
    pass

def dbusError(e):
    print 'error %s' % str(e)
