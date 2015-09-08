import dbus
import dbus.mainloop.glib
import gobject
import os
from random import randint
import sys
import time
from optparse import OptionParser
import picamera

proxSensorsVal = [0, 0, 0, 0, 0, 0, 0]
accVal = [0, 0, 0]
ground = [0, 0]
ref = [0, 0]
amb = [0, 0]
temp = [0]
mic = [0]
CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'test.aesl')


def Test():
    # get the values of the sensors
    robot.GetVariable("thymio-II", "prox.horizontal", reply_handler=get_variables_reply,
                      error_handler=get_variables_error)

    robot.GetVariable("thymio-II", "acc", reply_handler=accget_variables_reply, error_handler=get_variables_error)

    robot.GetVariable("thymio-II", "prox.ground.delta", reply_handler=deltaget_variables_reply,
                      error_handler=get_variables_error)
    robot.GetVariable("thymio-II", "prox.ground.reflected", reply_handler=reftaget_variables_reply,
                      error_handler=get_variables_error)
    robot.GetVariable("thymio-II", "prox.ground.ambiant", reply_handler=ambtaget_variables_reply,
                      error_handler=get_variables_error)

    robot.GetVariable("thymio-II", "temperature", reply_handler=tempget_variables_reply,
                      error_handler=get_variables_error)
    robot.GetVariable("thymio-II", "mic.intensity", reply_handler=micget_variables_reply,
                      error_handler=get_variables_error)

    # print the proximity sensors value in the terminal
    print "---proximity sensors"
    print "front left: %d ; front-middle left: %d ; front-middle: %d ; front-middle right: %d ; front right: %d ; back left: %d ; back right: %d " % (
        proxSensorsVal[0], proxSensorsVal[1], proxSensorsVal[2], proxSensorsVal[3], proxSensorsVal[4], proxSensorsVal[
            5], proxSensorsVal[6])

    # print accelerometer sensor
    print "---accelerometer sensor"
    print "x-axis: %d , y-axis: %d , z-axis: %d " % (accVal[0], accVal[1], accVal[2])

    # print ground
    print "---ground distance sensors"
    print "ground.delta -> left: %d , right: %d ; ground.reflected -> left: %d , right: %d ; ground.ambiant -> left: %d, right; %d  " % (
        ground[0], ground[1], ref[0], ref[1], amb[0], amb[1])

    # print temp
    print "---temperature sensor"
    print temp[0]

    # print sound intensitivi
    # print "---microphone intensity"
    # print mic[0]

    print " "

    char = sys.stdin.read(1)
    if char == 'p':
        for x in range(0, 7):
            robot.SendEventName('PlaySound', [x], reply_handler=dbusReply, error_handler=dbusError)
            time.sleep(1)
    if char == 'l':
        for x in range(0, 32):
            robot.SendEventName('SetColor',
                                [randint(0, 32), randint(0, 32), randint(0, 32)],
                                reply_handler=dbusReply,
                                error_handler=dbusError)
            time.sleep(0.2)
        robot.SendEventName('SetColor', [0, 0, 0], reply_handler=dbusReply,
                            error_handler=dbusError)
    if char == 'o':
        robot.SendEventName('PlayFreq', [700, 0], reply_handler=dbusReply, error_handler=dbusError)
        time.sleep(1.5)
        robot.SendEventName('PlayFreq', [700, -1], reply_handler=dbusReply, error_handler=dbusError)
        time.sleep(0.1)
        robot.SendEventName('PlayFreq', [1000, 0], reply_handler=dbusReply, error_handler=dbusError)
        time.sleep(1.5)
        robot.SendEventName('PlayFreq', [1000, -1], reply_handler=dbusReply, error_handler=dbusError)
    if char == 'w':
        robot.SetVariable("thymio-II", "motor.left.target", [50])
        robot.SetVariable("thymio-II", "motor.right.target", [50])
        time.sleep(1)
        robot.SetVariable("thymio-II", "motor.left.target", [0])
        robot.SetVariable("thymio-II", "motor.right.target", [0])
    if char == 's':
        robot.SetVariable("thymio-II", "motor.left.target", [-300])
        robot.SetVariable("thymio-II", "motor.right.target", [-300])
        time.sleep(1)
        robot.SetVariable("thymio-II", "motor.left.target", [0])
        robot.SetVariable("thymio-II", "motor.right.target", [0])
    if char == 'a':
        robot.SetVariable("thymio-II", "motor.left.target", [-300])
        robot.SetVariable("thymio-II", "motor.right.target", [300])
        time.sleep(0.2)
        robot.SetVariable("thymio-II", "motor.left.target", [0])
        robot.SetVariable("thymio-II", "motor.right.target", [0])
    if char == 'd':
        robot.SetVariable("thymio-II", "motor.left.target", [300])
        robot.SetVariable("thymio-II", "motor.right.target", [-300])
        time.sleep(0.2)
        robot.SetVariable("thymio-II", "motor.left.target", [0])
        robot.SetVariable("thymio-II", "motor.right.target", [0])
    if char == "f":
        camera = picamera.PiCamera()
        camera.hflip = True
        camera.vflip = True
        camera.start_preview()
        time.sleep(2)
        camera.capture('image.jpg')

    return True


def get_variables_reply(r):
    global proxSensorsVal
    proxSensorsVal = r


def accget_variables_reply(r):
    global accVal
    accVal = r


def deltaget_variables_reply(r):
    global ground
    ground = r


def tempget_variables_reply(r):
    global temp
    temp = r


def reftaget_variables_reply(r):
    global ref
    ref = r


def ambtaget_variables_reply(r):
    global amb
    amb = r


def micget_variables_reply(r):
    global mic
    mic = r


def get_variables_error(e):
    print 'error:'
    print str(e)
    loop.quit()


def dbusReply():
    pass


def dbusError(e):
    print 'error %s'
    print str(e)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--system", action="store_true", dest="system", default=False,
                      help="use the system bus instead of the session bus")

    (options, args) = parser.parse_args()

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    if options.system:
        bus = dbus.SystemBus()
    else:
        bus = dbus.SessionBus()

    # Create Aseba network
    robot = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')

    # print in the terminal the name of each Aseba NOde
    print robot.GetNodesList()

    robot.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)

    # GObject loop
    # print 'starting loop'
    loop = gobject.MainLoop()
    # call the callback of test algorithm
    handle = gobject.timeout_add(100, Test)  # every 0.1 sec
    loop.run()
