import picamera
import cv2
import numpy as np
import sys, time, io, os
import dbus
import dbus.mainloop.glib
from helpers import *

try:
    CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
    AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()
    thymioController = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')
    thymioController.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)

    # switch thymio LEDs off
    thymioController.SendEventName('SetColor', [0, 0, 0, 0], reply_handler=dbusReply, error_handler=dbusError)

    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        camera.framerate = 30

        camera.start_preview()
        time.sleep(2)

        # capture into stream
        stream = io.BytesIO()
        counter = 0

        # if raw_input() == 'q':
        #     sys.exit(0)
        print "starting"

        for foo in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
            data = np.fromstring(stream.getvalue(), dtype=np.uint8)
            # "Decode" the image from the array, preserving colour
            image = cv2.imdecode(data, 1)

            # Convert BGR to HSV
            image = cv2.GaussianBlur(image, (5, 5), 0)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            filename = "camera/hsv_" + str(counter) + ".jpg"
            cv2.imwrite(filename, hsv)

            print "Done file %s" % filename

            stream.truncate()
            stream.seek(0)
            counter += 1

            time.sleep(3)

except Exception as e:
    print e
