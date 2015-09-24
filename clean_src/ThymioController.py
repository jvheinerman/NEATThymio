import dbus
import dbus.mainloop.glib
import glib
import gobject
import time
import os
import threading

import classes as cl
import CommandsListener

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')

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
