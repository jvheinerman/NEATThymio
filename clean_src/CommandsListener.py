import sys
import traceback
import threading
import socket
import struct
import pickle

import parameters as pr
from Simulation import *
from MessageReceiver import *

COMMANDS_LISTENER_HOST = ''
COMMANDS_LISTENER_PORT = 54321
LOCALHOST = '127.0.0.1'
PC_FIXED_IP = '192.168.1.100'
TRUSTED_CLIENTS = [LOCALHOST, PC_FIXED_IP]


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
                recvOptions = MessageReceiver.recvOneMessage(conn)
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
