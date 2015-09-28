import copy
import sys
import os
import errno
import random
import math
import time
import logging
import traceback
import threading
import socket
import struct
import pickle

import parameters as pr
import classes as cl
from ConfigParser import *
from CameraVision import *
from Inbox import *
from ConnectionsListener import *
from MessageSender import *
from MessageReceiver import *
from helpers import *

LEFT = 0
RIGHT = 1

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(CURRENT_FILE_PATH, 'config.json')
OUTPUT_PATH = os.path.join(CURRENT_FILE_PATH, 'output')

FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
LOCALHOST = '127.0.0.1'
PC_FIXED_IP = '192.168.1.100'
TRUSTED_CLIENTS = [LOCALHOST, PC_FIXED_IP]
COMMANDS_LISTENER_HOST = ''
COMMANDS_LISTENER_PORT = 54321
OUTPUT_FILE_RECEIVER_PORT = 23456  # 23456  # 24537
EOF_REACHED = 'EOF_REACHED'


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
        self.__threadCamera = CameraVision(False, self.__simLogger)

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

    def _runNetwork(self, evaluee):
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

        return motorspeed

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

        motorspeed = self._runNetwork(evaluee)

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

                        self.__threadCamera = CameraVision(False, self.__simLogger)

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
