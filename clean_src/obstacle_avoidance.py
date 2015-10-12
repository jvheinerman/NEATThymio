# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import gobject
import glib
import dbus
import dbus.mainloop.glib
import logging
import pickle
import parameters as pr
import classes
from helpers import *
from peas.networks.rnn import NeuralNetwork

EVALUATIONS = 1000
MAX_MOTOR_SPEED = 300
TIME_STEP = 0.005
ACTIVATION_FUNC = 'tanh'
POPSIZE = 1
GENERATIONS = 100
SOLVED_AT = EVALUATIONS * 2
EXPERIMENT_NAME = 'NEAT_obstacle_avoidance'

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
MAIN_LOG_PATH = os.path.join(CURRENT_FILE_PATH, 'log_main')
OUTPUT_PATH = os.path.join(CURRENT_FILE_PATH, 'output')
PICKLED_DIR = os.path.join(CURRENT_FILE_PATH, 'pickled')
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')


class NEATTask:
	def __init__(self, thymioController, debug = False, experimentName = 'ObstacleAvoidance'):
		self.thymioController = thymioController
		self.logger = logging.getLogger('simulationLogger')
		logLevel = logging.INFO
		if debug:
			logLevel = logging.DEBUG
		self.logger.setLevel(logLevel)
		outputDir = os.path.join(OUTPUT_PATH, experimentName)
		mkdir_p(outputDir)
		mkdir_p(PICKLED_DIR)
		logFilename = os.path.join(outputDir, experimentName + '_sim_debug.log')
		simHandler = logging.FileHandler(logFilename)
		simHandler.setFormatter(FORMATTER)
		self.logger.addHandler(simHandler)

	def evaluate(self, evaluee):
		raise NotImplemented('Evaluate method not implemented')

	def solve(self, evaluee):
		raise NotImplemented('Evaluate method not implemented')

	def getFitness(self, motorspeed, observation):
		raise NotImplemented('Evaluate method not implemented')

	def getLogger(self):
		return self.logger

class ObstacleAvoidance(NEATTask):

	def evaluate(self, evaluee):
		self.evaluations_taken = 0
		self.fitness = 0
		self.loop = gobject.MainLoop()
		def update_fitness(task, fit):
			task.fitness += fit
		def main_lambda(task):
			if task.evaluations_taken == EVALUATIONS:
				stopThymio(thymioController)
				task.loop.quit()
				return False 
			ret_value =  self.__step(evaluee, lambda (fit): update_fitness(self, fit))
			task.evaluations_taken += 1
			# time.sleep(TIME_STEP)
			return ret_value
		gobject.timeout_add(int(TIME_STEP * 1000), lambda: main_lambda(self))
		# glib.idle_add(lambda: main_lambda(self))
		self.loop.run()

		fitness = max(self.fitness, 1)
		print 'Fitness at end: %d' % fitness

		# self.thymioController.SendEventName('PlayFreq', [700, 0], reply_handler=dbusReply, error_handler=dbusError)
		# time.sleep(.3)
		# self.thymioController.SendEventName('PlayFreq', [700, -1], reply_handler=dbusReply, error_handler=dbusError)
		# time.sleep(0.1)
		time.sleep(1)

		return { 'fitness': fitness }

	def solve(self, evaluee):
		return int(self.evaluate(evaluee)['fitness'] >= SOLVED_AT)

	def __step(self, evaluee, callback):
		def ok_call(psValues):
			psValues = np.array([psValues[0], psValues[2], psValues[4], psValues[5], psValues[6], 1])

			left, right = list(NeuralNetwork(evaluee).feed(psValues)[-2:])

			motorspeed = { 'left': left, 'right': right }

			writeMotorSpeed(self.thymioController, motorspeed)

			callback(self.getFitness(motorspeed, psValues))

		def nok_call():
			print " Error while reading proximity sensors"

		getProxReadings(self.thymioController, ok_call, nok_call)
		return True

	def getFitness(self, motorspeed, observation):
		# Calculate penalty for rotating
		speedpenalty = 0
		if motorspeed['left'] > motorspeed['right']:
			speedpenalty = float((motorspeed['left'] - motorspeed['right'])) / float(pr.real_max_speed)
		else:
			speedpenalty = float((motorspeed['right'] - motorspeed['left'])) / float(pr.real_max_speed)
		if speedpenalty > 1:
			speedpenalty = 1

		# Calculate normalized distance to the nearest object
		sensorpenalty = 0
		for i, sensor in enumerate(observation[:-1]):
			distance = sensor / float(classes.SENSOR_MAX[i])
			if sensorpenalty < distance:
				sensorpenalty = distance
		if sensorpenalty > 1:
			sensorpenalty = 1

		# fitness for 1 timestep in [-2, 2]
		return float(motorspeed['left'] + motorspeed['right']) * (1 - speedpenalty) * (1 - sensorpenalty)



def getNextIDPath(path):
	nextID = 0
	filelist = sorted(os.listdir(path))
	if filelist and filelist[-1][0].isdigit():
		nextID = int(filelist[-1][0]) + 1
	return str(nextID)

def writeMotorSpeed(controller, motorspeed):
	controller.SetVariable("thymio-II", "motor.left.target", [motorspeed['left'] * MAX_MOTOR_SPEED])
	controller.SetVariable("thymio-II", "motor.right.target", [motorspeed['right'] * MAX_MOTOR_SPEED])


def getProxReadings(controller, ok_callback, nok_callback):
	controller.GetVariable("thymio-II", "prox.horizontal", reply_handler=ok_callback, error_handler=nok_callback)

def stopThymio(controller):
	writeMotorSpeed(controller, { 'left': 0, 'right': 0 })

def dbusReply():
    pass


def dbusError(e):
    print 'error %s' % str(e)

if __name__ == '__main__':
	from peas.methods.neat import NEATPopulation, NEATGenotype
	genotype = lambda: NEATGenotype(inputs=6, outputs=2, types=[ACTIVATION_FUNC])
	pop = NEATPopulation(genotype, popsize=POPSIZE)

	dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
	bus = dbus.SessionBus()
	thymioController = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')
	thymioController.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)

	# switch thymio LEDs off
	thymioController.SendEventName('SetColor', [0, 0, 0, 0], reply_handler=dbusReply, error_handler=dbusError)

	debug = True
	task = ObstacleAvoidance(thymioController, debug, EXPERIMENT_NAME)
	
	def epoch_callback(population):
		current_champ = population.champions[-1]
		print 'Champion: ' + str(current_champ.get_network_data())

		task.getLogger().info(', '.join([str(ind.stats['fitness']) for ind in population.population]))

		# current_champ.visualize(os.path.join(CURRENT_FILE_PATH, 'img/obstacle_avoid_%d.jpg' % population.generation))
		pickle.dump(current_champ, file(os.path.join(PICKLED_DIR, 'obstacle_avoid_%d.txt' % population.generation), 'w'))

	pop.epoch(generations=GENERATIONS, evaluator=task, solution=task, callback=epoch_callback)
