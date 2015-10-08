# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import gobject
import glib
import parameters as pr
import classes
from Simulation import *
from ThymioController import *
from peas.networks.rnn import NeuralNetwork

EVALUATIONS = 100


class ObstacleAvoidance(Simulation):

	# def __init__(self, thymioController, debug, experiment_name):
	# 	super(ObstacleAvoidance, self).__init__(thymioController, debug, experiment_name)
	def __init__(self, thymioController):
		self.thymioController = thymioController

	def evaluate(self, evaluee):
		self.evaluations_taken = 0
		self.fitness = 0
		self.loop = gobject.MainLoop()
		def update_fitness(task, fit):
			# print 'Doing update_fitness ' + str(task.fitness)
			task.fitness += fit
		def main_lambda(task):
			if task.evaluations_taken == EVALUATIONS:
				stopThymio(thymioController)
				task.loop.quit()
				return False 
			# print 'Doing __step'
			ret_value =  self.__step(evaluee, lambda (fit): update_fitness(self, fit))
			task.evaluations_taken += 1
			return ret_value
		handle = gobject.timeout_add(100, lambda: main_lambda(self))  # every 0.1 sec
		self.loop.run()

		print "Fitness at end: %d" % self.fitness

		# self.thymioController.SendEventName('PlayFreq', [700, 0], reply_handler=dbusReply, error_handler=dbusError)
		# time.sleep(.3)
		# self.thymioController.SendEventName('PlayFreq', [700, -1], reply_handler=dbusReply, error_handler=dbusError)
		# time.sleep(0.1)
		time.sleep(1)

		return { 'fitness': self.fitness }

	def solve(self, evaluee):
		return int(self.evaluate(evaluee)['fitness'] >= EVALUATIONS)

	def __step(self, evaluee, callback):
		# Read sensors: request to ThymioController
		# self.thymioController.readSensorsRequest()
		# self.waitForControllerResponse()
		# psValues = self.thymioController.getPSValues()
		# psValues = np.array([psValues[0], psValues[2], psValues[4], psValues[5], psValues[6]])
		# psValues = np.random.standard_normal(5)
		def ok_call(psValues):
			psValues = np.array([psValues[0], psValues[2], psValues[4], psValues[5], psValues[6], 1])
			# print 'Observation ' + str(psValues)

			left, right = list(NeuralNetwork(evaluee).feed(psValues)[-2:])

			motorspeed = { 'left': left, 'right': right }

			# self.thymioController.writeMotorspeedRequest((left, right))
			# self.waitForControllerResponse()
			# print 'Writing to motor ' + str(motorspeed)
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

		# fitness for 1 timestep in [-1000, 1000]
		return float(motorspeed['left'] + motorspeed['right']) * (1 - speedpenalty) * (1 - sensorpenalty)

class ObstacleAvoidance2(ObstacleAvoidance):

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

		# Only punish if object is really close
		if sensorpenalty < .8:
			sensorpenalty = 0

		# fitness for 1 timestep in [-1000, 1000]
		return float(motorspeed['left'] + motorspeed['right']) * (1 - speedpenalty) * (1 - sensorpenalty)


CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
MAIN_LOG_PATH = os.path.join(CURRENT_FILE_PATH, 'log_main')
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')

def getNextIDPath(path):
	nextID = 0
	filelist = sorted(os.listdir(path))
	if filelist and filelist[-1][0].isdigit():
		nextID = int(filelist[-1][0]) + 1
	return str(nextID)

def writeMotorSpeed(controller, motorspeed):
	speed_constant = 100
	controller.SetVariable("thymio-II", "motor.left.target", [motorspeed['left'] * speed_constant])
	controller.SetVariable("thymio-II", "motor.right.target", [motorspeed['right'] * speed_constant])


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
	genotype = lambda: NEATGenotype(inputs=6, outputs=2, types=['sigmoid'])
	pop = NEATPopulation(genotype, popsize=20)

	# Main logger for ThymioController and CommandsListener
	# mainLogger = logging.getLogger('mainLogger')
	# mainLogger.setLevel(logging.DEBUG)
	# mainLogFilename = getNextIDPath(MAIN_LOG_PATH) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '_main_debug.log'
	# mainHandler = logging.FileHandler(os.path.join(MAIN_LOG_PATH, mainLogFilename))
	# mainHandler.setFormatter(FORMATTER)
	# mainLogger.addHandler(mainHandler)

	# gobject.threads_init()
	# dbus.mainloop.glib.threads_init()

	# thymioController = ThymioController(mainLogger)
	# thymioController.run()
	dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
	bus = dbus.SessionBus()
	thymioController = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')
	# print thymioController.GetNodesList()
	thymioController.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)

	# debug = True
	# experiment_name = 'Experiment 001'
	# task = ObstacleAvoidance(thymioController, debug, experiment_name)
	task = ObstacleAvoidance2(thymioController)
	
	pop.epoch(generations=100, evaluator=task, solution=task)
