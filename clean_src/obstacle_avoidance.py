# -*- coding: utf-8 -*-

from helpers import *
from parameters import *
from neat_task import NEATTask
from CameraVision import *
import classes as cl
from peas.networks.rnn import NeuralNetwork

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
TIME_STEP = 0.005
ACTIVATION_FUNC = 'tanh'
POPSIZE = 5
GENERATIONS = 100
SOLVED_AT = EVALUATIONS * 2
EXPERIMENT_NAME = 'NEAT_obstacle_avoidance'

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
MAIN_LOG_PATH = os.path.join(CURRENT_FILE_PATH, 'log_main')
OUTPUT_PATH = os.path.join(CURRENT_FILE_PATH, 'output')
PICKLED_DIR = os.path.join(CURRENT_FILE_PATH, 'pickled')
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')


class ObstacleAvoidance(NEATTask):

	def _step(self, evaluee, callback):
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
