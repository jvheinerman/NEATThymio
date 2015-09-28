import numpy as np
import parameters as pr
import classes
from Simulation import *


class ObstacleAvoidance(Simulation):

	def __init__(self, arg):
		super(ObstacleAvoidance, self).__init__()
		self.arg = arg

	def evaluate(self, evaluee):
		fitness = 0
		for i in range(0, classes.total_eval):
			fitness += __runAndEvaluateForOneTimeStep(evaluee)
			print "Fitness at time %d: %d" % (i, fitness)

		return { 'fitness': fitness }

	def __runAndEvaluateForOneTimeStep(self, evaluee):
		# Read sensors: request to ThymioController
		self.__thymioController.readSensorsRequest()
		self.__waitForControllerResponse()
		psValues = self.__thymioController.getPSValues()

		psValues = np.array([psValues[0], psValues[2], psValues[4], psValues[5], psValues[6]])

		motorspeed = evaluee.feed(psValues)

		return getFitness(motorspeed, psValues)

	def getFitness(self, motorspeed, observation):
		# Calculate penalty for rotating
		speedpenalty = 0
		if motorspeed[LEFT] > motorspeed[RIGHT]:
			speedpenalty = float((motorspeed[LEFT] - motorspeed[RIGHT])) / float(pr.real_maxspeed)
		else:
			speedpenalty = float((motorspeed[RIGHT] - motorspeed[LEFT])) / float(pr.real_maxspeed)
		if speedpenalty > 1:
			speedpenalty = 1

		# Calculate normalized distance to the nearest object
		sensorpenalty = 0
		for sensor in observation:
			distance = sensor / float(classes.SENSOR_MAX)
			if sensorpenalty < distance:
				sensorpenalty = distance
		if sensorpenalty > 1:
			sensorpenalty = 1

		# fitness for 1 timestep in [-1000, 1000]
		return float(motorspeed[LEFT] + motorspeed[RIGHT]) * (1 - speedpenalty) * (1 - sensorpenalty)
