from helpers import *
from neat_task import NEATTask
from CameraVision import *
from peas.networks.rnn import NeuralNetwork

import gobject
import glib
import dbus
import dbus.mainloop.glib

EVALUATIONS = 1000
MAX_MOTOR_SPEED = 300
TIME_STEP = 0.005
ACTIVATION_FUNC = 'tanh'
POPSIZE = 1
GENERATIONS = 100
SOLVED_AT = EVALUATIONS * 2
EXPERIMENT_NAME = 'NEAT_foraging_task'

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')

class ForagingTask(NEATTask):

	def __init__(self, thymioController, debug=False, experimentName='NEAT_task', evaluations=1000, timeStep=0.005, activationFunction='tanh', popSize=1, generations=100, solvedAt=1000):
		NEATTask.__init__(self, thymioController, debug, experimentName, evaluations, timeStep, activationFunction, popSize, generations, solvedAt)
		self.camera = CameraVision(False, self.logger)
		print('Camera initialized')
		
	def _step(self, evaluee, callback):
		presence_box = self.camera.readPuckPresence()
		presence_goal = self.camera.readGoalPresence()
		presence = presence_box + presence_goal
		print('presence: ' + str(presence))

		inputs = np.hstack((presence, 1))

		out = NeuralNetwork(evaluee).feed(inputs)
		print(out)
		left, right = list(out[-2:])
		motorspeed = { 'left': left, 'right': right }
		writeMotorSpeed(self.thymioController, motorspeed)

		callback(self.getFitness(motorspeed, inputs))
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
	genotype = lambda: NEATGenotype(inputs=9, outputs=2, types=[ACTIVATION_FUNC])
	pop = NEATPopulation(genotype, popsize=POPSIZE)

	dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
	bus = dbus.SessionBus()
	thymioController = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')
	thymioController.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)

	# switch thymio LEDs off
	thymioController.SendEventName('SetColor', [0, 0, 0, 0], reply_handler=dbusReply, error_handler=dbusError)

	debug = True
	task = ForagingTask(thymioController, debug, EXPERIMENT_NAME)
	
	def epoch_callback(population):
		current_champ = population.champions[-1]
		print 'Champion: ' + str(current_champ.get_network_data())

		task.getLogger().info(', '.join([str(ind.stats['fitness']) for ind in population.population]))

		# current_champ.visualize(os.path.join(CURRENT_FILE_PATH, 'img/obstacle_avoid_%d.jpg' % population.generation))
		pickle.dump(current_champ, file(os.path.join(PICKLED_DIR, 'obstacle_avoid_%d.txt' % population.generation), 'w'))

	pop.epoch(generations=GENERATIONS, evaluator=task, solution=task, callback=epoch_callback)
