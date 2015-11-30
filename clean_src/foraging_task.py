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

INITIAL_ENERGY = 500
MAX_ENERGY = 1000
ENERGY_DECAY = 1

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
		self.presence = presence_box + presence_goal

		inputs = np.hstack((presence, 1))

		out = NeuralNetwork(evaluee).feed(inputs)
		left, right = list(out[-2:])
		motorspeed = { 'left': left, 'right': right }
		writeMotorSpeed(self.thymioController, motorspeed)

		callback(self.getEnergyDelta())
		return True

	def getFitness(self, motorspeed, observation):
		return max(self.evaluations_taken + self.energy, 1)

	def getEnergyDelta(self):
		print(self.presence)
		return fitness

	def evaluate(self, evaluee):
		self.evaluations_taken = 0
		self.energy = INITIAL_ENERGY
		self.fitness = 0
		self.loop = gobject.MainLoop()
		def update_energy(task, energy):
			task.energy += energy
		def main_lambda(task):
			if task.energy <= 0:
				stopThymio(thymioController)
				task.loop.quit()
				return False 
			ret_value =  self._step(evaluee, lambda (energy): update_energy(self, energy))
			task.evaluations_taken += 1
			task.energy -= ENERGY_DECAY
			# time.sleep(TIME_STEP)
			return ret_value
		gobject.timeout_add(int(self.timeStep * 1000), lambda: main_lambda(self))
		# glib.idle_add(lambda: main_lambda(self))
		self.loop.run()

		fitness = getFitness()
		print 'Fitness at end: %d' % fitness

		time.sleep(1)

		return { 'fitness': fitness }


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
