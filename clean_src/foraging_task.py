# -*- coding: utf-8 -*-

from helpers import *
from parameters import *
from neat_task import NEATTask
from CameraVision import *
import classes as cl
from peas.networks.rnn import NeuralNetwork

import gobject
import glib
import dbus
import dbus.mainloop.glib
import time

EVALUATIONS = 1000
MAX_MOTOR_SPEED = 300
TIME_STEP = 0.005
ACTIVATION_FUNC = 'tanh'
POPSIZE = 20
GENERATIONS = 100
SOLVED_AT = EVALUATIONS * 2
EXPERIMENT_NAME = 'NEAT_foraging_task'

INITIAL_ENERGY = 500
MAX_ENERGY = 1000
ENERGY_DECAY = 1

PUCK_BONUS_SCALE = 5
GOAL_BONUS_SCALE = 5
GOAL_REACHED_BONUS = INITIAL_ENERGY

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')

class ForagingTask(NEATTask):

    def __init__(self, thymioController, debug=False, experimentName='NEAT_task', evaluations=1000, timeStep=0.005, activationFunction='tanh', popSize=1, generations=100, solvedAt=1000):
        NEATTask.__init__(self, thymioController, debug, experimentName, evaluations, timeStep, activationFunction, popSize, generations, solvedAt)
        self.camera = CameraVisionVectors(False, self.logger)
        
    def _step(self, evaluee, callback):
        presence_box = self.camera.readPuckPresence()
        presence_goal = self.camera.readGoalPresence()

        # print presence_box, presence_goal
        if presence_goal and presence_box:
            self.prev_presence = self.presence
            self.presence = tuple(presence_box) + tuple(presence_goal)

            inputs = np.hstack(([x if not x == -np.inf else -10000 for x in self.presence], 1))
            inputs[::2] = inputs[::2] / self.camera.MAX_DISTANCE

            out = NeuralNetwork(evaluee).feed(inputs)
            left, right = list(out[-2:])
            motorspeed = { 'left': left, 'right': right }
            writeMotorSpeed(self.thymioController, motorspeed)
        else:
            time.sleep(.1)

        callback(self.getEnergyDelta())
        return True

    def getFitness(self):
        return max(self.evaluations_taken + self.energy, 1)

    def getEnergyDelta(self):
        self.presence = [x if not x == -np.inf else self.camera.MAX_DISTANCE for x in self.presence]
        self.prev_presence = [x if not x == -np.inf else self.camera.MAX_DISTANCE for x in self.prev_presence]

        if None in self.presence or None in self.prev_presence:
            return ENERGY_DECAY

        energy_delta = PUCK_BONUS_SCALE * (self.prev_presence[0] - self.presence[0])
        
        # print self.presence
        if self.presence[0] == 0:
            energy_delta = GOAL_BONUS_SCALE * (self.prev_presence[2] - self.presence[2])

        if self.camera.goal_reached():
            print '===== Goal reached!'
            stopThymio(self.thymioController)

            while self.camera.readPuckPresence()[0] == 0:
                print '===== Make sound'
                self.thymioController.SendEventName('PlayFreq', [700, 0], reply_handler=dbusReply, error_handler=dbusError)
                time.sleep(.3)
                self.thymioController.SendEventName('PlayFreq', [0, -1], reply_handler=dbusReply, error_handler=dbusError)
                time.sleep(.7)
            
            print '===== Exiting goal reached block'
            time.sleep(1)
            energy_delta = GOAL_REACHED_BONUS

        if energy_delta: print('Energy delta %d' % energy_delta)
        
        return energy_delta

    def evaluate(self, evaluee):
        self.evaluations_taken = 0
        self.energy = INITIAL_ENERGY
        self.fitness = 0
        self.presence = self.prev_presence = (None, None)
        self.loop = gobject.MainLoop()
        def update_energy(task, energy):
            task.energy += energy
        def main_lambda(task):
            if task.energy <= 0:
                stopThymio(thymioController)
                task.loop.quit()
                print 'Energy exhausted'
                return False 
            ret_value =  task._step(evaluee, lambda (energy): update_energy(task, energy))
            task.evaluations_taken += 1
            task.energy -= ENERGY_DECAY
            # time.sleep(TIME_STEP)
            return ret_value
        gobject.timeout_add(int(self.timeStep * 1000), lambda: main_lambda(self))
        # glib.idle_add(lambda: main_lambda(self))
        
        print 'Starting camera...'
        try:
            self.camera = CameraVisionVectors(False, self.logger)
            self.camera.start()
            # time.sleep(2)
        except RuntimeError, e:
            print 'Camera already started!'
        
        print 'Starting loop...'
        self.loop.run()

        fitness = self.getFitness()
        print 'Fitness at end: %d' % fitness

        stopThymio(self.thymioController)

        self.camera.stop()
        self.camera.join()
        time.sleep(1)

        return { 'fitness': fitness }


if __name__ == '__main__':
    from peas.methods.neat import NEATPopulation, NEATGenotype
    genotype = lambda: NEATGenotype(inputs=5, outputs=2, types=[ACTIVATION_FUNC])
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
