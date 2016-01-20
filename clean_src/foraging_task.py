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
from copy import deepcopy
import json
import time
import sys
import socket
import thread

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
MAX_STEPS = 10000

PUCK_BONUS_SCALE = 3
GOAL_BONUS_SCALE = 3
GOAL_REACHED_BONUS = INITIAL_ENERGY

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
AESL_PATH = os.path.join(CURRENT_FILE_PATH, 'asebaCommands.aesl')

class ForagingTask(NEATTask):

    def __init__(self, thymioController, commit_sha, debug=False, experimentName='NEAT_task', evaluations=1000, timeStep=0.005, activationFunction='tanh', popSize=1, generations=100, solvedAt=1000):
        NEATTask.__init__(self, thymioController, commit_sha, debug, experimentName, evaluations, timeStep, activationFunction, popSize, generations, solvedAt)
        self.camera = CameraVisionVectors(False, self.logger)
        self.ctrl_thread_started = False
        self.img_thread_started = False
        self.individuals_evaluated = 0
        
    def _step(self, evaluee, callback):
        presence_box = self.camera.readPuckPresence()
        presence_goal = self.camera.readGoalPresence()

        # print presence_box, presence_goal
        if presence_goal and presence_box:
            # self.frame_rate_counter += 1
            # print 'Camera test', self.frame_rate_counter

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
        global img_client
        if img_client and not self.camera.merged_binary is None:
            send_image(img_client, self.camera.merged_binary)

        self.presence = [x if not x == -np.inf else self.camera.MAX_DISTANCE for x in self.presence]
        self.prev_presence = [x if not x == -np.inf else self.camera.MAX_DISTANCE for x in self.prev_presence]

        if None in self.presence or None in self.prev_presence:
            return 0

        if self.presence[0] == 0 and self.prev_presence[0] != 0:
            self.getLogger().info(str(self.individuals_evaluated) + ' > Found puck')
        elif self.presence[0] != 0 and self.prev_presence[0] == 0:
            self.getLogger().info(str(self.individuals_evaluated) + ' > Lost puck')

        energy_delta = PUCK_BONUS_SCALE * (self.prev_presence[0] - self.presence[0])
        
        # print self.presence
        if self.presence[0] == 0:
            energy_delta = GOAL_BONUS_SCALE * (self.prev_presence[2] - self.presence[2])

        if self.camera.goal_reached() and self.presence[2] < 40:
            print '===== Goal reached!'
            stopThymio(self.thymioController)

            while self.camera.readPuckPresence()[0] == 0:
                self.thymioController.SendEventName('PlayFreq', [700, 0], reply_handler=dbusReply, error_handler=dbusError)
                time.sleep(.3)
                self.thymioController.SendEventName('PlayFreq', [0, -1], reply_handler=dbusReply, error_handler=dbusError)
                time.sleep(.7)
            
            time.sleep(1)
            energy_delta = GOAL_REACHED_BONUS

            self.getLogger().info(str(self.individuals_evaluated) + ' > Goal reached')

        # if energy_delta: print('Energy delta %d' % energy_delta)
        
        return energy_delta

    def evaluate(self, evaluee):
        self.frame_rate_counter = 0

        global ctrl_client
        if ctrl_client and not self.ctrl_thread_started:
            thread.start_new_thread(check_stop, (self, ))
            self.ctrl_thread_started = True

        self.evaluations_taken = 0
        self.energy = INITIAL_ENERGY
        self.fitness = 0
        self.presence = self.prev_presence = (None, None)
        self.loop = gobject.MainLoop()
        def update_energy(task, energy):
            task.energy += energy
        def main_lambda(task):
            if task.energy <= 0 or task.evaluations_taken >= MAX_STEPS:
                stopThymio(thymioController)
                task.loop.quit()
                
                if task.energy <= 0:
                    print 'Energy exhausted'
                else:
                    print 'Time exhausted'
                
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

        self.individuals_evaluated += 1

        return { 'fitness': fitness }

    def exit(self, value = 0):
        print 'Exiting...'
        # sys.exit(value)
        self.loop.quit()
        thread.interrupt_main()


def check_stop(task):
    global ctrl_client
    f = ctrl_client.makefile()
    line = f.readline()
    if line.startswith('stop'):
        release_resources(task.thymioController)
        task.exit(0)
    task.ctrl_thread_started = False

def release_resources(thymio):
    global ctrl_serversocket
    global ctrl_client
    ctrl_serversocket.close()
    if ctrl_client: ctrl_client.close()
    
    global img_serversocket
    global img_client
    img_serversocket.close()
    if img_client: img_client.close()

    stopThymio(thymio)

def write_header(client, boundary='thymio'):
    client.send("HTTP/1.0 200 OK\r\n" +
            "Connection: close\r\n" +
            "Max-Age: 0\r\n" +
            "Expires: 0\r\n" +
            "Cache-Control: no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0\r\n" +
            "Pragma: no-cache\r\n" +
            "Content-Type: multipart/x-mixed-replace; " +
            "boundary=" + boundary + "\r\n" +
            "\r\n" +
            "--" + boundary + "\r\n")

def send_image(client, image, boundary='thymio'):
    _, encoded = cv2.imencode('.png', image)
    image_bytes = bytearray(np.asarray(encoded))
    client.send("Content-type: image/png\r\n")
    client.send("Content-Length: %d\r\n\r\n" % len(image_bytes))
    client.send(image_bytes)
    client.send("\r\n--" + boundary + "\r\n")


if __name__ == '__main__':
    from peas.methods.neat import NEATPopulation, NEATGenotype
    genotype = lambda: NEATGenotype(inputs=5, outputs=2, types=[ACTIVATION_FUNC])
    pop = NEATPopulation(genotype, popsize=POPSIZE)

    log = { 'neat': {}, 'generations': [] }

    # log neat settings
    dummy_individual = genotype()
    log['neat'] = {
        'evaluations': EVALUATIONS,
        'activation_function': ACTIVATION_FUNC,
        'popsize': POPSIZE,
        'generations': GENERATIONS,
        'initial_energy': INITIAL_ENERGY,
        'max_energy': MAX_ENERGY,
        'energy_decay': ENERGY_DECAY,
        'max_steps': MAX_STEPS,
        'pucl_bonus_scale': PUCK_BONUS_SCALE,
        'goal_bonus_scale': GOAL_BONUS_SCALE,
        'goal_reached_bonus': GOAL_REACHED_BONUS,
        'elitism': pop.elitism,
        'tournament_selection_k': pop.tournament_selection_k,
        'initial_weight_stdev': dummy_individual.initial_weight_stdev,
        'prob_add_node': dummy_individual.prob_add_node,
        'prob_add_conn': dummy_individual.prob_add_conn,
        'prob_mutate_weight': dummy_individual.prob_mutate_weight,
        'prob_reset_weight': dummy_individual.prob_reset_weight,
        'prob_reenable_conn': dummy_individual.prob_reenable_conn,
        'prob_disable_conn': dummy_individual.prob_disable_conn,
        'prob_reenable_parent': dummy_individual.prob_reenable_parent,
        'weight_range': dummy_individual.weight_range
    }

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()
    thymioController = dbus.Interface(bus.get_object('ch.epfl.mobots.Aseba', '/'), dbus_interface='ch.epfl.mobots.AsebaNetwork')
    thymioController.LoadScripts(AESL_PATH, reply_handler=dbusReply, error_handler=dbusError)

    # switch thymio LEDs off
    thymioController.SendEventName('SetColor', [0, 0, 0, 0], reply_handler=dbusReply, error_handler=dbusError)

    debug = True
    commit_sha = sys.argv[-1]
    task = ForagingTask(thymioController, commit_sha, debug, EXPERIMENT_NAME)

    ctrl_serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ctrl_serversocket.bind((sys.argv[-2], 1337))
    ctrl_serversocket.listen(5)
    ctrl_client = None
    def set_client():
        global ctrl_client
        print 'Control server: waiting for socket connections...'
        (ctrl_client, address) = ctrl_serversocket.accept()
        print 'Control server: got connection from', address
    thread.start_new_thread(set_client, ())

    img_serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    img_serversocket.bind((sys.argv[-2], 31337))
    img_serversocket.listen(5)
    img_client = None
    def set_img_client():
        global img_client
        print 'Image server: waiting for socket connections...'
        (img_client, address) = img_serversocket.accept()
        print 'Image server: got connection from', address
        write_header(img_client)
    thread.start_new_thread(set_img_client, ())

    def epoch_callback(population):
        # update log
        generation = { 'individuals': [], 'gen_number': population.generation }
        for individual in population.population:
            copied_connections = { str(key): value for key, value in individual.conn_genes.items() }
            generation['individuals'].append({
                'node_genes': deepcopy(individual.node_genes),
                'conn_genes': copied_connections,
                'stats': deepcopy(individual.stats)
            })
        champion_file = task.experimentName + '_{}_{}.p'.format(commit_sha, population.generation)
        generation['champion_file'] = champion_file
        generation['species'] = [len(species.members) for species in population.species]
        log['generations'].append(generation)

        task.getLogger().info(', '.join([str(ind.stats['fitness']) for ind in population.population]))
        jsonLog = open(task.jsonLogFilename, "w")
        json.dump(log, jsonLog)
        jsonLog.close()

        current_champ = population.champions[-1]
        # print 'Champion: ' + str(current_champ.get_network_data())
        # current_champ.visualize(os.path.join(CURRENT_FILE_PATH, 'img/' + task.experimentName + '_%d.jpg' % population.generation))
        pickle.dump(current_champ, file(os.path.join(PICKLED_DIR, champion_file), 'w'))

    try:
        pop.epoch(generations=GENERATIONS, evaluator=task, solution=task, callback=epoch_callback)
    except KeyboardInterrupt:
        release_resources(task.thymioController)
        sys.exit(1)

    release_resources(task.thymioController)
    sys.exit(0)
