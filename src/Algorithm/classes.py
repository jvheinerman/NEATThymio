# classes.py created on March 12, 2015. Jacqueline Heinerman & Massimiliano Rango
# modified by Alessandro Zonta on June 25, 2015

# define global variables
global NB_DIST_SENS         # number of proximity sensors
global NB_CAM_SENS          # number of camera imput image
global NB_SENS              # total number of sensor used

global TIME_STEP            # time step of the simulation in seconds

global NN_WEIGHTS           # number of weights in the first NN = (15*2)+ 2 = 32 first layer
global NN_WEIGHTS_HIDDEN    # number of weights in the second layer NN

global NN_WEIGHTS_NO_HIDDEN # number of weights in NN without hidden layer

global HIDDEN_NEURONS       # number of hidden neurons
global TOTAL_WEIGHTS        # number of weights

global MAXSPEED             # maximum motor speed
global SENSOR_MAX           # max sensor value
global CAMERA_MAX           # max camera value

global SOUND                # sound emitted at goal

NB_DIST_SENS = 5  # 7
NB_CAM_SENS = 0  # 4 for pack color and 3 for target color

HIDDEN_NEURONS = 0

NN_WEIGHTS = (NB_DIST_SENS + NB_CAM_SENS + 1) * HIDDEN_NEURONS  # (5 + 4 + 1) * 4 = 40
NN_WEIGHTS_HIDDEN = HIDDEN_NEURONS * 2 + 2  # 4 * 2 + 2 = 10
TOTAL_WEIGHTS = NN_WEIGHTS + NN_WEIGHTS_HIDDEN  # 40 + 10 = 50

NN_WEIGHTS_NO_HIDDEN = (NB_DIST_SENS + NB_CAM_SENS + 1) * 2  # (5 + 4 + 1) * 2 = 20

MAXSPEED = 500
SENSOR_MAX = [3000, 3000, 3000, 3000, 3000]  # 4500  # XXX: found sensor with max value of 5100
TIME_STEP = 0.05  # = 50 milliseconds. IMPORTANT: keep updated with TIME_STEP constant in asebaCommands.aesl
CAMERA_MAX = [7000, 7000, 9000, 11000, 20000, 20000, 20000, 20000]  # left, right, center, bottom (red and black)

SOUND = 5


class Candidate(object):
    def __init__(self, memome, fitness, sigma):
        self.memome = memome
        self.fitness = fitness
        self.sigma = sigma


class RobotMemomeDataMessage(object):
    def __init__(self, fitness, memome):
        self.fitness = fitness
        self.memome = memome
