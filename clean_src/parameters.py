# parameters.py created on March 12, 2015. Jacqueline Heinerman & Massimiliano Rango
# modified by Alessandro Zonta on June 25, 2015
import os
import logging

EXPERIMENT_NAME = "WRITE_EXPERIMENT_NAME_HERE"
starter_number = 0

motorspeed = [0, 0]
NB_DIST_SENS = 5
SENSOR_MAX = [4000] * 5
TIME_STEP = 0.05
ps_value = [0.0 for x in range(NB_DIST_SENS)]

# system parameters
evolution = 1                # evolution
sociallearning = 1           # social learning
lifetimelearning = 1         # lifetime learning
threshold = 0.5              # fitness/maximum fitness value to exceed
total_evals = 1000           # one eval is lifetime/social or reevaluation
max_robot_lifetime = 1000    # either 200 or 100
seed = 0                     # experiment seed
real_speed_percentage = 0.3  # percentage of maximum speed used

# reevaluate parameters
eval_time = 2000                    # Evaluation time, in steps
tau = int(eval_time * 0.05)         # Recovery  period tau, in steps -> 5% of evaltime
tau_goal = int(eval_time * 0.25)    # Recovery after goal -> must be longer than normal tau -> 25% of evaltime
re_weight = 0.8                     # part or reevaluation fitness that stays the same

MAX_MOTOR_SPEED = 300
max_fitness = eval_time * 6
real_max_speed = real_speed_percentage * MAX_MOTOR_SPEED

MIN_FPS = 0.2                       # Minimum frame rate for camera updates

# Am I using hidden layer?
hidden_layer = 1

CURRENT_FILE_PATH = os.path.abspath(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(CURRENT_FILE_PATH, 'output')
PICKLED_DIR = os.path.join(CURRENT_FILE_PATH, 'pickled')
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')


