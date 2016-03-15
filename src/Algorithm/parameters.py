# parameters.py created on March 12, 2015. Jacqueline Heinerman & Massimiliano Rango
# modified by Alessandro Zonta on June 25, 2015
import classes

EXPERIMENT_NAME = "WRITE_EXPERIMENT_NAME_HERE"
starter_number = 10

motorspeed = [0, 0]
ps_value = [0.0 for x in range(classes.NB_DIST_SENS)]

# system parameters
evolution = 0                # evolution
sociallearning = 1           # social learning
lifetimelearning = 1         # lifetime learning
threshold = 0.5              # fitness/maximum fitness value to exceed
total_evals = 1000           # one eval is lifetime/social or reevaluation
max_robot_lifetime = 1000    # either 200 or 100
seed = 0                     # experiment seed
real_speed_percentage = 0.6  # percentage of maximum speed used

# memome parameters
range_weights = 4.0           # weights between [-range_weights, range_weights]
collected_memomes_total = 0   # number collected memomes
collected_memomes_max = 20    # max memome memory
sigmainitial = 1.0            # initital sigma
sigma_max = 4.0               # maximum sigma value
sigma_min = 0.01              # min sigma value
sigma_increase = 2.0          # sigma increase after not better solution

# reevaluate parameters
eval_time = 4000                    # Evaluation time, in steps
tau = int(eval_time * 0.005)         # Recovery  period tau, in steps -> 5% of evaltime
tau_goal = 0 #int(eval_time * 0.25)    # Recovery after goal -> must be longer than normal tau -> 25% of evaltime
re_weight = 0.8                     # part or reevaluation fitness that stays the same

real_max_speed = classes.MAXSPEED * real_speed_percentage
max_fitness = eval_time 

# Am I using hidden layer?
hidden_layer = 0


