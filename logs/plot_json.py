#!/usr/bin/env python2
import json
import sys
import matplotlib.pyplot as plt
import numpy as np

log = json.load(open(sys.argv[-1], "r"))

mean = []
stdev = []
maximum = []
for generation in log['generations']:

    fitnesses = []
    for individual in generation['individuals']:
        fitnesses.append(individual['stats']['fitness'])

    mean.append(np.mean(fitnesses))
    stdev.append(np.std(fitnesses))
    maximum.append(np.max(fitnesses))

rng = range(len(mean))

bounds = ((m - s, m + s) for m, s in zip(mean, stdev))
ymin, ymax = zip(*bounds)
plt.plot(rng, maximum, label="fitness", linewidth=1.0, color='k')
plt.plot(rng, mean, label="fitness", linewidth=1.0, color='b')
plt.fill_between(rng, ymax, ymin, alpha=.3, edgecolor="w", color='b')
plt.show()
