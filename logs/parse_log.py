#!/usr/bin/env python

import json
import sys

log = json.load(open(sys.argv[1], 'r'))

for i, g in enumerate(log['generations']):
	individuals = g['individuals']
	avg_generation = sum((individual['stats']['fitness'] for individual in individuals)) / len(individuals)
	print i, avg_generation
