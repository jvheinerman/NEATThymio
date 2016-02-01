#!/usr/bin/env python

import json
import sys

log = json.load(open(sys.argv[1], 'r'))

for g in log['generations']:
	individuals = g['individuals']
	avg_generation = sum((individual['stats']['fitness'] for individual in individuals)) / len(individuals)
	print 'Generation', g['gen_number']
	print 'Avg.', avg_generation
	all_fitnesses = [individual['stats']['fitness'] for individual in individuals]
	print 'Max', max(all_fitnesses)
	print 'Min', min(all_fitnesses)
	print ''
