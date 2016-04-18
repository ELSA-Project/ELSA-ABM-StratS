#!/usr/bin/env python
# -*- coding: utf-8 -*-

version='2.6.1'

import pickle

list_files=['ABMvars.py', 'iter_simO.py',  'simAirSpaceO.py', 'simulationO.py', 'plots.py', \
		'prepare_network.py', 'performance_plots.py', 'utilities.py',\
		 'list_of_files.py', 'study_overlap_and_swap.py']

with open('list_of_files.pic', 'w') as f:
	pickle.dump(list_files, f)
