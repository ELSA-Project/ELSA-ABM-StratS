#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1,'..')
sys.path.insert(1,'../abm_strategic_model1') # For import of G!
import os
from os.path import join as jn
import pickle

from libs.paths import main_dir, result_dir
from abm_strategic_model1.iter_simO import iter_airport_change, iter_sim
from abm_strategic_model1.utilities import read_paras
from abm_strategic_model1.performance_plots import get_results

if __name__=='__main__':
	if 1:
		paras_G_file = jn(main_dir, 'abm_strategic_model1/my_paras/my_paras_G_iter_DiskWorld.py')
		file_net = jn(result_dir, 'networks/DiskWorld/DiskWorld.pic')
		with open(file_net, 'r') as f:
			G = pickle.load(f)

		paras_G_iter = read_paras(paras_file=paras_G_file,
								  post_process=False)

		iter_airport_change(paras_G_iter, G)

	if 1:
		paras_file = jn(main_dir, 'abm_strategic_model1/my_paras/my_paras_iter_for_DiskWorld.py')
		paras_iter = read_paras(paras_file=paras_file,
								  post_process=True)

		iter_sim(paras_iter)

	# Save paras files 
	rep = jn(result_dir, 'model1/DiskWorld/consolidated')
	os.system('mkdir -p '+ rep)
	# Save iter paras of network builder
	os.system('cp ' + paras_G_file + ' ' + rep + '/')
	# Save paras of network builder
	os.system('cp ' + paras_G_iter['paras_file'] + ' ' + rep + '/')
	# Save iter paras of simulation
	os.system('cp ' + paras_file + ' ' + rep + '/')
	# Save paras of simulation
	os.system('cp ' + paras_iter['paras_file'] + ' ' + rep + '/')

	# Gather results and dump them on disk for external plotting
	results, results_global = get_results(paras_iter, Gname='DiskWorld')

	with open(jn(rep, 'results.pic'), 'w') as f:
		pickle.dump(results, f)

	with open(jn(rep, 'results_global.pic'), 'w') as f:
		pickle.dump(results_global, f)

	print "Files saved in:", rep