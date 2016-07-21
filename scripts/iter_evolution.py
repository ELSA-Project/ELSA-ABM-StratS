#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1,'..')
sys.path.insert(1,'../abm_strategic_model1')
import os
from os.path import join as jn
import pickle

from abm_strategic_model1.evolution import iter_evolution, get_results
from abm_strategic_model1.iter_simO import iter_airport_change
from libs.paths import main_dir, result_dir
from abm_strategic_model1.utilities import read_paras
from libs.general_tools import send_email_when_ready

if __name__=='__main__':
	name_network = 'DiskWorld'
	rep_results = 'evolution'

	send_email = False

	paras_G_file = jn(main_dir, 'abm_strategic_model1/my_paras/my_paras_G_iter_DiskWorld_evolution.py')
	paras_file = jn(main_dir, 'abm_strategic_model1/my_paras/my_paras_iter_for_DiskWorld_evolution.py')

	with send_email_when_ready(do=send_email, text='Finished simulations with ' + name_network + ' (' + 'evolution' + ')\n\n'):
		if 1:
			file_net = jn(result_dir, 'networks/' + name_network + '/' + name_network + '.pic')
			with open(file_net, 'r') as f:
				G = pickle.load(f)

			paras_G_iter = read_paras(paras_file=paras_G_file, post_process=False)
			iter_airport_change(paras_G_iter, G)

		if 1:
			paras = read_paras(paras_file=paras_file, post_process=True)
			iter_evolution(paras, n_iter=200, evolution_type='replicator')

		# Save paras files 
		rep = jn(result_dir, 'model1/3.1/' + name_network + '/' + rep_results)
		os.system('mkdir -p '+ rep)
		# Save iter paras of network builder
		# os.system('cp ' + paras_G_file + ' ' + rep + '/')
		# print "Copied", paras_G_file, "in", rep
		# # Save paras of network builder
		# os.system('cp ' + paras_G_iter['paras_file'] + ' ' + rep + '/')
		# print "Copied", paras_G_iter['paras_file'], "in", rep
		# Save iter paras of simulation
		os.system('cp ' + paras_file + ' ' + rep + '/')
		print "Copied", paras_file, "in", rep
		# Save paras of simulation
		os.system('cp ' + paras['paras_file'] + ' ' + rep + '/')
		print "Copied", paras['paras_file'], "in", rep

		# Gather results and dump them on disk for external plotting
		print "Preparing results..."
		results = get_results(paras, show_detailed_evo=True)

		#print results

		with open(jn(rep, 'results.pic'), 'w') as f:
			pickle.dump(results, f)

		print "Files saved in:", rep


