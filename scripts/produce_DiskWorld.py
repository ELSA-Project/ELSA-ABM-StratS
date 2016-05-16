#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1,'..')
sys.path.insert(1,'../abm_strategic_model1') # For import of G!
import os
from os.path import join as jn
import pickle


from libs.paths import main_dir, result_dir
from libs.general_tools import send_email_when_ready
from abm_strategic_model1.iter_simO import iter_airport_change, iter_sim
from abm_strategic_model1.utilities import read_paras
from abm_strategic_model1.performance_plots import get_results

if __name__=='__main__':
	name_network = 'DiskWorld'
	#rep_results = 'airports_sweep'
	rep_results = 'consolidated'
	#suff = '_higher_def'
	suff = '_quick'
	send_email = True

	paras_G_file = jn(main_dir, 'abm_strategic_model1/my_paras/my_paras_G_iter_' + name_network + suff + '.py')
	paras_file = jn(main_dir, 'abm_strategic_model1/my_paras/my_paras_iter_for_' + name_network + suff + '.py')

	with send_email_when_ready(do=send_email, text='Finished simulations with ' + name_network + ' (' + suff + ')\n\n'):
		if 1:
			file_net = jn(result_dir, 'networks/' + name_network + '/' + name_network + '.pic')
			with open(file_net, 'r') as f:
				G = pickle.load(f)

			paras_G_iter = read_paras(paras_file=paras_G_file, post_process=False)
			iter_airport_change(paras_G_iter, G)

		if 1:
			paras_iter = read_paras(paras_file=paras_file, post_process=True)
			iter_sim(paras_iter)

		# Save paras files 
		rep = jn(result_dir, 'model1/3.1/' + name_network + '/' + rep_results)
		os.system('mkdir -p '+ rep)
		# Save iter paras of network builder
		os.system('cp ' + paras_G_file + ' ' + rep + '/')
		print "Copied", paras_G_file, "in", rep
		# Save paras of network builder
		os.system('cp ' + paras_G_iter['paras_file'] + ' ' + rep + '/')
		print "Copied", paras_G_iter['paras_file'], "in", rep
		# Save iter paras of simulation
		os.system('cp ' + paras_file + ' ' + rep + '/')
		print "Copied", paras_file, "in", rep
		# Save paras of simulation
		os.system('cp ' + paras_iter['paras_file'] + ' ' + rep + '/')
		print "Copied", paras_iter['paras_file'], "in", rep

		# Gather results and dump them on disk for external plotting
		print "Preparing results..."
		results, results_global = get_results(paras_iter, Gname=name_network)

		with open(jn(rep, 'results.pic'), 'w') as f:
			pickle.dump(results, f)

		with open(jn(rep, 'results_global.pic'), 'w') as f:
			pickle.dump(results_global, f)

		print "Files saved in:", rep