# -*- coding: utf-8 -*-

import sys
sys.path.insert(1,'..')
from random import seed

from abm_strategic_model1.utilities import read_paras
from abm_strategic_model1.simulationO import do_standard

if __name__=='__main__':

	if 1:
		# Manual seed
		see_ = 10
		print "===================================="
		print "USING SEED", see_
		print "===================================="
		seed(see_)
	
	paras_file = 'my_paras_DiskWorld_test.py'
	paras = read_paras(paras_file=paras_file, post_process=True)

	# paras_G_file = 'my_paras_G_DiskWorld_test.py'
	# paras_G = read_paras(paras_file=paras_G_file, post_process=False)

	# paras_G['nairports_sec'] = 2

	# soft_infrastructure(paras['G'], paras_G)

	print
	paras['G'].describe(level=2)
	print


	# if 1:
	# 	# save network instance
	# 	paras['G'].name = 'DiskWorld_test'
	# 	paras['G'].comments['seed'] = see_
	# 	os.system('mkdir -p ' + jn(result_dir, 'networks/DiskWorld_test'))
	# 	with open(jn(result_dir, 'networks/DiskWorld_test/DiskWorld_test.pic'), 'w') as f:
	# 		pickle.dump(paras['G'], f)

	print 'ACs:', paras['AC']

	do_standard((paras, paras['G']), storymode=True)

	