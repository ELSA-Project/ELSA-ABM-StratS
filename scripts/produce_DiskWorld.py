#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1,'..')
sys.path.insert(1,'../abm_strategic_model1') # For import of G!
from os.path import join as jn
import pickle

from libs.paths import main_dir, result_dir
from abm_strategic_model1.iter_simO import iter_airport_change
from abm_strategic_model1.utilities import read_paras

if __name__=='__main__':
	paras_file = jn(main_dir, 'abm_strategic_model1/my_paras/my_paras_G_iter_DiskWorld.py')
	file_net = jn(result_dir, 'networks/DiskWorld/DiskWorld.pic')
	with open(file_net, 'r') as f:
		G = pickle.load(f)

	paras_G_iter = read_paras(paras_file=paras_file,
							  post_process=False)

	iter_airport_change(paras_G_iter, G)