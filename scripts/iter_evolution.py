# -*- coding: utf-8 -*-

import sys
sys.path.insert(1,'..')
sys.path.insert(1,'../abm_strategic_model1')
from os.path import join as jn

from abm_strategic_model1.evolution import iter_evolution
from libs.paths import main_dir
from abm_strategic_model1.utilities import read_paras

if __name__=='__main__':
	paras_file = jn(main_dir, 'abm_strategic_model1/my_paras/my_paras_iter_for_DiskWorld_evolution.py')
	paras = read_paras(paras_file=paras_file, post_process=True)

	iter_evolution(paras, n_iter=20, evolution_type='replicator')

