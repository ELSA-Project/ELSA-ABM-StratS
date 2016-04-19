#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TEMPLATE
"""

import sys
sys.path.insert(1,'..')
from abm_strategic_model1.utilities import read_paras

version = '3.0.0'

# ============================================================================ #
# =============================== Parameters ================================= #
# ============================================================================ #

paras_file = #'my_paras/my_paras_G_DiskWorld.py' TODO
paras = read_paras(paras_file=paras_file, post_process=False) # Import main parameters

# ---------------- Number of airports ---------------- #

nairports_sec_iter = range(10, 20, 2)

# --------------------System parameters -------------------- #
n_iter = 2 # Number of iterations for each set of parameters.

# --------------------- Paras to loop --------------- #
# Set the parameters to sweep by indicating their name. You can 
# put an empty list if you just want to have several iterations 
# of a single set of parameters.

paras_to_loop = ['nairports_sec']


##################################################################################
################################# Post processing ################################
##################################################################################
# DO NOT MODIFY THIS SECTION.

# -------------------- Post-processing -------------------- #
# Add new parameters to the dictionary.

for k,v in vars().items():
    if k[-4:]=='iter' and k[:-5] in paras_to_loop:
        paras[k] = v

paras['paras_to_loop'] = paras_to_loop
paras['n_iter'] = n_iter