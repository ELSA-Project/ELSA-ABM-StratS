# -*- coding: utf-8 -*-

"""
Tools to iterate the simulations with an evolution of the strategies

"""
import os
from os.path import join as jn
import matplotlib.pyplot as plt
import pandas as pd
from random import seed
import pickle
import numpy as np

from simulationO import do_standard
from utilities import read_paras, post_process_paras
from prepare_network import soft_infrastructure
from libs.general_tools import counter
from libs.paths import result_dir

#pd.options.display.mpl_style = 'default'

class EvolutionCourse(object):
	def __init__(self, paras, evolution_type='constant_number', fitness_var='satisfaction',\
		n_iter=50, def_pop1=(1., 0., 0.000001), def_pop2=(1., 0., 1000000.)):
		"""
		Parameters
		----------
		paras : Paras object
		evolution_type : string,
			Can be:
			- 'constant_number': total number of flights is constant.
		fitness_var : string,
			variable to be considered as the fitness, e.g. satisfaction.

		"""

		self.paras = paras # Set of initial parameters
		self.evolution_type = evolution_type
		self.fitness_var = fitness_var
		self.record = {'nA':[], 'fitness':[]}
		self.compute_bounds_on_mix()
		self.n_iter = n_iter

		# Add check about populations here
		# add definition of population here.
		self.pop1 = def_pop1
		self.pop2 = def_pop2

	def fight(self, paras):
		"""
		Run simulation based on the current state of the paras dict.
		"""

		results = do_standard((paras, paras['G']))

		return results

	def evolve(self, paras, results):
		"""
		Based on the results, compute te fitness and change the paras.
		"""

		fitness_diff = self.fitness(results)

		if self.evolution_type=='constant_number':
			#paras['nA'] = min(1., max(0., paras['nA']*(1.+fitness_diff)))
			#new_value = max(self.nA_min, min(self.nA_max, fitness_diff))
			new_value = fitness_diff
		elif self.evolution_type=='proportional':
			s1 = results[self.fitness_var][self.pop1]
			s2 = results[self.fitness_var][self.pop2]
			nA = paras['nA']
			new_value = (nA*(1.+s1))/(nA*(1.+s1) + (1.-nA)*(1.+s2))
		else:
			raise Exception('Unknown evolution_type:', self.evolution_type)

		new_value = max(self.nA_min, min(self.nA_max, new_value))
		paras.update('nA', new_value)
		paras = post_process_paras(paras)

		return paras

	def compute_bounds_on_mix(self, eps=0.0001):
		ntot = self.paras['ACtot']
		nA_min = 1./ntot + eps
		nA_max = 1. - 1./ntot - eps

		self.nA_min = nA_min
		self.nA_max = nA_max

	def fitness(self, results):
		# fitness_diff = (results[self.fitness_var][self.pop1] - results[self.fitness_var][self.pop2])/\
		# 				(results[self.fitness_var][self.pop1] + results[self.fitness_var][self.pop2])
		fitness_diff = (results[self.fitness_var][self.pop1] - results[self.fitness_var][self.pop2])/2.
		return (1.+fitness_diff)/2.

	def run(self):
		paras = self.paras
		for i in range(self.n_iter):
			counter(i, self.n_iter, message='Evolving...')
			# print 'i=', i
			
			results = self.fight(paras)
			# print 'sat S=', results[self.fitness_var][self.pop1]
			# print 'sat R=', results[self.fitness_var][self.pop2]
			# print 'fitness (of S)=', self.fitness(results)
			paras = self.evolve(paras, results)
			# print 'New nA=', paras['nA']
			self.append_to_record(paras, results)
			# print

	def append_to_record(self, paras, results):
		self.record['nA'].append(paras['nA'])
		self.record['fitness'].append(self.fitness(results))


if __name__=='__main__':
	# paras_file = '../tests/paras_test_evolution.py'
	# paras = read_paras(paras_file=paras_file, post_process=True)

	if 1:
		# Manual seed
		see_ = 10
		print "===================================="
		print "USING SEED", see_
		print "===================================="
		seed(see_)
	
	paras_file = 'my_paras/my_paras_DiskWorld_evolution.py'
	paras = read_paras(paras_file=paras_file, post_process=True)

	paras_G_file = 'my_paras/my_paras_G_DiskWorld.py'
	paras_G = read_paras(paras_file=paras_G_file, post_process=False)

	paras_G['nairports_sec'] = 2

	soft_infrastructure(paras['G'], paras_G)

	print
	paras['G'].describe(level=2)
	print


	if 0:
		# save network instance
		paras['G'].name = 'DiskWorld_test'
		paras['G'].comments['seed'] = see_
		os.system('mkdir -p ' + jn(result_dir, 'networks/DiskWorld_test'))
		with open(jn(result_dir, 'networks/DiskWorld_test/DiskWorld_test.pic'), 'w') as f:
			pickle.dump(paras['G'], f)

	print 'ACs:', paras['AC']

	EC = EvolutionCourse(paras, n_iter=100, evolution_type='proportional')
	EC.run()

	#print EC.record

	# simple plot
	plt.plot(range(EC.n_iter), EC.record['nA'], '-b')

	eq = np.mean(EC.record['nA'][15:])
	plt.plot(range(EC.n_iter), [eq]*EC.n_iter, '--r')
	#plt.plot(range(EC.n_iter), EC.record['fitness'], '-b')
	plt.show()
