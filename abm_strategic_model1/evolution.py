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

from simulationO import do_standard, build_path as build_path_single
from utilities import read_paras, post_process_paras
from prepare_network import soft_infrastructure
from abm_strategic_model1.iter_simO import average, loop
from libs.general_tools import counter, clock_time, fit, r_squared
from libs.paths import result_dir

#pd.options.display.mpl_style = 'default'

version = '3.1.0'
main_version = version.split('.')[0] + '.' + version.split('.')[1]

class EvolutionCourse(object):
	def __init__(self, paras, evolution_type='constant_number', fitness_var='satisfaction',\
		n_iter=50, def_pop1=(1., 0., 0.000001), def_pop2=(1., 0., 1000000.), n_eq=None):
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

		if n_eq==None:
			self.n_eq = min(int(n_iter/2.), 40)
		else:
			self.n_eq = n_eq

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

		if self.evolution_type=='constant_number':
			fitness_diff = self.fitness(results)
			#paras['nA'] = min(1., max(0., paras['nA']*(1.+fitness_diff)))
			#new_value = max(self.nA_min, min(self.nA_max, fitness_diff))
			new_value = fitness_diff
		elif self.evolution_type=='proportional':
			s1 = results[self.fitness_var][self.pop1]
			s2 = results[self.fitness_var][self.pop2]
			nA = paras['nA']
			new_value = (nA*(1.+s1))/(nA*(1.+s1) + (1.-nA)*(1.+s2))
		elif self.evolution_type=='replicator':
			s1 = results[self.fitness_var][self.pop1]
			s2 = results[self.fitness_var][self.pop2]
			nA = paras['nA']
			new_value = nA + nA*(1.-nA)*(s1-s2)
		else:
			raise Exception('Unknown evolution_type:', self.evolution_type)

		new_value = max(self.nA_min, min(self.nA_max, new_value))
		paras.update('nA', new_value)
		paras = post_process_paras(paras)

		return paras

	def fit_nA(self):
		initial_n = self.record['nA'][0]
		def f_fit(x, a, b):
			return initial_n + a*(1.-np.exp(-x/b))

		popt, pcov, f_fit_opt = fit(list(range(len(self.record['nA']))), self.record['nA'],
									f_fit=f_fit,
									#p0=(0.3, 40.)
									)

		r2 = r_squared(self.record['nA'], np.vectorize(f_fit_opt)(self.record['nA']))

		return popt, pcov, f_fit_opt, r2

	def compute_results(self):
		eq = np.mean(self.record['nA'][-self.n_eq:])
		std = np.std(self.record['nA'][-self.n_eq:])

		eq_fit = np.mean(self.record['fitness'][-self.n_eq:])
		std_fit = np.std(self.record['fitness'][-self.n_eq:])

		# Compute time to equilibrium. This assumes an exponential
		# relaxation and might not be super accurate.
		# popt, pcov, f_fit_opt, r2 = self.fit_nA()
		
		# if r2>0.5:
		# 	time = popt[1]
		# else:
		# 	time = np.nan

		self.results = {'fS_eq':eq, 'fS_std':std, 'fit_eq':eq_fit, 'fit_std':std_fit, 'values':np.array(self.record['nA'])}

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

	def run(self, verbose=True):
		paras = self.paras
		for i in range(self.n_iter):
			if verbose:
				counter(i, self.n_iter, message='Evolving...')
			# print 'i=', i
			results = self.fight(paras)
			# print 'sat S=', results[self.fitness_var][self.pop1]
			# print 'sat R=', results[self.fitness_var][self.pop2]
			# print 'fitness (of S)=', self.fitness(results)
			paras = self.evolve(paras, results)
			# print 'New nA=', paras['nA']
			self.append_to_record(paras, results)

	def append_to_record(self, paras, results):
		self.record['nA'].append(paras['nA'])
		self.record['fitness'].append(self.fitness(results))

def build_pat(paras, vers=main_version, in_title=['tau', 'par', 'ACtot', 'nA'], rep=result_dir):
	"""
	Build the path for results.
	"""
	return build_path_single(paras, vers=vers, rep=rep, name_G=paras['G'].name_generic) + '_iter' + str(paras['n_iter']) + '_evolution.pic'

def do_evo((paras, kwargs)):
	EC = EvolutionCourse(paras, **kwargs)
	EC.run(verbose=False)
	EC.compute_results()
	return EC.results

def aggregate_results(results_list):
	results = {}
	for met in results_list[0].keys():
		#print met, np.mean([v[met] for v in results_list], axis=0)
		results[met] = {'avg':np.mean([v[met] for v in results_list], axis=0), 'std':np.std([v[met] for v in results_list], axis=0)}
	
	return results	

def iter_evolution(paras, **kwargs):
	n_iter_tot = np.prod([len(paras[p + '_iter']) for p in paras['paras_to_loop']])
	paras['n_iter_tot'] = n_iter_tot
	print "Total number of iterations over the parameters:", n_iter_tot

	loop({p:paras[p + '_iter'] for p in paras['paras_to_loop']},
			paras['paras_to_loop'],
			paras,
			tot_lvl=len(paras['paras_to_loop']),
			thing_to_do=average,
			args_do=(paras, kwargs), 
			do=do_evo,
			build_pat=build_pat,
			args_pat=(paras, ),
			aggregator=aggregate_results,
			parallel=paras['parallel'],
			force=paras['force'],
			n_iter=paras['n_iter'])
	print

def load_results(paras=None, build_pat=None, args_pat=None, kwargs_pat={}, verbose=False):
	file_pat = build_pat(*args_pat, **kwargs_pat)
	with open(file_pat, 'r') as f:
		results = pickle.load(f)
	return results

def get_results(paras, vers=main_version, show_detailed_evo=False):
	it, results = loop({p:paras[p + '_iter'] for p in paras['paras_to_loop']}, 
									paras['paras_to_loop'], 
									paras,
									results={},
									thing_to_do=load_results,
									paras=paras,
									args_pat=(paras, ),
									build_pat=build_pat,
									kwargs_pat = {'vers':vers},
									show_detailed_evo=show_detailed_evo)
	print
	return results

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

	paras['nA'] = 0.5

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

	save_rep = jn(result_dir, 'evolution/3.1')
	os.system("mkdir -p " + save_rep)

	EC = EvolutionCourse(paras, n_iter=100, evolution_type='replicator')
	EC.run()

	#print EC.record

	# 
	popt, pcov, f_fit_opt = EC.fit_nA()

	print popt
	print pcov
	print np.trace(pcov)
	print f_fit_opt(0)

	# simple plot
	plt.plot(range(EC.n_iter), EC.record['nA'], '-b')
	plt.plot(range(EC.n_iter), np.vectorize(f_fit_opt)(list(range(EC.n_iter))), '-g')
	eq = np.mean(EC.record['nA'][-40:])
	plt.plot(range(EC.n_iter), [eq]*EC.n_iter, '--r')
	#plt.plot(range(EC.n_iter), EC.record['fitness'], '-b')
	plt.savefig(jn(save_rep, 'single_evolution_Dt1_d20.png'))
	plt.show()
