#! /usr/bin/env python
# -*- coding: utf-8 -*-

from ABMvars import paras
from performance_plots import build_path, get_results, loc, nice_colors

import os
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import rc
from copy import copy
import pickle

def get_diff_pure_population(paras):
	"""
	Get the difference between the S and R companies in the pure 
	population case as a function of delta_t
	"""
	#paras['Delta_t_iter'] = list(np.arange(10))

	paras['par_iter'] = [[[1.,0.,10.**_e], [1.,0.,1.]] for _e in [-3,3]]
	paras['par_iter']=tuple([tuple([tuple([float(_v) for _v in _pp])  for _pp in _p])  for _p in paras['par_iter']])

	paras['paras_to_loop'] = ['Delta_t', 'par']

	results, results_global = get_results(paras, vers = '2.6')

	dT = sorted(results.keys())

	sat_R = [results[T][((1.0, 0.0, 1000.0), (1.0, 0.0, 1.0))]['satisfaction'][(1.0, 0.0, 1000.0)]['avg'] for T in dT]
	sat_S = [results[T][((1.0, 0.0, 0.001), (1.0, 0.0, 1.0))]['satisfaction'][(1.0, 0.0, 0.001)]['avg'] for T in dT]

	sat_R_std = [results[T][((1.0, 0.0, 1000.0), (1.0, 0.0, 1.0))]['satisfaction'][(1.0, 0.0, 1000.0)]['std'] for T in dT]
	sat_S_std = [results[T][((1.0, 0.0, 0.001), (1.0, 0.0, 1.0))]['satisfaction'][(1.0, 0.0, 0.001)]['std'] for T in dT]

	diff = [sat_S[i] - sat_R[i] for i in range(len(sat_R))]

	diff_std = [sqrt(sat_S_std[i]**2 + sat_R_std[i]**2) for i in range(len(sat_R))]

	#print dT

	# print sat_R

	# print sat_S

	return diff, diff_std, dT

def get_equilibrium_point(paras, errors='sampling'):
	"""
	Compute the equilibrium point, which is the point crossing the y = 0
	in a graph with ordinate S-R and abscissa f_S.
	"""
	#paras['Delta_t_iter'] = list(np.arange(10))
	#paras_copy = copy(paras)

	assert errors in ['sampling', 'statistical']

	paras['par'] = ((1.0, 0.0, 0.001), (1.0, 0.0, 1000.))
	#paras['nA_iter'] = list(np.arange(0.1,0.91,0.1))
	paras['paras_to_loop'] = ['Delta_t', 'nA']

	results, results_global = get_results(paras, vers = '2.6') 

	for nA in paras['nA_iter']:
		paras.update('nA', nA)
		#print 'AC when nA = ', nA, ':', paras['AC']

	dT = sorted(results.keys())

	fS_cross = []
	fS_cross_std = []

	for T in dT:
		print "dT=", T
		fS = sorted(results[T].keys())

		print 'fS=', fS
		sat_R = [results[T][fSS]['satisfaction'][(1.0, 0.0, 1000.0)]['avg'] for fSS in fS]
		sat_S = [results[T][fSS]['satisfaction'][(1.0, 0.0, 0.001)]['avg'] for fSS in fS]
		sat_R_std = [results[T][fSS]['satisfaction'][(1.0, 0.0, 1000.0)]['std'] for fSS in fS]
		sat_S_std = [results[T][fSS]['satisfaction'][(1.0, 0.0, 0.001)]['std'] for fSS in fS]

		for i in range(len(fS)):
			if fS[i] == -1.:
				fS[i] = 0.
			elif fS[i] == 2.:
				fS[i] = 1.

		diff = np.array([sat_S[i] - sat_R[i] for i in range(len(sat_R))])
		diff_std = np.array([sat_S_std[i] + sat_R_std[i] for i in range(len(sat_R))])

		# Upper estimation of diff
		upper_diff = diff + diff_std
		# lower estimation of diff.
		lower_diff = diff - diff_std

		print 'diff=', diff
		print 'upper_diff=', upper_diff
		print 'lower_diff=', lower_diff

		if False in list(diff<0):
			if True in list(diff<0):# and  list(diff<0).index(True)>3:
				j = list(diff<0).index(True)
				i = j-1
				print ('j=', j, 'i=', i )
				if j!=0:
					print 'Points before and after crossing:', fS[i], fS[j]

					fS_cross.append((diff[j]*fS[i] - diff[i]*fS[j])/(diff[j]-diff[i]))

					# Sampling error
					if errors=='sampling':
						fS_cross_std.append(fS[j] - fS[i])

				else:
					print "Warning: diff of sat is not monotoneous with fS."
					fS_cross.append(fS[0]/2.)
					if errors=='sampling':
						fS_cross_std.append(fS[0])
			else:
				fS_cross.append((fS[-1] + 1.)/2.)
				if errors=='sampling':
					fS_cross_std.append(1. - fS[-1])
		else:
			fS_cross.append(fS[0]/2.)
			if errors=='sampling':
				fS_cross_std.append(fS[0])


		if errors=='statistical':
			if False in list(upper_diff<0):
				if True in list(upper_diff<0):
					j = list(upper_diff<0).index(True)
					i = j-1
					print ('j=', j, 'i=', i )
					if j!=0:
						print 'Points before and after crossing (upper):', fS[i], fS[j]
						# Upper estimation of the crossing (with upper part of error)
						upper_fS_cross = (upper_diff[j]*fS[i] - upper_diff[i]*fS[j])/(upper_diff[j]-upper_diff[i])
						print upper_fS_cross

					else:
						print "Warning: diff of sat is not monotoneous with fS (upper estimation)."
						upper_fS_cross = fS[0]
				else:
					upper_fS_cross = 1. - fS[-1]
			else:
				upper_fS_cross = fS[0]


			if False in list(lower_diff<0):
				if True in list(lower_diff<0):
					j = list(lower_diff<0).index(True)
					i = j-1
					print ('j=', j, 'i=', i )
					if j!=0:
						print 'Points before and after crossing (upper):', fS[i], fS[j]
						# Lower estimation of the crossing (with lower part of error)
						lower_fS_cross = (lower_diff[j]*fS[i] - lower_diff[i]*fS[j])/(lower_diff[j]-lower_diff[i])
						print lower_fS_cross

					else:
						print "Warning: diff of sat is not monotoneous with fS (lower estimation)."
						lower_fS_cross = fS[0]
				else:
					lower_fS_cross = 1. - fS[-1]
			else:
				lower_fS_cross = fS[0]
						
			# Statistical error
			fS_cross_std.append(upper_fS_cross - lower_fS_cross)

		print "Crossing point:", fS_cross[-1]
		print "Error:", fS_cross_std[-1]
		print 

	return fS_cross, fS_cross_std, dT

def get_global_optimum(paras):
	#paras = deep_copy(paras)
	paras['paras_to_loop'] = ['Delta_t', 'nA']

	results, results_global = get_results(paras, vers = '2.6')

	dT = sorted(results.keys())

	max_fS = []
	max_fS_std = [[],[]]
	for T in dT:
		#print 'dT =', T 
		fS = sorted(results[T].keys())
		#print ('fS=', fS)
		sat_R = [results[T][fSS]['satisfaction'][(1.0, 0.0, 1000.0)]['avg'] for fSS in fS]
		sat_S = [results[T][fSS]['satisfaction'][(1.0, 0.0, 0.001)]['avg'] for fSS in fS]

		sat_R_std = [results[T][fSS]['satisfaction'][(1.0, 0.0, 1000.0)]['std']/sqrt(paras['n_iter']) for fSS in fS]
		sat_S_std = [results[T][fSS]['satisfaction'][(1.0, 0.0, 0.001)]['std']/sqrt(paras['n_iter']) for fSS in fS]

		for i in range(len(fS)):
			if fS[i] == -1.:
				fS[i] = 0.
			elif fS[i] == 2.:
				fS[i] = 1.

		sat_G = [sat_R[i]*(1.-fS[i]) + sat_S[i]*fS[i] for i in range(len(fS))]

		sat_G_std = [sat_R_std[i]*(1.-fS[i]) + sat_S_std[i]*fS[i] for i in range(len(fS))]


		

		#sat_G = esults[met][p][k]*parass['AC_dict'][p]/float(parass['ACtot'])

		# print 'Global satisfaction:'
		# print sat_G
		# print 'Standard deviation of global satisfaction'
		# print sat_G_std

		# print 'sat_R'
		# print sat_R
		# print 'sat_S'
		# print sat_S

		i_best = np.argmax(sat_G)
		#print ('i_best = ', i_best)
		fS_best = fS[i_best]
		#print ('fS_best = ', fS_best)
		max_fS.append(fS_best)

		#Looking for lower bound or error:
		found = False
		i = i_best - 1
		while i>=0 and not found:
			min_previous_value = sat_G[i+1] - sat_G_std[i+1] 
			max_current_value = sat_G[i] + sat_G_std[i]
			found = min_previous_value > max_current_value
			i -= 1


		low_bound_wide = fS[i+1]
		low_bound_sharp = fS[i+2]

		# print 'low_bound_wide, low_bound_sharp', low_bound_wide, low_bound_sharp
		# print 'Corresponding indices:', i+1, i+2
		
		x1, x2 = low_bound_wide, low_bound_sharp
		y1, y2 = sat_G[i+1], sat_G[i+2]
		d1, d2 = sat_G_std[i+1], sat_G_std[i+2]

		low_bound = (y2*x2- x2*y1 - d2*x2 + 2*d2*x1 - x2*d1)/(y2 -y1 + d2 -d1)

		found = False
		i = i_best +1
		while i<len(fS) and not found:
			#print 'i = ', i
			min_previous_value = sat_G[i-1] - sat_G_std[i-1] 
			max_current_value = sat_G[i] + sat_G_std[i]
			found = min_previous_value > max_current_value
			i += 1

		high_bound_wide = fS[i-1]
		high_bound_sharp = fS[i-2]

		# print 'high_bound_wide, high_bound_sharp', high_bound_wide, high_bound_sharp
		# print 'Corresponding indices:', i-1, i-2
		

		x1, x2 = high_bound_sharp, high_bound_wide
		y1, y2 = sat_G[i-2], sat_G[i-1]
		d1, d2 = sat_G_std[i-2], sat_G_std[i-1]

		high_bound = (y2*x2- x2*y1 - d2*x2 + 2*d2*x1 - x2*d1)/(y2 -y1 + d2 -d1)

		# print 'low_bound=', low_bound, '; high_bound=', high_bound

		max_fS_std[0].append(fS_best - low_bound)
		max_fS_std[1].append(high_bound - fS_best)

		# if i_best ==0:
		# 	max_fS_std.append(fS[1] - fS[0])
		# elif i_best == len(fS)-1:
		# 	max_fS_std.append(fS[-1] - fS[-2])
		# else:
		# 	max_fS_std.append(fS[i_best+1] - fS[i_best-1])

	return max_fS, max_fS_std, dT

def plot_cross_vs_pure_adv(paras, diff, fS_cross, diff_std, fS_cross_std, std_error = True, rep = '.'):
	plt.figure()

	plt.ylabel(r'$f_S^{eq}$')
	plt.xlabel(r'$\Delta Sat$')

	formats = ['o-', '^-']
	for i,d in enumerate(fS_cross.keys()):
		plt.plot(diff[d], fS_cross[d], 'ro')
		plt.errorbar(diff[d], fS_cross[d], xerr=diff_std[d], yerr = fS_cross_std[d], fmt='ro')

	plt.savefig(rep + "/cross_vs_pure_adv.png") 

def plot_cross_eq_glob(paras, fS_cross, fS_cross_std, dT, max_fS, max_fS_std,  rep = '.', leg=False, figsize=(10, 7), xlim=(-1, 24), ylim=(0., 1.), loc=loc['lr'], anchor_box=None):
	"""
	Beware, same than next function, except that is only for one pair of airports!

	fS_cross, etc are directly lists of values.

	"""
	plt.figure(figsize=figsize)

	plt.ylabel(r'$f_S^{eq}$', fontsize=24)
	plt.xlabel(r'$\Delta t$', fontsize=24, labelpad=-10)
	plt.tick_params(labelsize=18)

	#plt.plot(dT, fS_cross, 'ro-')
	#colors = ['r', 'b']
	#formats = ['o-', '^-']
	#for i,d in enumerate(fS_cross.keys()):
	plt.errorbar(dT, fS_cross, yerr = fS_cross_std, fmt = 'o-', color = 'r', label = r'$f_{S,eq}$')#, d = ' + str(d))
	#plt.errorbar(dT, max_fS, yerr = max_fS_std, fmt = '^-', color = 'b', label = r'$f_{S,max}$')#, d = ' + str(d))

	plt.xlim(xlim)
	plt.ylim(ylim)
	if leg:
		if loc!=0:
			plt.legend(loc=loc, fontsize=16, fancybox=True, shadow=True)  
		else:
			plt.legend(fontsize=16, fancybox=True, shadow=True, bbox_to_anchor=anchor_box)  
	#plt.legend(loc = loc['lr'])
	plt.savefig(rep + "/plot_cross_eq_glob.png", close=False)
	plt.savefig(rep + "/plot_cross_eq_glob.svg") 

def plot_cross_several_densities(paras, fS_cross, fS_cross_std, dT, max_fS, max_fS_std,  rep = '.', labelpad=-5):
	"""
	Plot all the curves for the different pairs of airports independently.

	fS_cross: dictionnary
		keys are the airports, values are lists of values of the point of equilibrium.
	fS_cross_std: dictionnary
		keys are the airports, values are lists of std of the values of the point of equilibrium.
	dT: dictionnary,
		keys are the airports, values are lists of values of dt.
	max_fS: dictionnary,
		keys are the airports, values are lists of the point where the satisfaction is maximum.
	max_fS: dictionnary, UNUSED
		keys are the airports, values are lists of the std deviation of the point where the satisfaction is maximum.
	"""

	plt.figure(figsize=(10,7))

	plt.ylabel(r'$f_S^{eq}$', fontsize=24)
	plt.xlabel(r'$\Delta T$', fontsize=24, labelpad=labelpad)

	plt.tick_params(labelsize = 18)
	#plt.plot(dT, fS_cross, 'ro-')
	#colors = ['r', 'b']
	#formats = ['o-', '^-']
	cols = plt.rcParams['axes.color_cycle']
	x, y, yerr = [], [], []
	for i,d in enumerate(fS_cross.keys()):
		x.append(dT[d])
		y.append(fS_cross[d])
		yerr.append(fS_cross_std[d])

		plt.errorbar(dT[d], fS_cross[d], 
			yerr = fS_cross_std[d], 
			#fmt = formats[i], 
			fmt = 'o-', 
			color = 'r'#cols[i%len(cols)],#plt.rcParams['axes.color_cycle']
			#color = nice_colors[i%len(nice_colors)],
			#label = r'$d =$' + str(d)
			)

	plt.xlim([-0.5, max(paras['Delta_t_iter'])+0.5])
	plt.ylim([-0.05, 1.05])
	plt.legend(loc = loc['lr'])
	plt.savefig(rep + "/plot_cross_several_densities.png") 

	with open(rep + "/plot_cross_several_densities.pic", 'w') as f:
		pickle.dump((x, y, yerr), f)

def plot_cross_average_densities(paras, fS_cross, fS_cross_std, dT, max_fS, max_fS_std,  rep = '.', labelpad=-5):
	plt.figure(figsize=(10,7))

	plt.ylabel(r'$f_S^{eq}$', fontsize=24)
	plt.xlabel(r'$\Delta T$', fontsize=24, labelpad=labelpad)

	plt.tick_params(labelsize = 18)
	
	y, yerr = [], []
	cols = plt.rcParams['axes.color_cycle']
	for i, d in enumerate(fS_cross.keys()):
		y.append(fS_cross[d])
		yerr.append(fS_cross_std[d])
		
	y = np.array(y).mean(axis=0)
	yerr = np.array(yerr).mean(axis=0)
	plt.errorbar(dT[d], y, yerr=yerr, fmt='k--')

	plt.xlim([-0.5, max(paras['Delta_t_iter'])+0.5])
	plt.ylim([-0.05, 1.05])
	#plt.legend(loc = loc['lr'])
	plt.savefig(rep + "/plot_cross_average_densities.png") 

	with open(rep + "/plot_cross_average_densities.pic", 'w') as f:
		pickle.dump((dT[d], y, yerr), f)

def plot_cross_median_densities(paras, fS_cross, fS_cross_std, dT, max_fS, max_fS_std,  rep = '.', labelpad=-5):
	plt.figure(figsize=(10,7))

	plt.ylabel(r'$f_S^{eq}$', fontsize=24)
	plt.xlabel(r'$\Delta T$', fontsize=24, labelpad=labelpad)

	plt.tick_params(labelsize = 18)
	
	y, yerr = [], []
	cols = plt.rcParams['axes.color_cycle']
	for i,d in enumerate(fS_cross.keys()):
		y.append(fS_cross[d])
		yerr.append(fS_cross_std[d])
		
	y_low=np.percentile(np.array(y), 25, axis=0)
	y_high=np.percentile(np.array(y), 75, axis=0)
	y = np.median(np.array(y), axis=0)
	#yerr = np.median(np.array(yerr), axis=0)
	
	yerr = y_high - y_low
	plt.errorbar(dT[d], y, yerr=yerr, fmt='k--')
	plt.fill_between(dT[d], y_low, y_high, facecolor='k', alpha=0.25)

	plt.xlim([-0.5, max(paras['Delta_t_iter'])+0.5])
	plt.ylim([-0.05, 1.05])
	#plt.legend(loc = loc['lr'])
	plt.savefig(rep + "/plot_cross_median_densities.png") 

	with open(rep + "/plot_cross_median_densities.pic", 'w') as f:
		pickle.dump((dT[d], y, y_low, y_high), f)

if __name__ == '__main__':
	

	# This is the one used for the article:
	#rep = '../results/' + paras['G'].name + '_equilibrium/Fixed_density_several_airports'

	rep = '../results/' + paras['G'].name + '_equilibrium2/Fixed_density_several_airports'
	

	#rep = '../results/' + paras['G'].name + '_equilibrium/Fixed_density'

	os.system('mkdir -p ' + rep)
	os.system('cp ABMvars.py ' + rep + '/')

	diff, diff_std, dT1 = {}, {}, {}
	fS_cross, fS_cross_std, dT2 = {}, {}, {}
	max_fS, max_fS_std, dT3 = {}, {}, {}
	#paras_init = copy(paras)

	#for d in [5.]:#, 10.]:
	for d in paras['airports_iter']:
		#print ('density=', d)
		print ('airports=', d)
		#paras = copy(paras_init)
		paras.update('airports', d)

		#diff[d], diff_std[d], dT1[d] = get_diff_pure_population(paras)
		fS_cross[d], fS_cross_std[d], dT2[d] = get_equilibrium_point(paras, errors='sampling')
		max_fS[d], max_fS_std[d], dT3[d] = get_global_optimum(paras)
		print
		print

		#assert dT1==dT2

	with open(rep + "/plot_cross_densities.pic", 'w') as f:
		pickle.dump((fS_cross, fS_cross_std, dT2), f)
	with open(rep + "/plot_max_densities.pic", 'w') as f:
		pickle.dump((max_fS, max_fS_std, dT3), f)

	#plot_cross_vs_pure_adv(paras, diff, fS_cross, diff_std, fS_cross_std, rep = rep)
	plot_cross_several_densities(paras, fS_cross, fS_cross_std, dT2, max_fS, max_fS_std, rep = rep)
	plot_cross_average_densities(paras, fS_cross, fS_cross_std, dT2, max_fS, max_fS_std, rep = rep)
	plot_cross_median_densities(paras, fS_cross, fS_cross_std, dT2, max_fS, max_fS_std, rep = rep)
	
	assert dT2 == dT3
	#d = 5.
	plot_cross_eq_glob(paras, fS_cross[d], fS_cross_std[d], dT2[d], max_fS[d], max_fS_std[d], rep = rep)
	print "Plots saved in", rep

	#plt.show()
