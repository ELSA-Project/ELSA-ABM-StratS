#!/usr/bin/env python

import sys
#sys.path.insert(1,'/home/earendil/Documents/ELSA/Modules')
from os.path import join as jn

import networkx as nx
#from random import getstate, setstate, 
from random import shuffle, uniform,  sample, seed
import pickle
from string import split
import matplotlib.pyplot as plt
import os
import numpy as np
import copy
from math import ceil

from simAirSpaceO import AirCompany, Network_Manager
from utilities import draw_network_map, read_paras

from libs.paths import result_dir
from libs.general_tools import delay, date_st, header, clock_time, draw_network_and_patches

#from tools_airports import map_of_net

import warnings

version = '3.0.0'
main_version = split(version,'.')[0] + '.' + split(version,'.')[1]

if 0:
	print 'Caution! Seed!'
	seed(2)


class Simulation:
	"""
	Class Simulation. 
	=============
	Main object for the simulation. Initialize loads, prepares air companies, 
	calls network manager, keeps the M1 queue in memory, calls for possible 
	shocks.
	"""

	def __init__(self, paras, G=None, verbose=False):
		"""
		Initialize the simulation, build the network if none is given in argument, set the verbosity

		Parameters
		----------
		paras : Paras object, dict-like
			Gathers all parameters of the simulation.
		G : hybrid network
			compulsory, None is deprecated TODO.
		verbose : boolean, optional
			set verbosity.

		Notes
		-----
		New in 3.0.0: taken and adapted from Model 2

		(From Model 2)
		Change in 2.9.3: self.make_times is computed at each simulations.
		Changed in 2.9.3: update shortest paths if there is a change of Nsp_nav.
		
		"""

		self.paras = paras

		for k in ['AC', 'Nfp', 'na', 'tau', 'departure_times', 'ACtot', 'N_shocks','Np', \
			'ACsperwave','Delta_t', 'width_peak', 'old_style_allocation', 'flows', 'nA', \
			'day', 'noise', 'STS', 'starting_date', 'bootstrap_mode', 'bootstrap_only_time',\
			'discard_first_and_last_node']:
			if k in paras.keys():
				setattr(self, k, paras[k])
	
		self.make_times()
		self.pars = paras['par']

		self.G = paras['G'].copy()

		assert check_object(self.G)
		assert self.G.Nfp==paras['Nfp']
		
		self.verb = verbose
		self.rep = build_path(paras)
		
		#os.system('mkdir -p ' + self.rep) TODO: see this!
		
	def make_simu(self, clean=False, storymode=False):
		"""
		Do the simulation, clean afterwards the network (useful for iterations).

		Parameters
		----------
		clean : boolean, optional
			if True, ask the network manager to initialize load on the network 
			beforehand.
		storymode : boolean, optional
			sets verbosity.

		Notes
		-----
		Changed in 3.0.0: taken from Model 2

		(From Model 2)
		Changed in 2.9.6: added the shuffle_departure.
		
		"""

		if self.verb:
			print 'Doing simulation...'

		Netman = Network_Manager(old_style=self.old_style_allocation, discard_first_and_last_node=self.discard_first_and_last_node)
		self.storymode = storymode

		#------------------------------------------------------# 

		Netman.initialize_load(self.G, length_day=int(self.day/60.))

		if self.flows == {}:
			self.build_ACs()
		else:
			self.build_ACs_from_flows()

		# self.show_ACs()
		# raise Exception()
		
		if clean:
			Netman.initialize_load(self.G, length_day=int(self.day/60.)) # TODO: check why I am doing this again.
		
		self.queue = Netman.build_queue(self.ACs)

		self.shuffle_departure_times()

		Netman.allocate_queue(self.G, self.queue, storymode=storymode)

		self.mark_best_of_queue()

		self.M0_queue = copy.deepcopy(self.queue)
		#Netman.M0_to_M1(self.G, self.queue, self.N_shocks, self.tau, self.Nsp_nav, storymode=True)
		Netman.M0_to_M1_quick(self.G, self.queue, self.N_shocks, self.tau, storymode=True, sectors_to_shut = self.STS)

	def build_ACs(self):
		"""
		Build all the Air Companies based on the number of Air Companies given in paras.
		If n_AC is an integer and if several sets of parameters are given for the utility 
		function, make the same number of AC sor each set. If n_AC is an array of integers 
		of the same size than the number of set of parameters, populates the different
		types with this array. 

		Request the computation of all flight plans for all ACs.
		
		Examples:
		---------
		self.AC=30, self.pars=[[1,0,0]] 
		gives 30 ACs with parameters [1,0,0].
		
		self.AC=30, self.pars=[[1,0,0], [0,0,1]] 
		gives 15 ACs with parameters [1,0,0] and 15 ACs with parameters [0,0,1].
		
		self.AC=[10,20], self.pars=[[1,0,0], [0,0,1]] 
		gives 10 ACs with parameters [1,0,0] and 20 ACs with parameters [0,0,1].

		Raises
		------
		Exception
			if self.AC is a list with a different length from self.pars

		Notes
		-----
		Changed in 3.0.0: taken and adapted from Model 2

		(From Model 2)
		Changed in 2.9: pairs given to ACs are navpoints, not sectors.

		"""
		
		if type(self.AC)==int:
			self.AC = [self.AC/len(self.pars) for i in range(len(self.pars))]
		
		try:
			assert len(self.AC)==len(self.pars)
		except:
			raise Exception('AC should have the same length than the parameters, or be an integer')
	
		self.ACs={}
		k=0
		shuffle(self.t0sp)
		for i,par in enumerate(self.pars):
			for j in range(self.AC[i]):
				self.ACs[k] = AirCompany(k, self.Nfp, self.na, self.G.connections(), par)
				self.ACs[k].fill_FPs(self.t0sp[k], self.tau, self.G)
				k+=1

	def build_ACs_from_flows(self): 
		"""
		Build the list of ACs from the flows. 

		Notes
		-----
		Changed in 3.0.0: taken and adapted from Model 2

		(From Model 2)
		New in 2.9.2 
		Changed in 2.9.5: record the day of simulation.
		Changed in 2.9.6: added bootstrap_mode. Changed times of departure from integers to floats.
		
		"""

		self.ACs = {}
		all_times = [time for ((source, destination), times) in self.flows.items() for time in times]
		
		if not self.bootstrap_mode:
			flows = self.flows
		else:
			# Note: at this stage, self.ACtot is already either the number of flights from data
			# or some user defined number.
			if self.bootstrap_only_time:
				# resampling times keeping number of flights constant per each entry/exit (scaling them up)
				tot_AC = len(all_times)
				flows = {(entry, exit):sample(all_times, int(len(times)*self.ACtot/float(tot_AC))) for (entry, exit), times in self.flows.items()}
			else:
				# resampling the pairs of entry/exit
				new_pairs = np.array(sample(self.flows.keys(), self.ACtot))
				flows = {}
				for pair in new_pairs:
					flows[pair] = flows.get(pair, []) + [choice(all_times)]
		
		all_times = [delay(time) for time in all_times]
		# Guess the starting date.
		min_time = date_st(min(all_times))
		self.starting_date = [min_time[0], min_time[1], min_time[2], 0, 0, 0]
		k=0

		for ((source, destination), times) in flows.items():
			idx_s = self.G.idx_nodes[source]
			idx_d = self.G.idx_nodes[destination]
			if idx_s in self.G.get_airports() and idx_d in self.G.get_airports() and self.G.short.has_key((idx_s, idx_d)):    
				# For each OD, put nA fraction of the flights as company A and 1-nA as company B
				n_flights_tot = len(times)
				n_flights_A = int(self.nA*n_flights_tot)
				n_flights_B = n_flights_tot - int(self.nA*n_flights_tot)
				AC = [n_flights_A, n_flights_B]
				l = 0
				for i, par in enumerate(self.pars):
					for j in range(AC[i]):
						time = times[l]
						self.ACs[k] = AirCompany(k, self.Nfp, self.na, self.G.short.keys(), par)
						time = delay(time, starting_date=self.starting_date)/60.
						self.ACs[k].fill_FPs([time], self.tau, self.G, pairs=[(idx_s, idx_d)])
						k+=1
						l+=1
			else:
				if self.verb:
					print ("I do " + (not idx_s in self.G.get_airports())*'not', "find", idx_s, ", I do " + (not idx_d in self.G.get_airports())*'not', "find", idx_d,\
					 'and the couple is ' + (not self.G.short.has_key((idx_s, idx_d)))*'not', 'in pairs.')
					print ('I skip this flight.')
			
	def compute_flags(self):
		"""
		Computes flags, bottlenecks and overloadedFPs for each AC and in total.
		The queue is used to sort the flights, even within a given AC.
		"""
		for ac in self.ACs.values():
			ac.flag_first,ac.bottlenecks, ac.overloadedFPs = [], [],[]
		self.bottlenecks, self.flag_first, self.overloadedFPs= [], [], []
		for f in self.queue:
			f.make_flags()
		
			self.ACs[f.ac_id].flag_first.append(f.flag_first)
			self.ACs[f.ac_id].bottlenecks.append(f.bottlenecks)
			self.ACs[f.ac_id].overloadedFPs.append(f.overloadedFPs)
			
			self.flag_first.append(f.flag_first)
			self.overloadedFPs.append(f.overloadedFPs)
			self.bottlenecks.append(f.bottlenecks)
		
		self.results={p:{ac.id:{'flags':ac.flag_first,'overloadedFPs':ac.overloadedFPs, 'bottlenecks':ac.bottlenecks}\
					for ac in self.ACs.values() if ac.par==p} for p in self.pars}
		self.results['all']=self.flag_first, self.overloadedFPs, self.bottlenecks
		
		# if self.verb:
			#print 'flags', self.flag_first
			#print 'overloadedFPs', self.overloadedFPs
			#print 'bottlenecks', self.bottlenecks
			# print
			# print
			
	def save(self, rep='', split=False, only_flags=False):
		"""
		Save the network in a pickle file, based on the paras.
		Can be saved in a single file, or different files to speed up the post-treatment.
		
		Notes
		-----
		This is a bit dusty maybe.

		"""

		if rep=='':
			rep=build_path(self.paras)

		if only_flags:
			f=open(rep + '_flags.pic','w')
			pickle.dump((self.flag_first, self.bottlenecks, self.overloadedFPs),f)
			f.close()
		else:
			if not split:
				print ('Saving whole object in ', rep)
				with open(rep + '/sim.pic','w') as f:
					pickle.dump(self,f)
			else:
				print ('Saving split object in ', rep)
				with open(rep + '_ACs.pic','w') as f:
					pickle.dump(self.ACs,f)
				
				with open(rep + '_G.pic','w') as f:
					pickle.dump(self.G,f)
				
				with open(rep + '_flags.pic','w') as f:
					pickle.dump((self.flag_first, self.bottlenecks, self.overloadedFPs),f)
			
	def load(self, rep=''):
		"""
		Load a splitted Simulation from disk.

		Notes
		-----
		This is a bit dusty maybe.

		"""            
		
		if rep=='':
			rep=build_path(self.paras)
		if self.verb:
			print ('Loading splitted simulation from', rep)

		with open(rep + '_ACs.pic','r') as f:
			self.ACs=pickle.load(f)
		
		with open(rep + '_G.pic','r') as f:
			self.G=pickle.load(f)
		
		with open(rep + '_flags.pic','r') as f:
			(self.flag_first, self.bottlenecks, self.overloadedFPs)=pickle.load(f)
		
	def make_times(self):
		"""
		Prepares t0sp, the matrix of desired times. Three modes are possible, controlled
		by attribute 'departure_times'. 

		If 'zeros', all departure times are set to 0,
		if 'uniform', departure times are drawn uniformly between zero and self.day,
		if 'square_waves', departure times are drawn from a distribution of in wave shaped. The
		attribute 'width_peak' controls the width of the waves and 'Delta_t' controls the time
		between the end of the a wave and the beginning of the following. Note that for Delta_t=0.,
		the distribution is not not completely equivalent to a uniform distribution, since the 
		flights are evenly spread between waves anyway.

		Raises
		------
		Exception
			if ACs are requested to have more than one flight and self.departure_times='square_waves'.

		Notes
		-----
		Changed in 3.0.0: taken from Model 2 (unchanged).
		(From Model 2)
		Changed in 2.9: added departures from data.
		Changed in 2.9.3: added noise.
		Changed in 2.9.5(?): removed departures from data

		"""

		if self.departure_times=='zeros':
			self.t0sp = [[0 for j in range(self.na)] for i in range(self.ACtot)]     
		elif self.departure_times=='uniform':
			self.t0sp = [[uniform(0, self.day) for j in range(self.na)] for i in range(self.ACtot)]
		elif self.departure_times=='square_waves':
			self.t0sp = []
			if self.na==1:
				for i in range(self.Np):
					for j in range(self.ACsperwave):
						self.t0sp.append([uniform(i*(self.width_peak+self.Delta_t),i*(self.width_peak+self.Delta_t)+self.width_peak)])
				
				# add the reamining flights in the last wave
				while len(self.t0sp)<self.ACtot:
					self.t0sp.append([uniform(i*(self.width_peak+self.Delta_t),i*(self.width_peak+self.Delta_t)+self.width_peak)])
			else:
				raise Exception('na!=1 is not implemented yet...')
			
	def mark_best_of_queue(self):
		"""
		Records the cost the best flight plan (first one) for each flight.

		Notes
		-----
		This method belongs here and not in the Flight object. The reason is that when 
		there are some reallocations, hence some recomputations of the flight plans by 
		the flights, the best costs should not be recomputed.

		"""
		
		for f in self.queue:
			f.best_fp_cost=f.FPs[0].cost
			
	def shuffle_departure_times(self):
		"""
		Add a gaussian noise on the departure times. The standard deviation is
		given by attribute 'noise'.

		Notes
		-----
		New in 3.0.0: taken from Model 2 (unchanged).

		"""
		
		if self.noise!=0:
			for f in self.queue:
				f.shift_desired_time(gauss(0., self.noise))
	 
	def show_ACs(self):
		for k, ac in self.ACs.items():
			print 'AC', ac.id
			ac.show_flights()


def build_path(paras, vers=main_version, in_title=['Nfp', 'tau', 'par', 'ACtot', 'nA', 'departure_times',\
	'old_style_allocation', 'noise'], rep=result_dir, name_G=None):
	"""
	Used to build name + path from a set of paras. 

	Parameters
	----------
	paras : dict
		standard paras dict used by the model
	vers : string, optional
		Normally formatted like 'X.Y'
	in_title : list of str
		lisf of keys to include in the title.
	rep : string
		path to (left-)append to the name of the file. Use '' to have only the name.

	Returns
	-------
	name : string
		full path if rep is different from '', only name of the file otherwise.

	Notes
	-----
	Changed in 3.0.0: taken from Model 2 (unchanged). 
	(From Model 2)
	Changed 2.2: is only for single simulations.
	Changed 2.4: takes different departure times patterns. Takes names.
	Chnaged in 2.5: added N_shocks and improved the rest.

	"""

	if name_G==None:
		name_G = paras['G'].name

	name = 'model1/' + name_G + '/Sim_v' + vers + '_' + paras['G'].name
	name = jn(rep, name)
	
	in_title = list(np.unique(in_title))
		
	if paras['departure_times']!='zeros':
		try:
			in_title.remove('ACtot')
		except ValueError:
			pass
		in_title.insert(1,'density')
		in_title.insert(2,'day')
		if paras['departure_times']=='square_waves':
			in_title.insert(1,'Delta_t')
			
	if paras['N_shocks']==0 and paras['mode_M1']=='standard':
		try:
			in_title.remove('N_shocks')
		except ValueError:
			pass      
	elif paras['N_shocks']!=0 and paras['mode_M1']=='standard':
		in_title.insert(-1,'N_shocks')
	elif paras['mode_M1']=='sweep':
		in_title.insert(-1,'STS')
	
	in_title = np.unique(in_title)
	
	for p in in_title:
		if p=='par':
			if len(paras[p])==1 or paras['nA']==1.:
				coin = str(float(paras[p][0][0])) + '_' + str(float(paras[p][0][1])) + '_' + str(float(paras[p][0][2]))
			elif len(paras[p])==2:
				coin = str(float(paras[p][0][0])) + '_' + str(float(paras[p][0][1])) + '_' + str(float(paras[p][0][2])) + '__' +str(float(paras[p][1][0])) + '_' + str(float(paras[p][1][1])) + '_' + str(float(paras[p][1][2])) 
			else:
				coin = '_several'
		else:
			coin = str(paras[p])
		name += '_' + p + coin

	return name

def check_object(G):
	"""
	Use to check if you have an old object.
	"""
	
	return hasattr(G, 'comments')

def post_process_queue(queue):
	"""
	Used to post-process results. Conpute the satisfaction of each flight, the regulated flight plans,
	the regulated flights.

	Parameters
	----------
	queue : list of FlightPlan objects

	Returns
	-------
	queue : list of FlightPlan objects
		With additional flags computed.

	Notes
	-----
	Changed in 3.0.0: taken from Model 2 (unchanged).

	Every processes between the simulation and the plots should be here.
	Changed in 2.4: add satisfaction, regulated flight & regulated flight plans. On level of iteration added (on par).
	Changed in 2.5: independent function.
	Changed in 2.7: best cost is not the first FP's one.
	Changed in 2.9.3: added regulated_1FP
	
	"""
	
	for f in queue:   
		# Make flags
		f.make_flags() # Maybe useless
		
		bestcost = f.best_fp_cost
		acceptedFPscost = [FP.cost for FP in f.FPs if FP.accepted]
				
		if len(acceptedFPscost) != 0:
			f.satisfaction = bestcost/min(acceptedFPscost)
		else:
			f.satisfaction = 0.                    
			
		# Regulated flight plans
		f.regulated_FPs=len([FP for FP in f.FPs if not FP.accepted])

		# At least one flight plan regulated
		f.regulated_1FP = float(len([FP for FP in f.FPs if not FP.accepted])!=0)

		# Regulated flights
		if len([FP for FP in f.FPs if FP.accepted])==0:
			f.regulated_F = 1.
		else:
			f.regulated_F = 0.

	return queue

def extract_aggregate_values_on_queue(queue, types_air_companies, mets=['satisfaction', 'regulated_F', 'regulated_FPs']):
	"""
	Computes some aggregated metrics on queue. The results are gathered by metrics and types 
	of air companies. Only the average metrics are computed.

	Parameters
	----------
	queue : list of Flight objects
	types_air_companies : list of tuples (float, float, float)
		Reprenting the different types of companies in the simulations. The tuples are the 
		parameters of the cost function of each company.
	mets : list of string, optional
		of attributes to extract from the Flight object in queue.

	Returns:
	--------
	results : dict of dict
		first levels of keys are the metrics, second levels are the types of air companies. 

	Notes
	-----
	If some flights have a type (i.e. a tuple for cost function) which is not in types_air_companies,
	they are ignored.

	"""

	results = {}
	for m in mets:
		results[m] = {}
		for tac in types_air_companies:
			pouet = [getattr(f,m) for f in queue if tuple(f.par)==tuple(tac)]
			if pouet!=[]:
				results[m][tuple(tac)] = np.mean(pouet)
			else:
				results[m][tuple(tac)] = 0.
		
	return results                
			
def extract_aggregate_values_on_network(G):
	"""
	Extract aggregated values concerning the network concerning the traffic. Right 
	now, the load (see the definition in simAirSpaceO.Network_Manager object) and the 
	load/capacity are computed.

	TODO: Only works with the old, instantaneous definition of the load. Needs an update.

	Parameters
	----------
	G : hybrid network

	Returns
	-------
	dict : dictionary
		values are averages over all nodes and all times of the loads of the network.

	Notes
	-----
	Changed in 3.0.0: taken from Model 2 (unchanged).

	"""

	coin1 = []
	coin2 = []
	for n in G.nodes():
		if len(G.node[n]['load'])>2:
			avg = np.average(G.node[n]['load'])
			coin1.append(avg)
			coin2.append(avg/float(G.node[n]['capacity']))
		else:
			coin1.append(0.)
			coin2.append(0.)

	return {'loads': np.mean(coin1), 'loads_norm':np.mean(coin2)}

def plot_times_departure(queue, rep='.'):
	t_pref=[f.FPs[0].t for f in queue]
	t_real=[f.fp_selected.t for f in queue if f.fp_selected!=None]
	
	plt.figure(1)
	bins=range(int(ceil(max(t_real + t_pref))) + 10)
	plt.hist(t_pref,label='pref',facecolor='green', alpha=0.75, bins=bins)
	#plt.hist(t_real,label='real',facecolor='blue', alpha=0.25, bins=bins)
	#plt.legend()
	plt.ylabel('Number of departing flights')
	plt.xlabel('Time')
	plt.savefig(rep + '/departure_times.png')
	plt.show()

def do_standard((paras, G), storymode=False):
	"""
	Make the simulation and extract aggregate values. Used for automatic entry, 
	in particular by iter_sim.

	Parameters
	----------
	paras : dictionary
		containing the parameters for the simulations.
	G : Net object

	Returns
	-------
	results : dictionary
		gathers results from extract_aggregate_values_on_queue and 
		extract_aggregate_values_on_network.
	
	Notes
	-----
	New in 3.0.0: taken from Model 2 (unchanged).

	(From Model 2)
	New in 2.9.2: extracted from average_sim
	Changed in 2.9.4: taken from iter_simO
	Changed in 2.9.6: changed signature, removed integer i.

	"""
	
	results = {} 
	sim = Simulation(paras, G=G.copy(), verbose=False)
	sim.make_simu(storymode=storymode)
	sim.queue = post_process_queue(sim.queue)
	
	results_queue = extract_aggregate_values_on_queue(sim.queue, paras['par'])
	results_G = extract_aggregate_values_on_network(sim.G)
	
	for met in results_G:
		results[met] = results_G[met]
			
	for met in results_queue:
		results[met] = {tuple(p):[] for p in paras['par']}
		for p in paras['par']:
			results[met][tuple(p)] = results_queue[met][tuple(p)]

	del sim
	return results

if __name__=='__main__': 
	"""
	===========================================================================
	Manual single simulation
	===========================================================================
	"""
	paras_file = None if len(sys.argv)==1 else sys.argv[1]
	paras = read_paras(paras_file=paras_file)

	GG = paras['G'] #ABMvars.G

	print (header(paras,'SimulationO', version, paras_to_display=['ACtot']))

	with clock_time():
		sim = Simulation(paras, G=GG, verbose=True)
		sim.make_simu(storymode=False)
		sim.compute_flags()
		queue = post_process_queue(sim.queue)
		M0_queue = post_process_queue(sim.M0_queue)

	"""
	===========================================================================
	Some snippets to view the results.
	===========================================================================
	"""
	if 1:
		for n in sim.G.nodes():
			#print n, sim.G.node[n]['capacity'], sim.G.node[n]['load']
			if max(sim.G.node[n]['load']) == sim.G.node[n]['capacity']:
				print "Capacity's reached for sector:", n
				#pass
			if max(sim.G.node[n]['load']) > sim.G.node[n]['capacity']:
				print "Capacity overreached for sector:", n, '!'
				#pass
		#draw_network_map(sim.G.G_nav, title=sim.G.G_nav.name, load=False, generated=True,\
		#        airports=True, stack=True, nav=True, queue=sim.queue)

	if 1:
		trajectories = []
		for f in sim.queue:
			try:
				trajectories.append(f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p) 
			except ValueError:
				pass

		#  Real trajectories
		# trajectories_real = []
		# for f in sim.G.flights_selected:
		#     navpoints = set([sim.G.G_nav.idx_nodes[p[0]] for p in f['route_m1']])
		#     if navpoints.issubset(set(sim.G.G_nav.nodes())) and (sim.G.G_nav.idx_nodes[f['route_m1'][0][0]], sim.G.G_nav.idx_nodes[f['route_m1'][-1][0]]) in sim.G.G_nav.short.keys():
		#         trajectories_real.append([sim.G.G_nav.idx_nodes[p[0]] for p in f['route_m1']])

		draw_network_and_patches(sim.G, None, sim.G.polygons, name='trajectories', flip_axes=True, trajectories=trajectories, trajectories_type='sectors', rep=sim.rep)
		#draw_network_and_patches(sim.G, sim.G.G_nav, sim.G.polygons, name='trajectories_real', flip_axes=True, trajectories=trajectories_real, trajectories_type='navpoints', save = True, rep = build_path(sim.paras), dpi = 500)

	if 0:
		#draw_network_and_patches(sim.G,sim.G.G_nav,sim.G.polygons, name='network_small_beta', flip_axes=True, trajectories=trajectories_nav, trajectories_type='navpoints')
		p0 = 20
		p1 = 65
		#all_possible_trajectories=sorted([path for paths in sim.G.short_nav[(p0,p1)].values() for path in paths], key= lambda p: sim.G.G_nav.weight_path(p))
		possible_trajectories = [p for p in sim.G.G_nav.short[(p0,p1)]]
		if 0:
			#print 'All possible trajectories:'
			for path in all_possible_trajectories:
				pass
				#print path
			#print
			#print 'Possible trajectories:'
			for path in possible_trajectories:
				pass
				#print path
		
		#draw_network_and_patches(sim.G,sim.G.G_nav,sim.G.polygons, name='all_possible_trajectories', flip_axes=True, trajectories=all_possible_trajectories, trajectories_type='navpoints')
		draw_network_and_patches(sim.G,sim.G.G_nav,sim.G.polygons, name='possible_trajectories', flip_axes=False, trajectories=possible_trajectories, trajectories_type='navpoints', rep = sim.rep)
		#draw_network_and_patches(sim.G, None, sim.G.polygons, name='numbers', , flip_axes=True, numbers=True)

		#draw_network_map(sim.G, title=sim.G.name, load=False, generated=True,\
		#         airports=True, trajectories=trajectories, add_to_title='_high_beta')
	# if 1
	#     draw_network_map(sim.G, title=sim.G.name, load=False, generated=True,\
	#             airports=True, trajectories=trajectories, add_to_title='_high_beta')#stack=True
	#     draw_network_map(sim.G.G_nav, title=sim.G.name, load=False, generated=True,\
	#             airports=True, trajectories=trajectories_nav, add_to_title='_nav_high_beta')
	# if 0:
	#     #print sim.queue[0].FPs[0].p_nav
	#     for p in sim.queue[0].FPs[0].p_nav:
	#         #print sim.G.G_nav.node[p]['sec']
	#     #print
	#     #print sim.queue[0].FPs[1].p_nav
	#     for p in sim.queue[0].FPs[2].p_nav:
	#         #print sim.G.G_nav.node[p]['sec']
	#     sim.queue[0].FPs[1].p_nav
	#     draw_network_map(sim.G.G_nav, title=sim.G.name, load=False, generated=True,\
	#             airports=True, trajectories=[sim.queue[0].FPs[0].p_nav], add_to_title='_nav_high_beta', polygons=sim.G.polygons.values())
	#     draw_network_map(sim.G.G_nav, title=sim.G.name, load=False, generated=True,\
	#             airports=True, trajectories=[sim.queue[0].FPs[2].p_nav], add_to_title='_nav_high_beta', polygons=sim.G.polygons.values())
	#     plt.show()
		
	#sim.save()