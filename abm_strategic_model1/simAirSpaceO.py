#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../libs/YenKSP')

import os
from os.path import join

import networkx as nx
from random import sample, uniform, gauss, shuffle
import numpy as np
from numpy import sqrt, exp, log
from numpy.random import lognormal
import matplotlib.delaunay as triang
import pickle
from itertools import takewhile

from libs.general_tools import delay, build_triangular, clock_time
from libs.YenKSP.graph import DiGraph
from libs.YenKSP.algorithms import ksp_yen

version = '3.2.0'

class ModelException(Exception):
	pass

class Network_Manager:
	"""
	Class Network_Manager 
	=====================
	The network manager receives flight plans from air companies and tries to
	fill the best ones on the network, by increasing order of cost. If the 
	flight plan does not overreach the capacity of any sector, it is 
	allocated to the network and the sector loads are updated. 
	The network manager can also inform the air companies if a shock occurs on
	the network, i.e if some sectors are shut. It asks for a new bunch of 
	flights plans from the airlines impacted.

	Notes
	-----
	New in 3.0.0: taken and adapted from Model 2

	(From Model 2)
	New in 2.9.2: gather methods coming from class Net (and Simulation) to make a proper agent.

	"""
	
	def __init__(self, old_style=False, discard_first_and_last_node=True):
		"""

		Parameters
		----------
		old_style: booelan
			'Old style' means that we detect the peaks in the loads (we have the full 
			profile of the load with time, and we detect if the maximum is smaller than the capacity)
			The 'new style computes' just the number of flights which cross the sector during a hour, 
			and then detect if this number is smaller than the capacity.

		Notes
		-----
		Not changed w.r.t Model 2
		
		"""
		
		self.discard_first_and_last_node = discard_first_and_last_node
		self.old_style = old_style 
		if not old_style:
			self.overload_sector = self.overload_sector_hours
			self.allocate = self.allocate_hours
			self.deallocate = self.deallocate_hours
			self.overload_airport = self.overload_airport_hours
		else:
			self.overload_sector = self.overload_sector_peaks
			self.allocate = self.allocate_peaks
			self.deallocate = self.deallocate_peaks
			self.overload_airport = self.overload_airport_peaks

	def initialize_load(self, G, control_time_window):
		"""
		Initialize loads for network G. If the NM is new style, it creates for each node a 
		list of length length_day which represents loads. Load is increased by one when a 
		flights crosses the airspace in the corresponding hour. Note that there is no explicit
		dates. If the NM is old style, the load is represented by a list of the 2-lists. The
		is a float representing the time at which the load changes, the second element is the
		load of the sector starting from the first element to the first element of the next 
		2-list.

		The load of normal sectors and airport sectors are tracked separately.

		Examples:
		---------
		This sequence:
		G.node[n]['load_old'] = [[0, 0], [10., 1], [10.5, 2], [15.5, 1], [20., 0], [10**6,0]]
		Means that there is no flight between 0. and 10., one flight between 10. and 10.5, 
		two between 10.5 and 15.5, one again between 15.5 and 20. and no flight afterwards.

		Parameters
		----------
		G : Net object
			or networkx
		control_time_window : int
			Number of hours tracked by the Network Manager. Should be greater than the length
			of the day because of the shifting behaviour of the companies.

		Notes
		-----
		Unchanged w.r.t Model 2.

		Changed in 2.2: keeps in memory only the intervals.
		Changed in 2.5: t_max is set to 10**6.
		Changed in 2.7: load of airports added.
		Changed in 2.9: no more load of airports. Load is an array giving the load for each hour.
		Changed in 2.9.3: airports again :).
		Changed in 3.1.0: the control_time_window is now compulsory.

		"""

		self.control_time_window = control_time_window

		if not self.old_style:
			for n in G.nodes():
				G.node[n]['load']=[0 for i in range(control_time_window)]
			for a in G.airports:
				G.node[a]['load_airport']=[0 for i in range(control_time_window)]
		else:
			for n in G.nodes():
				G.node[n]['load']=[[0,0],[10**6,0]] 
			for a in G.airports:
				G.node[a]['load_airport']=[[0,0],[10**6,0]] 

	def build_queue(self, ACs):
		"""
		Add all the flights of all ACs to a queue, in random order.

		Parameters
		----------
		ACs : a list of AirCompany objects 
			The AirCompanys must have their flights and flight plans computed. 

		Returns
		-------
		queue : a list of objects flights
			This is the priority list for the allocation of flights.

		Notes
		-----
		Unchanged w.r.t Model 2

		"""

		queue=[]
		for ac in ACs.values():
			for f in ac.flights:
				queue.append(f)
		shuffle(queue)

		return queue

	def allocate_queue(self, G, queue, storymode=False):
		"""
		For each flight of the queue, tries to allocate it to the airspace.

		Notes
		-----
		Unchanged w.r.t Model 2

		"""

		for i,f in enumerate(queue):
			if storymode:
				print "Flight with position", i, "from", f.source, "to", f.destination, "of company", f.ac_id
				print "with parameters", f.par
				print "tries to be allocated."
			f.pos_queue = i
			self.allocate_flight(G, f, storymode=storymode)
			if storymode:
				print "flight accepted:", f.fp_selected!=None
				if f.fp_selected==None:
					print 'because '
				print
				print 
				
	def allocate_flight(self, G, flight, storymode=False):
		"""
		Tries to allocate the flights by sequentially checking if each flight plan does not overload any sector,
		beginning with the best ones. The rejection of the flights is kept in memory, as
		well as the first sector overloaded (bottleneck), and the flight plan selected.

		Parameters
		----------
		G : Net Object
			(Sector) Network on which on which the flights will be allocated. 
		flight : Flight object
			The flight which has to allocated to the network.
		storymode : bool, optional
			Used to print very descriptive output.

		Notes
		-----
		Unchanged w.r.t. Model 2

		Changed in 2.2: using intervals.
		Changed in 2.7: sectors of airports are checked independently.
		Changed in 2.9: airports are not checked anymore.
		Changed in 2.9.3: airports are checked again :).

		"""

		i = 0
		found = False
		#print 'flight id', flight.ac_id
		while i<len(flight.FPs) and not found:
			# print 'fp id', i
			fp = flight.FPs[i]
			#print 'fp.p', fp.p
			self.compute_flight_times(G, fp)
			path, times = fp.p, fp.times

			if storymode:
				print "     FP no", i, "tries to be allocated with trajectory (sectors):"
				print fp.p
				print "and crossing times:"
				print fp.times

			if self.discard_first_and_last_node:
				first = 1 ###### ATTENTION !!!!!!!!!!!
				last = len(path)-1 ########## ATTENTION !!!!!!!!!!!
			else:
				first = 0 ###### ATTENTION !!!!!!!!!!!    
				last = len(path) ########## ATTENTION !!!!!!!!!!!
			
			j = first
			while j<last and not self.overload_sector(G, path[j],(times[j],times[j+1])):#and self.node[path[j]]['load'][j+time] + 1 <= self.node[path[j]]['capacity']:
				j += 1 

			fp.accepted = not ((j<last) or self.overload_airport(G, path[0],(times[0],times[1])) or self.overload_airport(G, path[-1],(times[-2],times[-1])))
				  
			path_overload = j<last
			source_overload = self.overload_airport(G, path[0],(times[0],times[1]))
			desetination_overload = self.overload_airport(G, path[-1],(times[-2],times[-1]))

			if storymode:
				print "     FP has been accepted:", fp.accepted
				if not fp.accepted:
					if path_overload: 
						print "     because sector", path[j], "was full."
					if source_overload:
						print "     because source airport was full."
						print G.node[path[0]]['load_airport']
						print G.node[path[0]]['capacity_airport']
					if desetination_overload:
						print "     because destination airport was full."

			if fp.accepted:
				self.allocate(G, fp, storymode=storymode, first=first, last=last)
				flight.fp_selected=fp
				flight.accepted = True
				found=True
			else:
				if j<last:
					fp.bottleneck=path[j]
			i+=1 

		if not found:
			flight.fp_selected=None
			flight.accepted = False
		
	def compute_flight_times(self, G, fp):
		"""
		Compute the entry times and exit times of each sector for the trajectory of the given flight
		plan. Store them in the 'times' attribute of the flight plan.

		Parameters
		----------
		G : hybrid network.
		fp : FlightPlan object.
		
		Notes
		-----
		Changed in 3.0.0: adapted for Model 1

		(From Model 2)
		Changed in 2.8: based on navpoints.

		"""
		
		ints = [0.]*(len(fp.p)+1)
		ints[0] = fp.t
		road = fp.t
		for i in range(1,len(fp.p)):
			w = G[fp.p[i-1]][fp.p[i]]['weight']
			ints[i] = road + w/2.
			road += w
		ints[len(fp.p)] = road        
		fp.times = ints
		
	def overload_sector_hours(self, G, n, (t1, t2)):
		"""
		Check if the sector n would be overloaded if an additional flight were to 
		cross it between times t1 and t2.

		Parameters
		----------
		G : Net object
			Unmodified.
		n : int or string
			sector to check.
		(t1, t2) : (float, float)
			times of entry and exit of the flight in the sector n.
		
		Returns
		-------
		overload : boolean,
			True if the sector would be overloaded with the allocation of this flight plan.

		Raises
		------
		ModelException
			when the flight plan is partly outside of the control window.

		Notes
		-----
		Unchanged w.r.t. Model 2.
		Changed in 2.9: it does not check anymore if the maximum number of flights have 
		overreached the capacity, but if the total number of flights during an hour 
		(counting those already there at the beginning and those still there at the end)
		is greater than the capacity. 
		Changed in 2.9.8: added the condition h<len(G.node[n]['load']). There is now an 
		absolute reference in time and the weights of the network are in minutes
		Changed in 3.1.0: added an exception when the flight is partly outside of the 
		control window.

		"""

		overload = False
		h = 0
		if t2/60.>=len(G.node[n]['load']):
			raise ModelException("The arrival of a flight is after the end of the control window!\n\
								Arrival time: " + str(t2/60.) + ' ; End of time window: ' + str(len(G.node[n]['load'])))

		while float(h) <= t2/60. and h<len(G.node[n]['load']) and not overload:
			if h+1 > t1/60. and G.node[n]['load'][h]+1>G.node[n]['capacity']:
				overload = True
			h += 1

		return overload

	def overload_sector_peaks(self, G, n, (t1, t2)):
		"""
		Old version (2.8.2) of previous method. Based on maximum number of 
		planes in a given sector at any time. See initialize_load method
		for more details.

		Notes
		-----
		Unchanged w.r.t. Model 2.
		Was previously called overload_capacity

		"""
		
		ints = np.array([p[0] for p in G.node[n]['load']])
		
		caps = np.array([p[1] for p in G.node[n]['load']])
		i1 = max(0,list(ints>=t1).index(True)-1)
		i2 = list(ints>=t2).index(True)
		
		pouet = np.array([caps[i]+1 for i in range(i1,i2)])

		return len(pouet[pouet>G.node[n]['capacity']]) > 0

	def overload_airport_hours(self, G, n, (t1, t2)):
		"""
		Same than overload_sector_hours, for airports.

		Notes
		------
		Unchanged w.r.t to Model 2
		Changed in 2.9.8: added the condition h<len(G.node[n]['load'])
		"""
		
		overload = False
		h = 0 
		while float(h) <= t2/60. and h<len(G.node[n]['load']) and not overload:
			if h+1 > t1/60. and G.node[n]['load_airport'][h]+1>G.node[n]['capacity_airport']:
				overload = True
			h += 1
		return overload

	def overload_airport_peaks(self, G, n, (t1, t2)):
		"""
		Same then overload_sector_peaks, for airports.

		Notes
		-----
		Unchanged w.r.t. Model 2.
		"""

		ints = np.array([p[0] for p in G.node[n]['load_airport']])
		
		caps = np.array([p[1] for p in G.node[n]['load_airport']])
		i1 = max(0,list(ints>=t1).index(True)-1)
		i2 = list(ints>=t2).index(True)
		
		pouet = np.array([caps[i]+1 for i in range(i1,i2)])

		return len(pouet[pouet>G.node[n]['capacity_airport']]) > 0
		
	def allocate_hours(self, G, fp, storymode=False, first=0, last=-1):
		"""
		Fill the network with the given flight plan. For each sector of the flight plan, 
		add one to the load for each slice of time (one hour slices) in which the flight 
		is present in the sector. 
		The 'first' and 'last' optional arguments are used if the user wants to avoid 
		to load the first and last sectors, as it was in the first versions.

		Parameters
		----------
		G : Net objeect
			Modified in output with loads updated.
		fp : FlightPlan object
			flight plan to allocate.
		storymode : boolean, optional
			verbosity.
		first : int, optional
			position of the first sector to load in the trajectory. Deprecated.
		last : int, optional
			position of the last sector to load. Deprecated.

		Notes
		-----
		Unchanged w.r.t. Model 2

		(From Model 2)
		Changed in 2.9: completely changed (count number of flights per hour).
		Changed in 2.9.5: does change the load of the first and last sector.
		Changed in 2.9.8: added condition h<G.node[n]['load']

		"""

		if storymode:
			print "NM allocates the flight."
		path, times = fp.p, fp.times
		#for i in range(1,len(path)-1):
		if last==-1: 
			last = len(path)
		for i in range(first, last):
			n = path[i]
			t1, t2 = times[i]/60.,times[i+1]/60.
			h = 0
			while h<t2 and h<len(G.node[n]['load']):
				if h+1>t1:
					if storymode:
						print "Load of sector", n, "goes from",  G.node[n]['load'][h], "to", G.node[n]['load'][h]+1, "for interval", h, "--", h+1
					G.node[n]['load'][h] += 1
				h+=1

	def allocate_peaks(self, G, fp, storymode=False, first=0, last=-1):
		"""
		Old version of previous method.

		Notes
		-----
		Unchanged w.r.t Model 2.
		
		"""

		path, times = fp.p, fp.times
		if last==-1: 
			last = len(path)
		#for i, n in enumerate(path):
		for i in range(first, last):
			n = path[i]
			t1, t2 = times[i],times[i+1]
			ints = np.array([p[0] for p in G.node[n]['load']])
			caps = np.array([p[1] for p in G.node[n]['load']])
			i1 = list(ints>=t1).index(True)
			i2 = list(ints>=t2).index(True)
			if ints[i2]!=t2:
				G.node[n]['load'].insert(i2,[t2,caps[i2-1]])
			if ints[i1]!=t1:
				G.node[n]['load'].insert(i1,[t1,caps[i1-1]])
				i2 += 1
			for k in range(i1,i2):
				G.node[n]['load'][k][1] += 1

	def deallocate_hours(self, G, fp, first=0, last=-1):
		"""
		Used to deallocate a flight plan not legit anymore, for instance because one 
		sector has been shutdown.
		
		Parameters
		----------
		G : Net object,
			Loads are modified as output
		fp : FlightPlan object
			Flight plan to deallocate.
		first : int, optional
			position of the first sector to unload in the trajectory. Must be consistent
			with the allocation's parameters. Deprecated.
		last : int, optional
			position of the last sector to unload. Must be consistent
			with the allocation's parameters. Deprecated.

		Notes
		-----
		Unchanged w.r.t. Model 2.

		New in 2.5
		Changed in 2.9: completely changed, based on hour slices.

		"""
		
		path,times = fp.p,fp.times
		if last==-1: 
			last = len(path)
		#for i in range(1, len(path)-1):
		for i in range(first, last):
			n = path[i]
			t1,t2 = times[i]/60.,times[i+1]/60.
			h=0
			while h<t2:
				if h+1>t1:
					G.node[n]['load'][h]-=1
				h+=1

	def deallocate_peaks(self, fp):
		"""
		Old version of previous deallocate_hours. See description of initialize_load.
		
		Notes
		-----
		Unchanged w.r.t. Model 2.
		"""

		path,times=fp.p,fp.times
		for i,n in enumerate(path):
			t1,t2=times[i],times[i+1]
			ints=np.array([p[0] for p in G.node[n]['load_old']])
			i1=list(ints==t1).index(True)
			i2=list(ints==t2).index(True)
			for k in range(i1,i2):
				G.node[n]['load_old'][k][1]-=1
			
			if G.node[n]['load_old'][i2-1][1]==G.node[n]['load_old'][i2][1]:
				G.node[n]['load_old'].remove([t2,G.node[n]['load_old'][i2][1]])
			if G.node[n]['load_old'][i1-1][1]==G.node[n]['load_old'][i1][1]:
				G.node[n]['load_old'].remove([t1,G.node[n]['load_old'][i1][1]])

	def M0_to_M1(self, G, queue, N_shocks, tau, storymode=False):
		"""
		Routine aiming at modelling the shut down of sectors due to bad weather or strikes. Some 
		sectors are shut down at random. Flight plans crossing these sectors are deallocated. 
		Shortest paths are recomputed. Finally, deallocated flights are reallocated on the 
		new network, with the same initial order. This procedure is repeated after each sector 
		is shut down.
		
		Parameters
		----------
		G : hybrid network
		queue : list of Flight objects
			initial queue before the shocks.
		N_shocks : int
			number of sectors to shut down.
		tau : float
			parameter of shift in time (TODO: should not really be here...).
		
		Notes
		-----
		Chnaged in 3.0.0: apdated from Model 2.

		(From Model 2)
		Changed in 2.9: updated for navpoints and can now shut down sectors containing airports.
		Transferred from simulationO.

		"""
		
		sectors_to_shut = sample(G.nodes(), N_shocks)

		for n in sectors_to_shut:
			flights_to_reallocate = []
			flights_suppressed = []          
			sec_pairs_to_compute = []
			#nav_pairs_to_compute = []
			for f in queue:
				if f.accepted:
					path_sec = f.fp_selected.p
					if n in path_sec:
						if path_sec[0]==n or path_sec[-1]==n:
							flights_suppressed.append(f) # The flight is suppressed if the source or the destination is within the shut sector
						else:
							flights_to_reallocate.append(f)
							sec_pairs_to_compute.append((path_sec[0], path_sec[-1]))
							#nav_pairs_to_compute.append((f.fp_selected.p_nav[0], f.fp_selected.p_nav[-1]))

			sec_pairs_to_compute = list(set(sec_pairs_to_compute))
			#nav_pairs_to_compute = list(set(nav_pairs_to_compute))
					
			if storymode:
				first_suppressions = len(flights_suppressed)
				print
				print 'Shutting sector', n
				print 'Number of flights to be reallocated:', len(flights_to_reallocate)
				print 'Number of flights suppressed:', len(flights_suppressed)

			for f in flights_to_reallocate + flights_suppressed:
				self.deallocate(G, f.fp_selected)
				#queue.remove(f)
			
			G.shut_sector(n)
			G.build_H()
			#G.G_nav.build_H()

			if storymode:
				print 'Recomputing shortest paths...'

			G.compute_shortest_paths(G.Nfp, repetitions=False, delete_pairs=False)
			#G.compute_all_shortest_paths(Nsp_nav, perform_checks=True, sec_pairs_to_compute=sec_pairs_to_compute, nav_pairs_to_compute=nav_pairs_to_compute)                
			
			for f in flights_to_reallocate:
				if not (f.source, f.destination) in G.short.keys():
					flights_suppressed.append(f)
				else:
					f.compute_flightplans(tau, G)
					self.allocate_flight(G, f)

			if storymode:
				print 'There were', len(flights_suppressed) - first_suppressions, 'additional flights which can not be allocated.'
			for f in flights_suppressed:
				for fp in f.FPs:
					fp.accepted = False
				f.accepted = False

	def M0_to_M1_quick(self, G, queue, N_shocks, tau, storymode=False, sectors_to_shut=None):
		"""
		Same method than previous one, but closes all sectors at the same time, 
		then recomputes the shortest paths.

		Parameters
		----------
		G : Net object
		queue : list of Flight objects
			initial queue before the shocks.
		N_shocks : int
			number of sectors to shut down.
		tau : float
			parameter of shift in time (TODO: should not really be here...).
		
		Notes
		-----
		Changed in 3.0.0: Adapted from Model 2.

		(From Model 2)
		New in 2.9.7.
		
		"""
		
		#if storymode:
		#    print "N_shocks:", N_shocks
		if sectors_to_shut==None:
			#sectors_to_shut = shock_sectors(G, N_shocks)#sample(G.nodes(), N_shocks)
			sectors_to_shut = sample(G.nodes(), int(N_shocks))
		else:
			sectors_to_shut = [sectors_to_shut]

		if sectors_to_shut!=[]:
			flights_to_reallocate = []
			flights_suppressed = []          
			sec_pairs_to_compute = []
			#nav_pairs_to_compute = []
			for f in queue:
				if f.accepted:
					path_sec = f.fp_selected.p
					if set(sectors_to_shut).intersection(set(path_sec))!=set([]):
						if path_sec[0] in sectors_to_shut or path_sec[-1] in sectors_to_shut:
							flights_suppressed.append(f) # The flight is suppressed if the source or the destination is within the shut sector
						else:
							flights_to_reallocate.append(f)
							sec_pairs_to_compute.append((path_sec[0], path_sec[-1]))
							#nav_pairs_to_compute.append((f.fp_selected.p_nav[0], f.fp_selected.p_nav[-1]))

			sec_pairs_to_compute = list(set(sec_pairs_to_compute))
			#nav_pairs_to_compute = list(set(nav_pairs_to_compute))
					
			if storymode:
				first_suppressions = len(flights_suppressed)
				print
				print 'Shutting sectors', sectors_to_shut
				print 'Number of flights to be reallocated:', len(flights_to_reallocate)
				print 'Number of flights suppressed:', len(flights_suppressed)

			for f in flights_to_reallocate + flights_suppressed:
				self.deallocate(G, f.fp_selected)
				#queue.remove(f)
			
			for n in sectors_to_shut:
				G.shut_sector(n)

			G.build_H()
			#G.G_nav.build_H()

			if storymode:
				print 'Recomputing shortest paths...'
			G.compute_shortest_paths(G.Nfp, repetitions=False, delete_pairs=False)
			#G.compute_all_shortest_paths(Nsp_nav, perform_checks=True, sec_pairs_to_compute=sec_pairs_to_compute, nav_pairs_to_compute=nav_pairs_to_compute, verb = storymode)                
			
			for f in flights_to_reallocate:
				if not (f.source, f.destination) in G.short.keys():
					flights_suppressed.append(f)
				else:
					f.compute_flightplans(tau, G)
					self.allocate_flight(G, f)

			if storymode:
				print 'There were', len(flights_suppressed) - first_suppressions, 'additional flights which can not be allocated.'
				print
			for f in flights_suppressed:
				for fp in f.FPs:
					fp.accepted = False
				f.accepted = False

class FlightPlan:
	"""
	Class FlightPlan. 
	=============
	Keeps in memory its path, time of departure, cost and id of AC.

	Notes
	-----
	Changed in 3.0.0: adapted from Model 2

	(From Model 2)
	Changed in 2.8: added p_nav.
	Changed in 2.9.6: added shift_time method.
	
	"""
	
	def __init__(self, path, time, cost, ac_id):
		"""
		Parameters
		----------
		path : list of sectors
		time : float
			time of departure in minutes.
		cost : float
			Cost of the nav-path given by the utility function of the company.
		ac_id : int
			Id of the Air Company.

		"""

		self.p = path # path in sectors
		self.t = time # of departure
		self.cost = cost # cost given the utility function
		self.ac_id = ac_id # id of the air company
		self.accepted = True # if the flight plan has been accepted by the NM.
		self.bottleneck = -1 # for post-processing.

	def shift_time(self, shift):
		"""
		Shift the time of departure by shift (in minutes).
		"""
		self.t += shift

	def __repr__(self):
		return 'FP with departure ime: ' + str(self.t) + ' ; and path:' + str(self.p)

class Flight:
	"""
	Class Flight. 
	=============
	Keeps in memory its id, source, destination, prefered time of departure and id of AC.
	Thanks to AirCompany, keeps also in memory its flight plans (self.FPs).

	Notes
	-----
	Changed in 3.0.0: adapted from model 2.

	(From Model 2)
	Changed in 2.9.6: Compute FPs added (coming from AirCompany object).
	New in 2.9.6: method shift_desired_time.
	
	"""
	
	def __init__(self, Id, source, destination, pref_time, ac_id, par, Nfp):
		"""
		Parameters
		----------
		Id : int
			Identifier of the flight, relative to the AirCompany
		source : int or string
			label of origin node
		destination : int or string
			label of destination node.
		pref_time : float
			Preferred time of departure, in minutes (from time 0, beginning of the day)
		ac_id : int
			Unique Id of the AirCompany.
		par : tuple (float, float, float)
			behavioral parameter of the AirCompany for utility function.
		Nfp : int
			Maximum number of flights plans that the AirCompany is going to submit 
			for this flight. 

		Notes
		-----
		Changed in 2.9.6: added Nfp.

		"""
		self.id = Id
		self.source = source
		self.destination = destination
		self.pref_time = pref_time
		self.ac_id = ac_id 
		self.par = par
		self.Nfp = Nfp

	def compute_flightplans(self, tau, G): 
		"""
		Compute the flight plans for a given flight, based on Nfp and the best paths,
		 and the utility function.
		
		Parameters
		----------
		tau : float
			The different flight plans of the flight will be shifted by this amount (in minutes).
		G : Net object
			Used to compute cost of paths. Not modified.

		Raises
		------
		Exception
			If some pairs in the network to not have enough shortest paths
		Exception
			If the list of flight plans in output is smaller than self.Nfp

		Notes
		-----
		New in 3.0.0: adapted from Model 2

		(From Model 2)
		Changed in 2.2: tau introduced.
		Changed in 2.8: paths made of navpoints and then converted.
		Changed in 2.9: ai and aj are navpoints, not sectors.
		Changed in 2.9.6: use the convert_path method of the Net object.
		New in 2.9.6: comes from AirCompany object.
		Changed in 2.9.7: ai and aj are source and destination.
		Changed in 2.9.9: resolved a serious bug of references on paths.
		Changed in 3.1.1: slightly optimized.
		
		"""

		ai, aj = self.source, self.destination
		t0sp = self.pref_time

		# Check that all origin-destination pairs in the network
		# have a number of shortest paths exactly equal to the number 
		# of flight plans to be submitted.
		try:
			for k, v in G.short.items():
				assert len(v)==G.Nfp
		except:
			raise Exception("OD Pair", k, "have", len(v), "shortest paths whereas", G.Nfp, "were required.")

		# For each shortest path, compute the path in sectors and the total weight of the nav-path
		SP = [(p, G.weight_path(p)) for p in G.short[(ai,aj)]]

		# # Compute the cost of the worst path (with desired time).
		uworst = utility(self.par, SP[0][-1], t0sp, SP[-1][-1], t0sp)

		# Compute the cost of all paths which have a cost smaller than uworst 
		# Old one (slightly less efficient)
		# u = [[(p, t0sp + i*tau, utility(self.par, SP[0][-1], t0sp, c, t0sp + i*tau)) for p, c in SP] for i in range(self.Nfp)\
		# 	if utility(self.par,SP[0][-1], t0sp, SP[0][-1],t0sp + i*tau)<=uworst]

		u = [list(takewhile(lambda (x, y, cost): cost<=uworst, ((p, t0sp+i*tau, utility(self.par, SP[0][-1], t0sp, c, t0sp+i*tau)) for i in range(self.Nfp)))) for p, c in SP]

		# Select the Nfp flight plans less costly, ordered by increasing cost.
		fp = [FlightPlan(a[0][:], a[1], a[2], self.id) for a in sorted([item for sublist in u for item in sublist], key=lambda a: a[2])[:self.Nfp]]
				
		if len(fp)!=self.Nfp:
			raise Exception('Problem: there are', len(fp), 'flights plans whereas there should be', self.Nfp)
	
		if not G.weighted:
			# Shuffle the flight plans with equal utility function
			uniq_util = np.unique([item.cost for item in fp])
			sfp = []
			for i in uniq_util:
				v = [item for item in fp if item.cost==i]
				shuffle(v)
				sfp = sfp+v
			fp = sfp
		
		self.FPs = fp
		
	def make_flags(self):
		"""
		Used for post-processing.
		Used to remember the flight plans which were overloading the network, 
		as well as the first sector to be overloaded on the trajectories.
		"""
		try:
			self.flag_first = [fp.accepted for fp in self.FPs].index(True)
		except ValueError:
			self.flag_first = len(self.FPs)
			
		self.overloadedFPs = [self.FPs[n].p for n in range(0, self.flag_first)]
		self.bottlenecks = [fp.bottleneck for fp in self.FPs if fp.bottleneck!=-1]
	   
	def shift_desired_time(self, shift):
		"""
		Shift the desired time of all flight plans of the flight.

		Parameters
		----------
		shift : float
			Amount of time in minutes.
		
		Notes
		-----
		New in 3.0.0: taken from Model 2 (unchanged).

		"""
		
		shift = int(shift)
		self.pref_time += shift
		for fp in self.FPs:
			fp.shift_time(shift)

	def __repr__(self):
		return 'Flight number ' + str(self.id) + ' from AC number ' + str(self.ac_id) +\
			' from ' + str(self.source) + ' to ' + str(self.destination)

	def show_fps(self):
		print 'flight', self.id, 'with parameters', self.par
		for fp in self.FPs:
			print fp
		
class AirCompany:
	"""
	Class AirCompany
	================
	Keeps in memory the underliying network and several parameters, in particular the 
	coefficients for the utility function and the pairs of airports used.

	Notes
	-----
	Only slightly modified by Model 2. Removed a couple of unused methods.

	"""
	
	def __init__(self, Id, Nfp, na, pairs, par):
		"""
		Initialize the AirCompany.

		Parameters
		----------
		Id: integer
			unique identifier of the company.
		Nfp : integer
			Number of flights plans that flights will submit
		na: integer
			Number of flights per destination-origin operated by the Air company.
			Right now the Model supports only na=1
		pairs : list of tuple with origin-destination
			departure/arrivale point will be drawn from them if not specified in fill_FPs
		par : tuple (float, float, float)
			Parameters for the utility function

		"""

		try:
			assert na==1
		except AssertionError:
			raise Exception("na!=1 is not supported by the model.")
		self.Nfp = Nfp
		self.par = par
		self.pairs = pairs
		self.na = na
		self.id = Id
		
	def fill_FPs(self,t0spV, tau, G, pairs=[]):
		"""
		Fill na flights with Nfp flight plans each, between airports given by pairs.

		Parameters
		----------
		t0spV : iterable with floats
			Desired times of departure for each origin-destination pair.
		tau : float
			Amount of time (in seconds) used to shift the flights plans
		G : Net Object
			(Sector) Network on which on which the flights will be allocated. 
			Needs to have an attribute G_nav which is the network of navpoints.
		pairs : list of tuples (int, int), optional
			If given, it is used as the list origin destination for the flights. 
			Otherwise self.pairs is used.

		Notes
		-----
		Changed in 3.0.0: taken from Model 2 (unchanged)

		New in 2.9.5: can specify a pair of airports.
		Changed in 2.9.6: the flight computes the flight plans itself.

		"""
			
		if pairs==[]:
			assigned_airports = sample(self.pairs, self.na) 
		else:
			assigned_airports = pairs

		self.flights = []
		i = 0
		for (ai,aj) in assigned_airports:
			self.flights.append(Flight(i, ai, aj, t0spV[i], self.id, self.par, self.Nfp))
			self.flights[-1].compute_flightplans(tau, G)
			i += 1

	def __repr__(self):
		return 'AC with para ' + str(self.par)

	def show_flights(self):
		for f in self.flights:
			f.show_fps()
		
class Net(nx.Graph):
	"""
	Class Net
	=========
	Derived from nx.Graph. Several methods added to build it, generate, etc...

	Notes
	----
	Changes in 3.0.0: sorted methods and imported methods from Model 2.

	"""
	
	def __init__(self):
		super(Net, self).__init__()

	def describe(self, level=0):
		print "Net", self.name
		print "================"
		print "Number of nodes:", len(self.nodes())
		print "Number of airports:", len(self.get_airports())
		print "Number of connections:", len(self.connections())
		if level>0:
			print "Airports:", self.get_airports()
			print "Connections:", self.connections()
		if level>1:
			print "Shortest paths with weights:"
			for k, v in self.short.items():
				print k, ':'
				for p in v:
					print ' - ', p, len(p), self.weight_path(p)
	
	def add_airports(self, airports, C_airport=10):
		"""
		Add airports given by user. 

		Parameters
		----------
		C_airport : int, optional
			Capacity of the sectors which are airports. They are used only by flights which are
			departing or lending in this area. It is different from the standard capacity key
			which is used for flights crossing the area, which is set to 10000.
		
		Notes
		-----
		New in 3.0.0: taken from Model 2
		Changed in 3.2.0: removed connections generation.

		(From Model 2)
		Changed in 2.9.8: changed name to add_airports. Now the airports are added
		to the existing airports instead of overwriting the list.

		"""

		if not hasattr(self, "airports"):
			self.airports = airports
		else:
			self.airports = np.array(list(set(list(self.airports) + list(airports))))
			
		# if not hasattr(self, "short"):
		# 	self.short = {}

		# if pairs==[]:
		# 	for ai in self.airports:
		# 		for aj in self.airports:
		# 			if len(nx.shortest_path(self, ai, aj))-2>=min_dis and ((not singletons and ai!=aj) or singletons):
		# 				if not self.short.has_key((ai,aj)):
		# 					self.short[(ai, aj)] = []
		# else:
		# 	for (ai,aj) in pairs:
		# 		 if ((not singletons and ai!=aj) or singletons):
		# 			if not self.short.has_key((ai,aj)):
		# 				self.short[(ai, aj)] = []

		for a in airports:
			#self.node[a]['capacity']=100000                # TODO: check this.
			self.node[a]['capacity_airport'] = C_airport

	def basic_statistics(self, rep='.', name=None):
		"""
		Computes basic stats on degree, weights and capacities. 

		Parameters
		----------
		rep : string, optional
			directory in which the stats are saved.
		name : string, optional
			name of the file (without extension).

		Notes
		-----
		TODO: expand this.
		Changed in 3.0.0: taken from Model 2 (unchanged)

		"""
		if name==None:
			name = self.name

		os.system('mkdir -p ' + rep)
		print 'basic_statistics', join(rep, name + '_basic_stats_net.txt')
		with open(join(rep, name + '_basic_stats_net.txt'), 'w') as f:
			print >>f, 'Mean/std degree:', np.mean([self.degree(n) for n in self.nodes()]), np.std([self.degree(n) for n in self.nodes()])
			print >>f, 'Mean/std weight:', np.mean([self[e[0]][e[1]]['weight'] for e in self.edges()]), np.std([self[e[0]][e[1]]['weight'] for e in self.edges()])
			print >>f, 'Mean/std capacity:', np.mean([self.node[n]['capacity'] for n in self.nodes() if not n in self.airports]),\
				np.std([self.node[n]['capacity'] for n in self.nodes() if not n in self.airports])
		
	def build(self, N, Gtype='D', mean_degree=6, prelist=[], put_nodes_at_corners=False):
		"""
		Build a graph of the given type, nodes, edges.
		Essentially gathers build_nodes and build_net methods and add the possibility 
		of building a simple triangular network (not sure it is up to date though).

		Parameters
		----------
		N : int
			Number of nodes to produce.
		Gtype : string, 'D', 'E', or 'T'
			type of graph to be generated. 'D' generates a dealunay triangulation 
			which in particular is planar and has the highest degree possible for a planar 
			graph. 'E' generates an Erdos-Renyi graph. 'T' builds a triangular network
		mean_degree : float
			mean degree for the Erdos-Renyi graph.
		prelist : list of 2-tuples, optional
			Coordinates of nodes to add to the nodes previously generated
		put_nodes_at_corners : boolean, optional
			if True, put a nodes at each corner for the 1x1 square

		Notes
		-----
		Changed in 3.0.0: taken and adapted from Model 2.

		"""

		print 'Building random network of type', Gtype
		
		if Gtype=='D' or Gtype=='E':
			self.build_nodes(N, prelist=prelist, put_nodes_at_corners=put_nodes_at_corners)
			self.build_net(Gtype=Gtype, mean_degree=mean_degree)  
		elif Gtype=='T':
			xAxesNodes = int(np.sqrt(N/float(1.4)))
			self.import_from(build_triangular(xAxesNodes, x_shift=-1., y_shift=-1., side=2.))
	
	def build_H(self): 
		"""
		Build the DiGraph object used in the ksp_yen algorithm.
		"""
		self.H = DiGraph()
		self.H._data = {}
		for n in self.nodes():
			self.H.add_node(n)
		for e in self.edges():
			self.H.add_edge(e[0],e[1], cost=self[e[0]][e[1]]['weight'])
			self.H.add_edge(e[1],e[0], cost=self[e[1]][e[0]]['weight'])

	def build_nodes(self, N, prelist=[], put_nodes_at_corners=False, small=1.e-5):
		"""
		Add N nodes to the network, with coordinates taken uniformly in a square. 
		Alternatively, prelist gives the list of coordinates.

		Parameters
		----------
		N : int
			Number of nodes to produce
		prelist : list of 2-tuples, optional
			Coordinates of nodes to add to the nodes previously generated
		put_nodes_at_corners : boolean, optional
			if True, put a nodes at each corner for the 1x1 square
		small : float, optional
			Used to be sure that the nodes in the corner are strictly within the
			square

		Notes
		-----
		New in 3.0.0: taken from Model 2 (unchanged)
		(From Model 2)
		Remark: the network should not have any nodes yet.
		New in 2.8.2

		"""
		for i in range(N):
			self.add_node(i,coord=[uniform(-1.,1.),uniform(-1.,1.)])  
		for j,cc in enumerate(prelist):
			self.add_node(N+j,coord=cc)
		if put_nodes_at_corners:
			self.add_node(N+len(prelist), coord=[1.-small, 1.-small])
			self.add_node(N+len(prelist)+1, coord=[-1.+small, 1.-small])
			self.add_node(N+len(prelist)+2, coord=[-1.+small, -1.+small])
			self.add_node(N+len(prelist)+3, coord=[1.-small, -1.+small])

	def build_net(self, Gtype='D', mean_degree=6):
		"""
		Build edges, based on Delaunay triangulation or Erdos-Renyi graph. 
		No weight is computed at this point.

		Parameters
		----------
		Gtype : string, 'D' or 'E'
			type of graph to be generated. 'D' generates a dealunay triangulation 
			which in particular is planar and has the highest degree possible for a planar 
			graph. 'E' generates an Erdos-Renyi graph.
		mean_degree : float
			mean degree for the Erdos-Renyi graph.

		Notes
		-----
		New in 3.0.0: taken from Model 2 (unchanged)

		Changed in 2.9.10: removed argument N.

		"""

		if Gtype=='D':
			x,y =  np.array([self.node[n]['coord'][0] for n in self.nodes()]),np.array([self.node[n]['coord'][1] for n in self.nodes()])   
			cens, edg, tri, neig = triang.delaunay(x,y)
			for p in tri:
				self.add_edge(p[0],p[1])
				self.add_edge(p[1],p[2])
				self.add_edge(p[2],p[0])
		elif Gtype=='E':  
			N = len(self.nodes())
			prob = mean_degree/float(N-1) # <k> = (N-1)p - 6 is the mean degree in Delaunay triangulation.
			for n in self.nodes():
				for m in self.nodes():
					if n>m:
						if np.random.rand()<=prob:
							self.add_edge(n,m)

	def capacities(self):
		return {n:self.node[n]['capacity'] for n in self.nodes()}

	def compute_shortest_paths(self, Nfp, repetitions=True, pairs=[], 
		verb=1, delete_pairs=True):
		"""
		Pre-Build Nfp weighted shortest paths between each pair of airports. 
		If the function dooes not find enough paths, the corresponding source/destination pair can be deleted.
		
		Parameters
		----------
		Nfp : int
			Number of shortest paths to compute between each pair of airports.
		repetitions : boolean, optional
			If True, a path can have a given node twice or more. Otherwise, the function makes
			several iterations, considering longer and longer paths until it finds a path which doesn't have any 
			repeated sector.
		use_sector_path : boolean, optional
			If True, the nav-paths are generated so that the sector paths do not have repeated sectors. Does not 
			have any effect if repetitions is True.
		old : boolean, optional
			Should always be false. Used to compare with previous algorithm of YenKSP.
		pairs : list of 2-tuples, optional
			list of origin-destination for which the shortest paths will be computed. If [], all shortest paths
			will be computed.
		verb : int, optional
			verbosity
		delete_pairs : boolean, optional
			if True, all pairs for which not enough shortest paths have been found are deleted.

		Notes
		-----
		Changed in 3.0.0: taken and adapted from Model 2
		Changed in 3.0.0: fixed a bug where the paths were not correctly sorted in output.

		(From Model 2)
		Changed in 2.9: added singletons option. Added repetitions options to avoid repeated sectors in paths.
		Changed in 2.9.4: added procedure to have always 10 distinct paths (in sectors).
		Changed in 2.9.7: modified the location of the not enough_path loop to speed up the process. Added
		pairs_to_compute, so that it does not necesseraly recompute every shortest paths.
		Changed in 2.9.8: Added n_tries in case of use_sector_path.
		Changed in 2.9.10: added option to remove pairs which do not have enough paths. If disabled, the last paths 
		is directed until Nfp is reached.

		"""

		if pairs==[]:
			pairs = self.short.keys()[:]
		
		deleted_pairs = []
		if repetitions:
			for (a,b) in pairs:
				enough_paths = False
				Nfp_init = Nfp
				while not enough_paths:
					enough_paths=True
					#self.short={(a,b):self.kshortestPath(a, b, Nfp, old=old) for (a,b) in self.short.keys()}
					paths = self.kshortestPath(a, b, Nfp)
					if len(paths) < Nfp_init:
						enough_paths = False
				self.short[(a,b)] = paths[:]
				Nfp = Nfp_init
		else:
			for it, (a,b) in enumerate(pairs):
				# print "Shortest path for", (a, b)
				#if verb:
				#    counter(it, len(pairs), message='Computing shortest paths...')
				if a!=b:
					enough_paths = False
					Nfp_init = Nfp
					while not enough_paths:
						enough_paths = True
						paths = self.kshortestPath(a, b, Nfp) #Initial set of paths
						previous_duplicates = 1
						duplicates = []
						n_tries = 0
						while len(duplicates)!=previous_duplicates and n_tries<50:
							previous_duplicates = len(duplicates)
							duplicates = []
							for sp in paths:
								if len(np.unique(sp))<len(sp): # Detect if some sectors are duplicated within sp
									duplicates.append(sp)

							if len(duplicates)!=previous_duplicates: # If the number of duplicates has changed, compute some more paths.
								paths = self.kshortestPath(a, b, Nfp+len(duplicates))
							n_tries += 1

						for path in duplicates:
							paths.remove(path)

						try:
							try:
								assert n_tries<50
							except AssertionError:
								print "I hit the maximum number of iterations."
								raise

							assert len(paths)==Nfp and len(duplicates)==previous_duplicates
							enough_paths=True
							p_paths = sorted(list(set([tuple(vv) for vv in paths])), key=lambda p:self.weight_path(p))
							paths = [list(vvv) for vvv in p_paths][:Nfp_init]
							if len(paths) < Nfp_init:
								enough_paths = False
								print 'Not enough paths, doing another round (' + str(Nfp +1 - Nfp_init), 'additional path(s)).'
							Nfp += 1
						except AssertionError:
							#print 'a:', a, 'b:', b, 'len(self.short[(a,b)]):', len(self.short[(a,b)])
							print "WARNING: kspyen can't find enough paths (only " + str(len(paths)) + ')', "for the pair", a, b,
							#print 'Number of duplicates:', len(duplicates)
							#print 'Number of duplicates:', len(duplicates)
							#print 'Number of paths with duplicates:', len(paths_init)
							if delete_pairs:
								print "I delete this pair."
								deleted_pairs.append((a,b))
								del self.short[(a,b)]
								break
							else:
								print
								print "I don't take any action and keep in memory the shortest paths,"
								print "but you should not use the network with this Nfp. Try to run it"
								print "again with a smaller Nfp."

					Nfp = Nfp_init
					if self.short.has_key((a,b)):
						self.short[(a,b)] = paths[:]      

					if not delete_pairs:
						if len(self.short[(a,b)])<Nfp:
							print  "Pair", (a,b), "do not have enough path, I duplicate the last one..."
						if len(self.short[(a,b)])>Nfp:
							# Should only happen when the maximum number of iteration has been hit
							self.short[(a,b)] = self.short[(a,b)][:10]
						while len(self.short[(a,b)])<Nfp:
							self.short[(a,b)].append(self.short[(a,b)][-1])
						assert len(self.short[(a,b)])==Nfp
				else:
					# for the case where a==b. Might want to remove this in Model 1.
					self.short[(a,b)] = [[a] for i in range(Nfp)]

	def connections(self):
		"""
		Notes
		-----
		New in 3.0.0: taken from Model 2 (unchanged)

		(From Model 2)
		New in 2.9.8: returns the possible connections between airports.
		
		"""
		
		return self.short.keys()

	def fix_airports(self, *args, **kwargs):
		"""
		Used to reset the airports and then add the new airports.

		Notes
		-----
		Changed in 3.0.0: taken from Model 2 (unchanged).

		"""
		
		if hasattr(self, "airports"):
			self.airports = []
			self.short = {}

		self.add_airports(*args, **kwargs)

	def fix_capacities(self, capacities, **kwargs):
		"""

		Notes
		-----
		New in 3.2.0

		"""
		
		for n, c in capacities.items():
			self.node[n]['capacity'] = c

	def fix_weights(self, weights, **kwargs):
		"""

		Notes
		-----
		New in 3.2.0

		"""
		
		for (n1, n2), w in weights.items():
			self[n1][n2]['weight'] = w

	def generate_connections(self, typ=None, options={}, min_dis=2):
		"""
		Generate available pairs of airports.

		Parameters
		==========
		typ: string, optional
			Type of airpor network. Leave None for complete graph, 'BA'
			for Barabasi-Albert network.
		options: dict, optional
			Dictionary of options for the generation. If typ=='BA', this 
			dictionary should have the key 'mean_k' at least
		min_dis: int, optional
			Minimum distance in nodes between two connected airports
			(without counting origin and destination)

		Notes
		-----
		New in 3.2.0

		"""

		if typ==None:
			self.short = {(ai,aj):[] for ai in self.airports for aj in self.airports if len(nx.shortest_path(self, ai, aj))-2>=min_dis}
		elif typ=='BA':
			# Compute a Barabasi-Albert graph
			AG = nx.barabasi_albert_graph(len(self.airports), int(options['mean_k']/2.))
			# Translate the links ids in names of airports
			links = [(self.airports[i], self.airports[j]) for i, j in AG.edges()]
			# Because AG is undirected
			links2 = [(self.airports[j], self.airports[i]) for i, j in AG.edges()]
			self.short = {(ai,aj):[] for ai, aj in links+links2}

	def generate_airports(self, nairports, C_airport=10):
		"""
		Generate nairports airports. Build the accessible pairs of airports for this network
		with a  minimum distance min_dis.

		Notes
		-----
		Changed in 3.0.0: updated with stuff from Model 2
		Changed in 3.2.0: removed short generation
		
		"""

		self.airports = sample(self.nodes(),nairports)
		
		for a in self.airports:
			self.node[a]['capacity']=100000                 # TODO: check this.
			self.node[a]['capacity_airport'] = C_airport

	def generate_weights(self, typ='coords', par=[1.,0.01], values=[]):
		"""
		Generates weights with a gaussian distribution or given by the euclidean distance
		between nodes, tuned so that the average matches the one given as argument.

		Parameters
		----------
		typ: str
			should be 'gauss' or 'coords'. The first produces gaussian weights with mean 
			given by the first element of par and the deviation given by the second element
			of par. 'coords' computes the euclideian distance between nodes (based on key 
			coord) and adjust it so the average weight over all edges matches the float given
			by par.
		par: list or float
			If typ is 'gauss', gives the mean and deviation. Otherwise, should be a float giving 
			the average weight.

		Notes
		-----
		Changed in 3.0.0: taken from Model 2 (unchanged).
		Changed in 3.0.0: default optional for typ is now "coords".

		"""

		assert typ in ['constant', 'gauss', 'lognormal', 'coords']

		self.typ_weights, self.par_weights = typ, par
		if typ=='gauss':
			mu = par[0]
			sigma = par[1]
			for e in self.edges():
				self[e[0]][e[1]]['weight'] = max(gauss(mu, sigma), 0.00001)
		elif typ=='coords':
			for e in self.edges():
				#self[e[0]][e[1]]['weight']=sqrt((self.node[e[0]]['coord'][0] - self.node[e[1]]['coord'][0])**2 +(self.node[e[0]]['coord'][1] - self.node[e[1]]['coord'][1])**2)
				self[e[0]][e[1]]['weight'] = np.linalg.norm(np.array(self.node[e[0]]['coord']) - np.array(self.node[e[1]]['coord']))
			avg_weight = np.mean([self[e[0]][e[1]]['weight'] for e in self.edges()])
			for e in self.edges():
				self[e[0]][e[1]]['weight'] = par*self[e[0]][e[1]]['weight']/avg_weight
		elif typ=='constant':
			for e in self.edges():
				self[e[0]][e[1]]['weight'] = par
		elif typ=='lognormal':
			mu_t = par[0]
			sig_t = par[1] 

			mu = log(mu_t/sqrt((sig_t/mu_t**2) + 1.))
			sig = sqrt(log((sig_t/mu_t**2) + 1.))
			
			for e in self.edges():
				self[e[0]][e[1]]['weight'] = max(lognormal(mu, sig), 0.00001)

		self.weighted = True
			
	def generate_capacities(self, typ='constant', C=5, par=[1]):
		"""
		Generates capacities with different distributions.
		If typ is 'constant', all nodes have the same capacity, given by C.
		If typ is 'gauss', the capacities are taken from a normal distribution with mean C
		and standard deviation par[0]
		If typ is 'uniform', the capacities are taken from a uniform distribution, with 
		bounds C-par[0]/2.,C+par[0]/2.
		If typ is 'areas', the capacities are proportional to the square root of the area of 
		the sector, with the proportionality factor set so at to have a mean close to C. 
		This requires that each node has a key 'area'.
		If typ is 'lognormal', the capacities are taken from a lognormal distribution.

		Capacities are integers, minimum 1.

		Parameters
		----------
		typ : string
			type of distribution, see description.
		C : int
			main parameter of distribution, see description.
		par : list of int or float
			other parameters, see description.

		Notes
		----- 
		Changed in 3.0.0: taken from Model 2 (unchanged).

		(From Model 2)
		New in 2.7: added lognormal and areas.
		Changed in 2.9.8: removed "manual"

		"""

		assert typ in ['constant', 'gauss', 'uniform', 'lognormal', 'areas']
		self.C, self.typ_capacities, self.par_capacities = C, typ, par
		if typ=='constant':
			for n in self.nodes():
				self.node[n]['capacity'] = C
		elif typ=='gauss':
			for n in self.nodes():
				self.node[n]['capacity'] = max(1, int(gauss(C,par[0])))
		elif typ=='uniform':
			for n in self.nodes():
				self.node[n]['capacity'] = max(1, int(uniform(C-par[0]/2.,C+par[0]/2.)))
		elif typ=='lognormal':
			for n in self.nodes():
				self.node[n]['capacity'] = max(1, int(lognormal(log(C),par[0])))
		elif typ=='areas':
			if par[0]=='sqrt':
				area_avg = np.mean([sqrt(self.node[n]['area']) for n in self.nodes()])
				alpha = C/area_avg
				for n in self.nodes():
					self.node[n]['capacity'] = max(1, int(alpha*sqrt(self.node[n]['area'])))

	def get_airports(self):
		"""
		Notes
		-----
		New in 3.0.0: taken from Model 2(unchanged)
		(From Model 2)
		New in 2.9.8: returns the airports based on connections.
		
		"""
		
		return set([e for ee in self.connections() for e in ee])

	def import_from(self, G, numberize=False, verb=False):
		"""
		Used to import the data of an already existing graph (networkx) in a Net obect.
		Weights are conserved. 

		Parameters
		----------
		G : a networkx object
			all keys attached to nodes will be preserved. Network needs to be completely weighted, 
			or none at all.
		numberize : boolean, optional
			if True, nodes of G will not be used as is, but an index will be generated instead.
			The real name in stored in the 'name' key of the node. A dictionnary idx_nodes is 
			also attached to the network for easy (reverse) mapping.
		verb : boolean, optional
			verbosity.

		Notes
		-----
		Changed in 3.0.0: taken from Model 2 (unchanged).
		(From Model 2)
		Changed in 2.9: included case where G is empty.
		TODO: preserve attributes of edges too.

		"""
		
		if verb:
			print 'Importing network...'

		if len(G.nodes())!=0:
			if not numberize:
				self.add_nodes_from(G.nodes(data=True))
				if len(G.edges())>0:
					e1, e2 = G.edges()[0]
					if 'weight' in G[e1][e2].keys():
						self.add_weighted_edges_from([(e[0],e[1], G[e[0]][e[1]]['weight']) for e in G.edges()])
					else:
						self.add_weighted_edges_from([(e[0],e[1], 1.) for e in G.edges()])
			else:
				self.idx_nodes={s:i for i,s in enumerate(G.nodes())}
				for n in G.nodes():
					self.add_node(self.idx_nodes[n], name=n, **G.node[n])

				e1, e2 = G.edges()[0]
				if len(G.edges())>0:
					if 'weight' in G[e1][e2].keys():
						for e in G.edges():
							e1 = self.idx_nodes[e[0]]
							e2 = self.idx_nodes[e[1]]
							self.add_edge(e1, e2, weight=G[e1][e2]['weight'])
					else:
						for e in G.edges():
							self.add_edge(self.idx_nodes[e[0]], self.idx_nodes[e[1]], weight=1.)

			if len(self.edges())>0:
				e1 = self.edges()[0]
				e2 = self.edges()[1]
				self.weighted = not (self[e1[0]][e1[1]]['weight']==self[e2[0]][e2[1]]['weight']==1.)
			else:
				print "Network has no edge!"
				self.weighted = False

			if verb:
				if self.weighted:
					print 'Network was found weighted'
				else:
					print 'Network was found NOT weighted'
					if len(self.edges())>0:
						print 'Example:', self[e1[0]][e1[1]]['weight']     
		else:
			print 'Network was found empty!'   

	def kshortestPath(self, i, j, k): 
		"""
		Return the k weighted shortest paths on the network thanks to YenKSP algorithm. Uses the DiGraph,
		computed by build_H.

		Parameters
		----------
		i : int
			origin
		j : int
			destination
		k : int
			Number of shortest paths to compute


		Returns
		-------
		spath_new: list
			list of the 

		Notes
		-----
		Changed in 3.0.0: the index of the loop was i too!
		TODO: See what is this fucking loop.

		"""
		
		# Compute the k-shortest paths with Yen-KSP algorithm
		spath = [a['path'] for a in  ksp_yen(self.H, i, j, k)]
		
		# Sort the paths by increasing (weighted) length
		spath = sorted(spath, key=lambda a:self.weight_path(a))
		
		# Not sure what this loop is doing...
		spath_new, ii = [], 0
		while ii<len(spath): 
			w_old = self.weight_path(spath[ii])
			a = [spath[ii][:]]
			ii += 1
			while ii<len(spath) and abs(self.weight_path(spath[ii]) - w_old)<10**(-8.):
				a.append(spath[ii][:])
				ii += 1
			#shuffle(a)
			spath_new += a[:]

		return spath_new

	def set_connections(self, connections, min_dis=2):
		self.short = {(ai,aj):[] for ai, aj in connections if len(nx.shortest_path(self, ai, aj))-2>=min_dis}
		
	def shut_sector(self,n):
		for v in nx.neighbors(self,n):
			self[n][v]['weight']=10**6

	def stamp_airports(self):
		"""

		Notes
		-----
		New in 3.0.0: taken from Model 2 (unchanged).

		(From Model 2)
		New in 2.9.8: compute the list of airports based on short.
		
		"""
		
		self.airports = list(self.get_airports())

	def weight_path(self,p): 
		"""
		Return the weight of the given path.
		"""
		return sum([self[p[i]][p[i+1]]['weight'] for i in range(len(p)-1)])
		

def utility((alpha,betha1,betha2), Lsp, t0sp, L, t0):
	"""
	the inputs of this function are all supposed to be NumPy arrays
	   
	Call: U=UTILITY(ALPHA,BETHA1,BETHA2,LSP,T0SP,L,T0);
	the function utility.m computes the utility function value, comparing two
	paths on a graph;
	
	INPUTS
	
	alpha, betha1, betha2 -> empirically assigned weight parameters,
	ranging from 0 to 1;
	
	Lsp -> length of the shortest path;
	
	t0sp -> departure time of the motion along the shortest path;
	
	L -> length of the path which one wants to compare to the shortest
	one;
	
	t0 -> depature time of the motion on the path used in the
	coparison with the shortes one;
	
	OUTPUT
	
	U -> is the value of the utility function for the given choise of paths;
	
	"""
	
	return np.dot(alpha,L)+np.dot(betha1,np.absolute(t0+L-(t0sp+Lsp)))+np.dot(betha2,np.absolute(t0-t0sp))
	
	

		
	
	

