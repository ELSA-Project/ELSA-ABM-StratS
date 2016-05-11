# -*- coding: utf-8 -*-
"""
Used to visualize runs of the simulations
This simple version only displays things.

"""

import datetime as dt

from abm_strategic_model1.simulationO import Simulation
from abm_strategic_model1.simAirSpaceO import Network_Manager
from abm_strategic_model1.utilities import read_paras


class SimulationStory(Simulation):
	# def prepare_map(self):
	# 	map_of_net(G, colors='r', limits=(0,0,0,0), title='', size_nodes=1., size_edges=2., nodes=[], zone_geo=[], edges=True, fmt='svg', dpi=100, \
 #    save_file=None, show=True, figsize=(9,6), background_color='white', key_word_weight='weight', z_order_nodes=6, diff_edges=False, lw_map=0.8,\
 #    draw_mer_par=True)
		
	def prepare_simu(self):
		self.NM = Network_Manager(old_style=self.old_style_allocation, discard_first_and_last_node=self.discard_first_and_last_node)
		self.NM.initialize_load(self.G, length_day=int(self.day/60.))

		if self.flows == {}:
			self.build_ACs()
		else:
			self.build_ACs_from_flows()

		self.queue = self.NM.build_queue(self.ACs)
		self.shuffle_departure_times()

		self.current_flight_index = -1
		self.current_flight_plan_index = 0
		self.found = True

	def next_flight(self):
		self.current_flight_index += 1
		if self.current_flight_index<len(self.queue):
			self.current_flight_plan_index = 0
			self.found = False
			i = self.current_flight_index

			f = self.queue[self.current_flight_index]
			f.pos_queue = i
			#self.allocate_flight(G, f, storymode=storymode)

			#self.show_origin_destination(f.source, f.destination)
			fake_date = dt.datetime(2010, 5, 6, 0, 0, 0)
			ttt = dt.timedelta(minutes=f.pref_time) + fake_date
			fmt = "%D %H:%M:%S"
			strr = ttt.strftime(fmt)

			map_update_info = {'origin_destination':(f.source, f.destination)}
			text_story = "Flight " + str(f.pos_queue) + " from " + str(f.source) +\
						" to " + str(f.destination) + " with parameters " +\
						str(f.par) + " and pref. dep. time " + strr + " tries to be allocated."

			satisfaction = self.compute_satisfaction()
			return {'map_update_info':map_update_info, 'text_story':text_story, 'satisfaction':satisfaction}
		else:
			satisfaction = self.compute_satisfaction()
			return {'text_info':"Simulation finished, no more flights.", 'stop':True, 'satisfaction':satisfaction}

	def step(self):
		flight = self.queue[self.current_flight_index]

		if self.current_flight_plan_index<len(flight.FPs) and not self.found:
			d = self.next_flight_plan(self.G)
		else:
			if not self.found:
				map_update_info = {}
				text_story = "Flight " + str(flight.pos_queue) + " has been rejected!\n"
				flight.fp_selected = None
				flight.accepted = False
				self.found = True

				satisfaction = self.compute_satisfaction()

				d = {'map_update_info':map_update_info, 
						'text_story':text_story, 
						'satisfaction':satisfaction}
			else:
				d = self.next_flight()
		
		d['queue'] = self.queue 
		return d

	def next_flight_plan(self, G):
		i = self.current_flight_plan_index
		flight = self.queue[self.current_flight_index]

		map_update_info = {'origin_destination':(flight.source, flight.destination)}
		
		fp = flight.FPs[i]
		self.NM.compute_flight_times(G, fp)
		path, times = fp.p, fp.times

		if self.NM.discard_first_and_last_node:
			first = 1 ###### ATTENTION !!!!!!!!!!!
			last = len(path)-1 ########## ATTENTION !!!!!!!!!!!
		else:
			first = 0 ###### ATTENTION !!!!!!!!!!!    
			last = len(path) ########## ATTENTION !!!!!!!!!!!
		
		j = first
		while j<last and not self.NM.overload_sector(G, path[j],(times[j],times[j+1])):#and self.node[path[j]]['load'][j+time] + 1 <= self.node[path[j]]['capacity']:
			j += 1 

		fp.accepted = not ((j<last) or self.NM.overload_airport(G, path[0],(times[0],times[1])) or self.NM.overload_airport(G, path[-1],(times[-2],times[-1])))
			  
		path_overload = j<last
		source_overload = self.NM.overload_airport(G, path[0],(times[0],times[1]))
		desetination_overload = self.NM.overload_airport(G, path[-1],(times[-2],times[-1]))

		#print "FP has been accepted:", fp.accepted
		fake_date = dt.datetime(2010, 5, 6, 0, 0, 0)
		ttt = dt.timedelta(minutes=fp.t) + fake_date
		fmt = "%H:%M:%S"
		strr = ttt.strftime(fmt)
		text_story = " - FP " + str(i) + " with departure time " +\
					strr + " "# " tried to be allocated "
		map_update_info['trajectory'] = fp.p
		if not fp.accepted:
			text_story += "has been rejected because "
			map_update_info['color_trajectory'] = 'r'
			if path_overload: 
				text_story += "sector " + str(path[j]) + " was full."
				map_update_info['overloaded_sector'] = path[j]
			if source_overload:
				text_story += "because origin airport was full."
				map_update_info['overloaded_sector'] = path[0]
			if desetination_overload:
				text_story += "because destination airport was full."
				map_update_info['overloaded_sector'] = path[-1]

		if fp.accepted:
			text_story += "has been allocated.\n"
			map_update_info['color_trajectory'] = 'g'
			self.NM.allocate(G, fp, first=first, last=last)
			flight.fp_selected = fp
			flight.accepted = True
			self.found=True
		else:
			if j<last:
				fp.bottleneck=path[j]
			self.current_flight_plan_index += 1

		return {'map_update_info':map_update_info, 'text_story':text_story}

	def compute_satisfaction(self):
		satisfaction = []
		for f in self.queue:
			if hasattr(f, 'accepted'):
				best_cost = f.FPs[0].cost
				acceptedFPscost = [FP.cost for FP in f.FPs if FP.accepted]
				if len(acceptedFPscost) != 0:
					sat = best_cost/min(acceptedFPscost)
				else:
					sat = 0.

				satisfaction.append((sat, f.par))

		return satisfaction
				

