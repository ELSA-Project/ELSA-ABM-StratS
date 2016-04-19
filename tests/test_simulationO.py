#!/usr/bin/env python

import sys
sys.path.insert(1, '..')
import unittest
import os

from abm_strategic_model1.simAirSpaceO import Net, Flight
from abm_strategic_model1.simulationO import *
from abm_strategic_model1.utilities import post_process_paras

# TODO: high level tests

class SimulationTest(unittest.TestCase):
	def setUp(self):
		self.prepare_network()

	def prepare_network(self):
		# Sectors
		self.G = Net()
		self.G.add_node(0, coord=(0., 0.))
		self.G.add_node(1, coord=(1., 0.))
		self.G.add_node(2, coord=(-0.5, np.sqrt(3.)/2.))
		self.G.add_node(3, coord=(0.5, np.sqrt(3.)/2.))
		self.G.add_node(4, coord=(1.5, np.sqrt(3.)/2.))
		self.G.add_node(5, coord=(0., np.sqrt(3.)))
		self.G.add_node(6, coord=(1., np.sqrt(3.)))

		# Sec-edges.
		self.G.add_edge(0, 1, weight=1.)
		self.G.add_edge(1, 4, weight=1.)
		self.G.add_edge(4, 6, weight=1.)
		self.G.add_edge(6, 5, weight=1.)
		self.G.add_edge(5, 2, weight=2.5)

		self.G.weighted = True
		#self.G.add_edge(2, 0)

		for i in [0, 1, 2, 4, 5, 6]:
			self.G.add_edge(3, i, weight=1.)

		self.G.airports = []
		
		# Create fake shortest paths
		self.G.short = {}
		self.G.short[(1, 5)] = [[1, 4, 6, 5], [1, 3, 2, 5]]
		self.G.Nfp = 2

		self.assertTrue(self.G.weight_path(self.G.short[(1, 5)][0])<self.G.weight_path(self.G.short[(1, 5)][1]))
		#print self.G.G_nav.weight_path(self.G.G_nav.short[(14, 11)][0]), self.G.G_nav.weight_path(self.G.G_nav.short[(14, 11)][1])
		self.G.airports = [1, 5]

		for n in self.G.nodes():
			self.G.node[n]['capacity'] = 5
		for a in self.G.airports:
			self.G.node[a]['capacity_airport'] = 10000

		self.G.comments = []

class FunctionsTest(SimulationTest):
	def test_build_path(self):
		self.paras = {}

		# Fo S Companies
		self.paras['par'] = [(1., 0., 0.), (1., 0., 0.)]
		self.paras['Nfp'] = 2
		self.paras['departure_times'] = 'zeros'
		self.paras['N_shocks'] = 0.
		self.paras['mode_M1'] = 'standard'

		class Pouet:
			pass

		self.paras['G'] = Pouet()
		self.paras['G'].name = 'GNAME'

		coin = build_path(self.paras, vers='0.0', in_title=[], rep='')

		self.assertTrue(coin=='Sim_v0.0_GNAME')

	def test_post_process_queue(self):
		queue = []
		queue.append(Flight(0, 1, 5, 0., 0, (1., 0., 1.), 2))
		queue.append(Flight(1, 1, 5, 0., 1, (1., 0., 1.), 2))

		for f in queue:
			f.compute_flightplans(20., self.G)
			f.best_fp_cost = f.FPs[0].cost

		queue[1].FPs[0].accepted = False
		queue[1].FPs[1].accepted = True

		queue = post_process_queue(queue)

		self.assertTrue(queue[0].satisfaction==1.)
		self.assertTrue(queue[0].regulated_1FP==0.)
		self.assertTrue(queue[0].regulated_FPs==0)
		self.assertTrue(queue[0].regulated_F==0.)

		self.assertTrue(queue[1].satisfaction==3./4.5)
		self.assertTrue(queue[1].regulated_1FP==1.)
		self.assertTrue(queue[1].regulated_FPs==1)
		self.assertTrue(queue[1].regulated_F==0.)

class SimulationTimesZeros(SimulationTest):
	def setUp(self):
		super(SimulationTimesZeros, self).setUp()
		self.prepare_paras()

	def prepare_paras(self):
		self.paras = {}

		self.paras['par'] = [(1., 0., 0.), (0., 0., 1.)] # For S and R Companies
		self.paras['Nfp'] = 2
		self.paras['departure_times'] = 'zeros'
		self.paras['N_shocks'] = 0.
		self.paras['mode_M1'] = 'standard'
		self.paras['ACtot'] = 2
		self.paras['na'] = 1
		self.paras['G'] = self.G
		self.paras['tau'] = 20.
		self.paras['nA'] = 1
		self.paras['old_style_allocation'] = False
		self.paras['noise'] = 0.
		self.paras['AC'] = 2

	def test_initialization(self):
		sim = Simulation(self.paras, G=self.G, verbose=False)

		self.assertTrue(len(sim.t0sp)==self.paras['ACtot'])
		for p in sim.t0sp:
			self.assertTrue(len(p)==1)
			self.assertTrue(p[0]==0.)

	def test_build_ACs(self):
		self.paras['AC'] = 2
		self.paras['par'] = [(1., 0., 0.)]
		sim = Simulation(self.paras, G=self.G, verbose=True)
		sim.build_ACs()

		self.assertTrue(len(sim.ACs)==self.paras['AC'])
		
		for ac in sim.ACs.values():
			self.assertTrue(ac.par==(1., 0., 0.))
			self.assertTrue(ac.flights[0].pref_time==0.)

	def test_build_ACs2(self):
		self.paras['AC'] = 2
		self.paras['par'] = [(1., 0., 0.), (0., 0., 1.)]
		sim = Simulation(self.paras, G=self.G, verbose=True)
		sim.build_ACs()

		self.assertTrue(len(sim.ACs)==self.paras['AC'])
		
		self.assertTrue(sim.ACs[0].par==(1., 0., 0.))
		self.assertTrue(sim.ACs[1].par==(0., 0., 1.))

	def test_build_ACs3(self):
		self.paras['AC'] = [2, 0]
		self.paras['par'] = [(1., 0., 0.), (0., 0., 1.)]
		sim = Simulation(self.paras, G=self.G, verbose=True)
		sim.build_ACs()

		self.assertTrue(len(sim.ACs)==sum(self.paras['AC']))
		
		self.assertTrue(sim.ACs[0].par==(1., 0., 0.))
		self.assertTrue(sim.ACs[1].par==(1., 0., 0.))

class SimulationFlows(SimulationTest):
	def setUp(self):
		super(SimulationFlows, self).setUp()
		self.prepare_paras()
		self.finish_network()

	def finish_network(self):
		self.G.idx_nodes = {1:1, 5:5}
		self.G.short[(5, 1)] = [[5, 6, 4, 1], [5, 2, 3, 1]]
		#self.G.G_nav.airports = [11, 14]
		# self.G.G_nav.short[(11, 14)] = [[11, 10, 9, 8, 7, 6, 17, 16, 15, 14], [11, 10, 2, 18, 0, 17, 16, 15, 14]]
		
	def prepare_paras(self):
		self.paras = {}

		self.paras['par'] = [(1., 0., 0.), (0., 0., 1.)] # For S and R Companies
		self.paras['Nfp'] = 2
		self.paras['departure_times'] = 'zeros'
		self.paras['N_shocks'] = 0.
		self.paras['mode_M1'] = 'standard'
		self.paras['ACtot'] = 4
		self.paras['na'] = 1
		self.paras['G'] = self.G
		self.paras['tau'] = 20.
		self.paras['nA'] = 1
		self.paras['old_style_allocation'] = False
		self.paras['noise'] = 0.
		#self.paras['AC'] = 2

	def test_build_ACs_from_flows(self):
		self.paras['flows'] = {}
		self.paras['flows'][(1, 5)] = [[2010, 1, 1, 0, 0, 0], [2010, 1, 1, 0, 10, 0]]
		self.paras['flows'][(5, 1)] = [[2010, 1, 1, 0, 20, 0], [2010, 1, 1, 0, 30, 0]]
		self.paras['bootstrap_mode'] = False
		self.paras['nA'] = 0.5
		self.paras['ACtot'] = 4

		sim = Simulation(self.paras, G=self.G, verbose=True)
		sim.build_ACs_from_flows()

		self.assertTrue(sim.starting_date==[2010, 1, 1, 0, 0, 0])

		print len(sim.ACs), self.paras['ACtot']
		self.assertTrue(len(sim.ACs)==self.paras['ACtot'])

		self.assertTrue(sim.ACs[0].par==(1., 0., 0.))
		self.assertTrue(sim.ACs[1].par==(0., 0., 1.))
		self.assertTrue(sim.ACs[2].par==(1., 0., 0.))
		self.assertTrue(sim.ACs[3].par==(0., 0., 1.))
	

		self.assertTrue(sim.ACs[0].flights[0].pref_time==0.)
		self.assertTrue(sim.ACs[1].flights[0].pref_time==10.)
		self.assertTrue(sim.ACs[2].flights[0].pref_time==20.)
		self.assertTrue(sim.ACs[3].flights[0].pref_time==30.)

	def test_build_ACs_from_flows_bootstrap(self):
		self.paras['flows'] = {}
		self.paras['flows'][(1, 5)] = [[2010, 1, 1, 0, 0, 0], [2010, 1, 1, 0, 10, 0]]
		self.paras['flows'][(5, 1)] = [[2010, 1, 1, 0, 20, 0], [2010, 1, 1, 0, 30, 0]]
		self.paras['bootstrap_mode'] = True
		self.paras['bootstrap_only_time'] = True
		self.paras['nA'] = 1.
		self.paras['ACtot'] = 6

		sim = Simulation(self.paras, G=self.G, verbose=True)
		sim.build_ACs_from_flows()

		self.assertTrue(sim.starting_date==[2010, 1, 1, 0, 0, 0])

		self.assertTrue(len(sim.ACs)==self.paras['ACtot'])

		for ac in sim.ACs.values():
			self.assertTrue(ac.flights[0].pref_time in [0., 10., 20., 30.])

class DoStandardTest(SimulationTest):
	def setUp(self):
		super(DoStandardTest, self).setUp()
		self.prepare_paras()

	def prepare_paras(self):
		self.paras = {}

		self.paras['par'] = [(1., 0., 0.), (0., 0., 1.)] # For S and R Companies
		self.paras['Nfp'] = 2
		self.paras['departure_times'] = 'zeros'
		self.paras['N_shocks'] = 0.
		self.paras['mode_M1'] = 'standard'
		self.paras['ACtot'] = 2
		self.paras['na'] = 1
		self.paras['G'] = self.G
		self.paras['tau'] = 20.
		self.paras['nA'] = 1
		self.paras['old_style_allocation'] = False
		self.paras['noise'] = 0.
		self.paras['AC'] = 2
		self.paras['day'] = 24.*60.
		self.paras['flows'] = {}
		self.paras['STS'] = None

	def test1(self):
		results = do_standard((self.paras, self.G))

		self.assertEqual(results['satisfaction'][(1., 0., 0.)], 1.)
		self.assertEqual(results['regulated_FPs'][(1., 0., 0.)], 0.)
		self.assertEqual(results['regulated_F'][(1., 0., 0.)], 0.)

	def test_with_paras_file(self):
		from abm_strategic_model1.paras_test import paras

		paras = post_process_paras(paras)

		results = do_standard((paras, paras['G']))

		self.assertEqual(results['satisfaction'][(1., 0., 0.)], 1.)
		self.assertEqual(results['regulated_FPs'][(1., 0., 0.)], 0.)
		self.assertEqual(results['regulated_F'][(1., 0., 0.)], 0.)

if __name__ == '__main__':
	# Manual tests
	#os.system('../abm_strategic_model1/simulationO.py paras_test.py')
	#os.system('../abm_strategic/iter_simO.py paras_iter_test.py')
	
	# Put failfast=True for stopping the test as soon as one test fails.
	unittest.main(failfast=True)



