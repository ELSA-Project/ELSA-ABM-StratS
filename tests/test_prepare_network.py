import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../abm_strategic_model1')
import unittest
from networkx import Graph
import types
import os
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d
from random import seed

from abm_strategic_model1.prepare_network import *

# def assertCoordinatesAreEqual(l1, l2, thr=10**(-6.)):
# 	b1 = len(l1)==len(l2)
# 	if not b1:
# 		return False
# 	else:
# 		s = sum(((l1[i][0]-l2[i][0])**2 + (l1[i][1]-l2[i][1])**2) for i in range(len(l2)))
# 		b2 = s<thr
# 		return b2

# class TestLowFunctions(unittest.TestCase):
# 	def test_area(self):
# 		l = [(0., 0.), (0., 2.), (2., 2.), (2., 0.), (0., 0.)]
# 		self.assertEqual(area(l), 4.)

# 	def test_segments(self):
# 		l = [(0., 0.), (0., 2.), (2., 2.)]
# 		self.assertEqual(segments(l), [((0., 0.), (0., 2.)), ((0., 2.), (2., 2.)), ((2., 2.), (0., 0.))])

# class SimpleSectorNetworkCase(unittest.TestCase):

# 	def setUp(self):
# 		self.G = Graph()
# 		self.G.add_node(0, coord=(0., 0.))
# 		self.G.add_node(1, coord=(1., 0.))
# 		self.G.add_node(2, coord=(-0.5, np.sqrt(3.)/2.))
# 		self.G.add_node(3, coord=(0.5, np.sqrt(3.)/2.))
# 		self.G.add_node(4, coord=(1.5, np.sqrt(3.)/2.))
# 		self.G.add_node(5, coord=(0., np.sqrt(3.)))
# 		self.G.add_node(6, coord=(1., np.sqrt(3.)))

# 	def show_network(self, show=True):
# 		plt.scatter(*zip(*[self.G.node[n]['coord'] for n in self.G.nodes()]), marker='s', color='r', s=50)

# 		if show:
# 			plt.show()

# 	def show_polygons(self, show=True):
# 		for pol in self.G.polygons.values():
# 			plt.fill(*zip(*list(pol.exterior.coords)), alpha=0.4)

# 		if show:
# 			plt.show()

# 	#def test_dummy(self):
# 		#self.show_network(show=True)
# 		#self.show_polygons(show=True)

# class TestLowNetworkFunctions(SimpleSectorNetworkCase):
# 	def test_compute_voronoi(self):
# 		G, vor = compute_voronoi(self.G, xlims=(-1., 2.), ylims=(-1., 2.))
		
# 		coin = [list(pol.exterior.coords) for pol in G.polygons.values() if type(pol)==type(Polygon())]
# 		self.assertTrue(len(coin)==7)
# 		for pol in G.polygons.values():
# 			self.assertTrue(type(pol) == type(Polygon()))
# 		#for i, c in enumerate(coin):
# 		#	print i, c
# 		l = np.sqrt(3.)
# 		pouet = []
# 		center = np.array([0.5, l/2.])
# 		for i in range(6):
# 			angle = - np.pi/2.- float(i)/6.*2.*np.pi
# 			point = center + (1./l)*np.array([np.cos(angle), np.sin(angle)])
# 			pouet.append(list(point))
# 		pouet.append(pouet[0])
# 		self.assertTrue(len(coin[3])==7)
# 		self.assertTrue(assertCoordinatesAreEqual(coin[3], pouet))

# 	def test_reduce_airports_to_existing_nodes(self):
# 		airports = [0, 1, 10000]
# 		pairs = [(0, 1), (0, 100000), (10000, 1)]
# 		pairs, airports = reduce_airports_to_existing_nodes(self.G, pairs, airports)
# 		self.assertEqual(pairs, [(0, 1)])
# 		self.assertEqual(airports, [0, 1])

# 	def test_recompute_neighbors(self):
# 		G, vor = compute_voronoi(self.G, xlims=(-1., 2.), ylims=(-0.5, 2.5))
# 		G.add_edge(0, 6)
# 		recompute_neighbors(G)
# 		self.assertFalse(G.has_edge(0,6))

# 	# def test_extract_weights_from_traffic(self):
# 	# 	self.G.idx_nodes = {('node' + str(i)):i for i in self.G.nodes()}
# 	# 	self.G.add_edges_from([(0, 3), (3, 6), (3, 4)])
# 	# 	f1 = {'route_m1t':[('node0', (2010, 5, 6, 0, 0, 0)), ('node3', (2010, 5, 6, 0, 10, 0)), ('node6', (2010, 5, 6, 0, 25, 0))]}
# 	# 	f2 = {'route_m1t':[('node0', (2010, 10, 6, 0, 0, 0)), ('node3', (2010, 10, 6, 0, 15, 0)), ('node4', (2010, 10, 6, 0, 35, 0))]}
# 	# 	flights = [f1, f2]
		
# 	# 	weights = extract_weights_from_traffic(self.G, flights)

# 	# 	self.assertEqual(weights[(0, 3)], 12.5)
# 	# 	self.assertEqual(weights[(3, 6)], 15.)
# 	# 	self.assertEqual(weights[(3, 4)], 20.)

class TestWholeFunction(unittest.TestCase):
	def show_network(self, show=True):
		plt.scatter(*zip(*[self.G.node[n]['coord'] for n in self.G.nodes()]), marker='s', color='r', s=50)

		if show:
			plt.show()

	def show_polygons(self, show=True):
		for pol in self.G.polygons.values():
			plt.fill(*zip(*list(pol.exterior.coords)), alpha=0.4)

		if show:
			plt.show()

	def test_prepare_network_default(self):
		from paras_G_test import paras_G

		# Manual seed
		see_ = 31
		print "===================================="
		print "USING SEED", see_
		print "===================================="
		seed(see_)
		rep = '../example'
		G = prepare_network(paras_G, rep=rep, show=False)
		#os.system('rm Example.pic')
		#os.system('rm Example.png')
		#os.system('rm Example_basic_stats_net.txt')

		# print "Airports:", G.get_airports()
		# print "Connections:", G.connections()
		# print "Edges:"
		# for n1, n2 in G.edges():
		# 	print n1, n2
		# print 'Capacity of sectors:'
		# for n in G.nodes():
		# 	print n, ':', G.node[n]['capacity']
		# print 'Capacity of airports:'
		# for a in G.get_airports():
		# 	print a, ':', G.node[a]['capacity_airport']

		self.assertEqual(len(G.get_airports()), 2)
		self.assertEqual(len(G.connections()), 2)
		self.assertTrue((4, 5) in G.connections())
		self.assertTrue((5, 4) in G.connections())
		for n in G.nodes():
			self.assertEqual(G.node[n]['capacity'], 5)

		for a in G.get_airports():
			self.assertTrue('capacity_airport' in G.node[a].keys())
			self.assertEqual(G.node[a]['capacity_airport'], 100000)

		self.G = G

	def test_prepare_network_triangular(self):
		from paras_G_test import paras_G

		# Manual seed
		see_ = 31
		print "===================================="
		print "USING SEED", see_
		print "===================================="
		seed(see_)
		rep = '../example'

		paras_G['type_of_net'] = 'T'
		G = prepare_network(paras_G, rep=rep, show=False)
		#os.system('rm Example.pic')
		#os.system('rm Example.png')
		#os.system('rm Example_basic_stats_net.txt')

		# print "Airports:", G.get_airports()
		# print "Connections:", G.connections()
		# print "Edges:"
		# for n1, n2 in G.edges():
		# 	print n1, n2
		# print 'Capacity of sectors:'
		# for n in G.nodes():
		# 	print n, ':', G.node[n]['capacity']
		# print 'Capacity of airports:'
		# for a in G.get_airports():
		# 	print a, ':', G.node[a]['capacity_airport']

		self.assertEqual(len(G.get_airports()), 2)
		self.assertEqual(len(G.connections()), 2)
		self.assertTrue((4, 5) in G.connections())
		self.assertTrue((5, 4) in G.connections())
		for n in G.nodes():
			self.assertEqual(G.node[n]['capacity'], 5)

		for a in G.get_airports():
			self.assertTrue('capacity_airport' in G.node[a].keys())
			self.assertEqual(G.node[a]['capacity_airport'], 100000)



if __name__ == '__main__':
	#suite = unittest.TestLoader().loadTestsFromTestCase(TestLawNetworkFunctions)
	unittest.main(failfast=True)