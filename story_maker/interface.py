# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../abm_strategic_model1')

import pickle
from math import *
from numpy.random import lognormal, normal
import networkx as nx

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

#from libs.general_tools import map_of_net
from libs.general_tools import nice_colors
from abm_strategic_model1.utilities import draw_sector_map, read_paras
import design

from story_maker import SimulationStory

def graph_bateau():
	G = nx.Graph()
	G.add_node(0, coord=[0., 0.])
	G.add_node(1, coord=[10., 0.])
	G.add_node(2, coord=[5., 10.*sqrt(3.)/2.])
	G.add_edge(0, 1, weight=10.)
	G.add_edge(1, 2, weight=10.)
	G.add_edge(2, 0, weight=15.)
	return G

def load_network():
	fil = '/home/earendil/Documents/ELSA/ABM/results_new/networks/DiskWorld/DiskWorld.pic'
	with open(fil, 'r') as f:
		G = pickle.load(f)

	return G

class StrategicGUI(QMainWindow, design.Ui_MainWindow):
	def __init__(self, parent=None, simu=None):
		super(StrategicGUI, self).__init__(parent)
		self.setupUi(self)

		self.prepare_main_frame()

		#self.pause.clicked.connect(self.print_stuff)
		self.pause.clicked.connect(self.pause_simulation)
		self.play.clicked.connect(self.play_simulation)
		self.play_one_step.clicked.connect(self.play_one_step_simulation)

		#self.main_splitter.setStretchFactor(1, 10)
		# These are not working properly
		#self.main_splitter.setSizes([2, 1])
		self.main_splitter.setSizes([600, 200])
		self.text_splitter.setSizes([400, 200])

		self.simu = simu
		self.simu.prepare_simu()
		self.G = self.simu.G

		#self.G = load_network()
		self.draw_network()

	def print_information(self, text):
		self.information.append(text)

	def print_story(self, text):
		self.story.append(text)

	def on_pick(self, event):
		# The event received here is of the type
		# matplotlib.backend_bases.PickEvent
		#
		# It carries lots of information, of which we're using
		# only a small amount here.
		# 
		box_points = event.artist.get_bbox().get_points()
		msg = "You've clicked on a bar with coords:\n %s" % box_points

		QMessageBox.information(self, "Click!", msg)

	def on_draw_test(self):
		""" Redraws the figure
		"""
		#str = unicode(self.textbox.text())
		#self.data = map(int, str.split())

		#x = range(len(self.data))

		# clear the axes and redraw the plot anew
		#
		self.axes.clear()        
		#self.axes.grid(self.grid_cb.isChecked())

		n = 1000
		mu = 1.
		sig = 1.
		values = normal(mu, sig, size=n)

		self.axes.hist(values, bins=50
			#left=x, 
			#height=self.data, 
			#width=self.slider.value() / 100.0, 
			#align='center', 
			#alpha=0.44,
			#picker=5
			)

		self.canvas.draw()

	def draw_network(self, display=True):
		self.axes.clear()        

		draw_sector_map(self.G, 
						colors=nice_colors[0], 
						limits=(-1.1, -1.1, 1.1, 1.), 
						size_nodes=40, 
						size_edges=0.5, 
						nodes=self.G.nodes(), 
						zone_geo=[], 
						edges=True, 
						show=False, 
						background_color='white', 
						key_word_weight=None, 
						z_order_nodes=6, 
						size_airports=80,
						ax=self.axes,
						coords_in_minutes=False,
						polygons=False,
						airports=True,
						load=False,
						numbers=True,
						shift_numbers=(0.02, 0.005),
						size_numbers=8)

		if display:
			self.canvas.draw()

	def draw_trajectory(self, p, color=nice_colors[5]):
		"""
		Parameters
		==========
		p: list
			of labels of nodes.

		TODO: not working with projection.

		"""

		self.draw_network(display=False)
		x, y = zip(*[self.G.node[n]['coord'] for n in p])
		self.axes.plot(x, y, '-', lw=2, c=color)
		self.canvas.draw()

	def indicate_origin_destination(self, origin, destination, size=15., color=nice_colors[6], marker='h'):
		self.draw_network(display=False)
		x, y = zip(*[self.G.node[n]['coord'] for n in [origin, destination]])
		self.axes.plot(x, y, marker, ms=size, c=color, zorder=10)
		self.canvas.draw()

	def pause_simulation(self):
		self.print_information('Not implemented yet')

	def play_simulation(self):
		self.print_information('Not implemented yet')

	def play_one_step_simulation(self):
		#self.print_information('Trying to draw something')
		#self.on_draw_test()
		#self.draw_network()
		#self.draw_trajectory([3, 18, 17, 15])
		map_update_info, text_story, text_info = self.simu.step()
		self.update_map(**map_update_info)
		self.print_story(text_story)
		self.print_information(text_info)

	def update_map(self, trajectory=None, color_trajectory=nice_colors[5], clear=False, origin_destination=None):
		if not clear:
			if trajectory!=None:
				self.draw_trajectory(trajectory, color=color_trajectory)
			if origin_destination!=None:
				self.indicate_origin_destination(*origin_destination)
		else:
			self.draw_network()
	def prepare_main_frame(self):
		self.dpi = 100
		self.fig = Figure((11.0, 8.0), dpi=self.dpi)
		self.canvas = FigureCanvas(self.fig)
		self.canvas.setParent(self.main_frame)

		# Since we have only one plot, we can use add_axes 
		# instead of add_subplot, but then the subplot
		# configuration tool in the navigation toolbar wouldn't
		# work.
		#
		self.axes = self.fig.add_subplot(111)

		# Bind the 'pick' event for clicking on one of the bars
		#
		self.canvas.mpl_connect('pick_event', self.on_pick)

		# Create the navigation toolbar, tied to the canvas
		#
		self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

		# Other GUI controls
		# 
		# self.textbox = QLineEdit()
		# self.textbox.setMinimumWidth(200)
		# self.connect(self.textbox, SIGNAL('editingFinished ()'), self.on_draw)

		# self.draw_button = QPushButton("&Draw")
		# self.connect(self.draw_button, SIGNAL('clicked()'), self.on_draw)

		# self.grid_cb = QCheckBox("Show &Grid")
		# self.grid_cb.setChecked(False)
		# self.connect(self.grid_cb, SIGNAL('stateChanged(int)'), self.on_draw)

		# slider_label = QLabel('Bar width (%):')
		# self.slider = QSlider(Qt.Horizontal)
		# self.slider.setRange(1, 100)
		# self.slider.setValue(20)
		# self.slider.setTracking(True)
		# self.slider.setTickPosition(QSlider.TicksBothSides)
		# self.connect(self.slider, SIGNAL('valueChanged(int)'), self.on_draw)

		#
		# Layout with box sizers
		# 
		# hbox = QHBoxLayout()
		
		# # for w in [  self.textbox, self.draw_button, self.grid_cb,
		# # 			slider_label, self.slider]:
		# # 	hbox.addWidget(w)
		# # 	hbox.setAlignment(w, Qt.AlignVCenter)
		
		# vbox = QVBoxLayout()
		# vbox.addWidget(self.canvas)
		# vbox.addWidget(self.mpl_toolbar)
		# vbox.addLayout(hbox)
		
		# self.main_frame.setLayout(vbox)
		#self.setCentralWidget(self.main_frame)

def main(paras):
	simu = SimulationStory(paras, G=paras['G'])

	app = QApplication(sys.argv)
	form = StrategicGUI(simu=simu)
	form.show()

	app.exec_()

if __name__ == '__main__': 
	#paras_file = None if len(sys.argv)==1 else sys.argv[1]
	paras_file = '/home/earendil/Documents/ELSA/ABM/Old_strategic/Model1/tests/my_paras_DiskWorld_test.py'
	paras = read_paras(paras_file=paras_file)
	main(paras)
