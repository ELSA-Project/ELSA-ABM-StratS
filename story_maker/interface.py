# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../abm_strategic_model1')

import pickle
from numpy import *
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

"""
Command line to generate the python template:
pyuic4 interface_template.ui -o design.py

"""

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

class StrategicGUI(QMainWindow, design.Ui_StrategicLayer):
	def __init__(self, parent=None, simu=None, epsilon=0.05, normal_color_nodes=nice_colors[0],\
		overloaded_color_nodes=nice_colors[2], orig_dest_color=nice_colors[6], traj_color=nice_colors[5],\
		size_nodes_normal=40.):
		super(StrategicGUI, self).__init__(parent)
		self.setupUi(self)

		self.prepare_main_frame()
		self.epsilon = epsilon

		self.pause.clicked.connect(self.pause_simulation)
		self.play.clicked.connect(self.play_simulation)
		self.play_one_step.clicked.connect(self.play_one_step_simulation)
		self.capacitySlider.valueChanged.connect(self.update_map)
		self.showCapacities.stateChanged.connect(self.update_map)

		#self.main_splitter.setStretchFactor(1, 10)
		# These are pixels!
		self.main_splitter.setSizes([600, 200])
		self.text_splitter.setSizes([400, 200])

		self.simu = simu
		self.simu.prepare_simu()
		self.G = self.simu.G

		# colors
		self.normal_color_nodes = normal_color_nodes
		self.overloaded_color_nodes = overloaded_color_nodes
		self.orig_dest_color = orig_dest_color
		self.traj_color = traj_color

		# Sizes
		self.size_nodes_normal = size_nodes_normal

		self.current_map_update = {}

		self.draw_network()

	def show_capacities(self, shift_numbers_min=(-0.01, -0.01), shift_numbers_max=(-0.02, -0.02),\
		fontsize_min=8, fontsize_max=13):
		#time = event
		time = self.capacitySlider.sliderPosition()

		def size_function(load, capacity, min_size=3.*self.size_nodes_normal, max_size=12.*self.size_nodes_normal):
			return min_size + (float(load)/capacity)*(max_size-min_size)

		def fontsize_function(load, capacity, min_size=fontsize_min, max_size=fontsize_max):
			return min_size + (float(load)/capacity)*(max_size-min_size)

		def shift_function(load, capacity, min_pos=array(shift_numbers_min), max_pos=array(shift_numbers_max)):
			return min_pos + (float(load)/capacity)*(max_pos-min_pos)

		# plot only the nodes with traffic>0 for this time
		nodes_to_plot = [n for n in self.G.nodes() if self.G.node[n]['load'][time]>0]

		if len(nodes_to_plot)>0:
			# coordinates of nodes
			x, y = zip(*[self.G.node[n]['coord'] for n in nodes_to_plot])

			# size of nodes
			sizes = [size_function(self.G.node[n]['load'][time], self.G.node[n]['capacity']) for n in nodes_to_plot]

			# colors
			colors = []
			for n in nodes_to_plot:
				if self.G.node[n]['load'][time]<self.G.node[n]['capacity']:
					colors.append(self.normal_color_nodes)
				elif self.G.node[n]['load'][time]==self.G.node[n]['capacity']:
					colors.append(self.overloaded_color_nodes)
				else:
					raise Exception("Loads should not be superior to capacity!")
			
			self.axes.scatter(x, y, s=sizes, edgecolor='w', c=colors, zorder=13)

			# Load in points
			for n in nodes_to_plot:
				load, cap = self.G.node[n]['load'][time], self.G.node[n]['capacity']
				pos_point = array(self.G.node[n]['coord'])
				pos_text = pos_point + shift_function(load, cap)#array(shift_numbers)
				self.axes.annotate(str(load), pos_text,
									size=fontsize_function(load, cap), 
									xytext=pos_text,
									zorder=14,
									color='w')
		else:
			self.print_information("All loads are null.")

	def on_click(self, event):
		if event.button!=1:
			return
		else:
			sec = self.get_ind_under_point(event)
			if sec!=None:
				text = "Load of sector " + str(sec) + " is:\n"
				for h in range(len(self.G.node[sec]['load'])-1):
					text += '- ' + str(self.G.node[sec]['load'][h]) + " for interval " + str(h) + " -- " + str(h+1) + "\n"
				self.print_information(text)

	def get_ind_under_point(self, event):
		'get the index of the vertex under point if within epsilon tolerance'

		xt, yt = zip(*[self.G.node[n]['coord'] for n in self.G.nodes()])#xyt[:, 0], xyt[:, 1]
		d = sqrt((array(xt) - event.xdata)**2 + (array(yt) - event.ydata)**2)
		indseq = nonzero(equal(d, amin(d)))[0]
		ind = indseq[0]

		if d[ind] >= self.epsilon:
			ind = None

		return ind

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

	def draw_network(self):     
		draw_sector_map(self.G, 
						colors=self.normal_color_nodes, 
						limits=(-1.1, -1.1, 1.1, 1.), 
						size_nodes=self.size_nodes_normal, 
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
						#shift_numbers=(0.02, 0.005),
						shift_numbers=(0.03, 0.015),
						size_numbers=8)

	def draw_trajectory(self, p):
		"""
		Parameters
		==========
		p: list
			of labels of nodes.

		TODO: not working with projection.

		"""
		x, y = zip(*[self.G.node[n]['coord'] for n in p])
		self.axes.plot(x, y, '-', lw=2, c=self.traj_color)

	def draw_overloaded_sector(self, sec, size=150.):
		x, y = self.G.node[sec]['coord']
		self.axes.scatter([x], [y], marker='x', s=size, c=self.overloaded_color_nodes, edgecolor='w', zorder=10)

	def indicate_origin_destination(self, origin, destination, size=400., \
		marker='h', fontsize=10, shift_numbers=(-0.015, -0.015)):
		color = self.orig_dest_color
		x, y = self.G.node[origin]['coord']
		self.axes.scatter([x], [y], marker=marker, s=size, c=color, edgecolor='w', zorder=10)
		pos_text = array((x, y)) + array(shift_numbers)
		self.axes.annotate('O', (x,y), size=fontsize, xytext=pos_text, zorder=11, color='w')

		x, y = self.G.node[destination]['coord']
		self.axes.scatter([x], [y], marker=marker, s=size, c=color, edgecolor='w', zorder=10)
		pos_text = array((x, y)) + array(shift_numbers)
		self.axes.annotate('D', (x,y), size=fontsize, xytext=pos_text, zorder=11, color='w')

	def pause_simulation(self):
		self.print_information('Not implemented yet')

	def play_simulation(self):
		self.print_information('Not implemented yet')

	def play_one_step_simulation(self):
		map_update_info, text_story, text_info = self.simu.step()
		self.current_map_update = map_update_info
		self.update_map()
		self.print_story(text_story)
		self.print_information(text_info)

	def update_map(self):
		self.axes.clear()   
		
		trajectory = self.current_map_update.get('trajectory', None)
		origin_destination = self.current_map_update.get('origin_destination', None)
		overloaded_sector = self.current_map_update.get('overloaded_sector', None)
		
		self.draw_network()
		
		if trajectory!=None:
			self.draw_trajectory(trajectory)
		if origin_destination!=None:
			self.indicate_origin_destination(*origin_destination)
		if overloaded_sector!=None:
			self.draw_overloaded_sector(overloaded_sector)
		if self.showCapacities.checkState()==2:
			self.show_capacities()
		
		self.canvas.draw()

	def prepare_main_frame(self):
		self.dpi = 100
		self.fig = Figure((11.0, 8.0), dpi=self.dpi)
		self.canvas = FigureCanvas(self.fig)
		self.canvas.setParent(self.main_frame)

		self.canvas.setFocusPolicy(Qt.ClickFocus)
		self.canvas.setFocus()
		# to capture keyboard
		#self.canvas.mpl_connect('key_press_event', self.pouet)
		self.canvas.mpl_connect('button_press_event', self.on_click)

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
