# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../abm_strategic_model1')

import time
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
#import additional_window_design

from story_maker import SimulationStory

"""
Command line to generate the python template:
pyuic4 interface_template.ui -o design.py

"""

class StrategicGUI(QMainWindow, design.Ui_StrategicLayer):
	def __init__(self, parent=None, simu=None, epsilon=0.05, normal_color_nodes=nice_colors[0],\
		overloaded_color_nodes=nice_colors[2], orig_dest_color=nice_colors[6], traj_color=nice_colors[5],\
		size_nodes_normal=40., play_velocity=1., keep_history=False, save_file='history.pic'):
		super(StrategicGUI, self).__init__(parent)
		self.setupUi(self)

		self.prepare_main_frame()
		self.prepare_departure_times()
		self.prepare_satisfaction()

		self.epsilon = epsilon
		self.velocity = play_velocity

		self.pause.clicked.connect(self.pause_simulation)
		self.play.clicked.connect(self.play_simulation)
		self.play_one_step.clicked.connect(self.play_one_step_simulation)
		self.capacitySlider.valueChanged.connect(self.update_map)
		self.showCapacities.stateChanged.connect(self.update_map)
		self.normalized.stateChanged.connect(self.update_departure_times)
		self.satisfactionSlider.valueChanged.connect(self.update_satisfaction)
		self.speedSpinBox.valueChanged.connect(self.change_velocity)

		#self.departureTimes.clicked.connect(self.show_departure_times)

		#self.main_splitter.setStretchFactor(1, 10)
		# These are pixels!
		self.main_splitter.setSizes([600, 300, 200])
		#self.text_splitter.setSizes([400, 200])

		self.simu = simu
		self.simu.prepare_simu()
		self.G = self.simu.G

		self.print_information("Doing simulation with:")
		self.print_information("- Total number of flights: " + str(len(self.simu.queue)))
		self.print_information("- Proportion of companies S: " + str(self.simu.paras['nA']))
		self.print_information("- Departure pattern type: " + str(self.simu.paras['departure_times']))
		if 'Delta_t' in self.simu.paras.keys():
			self.print_information("- Delta_t: " + str(self.simu.paras['Delta_t']))
		self.print_information('')

		# colors
		self.normal_color_nodes = normal_color_nodes
		self.overloaded_color_nodes = overloaded_color_nodes
		self.orig_dest_color = orig_dest_color
		self.traj_color = traj_color

		# Sizes
		self.size_nodes_normal = size_nodes_normal

		self.current_map_update = {}

		self.draw_network()

		self.keep_history = keep_history
		self.history = []
		self.save_file = save_file

	def change_velocity(self, event):
		self.velocity = self.speedSpinBox.value()

	def show_departure_times(self):
		self.departure_times_window.show()
		self.departure_times_window.show_departure_times()

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
		"""
		get the index of the vertex under point if within epsilon tolerance
		"""

		xt, yt = zip(*[self.G.node[n]['coord'] for n in self.G.nodes()])#xyt[:, 0], xyt[:, 1]
		d = sqrt((array(xt) - event.xdata)**2 + (array(yt) - event.ydata)**2)
		indseq = nonzero(equal(d, amin(d)))[0]
		ind = indseq[0]

		if d[ind] >= self.epsilon:
			ind = None

		return ind

	def print_information(self, text):
		if text!=None:
			self.information.append(text)

	def print_story(self, text):
		if text!=None:
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
		self.axes.scatter([x], [y], marker=marker, s=size, c=color, edgecolor='w', zorder=20)
		pos_text = array((x, y)) + array(shift_numbers)
		self.axes.annotate('O', (x,y), size=fontsize, xytext=pos_text, zorder=21, color='w')

		x, y = self.G.node[destination]['coord']
		self.axes.scatter([x], [y], marker=marker, s=size, c=color, edgecolor='w', zorder=20)
		pos_text = array((x, y)) + array(shift_numbers)
		self.axes.annotate('D', (x,y), size=fontsize, xytext=pos_text, zorder=21, color='w')

	def pause_simulation(self):
		self.print_information('Not implemented yet')
		self.stop = True

	def play_simulation(self):
		#self.print_information('Not implemented yet')
		self.stop = False
		#self.print_information("Started playing...")
		self.print_information("Started playing...")
		while not self.stop:
			self.play_one_step_simulation()
			time.sleep(1./float(self.velocity))
			qApp.processEvents()
			#print "Value of self.pause.clicked: ", self.pause.clicked()

			#stop = self.pause.clicked()

		self.print_information("Stopped playing.")

	def play_one_step_simulation(self):
		update = self.simu.step()

		self.print_story(update.get('text_story', None))
		self.print_information(update.get('text_info', None))

		self.satisfaction = update.get('satisfaction', None)
		self.current_map_update = update.get('map_update_info', None)

		if not update.get('stop', False):			
			self.update_map()
			
			self.update_departure_times()

			self.update_satisfaction()

			if self.keep_history:
				self.history.append(update)
		else:
			self.stop = True
			if self.keep_history:
				self.dump_history()

	def dump_history(self):
		with open(self.save_file, 'w') as f:
			pickle.dump(self.history, f)

		self.print_information("History dumped as " + self.save_file) 

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

	def prepare_departure_times(self):
		self.fig_dt = Figure((5.0, 4.0), dpi=100)
		self.canvas_dt = FigureCanvas(self.fig_dt)
		self.canvas_dt.setParent(self.departureTimes)

		self.canvas_dt.setFocusPolicy(Qt.ClickFocus)
		self.canvas_dt.setFocus()
		self.axes_dt = self.fig_dt.add_subplot(111)

		self.mpl_toolbar_dt = NavigationToolbar(self.canvas_dt, self.departureTimes)

	def prepare_satisfaction(self):
		self.fig_sat = Figure((5.0, 4.0), dpi=100)
		self.canvas_sat = FigureCanvas(self.fig_sat)
		self.canvas_sat.setParent(self.satisfaction)

		self.canvas_sat.setFocusPolicy(Qt.ClickFocus)
		self.canvas_sat.setFocus()
		self.axes_sat = self.fig_sat.add_subplot(111)

		self.mpl_toolbar_sat = NavigationToolbar(self.canvas_sat, self.departureTimes)

	def update_satisfaction(self):
		popS = (1., 0., 0.000001)#self.par_companyS
		popR = (1., 0., 1000000.)#self.par_companyR

		if self.satisfaction!=None:
			self.axes_sat.clear()
			sat_S = zeros(len(self.satisfaction))
			sat_R = zeros(len(self.satisfaction))
			nS = zeros(len(self.satisfaction))
			nR = zeros(len(self.satisfaction))
			for i, (sat, typ) in enumerate(self.satisfaction):
				sat_S[i] = sat_S[i-1]
				sat_R[i] = sat_R[i-1]
				nS[i] = nS[i-1]
				nR[i] = nR[i-1]
				if typ==popS:
					sat_S[i] += sat
					nS[i] += 1.
				elif typ==popR:
					sat_R[i] += sat
					nR[i] += 1.

			sat_S = sat_S/nS
			sat_R = sat_R/nR

			typ_plot = self.satisfactionSlider.sliderPosition()
			if typ_plot==0:
				self.axes_sat.plot(range(len(self.satisfaction)), sat_S, label='S', color=nice_colors[0])
				self.axes_sat.plot(range(len(self.satisfaction)), sat_R, label='R', color=nice_colors[2])
				self.axes_sat.legend(fontsize=8, loc='lower left')
				self.axes_sat.set_ylabel('Satisfaction')
				self.axes_sat.set_ylim((0., 1.1))
			elif typ_plot==1:
				self.axes_sat.plot(range(len(self.satisfaction)), sat_S-sat_R, color=nice_colors[4])
				self.axes_sat.set_ylabel('Sat_S - Sat_R')
				self.axes_sat.set_ylim((-1.1, 1.1))
			
			self.axes_sat.set_xlabel('Number of flights')
			
			self.canvas_sat.draw()
		
	def update_departure_times(self):
		self.axes_dt.clear()   

		normed = self.normalized.checkState()==2

		pref_times = [f.pref_time/60. for f in self.simu.queue]
		current_times = [f.fp_selected.t/60. for f in self.simu.queue if hasattr(f, 'accepted') and f.accepted]

		self.axes_dt.hist(pref_times, bins=list(range(24)), color=nice_colors[0], alpha=0.5, label='Pref. times', normed=normed)
		if len(current_times)>0:
			self.axes_dt.hist(current_times, bins=list(range(24)), color=nice_colors[2], alpha=0.5, label='Actual times', normed=normed)
		self.axes_dt.set_xlim((0, 24))
		self.axes_dt.legend(fontsize=8)
		
		self.canvas_dt.draw()

def main(paras, **kwargs):
	simu = SimulationStory(paras, G=paras['G'])

	app = QApplication(sys.argv)
	form = StrategicGUI(simu=simu, **kwargs)
	form.show()

	app.exec_()

if __name__ == '__main__': 
	#paras_file = None if len(sys.argv)==1 else sys.argv[1]
	paras_file = '/home/earendil/Documents/ELSA/ABM/Old_strategic/Model1/tests/my_paras_DiskWorld_test.py'
	#paras_file = '/home/earendil/Documents/ELSA/ABM/Old_strategic/Model1/abm_strategic_model1/my_paras/my_paras_DiskWorld_for_story.py'
	paras = read_paras(paras_file=paras_file)
	history_save_file = '../tests/history_test.pic'
	main(paras, keep_history=True, save_file=history_save_file)
