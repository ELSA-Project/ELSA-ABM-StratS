# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../abm_strategic_model1')

import os
from os.path import join as jn
import time
import pickle
from numpy import *
from numpy.random import lognormal, normal
import networkx as nx
from random import seed

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure

from libs.general_tools import nice_colors
from abm_strategic_model1.utilities import draw_sector_map, read_paras
from abm_strategic_model1.simulationO import Simulation
from libs.paths import result_dir

import design

from story_maker import SimulationStory

"""
Command line to generate the python template:
pyuic4 interface_template.ui -o design.py

"""

class DummyNM(object):
	def __init__(self, control_time_window):
		self.control_time_window = control_time_window

class DummySimu(Simulation):
	"""
	Used for replay of dumped history

	"""
	def __init__(self, load_file, *args, **kwargs):
		#super(DummySimu, self).__init__(*args, **kwargs)
		super(DummySimu, self).__init__(*args, **kwargs)
		self.load_file = load_file

	def load_history(self):
		print 'Loading history...'
		with open(self.load_file, 'r') as f:
			self.updates, self.queue = pickle.load(f)
		self.idx = -1
		control_time_window = 30
		self.NM = DummyNM(control_time_window)

	def prepare_simu(self):
		"""
		To mimic the behaviour of SimulationStory

		"""
		self.load_history()

	def step(self):
		self.idx += 1
		return self.updates[self.idx]

class StrategicGUI(QMainWindow, design.Ui_StrategicLayer):
	def __init__(self, parent=None, simu=None, epsilon=0.05, normal_color_nodes=nice_colors[0],\
		overloaded_color_nodes=nice_colors[2], orig_dest_color=nice_colors[6], traj_color=nice_colors[5],\
		size_nodes_normal=40., play_velocity=1., keep_history=False, save_file='history.pic',\
		load_file=None, rep_res='.'):

		super(StrategicGUI, self).__init__(parent)
		self.setupUi(self)

		self.epsilon = epsilon
		self.velocity = play_velocity

		if load_file==None:
			self.replay = False
			simu = SimulationStory(paras, G=paras['G'])
			self.print_information("Running NEW simulation with parameters:")
		else:
			keep_history = False
			self.replay = True
			simu = DummySimu(jn(rep_res, load_file), paras, G=paras['G'])
			self.print_information("REPLAYING simulation from file:" + load_file + "\nwith parameters:")

		self.rep_res = rep_res

		self.simu = simu

		self.simu.prepare_simu()
		self.G = self.simu.G

		self.prepare_main_frame()
		self.prepare_departure_times()
		self.prepare_satisfaction()


		self.pause.clicked.connect(self.pause_simulation)
		self.play.clicked.connect(self.play_simulation)
		self.play_one_step.clicked.connect(self.play_one_step_simulation)
		self.oneStepBackward.clicked.connect(self.one_step_backward)
		self.oneStepBackward.setEnabled(self.replay)
		self.capacitySlider.valueChanged.connect(self.update_map)
		self.showCapacities.stateChanged.connect(self.update_map)
		self.normalized.stateChanged.connect(self.update_departure_times)
		self.satisfactionSlider.valueChanged.connect(self.update_satisfaction)
		self.speedSpinBox.valueChanged.connect(self.change_velocity)
		self.magicButton.clicked.connect(self.magic)
		self.saveGraphs.clicked.connect(self.save_figures)
		self.jumpSpinBox.valueChanged.connect(self.jump_to_step)
		self.jumpSpinBox.setEnabled(self.replay)
		if self.replay:
			self.jumpSpinBox.setMaximum(len(self.simu.updates)-1)
			

		# These are pixels!
		self.main_splitter.setSizes([600, 300, 200])

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

		self.draw_network()

		self.keep_history = keep_history
		self.history = []
		self.save_file = save_file

		self.update_status = {}

	def change_velocity(self, event):
		self.velocity = self.speedSpinBox.value()

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

	def dump_history(self):
		with open(jn(self.rep_res, self.save_file), 'w') as f:
			pickle.dump((self.history, self.simu.queue), f)

		self.print_information("History dumped as " + self.save_file)

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

	def get_queue(self):
		return self.update_status.get('queue', None)

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

	def jump_to_step(self):
		if self.replay:
			# Ensure that the button does no go over the maximum number of flights.
			self.print_information("Jumping to step", self.jumpSpinBox.value())
			self.print_story("\n\n Jump to step", self.jumpSpinBox.value(), "\n\n")
			self.simu.idx = self.jumpSpinBox.value() - 1
			self.play_one_step_simulation()

	def magic(self):
		current_times = [f.fp_selected.t/60. for f in self.get_queue() if hasattr(f, 'accepted') and f.accepted]
		self.print_information(current_times)

	def one_step_backward(self):
		if self.simu.idx>0:
			self.simu.idx -= 2
			self.print_information("Jumping to step", self.simu.idx+1)
			self.print_story("\n\n Jump to step", self.simu.idx+1, "\n\n")
			self.play_one_step_simulation()

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

	def pause_simulation(self):
		self.stop = True

	def play_one_step_simulation(self):
		self.update_status = self.simu.step()

		self.print_story(self.update_status.get('text_story', None))
		self.print_information(self.update_status.get('text_info', None))

		if self.keep_history:
			self.history.append(self.update)

		if not self.update_status.get('stop', False):			
			self.update_map()
			
			self.update_departure_times()

			self.update_satisfaction()

		else:
			self.stop = True
			if self.keep_history:
				self.dump_history()

	def play_simulation(self):
		self.stop = False
		self.print_information("Started playing...")
		while not self.stop:
			start_time = time.time()
			self.play_one_step_simulation()
			
			sim_duration = time.time()-start_time

			time.sleep(max(0., 1./float(self.velocity) - sim_duration))

			# Catch all the other events
			qApp.processEvents()

		self.print_information("Stopped playing.")

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

		self.mpl_toolbar_sat = NavigationToolbar(self.canvas_sat, self.satisfaction)

		# Pre compute preferred times 
		self.pref_times = [f.pref_time/60. for f in self.simu.queue]

	def print_information(self, *texts):
		if len(texts)>0 and texts[0]!=None:
			text_tot = ''
			for text in texts:
				text_tot += str(text)
			self.information.append(text_tot)

	def print_story(self, *texts):
		if len(texts)>0 and texts[0]!=None:
			text_tot = ''
			for text in texts:
				text_tot += str(text)
			self.story.append(text_tot)
	
	def save_figures(self):
		self.print_information("Saving graphs and history in " + self.rep_res)
		it = self.simu.idx
		pos_init = self.satisfactionSlider.sliderPosition()
		self.satisfactionSlider.setSliderPosition(0)
		self.fig_sat.savefig(jn(self.rep_res, 'satisfactions_step' + str(it) +  '.png'))
		self.satisfactionSlider.setSliderPosition(1)
		self.fig_sat.savefig(jn(self.rep_res, 'diff_sats_step' + str(it) +  '.png'))
		self.fig_dt.savefig(jn(self.rep_res, 'departure_times_step' + str(it) +  '.png'))

		# Put the satisfaction slider back in initial position
		self.satisfactionSlider.setSliderPosition(pos_init)

		if self.keep_history:
			self.dump_history()

	def show_departure_times(self):
		self.departure_times_window.show()
		self.departure_times_window.show_departure_times()

	def show_capacities(self, shift_numbers_min=(-0.01, -0.01), shift_numbers_max=(-0.02, -0.02),\
		fontsize_min=8, fontsize_max=13):
		#time = event
		time = self.capacitySlider.sliderPosition()
		if time>self.simu.NM.control_time_window:
			self.print_information("Outside of control window.")
		else:
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
		
	def update_departure_times(self):
		self.axes_dt.clear()   
		queue = self.get_queue()

		normed = self.normalized.checkState()==2

		current_times = [f.fp_selected.t/60. for f in queue if hasattr(f, 'accepted') and f.accepted]

		max_hour = int(max(current_times+self.pref_times))
		self.axes_dt.hist(self.pref_times, bins=arange(max_hour+2), color=nice_colors[0], alpha=0.5, label='Pref. times', normed=normed)
		if len(current_times)>0:
			self.axes_dt.hist(current_times, bins=arange(max_hour+2), color=nice_colors[2], alpha=0.5, label='Actual times', normed=normed)
		self.axes_dt.legend(fontsize=8)
		self.axes_dt.set_xlabel("Time of the day in hours")
		if not normed:
			self.axes_dt.set_ylabel("Number of departures")
		else:
			self.axes_dt.set_ylabel("Density of departures")

		self.canvas_dt.draw()

	def update_map(self):
		current_map_update = self.update_status.get('map_update_info', {})

		self.axes.clear()   
		
		trajectory = current_map_update.get('trajectory', None)
		origin_destination = current_map_update.get('origin_destination', None)
		overloaded_sector = current_map_update.get('overloaded_sector', None)
		
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

	def update_satisfaction(self):
		satisfaction = self.update_status.get('satisfaction', None)
		popS = (1., 0., 0.000001)#self.par_companyS
		popR = (1., 0., 1000000.)#self.par_companyR

		if satisfaction!=None:
			self.axes_sat.clear()
			sat_S = zeros(len(satisfaction))
			sat_R = zeros(len(satisfaction))
			nS = zeros(len(satisfaction))
			nR = zeros(len(satisfaction))
			for i, (sat, typ) in enumerate(satisfaction):
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
				self.axes_sat.plot(range(len(satisfaction)), sat_S, label='S', color=nice_colors[0])
				self.axes_sat.plot(range(len(satisfaction)), sat_R, label='R', color=nice_colors[2])
				self.axes_sat.legend(fontsize=8, loc='lower left')
				self.axes_sat.set_ylabel('Satisfaction')
				self.axes_sat.set_ylim((0., 1.1))
			elif typ_plot==1:
				self.axes_sat.plot(range(len(satisfaction)), sat_S-sat_R, color=nice_colors[4])
				self.axes_sat.set_ylabel('Sat_S - Sat_R')
				self.axes_sat.set_ylim((-1.1, 1.1))
			
			self.axes_sat.set_xlabel('Number of flights')
			
			self.canvas_sat.draw()


def main(paras, **kwargs):
	app = QApplication(sys.argv)
	form = StrategicGUI(**kwargs)
	form.show()

	app.exec_()

if __name__ == '__main__':
	if 1:
		# Manual seed
		see = 10
		print "===================================="
		print "USING SEED", see
		print "===================================="
		seed(see)
	#paras_file = None if len(sys.argv)==1 else sys.argv[1]
	paras_file = '/home/earendil/Documents/ELSA/ABM/Old_strategic/Model1/tests/my_paras_DiskWorld_test.py'
	save_rep = jn(result_dir, 'model1/3.1/DiskWorld/stories/big2')
	os.system("mkdir -p " + save_rep)
	#paras_file = '/home/earendil/Documents/ELSA/ABM/Old_strategic/Model1/abm_strategic_model1/my_paras/my_paras_DiskWorld_for_story.py'
	paras = read_paras(paras_file=paras_file)
	
	history_save_file = "test.pic"
	
	#history_load_file = '../tests/history_test.pic'
	history_load_file = None
	#history_load_file = jn(save_rep, "history_big2.pic")
	main(paras, keep_history=True, save_file=history_save_file, load_file=history_load_file, rep_res=save_rep)
