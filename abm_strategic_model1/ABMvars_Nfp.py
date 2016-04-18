# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:15:30 2013

@author: luca
"""
#import sys
#sys.path.insert(1,'../Distance')
import pickle as _pickle
#from networkx import shortest_path
import sys as _sys
import numpy as _np
from prepare_network import prepare_network as _prepare_network
from math import ceil
import networkx as nx
from random import sample, shuffle
from general_tools import counter
import os as _os
from copy import copy as _copy

version='2.6.7'

# class Paras(dict):
#     def __init__(self, dic):
#         for k,v in dic.items():
#             self[k]=v
#         self.to_update={}
#     def update(self, name_para, new_value):
#         paras[name_para]=new_value
#         print 'Update:', name_para, new_value
#         for k in self.update_priority:
#             (f,args)=self.to_update[k]
#             vals=[paras[a] for a in args] 
#             self[k]=f(*vals)
class NoAirportsLeft(Exception):
    pass


def check_and_repair_H(G):
    recompute = True
    for n1, n2 in G.edges():
        try:
            assert G[n1][n2]['weight']==G.H[n1][n2]
        except AssertionError:
            print "graph H does not match G, I recompute it"
            recompute = True
            break
    if recompute:
        G.build_H()

    return G



class Paras(dict):
    def __init__(self, dic):
        for k,v in dic.items():
            self[k]=v
        self.to_update={}

    def update(self, name_para, new_value):
        paras[name_para]=new_value
        # Everything before level_of_priority_required should not be updated, given the para being updated.
        lvl = self.levels.get(name_para, len(self.update_priority)) #level_of_priority_required
        #print name_para, 'being updated'
        #print 'level of priority:', lvl, (lvl==len(update_priority))*'(no update)'
        for j in range(lvl, len(self.update_priority)):
            k = self.update_priority[j]
            (f,args)=self.to_update[k]
            vals=[paras[a] for a in args] 
            self[k]=f(*vals)

    def analyse_dependance(self):
        """
        Detect the first level of priority hit by a dependance in each parameter.
        Those who don't need any kind of update are not in the dictionnary.
        """
        self.levels = {}
        for i, k in enumerate(self.update_priority):
            (f,args)=self.to_update[k]
            for arg in args:
                if arg not in self.levels.keys():
                    self.levels[arg] = i
            
def network_whose_name_is(name):
    with open('../networks/' + name + '.pic') as _f:    
        B=_pickle.load(_f)
    return B


# ---------------- Network --------------- #
update_priority=[]
to_update={}

# type_of_net='D'                 #type of graph ('D'=Delaunay triangulation, 'T'=triangular lattice, "E"=Erdos-Renyi random graph)
# N=30                            #order of the graph (in the case net='T' verify that the order is respected)        
# #airports=[65,20]                #IDs of the nodes used as airports
#airports=['LFMME3', 'LFFFTB']   #LFBBN2 (vers Perpignan)
# nairports=2                     #number of airports
# pairs=[]#[(22,65)]              #available connections between airports
# #[65,20]
# #[65,22]
# #[65,62]
# min_dis=2                       #minimum number of nodes between two airpors


# generate_weights=True
# typ_weights='coords'
# sigma=0.01
# generate_capacities=True
# typ_capacities='manual'
# C=5                             #sector capacity

file_capacities='capacities_sectors_Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0.pic'

paras_G={k:v for k,v in vars().items() if k[:1]!='_' and k!='version'}


fixnetwork=True               #if fixnetwork='True' the graph is loaded from file and is not generated at each iteration

def give_airports_to_network(G, airports, distance=7, repetitions=False, force_short_computation=False):
    #print G
    for a in G.airports:
        C_airport = G.node[a]['capacity']
        n_non_aiport = [n for n in G.nodes() if not n in G.airports][0]
        G.node[a]['capacity'] = G.node[n_non_aiport]['capacity'] #WARNING: WORKS ONLY FOR CONSTANT CAPACITY!
    G.fix_airports(airports, 2)
    for a in G.airports:
        G.node[a]['capacity'] = C_airport
    #G.compute_shortest_paths(G.Nfp, singletons=False, repetitions=False, old=False)
    #G.name = _name_network + '_' + str(min(airports)) + '_' + str(max(airports))# + '_spatial'
    # G.name = 'DEL_C' + '_' + str(min(airports)) + '_' + str(max(airports)) + '_spatial'
    
    name_first, name_last = '', ''
    for coin in G.name.split('_'):
        try:
            a = int(coin)
        except ValueError:
            if coin!='spatial' and coin:
                name_first += coin + '_'
            else:
                name_last += '_' + coin 
        
    #G.name = 'NEWDEL1000_C_' + str(min(airports)) + '_' + str(max(airports)) + '_spatial' 
    G.name = name_first + str(min(airports)) + '_' + str(max(airports)) + name_last 
    print 'Gave name:', G.name
    #print 'airports', airports
    _name_SP = '../networks/' + _name_network + '_SPs/' + G.name + '_SPs.pic'
    print _name_SP
    if _os.path.exists(_name_SP) and not force_short_computation:
        with open(_name_SP, 'r') as _f:
            short=_pickle.load(_f)
            G.short=short
    else:
        print 'Computing shortest paths for these airports.'
        G.compute_shortest_paths(G.Nfp, repetitions = repetitions)
        #print 'len(G.short)', len(G.short)
        #print 'len(G.pairs)', len(G.pairs)

        with open(_name_SP, 'w') as _f:
            _pickle.dump(G.short, _f)

    #print 'len(G.short)', len(G.short)
    #print 'len(G.pairs)', len(G.pairs)
    if G.pairs!=G.short:
        G.pairs=G.short
    if len(G.short)>=2:
        for airports, paths in G.short.items():
            #for path in paths:
            #try:
            assert len(paths[0])>=distance
            # except:
            #     # print
            #     # print "Weight of the best path:", G.weight_path(paths[0])
            #     # print "Weight of the path:", G.weight_path(path)
            #     # print airports
            #     # print path  
            #     raise  
    else:
        raise NoAirportsLeft()
    #print G
    return G

def set_Nf(G, Nfp):
    G.Nfp = Nfp
    G.initialize_load()#2*(nx.diameter(G)+G.Nfp))
    G.compute_shortest_paths(G.Nfp, repetitions = False)
    
    return G


if fixnetwork:
    if 1:  # Take existing network
        #_f=open('LF_C_NSE.pic', 'r')
        #G=network_whose_name_is('DEL_C_0_4_spatial')
        repetitions = False
        #_name_network = 'NEWDEL1000_C_spatial'
        #_name_network = 'DEL_C_65_22_spatial'
        _name_network = 'DEL_C_65_20'
        #_name_network = 'DEL_C_65_22_new_spatial'
        #_name_network = 'LF26_RC_FL350_DA0v3_Strong'
        G = network_whose_name_is(_name_network)
        G = check_and_repair_H(G)
        #_name_network = 'NEWDEL1000_C_spatial_no_repetitions'
        #_name_network = 'DEL_C_65_22_spatial'
        _os.system('mkdir -p ../networks/' + _name_network + '_SPs')
        #airports = G.airports

        if 0:
            distance=7
            n_max_pairs = 100
            force_short = False
            _name_file = '../networks/' + _name_network + 'pairs_airports_distance_' + str(distance) + '_' + str(n_max_pairs) + '.pic'
            if not _os.path.exists(_name_file) or 0:
                airports_iter = []
                pairs = [(n, m) for n in G.nodes() for m in G.nodes() if n<m]#G.edges()[:]
                shuffle(pairs)
                for (n, m) in pairs:
                #for i, n in enumerate(nodes):
                #    for j, m in enumerate(nodes):
                    counter(len(airports_iter), n_max_pairs, message='Computing possible pairs of airports with distance ' + str(distance) + '...')
                    if n<m:
                        _sp = nx.shortest_path(G, n, m, weight = 'weight')
                        if not repetitions and len(set(_sp))<len(_sp):
                            _H = _copy(G)
                            _H.compute_shortest_paths(1, repetitions = repetitions) 
                            _sp = _H.short[(n,m)][0]

                        if len(_sp) == distance:
                            airports_iter.append([n,m])
                            if len(airports_iter)>=n_max_pairs: 
                                break       
                            

                with open(_name_file, 'w') as _f:
                    _pickle.dump(airports_iter, _f)

                print 'Number of pairs with distance', distance, ':', len(airports_iter)
                #if len(airports_iter)>1000:
                    #airports_iter = sample(airports_iter, 1000)
                    #print 'I selected the first 1000.'
                    #print 'I selected 1000 pairs at random.'

                    #with open(_name_file + '_1000.pic', 'w') as _f:
                    #    _pickle.dump(airports_iter, _f)

            else:
                with open(_name_file, 'r') as _f:
                    airports_iter = _pickle.load(_f)

            airports_iter_copy = airports_iter[:]
            for airports in airports_iter_copy:
                try:
                    give_airports_to_network(G, airports, distance=distance, repetitions=False, force_short_computation=force_short)
                except NoAirportsLeft:
                    print "Deleting airports", airports
                    airports_iter.remove(airports)
                    continue

                #if len(airports_iter)>n_max_pairs:
                #    with open(_name_file + '_1000.pic', 'r') as _f:
                #        airports_iter = _pickle.load(_f)

        # with open('pairs_airports_retrieved.pic', 'r') as _f:
        #     airports_iter = _pickle.load(_f)

        # if 0:
        #     with open('NEWDEL1000_C_spatial_SPs/NEWDEL1000_C_spatial_pairs.pic', 'r') as f:
        #         airports_iter = _pickle.load(f)
        #         airports_iter = sample(airports_iter, 1000)
        # else:
        #     with open('pairs_airports_retrieved.pic', 'r') as f:
        #         airports_iter = _pickle.load(f)
        #print airports_iter
        #G_iter=[network_whose_name_is(n) for n in ['DEL_C_4A', 'DEL_C_65_20' , 'DEL_C_4A2']]#, 'DEL_C_6A']]

            to_update['G']=(give_airports_to_network,('G', 'airports'))
            update_priority+=['G']

            N_pairs_airports = 10
            airports_iter = tuple([tuple(_pair) for _pair in airports_iter[:N_pairs_airports]])
    else: # Prepare a new one
        G=_prepare_network(paras_G)
else:
    G=None

# ---------------- Companies ---------------- #

Nfp=10                  #number of flight plans to submit for a flight (pair of departing-arriving airports)
# try:
#     assert G.Nfp==Nfp
# except:
#     print 'Nfp should be the same than in the network. Nfp=', Nfp, ' ; G.Nfp=', G.Nfp
#     raise
Nfp_iter = [2, 5, 10, 15, 20, 30, 50]

if G.Nfp!=Nfp:
    print 'Nfp should be the same than in the network. Nfp=', Nfp, ' ; G.Nfp=', G.Nfp
    print "I recompute shortest paths for the network."
    G = set_Nf(G, Nfp)
    #, 70]

if 1:    
    to_update['G']=(set_Nf,('G','Nfp'))
    update_priority+=['G']

na=1                            #number of flights (pairs of departing-arriving airports) per company
density=30.0
#density_iter=[2.*_i for _i in range(1,11)]

#density_iter=[5., 10.]

tau=1.
tau_iter=_np.arange(0.0001,1.01,0.05)           # factor for shifting in time the flight plans.

departure_times='square_waves' #departing time for each flight, for each AC
assert departure_times in ['zeros', 'peaks', 'uniform', 'square_waves']

def func_density_vs_ACtot_na_day(ACtot, na, day):
    return ACtot*na/float(day)#/unit)

def func_Np(a,b,c):
    return int(ceil(a/float(b+c)))
    
def func_ACsperwave(a,b):
    return int(a/float(b))
    
def func_Delta_t(a):
    return a

with_flows = False

day = 24.
flows = {}

if with_flows:
    print 'Using flows... (' + str(len(G.flights_selected)) + ' flights)'
    
    ACtot = 0
    if ACtot >0:
        _fl = sample(G.flights_selected, ACtot)
    else:
        _fl = G.flights_selected

    for f in _fl:
        # _entry = G.G_nav.idx_navs[f['route_m1t'][0][0]]
        # _exit = G.G_nav.idx_navs[f['route_m1t'][-1][0]]
        _entry = f['sec_path'][0]
        _exit = f['sec_path'][-1]
        flows[(_entry, _exit)] = flows.get((_entry, _exit),[]) + [f['sec_path_t'][0]]

    departure_times = 'exterior'
    ACtot = sum([len(v) for v in flows.values()])
    given_density = False

    density=func_density_vs_ACtot_na_day(ACtot, na, day)
    to_update['density']=(func_density_vs_ACtot_na_day,('ACtot','na','day'))
    update_priority+=['AC', 'AC_dict']

else:
    if departure_times=='uniform':
        def func_ACtot(a,b,c):
            return int(a*b/float(c))
        day=24.
        ACtot=func_ACtot(density, day, na)
        update_priority+=['ACtot', 'AC', 'AC_dict']
        to_update['ACtot']=(func_ACtot, ('density', 'day', 'na'))
    elif departure_times=='zeros':
        update_priority+=['AC', 'AC_dict']
        day=1.
        Delta_t=1.
        ACtot=120
        #ACtot_iter=[20*i for i in range(1, 11)]
        ACtot_iter=[10*i for i in range(1, 11)]
    elif departure_times=='square_waves':
        def func_ACtot(a,b):
            return int(a*b)
        def func_density(a,b,c):
            return a*b/float(c)
        def func_ACsperwave(a,b,c):
            return int(float(a*b)/float(c))

        day=24
        Delta_t=23
        #Delta_t_iter=range(24)
        #Delta_t_iter=range(12)
        Delta_t_iter=[0,1,5,8,11,23]
        #Delta_t_iter=[0,23]
        #Delta_t_iter=[0,1,5,8,11,15,20,23]
        #Delta_t_iter=[16, 17, 18]
            
        ACsperwave=20
        #ACsperwave_iter=[2*_i for _i in range(1, 20)] + [5*_i for _i in range(8,21)]
        #ACsperwave_iter=[2*_i for _i in range(1, 20)] + [5*_i for _i in range(8,21)]
        ACsperwave_iter=[5*_i for _i in range(1, 11)]
        density=5.
        density_iter=[1.,5.,10.]
        Np=func_Np(day, tau, Delta_t)
        #print 'Np=',Np
        #ACtot=func_ACtot(density,day)
        #
        ACtot=func_ACtot(ACsperwave, Np)
        to_update['ACtot']=(func_ACtot,('ACsperwave','Np'))
        to_update['Np']=(func_Np,('day', 'tau', 'Delta_t'))
        if 0:
            #constant ACsperwave
            update_priority+=['Np', 'ACtot','density','AC', 'AC_dict']
            #update_priority=['Np','density', 'ACtot','AC', 'AC_dict']
            density=func_density(ACtot, na, day)
            to_update['density']=(func_density,('ACtot','na','day'))
        else:
            #constant density
            update_priority+=['Np','ACsperwave', 'ACtot','AC', 'AC_dict']
            ACsperwave=func_ACsperwave(density,day,Np)
            ACtot=func_ACtot(ACsperwave, Np)
            to_update['ACsperwave']=(func_ACsperwave,('density', 'day','Np'))
            
        #to_update['Delta_t']=(func_Delta_t,('Delta_t',))
        #to_update['ACsperwave']=(func_ACsperwave,('ACtot', 'Np'))
        #ACtot=func_ACtot(density, day, na)
        #ACsperwave=func_ACsperwave(ACtot,Np)

nA=1.                        # percentage of Flights of the AC number 1

_range1 = list(_np.arange(0.02,0.1,0.02))
_range2 = list(_np.arange(0.92,0.99,0.02))
_range3 = list(_np.arange(0.1,0.91,0.1))
_range4 = list(_np.arange(0.,1.01,0.1))
_range5 = list(_np.arange(0.,1.,0.2))
#nA_iter=_range1 + _range3 + _range2
#nA_iter=_range4#list(_np.arange(0.7,0.91,0.1))
nA_iter=[-1.] + _range1 + _range3 + _range2 + [2.]
#nA_iter=_range1 + _range3 + _range2
#nA_iter= [-1., 2.]

def func_AC(a, b):
    if 1>=a>=0:
        return [int(a*b),b-int(a*b)]  
    elif a<0:
        return [1, b-1]
    else:
        return [b-1, 1]

AC=func_AC(nA, ACtot)               #number of air companies/operators

#par_iter=[[[1.,0.,0.001], [1.,0.,1000.]],[[1.,0.,1.], [1.,0.,1.]], [[1.,0.,1000.], [1.,0.,1.]]]
#par_iter=[[[1.,0.,10.**_e], [1.,0.,1.]] for _e in range(-3,4)]
par_iter=[[[1.,0.,10.**_e], [1.,0.,1.]] for _e in _np.arange(-3.5, 4., 0.5)]
#par_iter=[[[1.,0.,10.**_e], [1.,0.,1.]] for _e in [-3,3]]

#par_iter=[[[1.,0.,0.], [1.,0.,1.]], [[0.,0.,1.], [1.,0.,1.]]]
#par_iter=[[[1., 0., 0.], [1., 0., 1.]]] + [[[1.,0.,10.**_e], [1.,0.,1.]] for _e in range(-3,4)] + [[[0., 0., 1.], [1., 0., 1.]]]
#par_iter=[[[0.,0.,1.], [1.,0.,1.]]]


#par_iter=[[[1.,0.,1000.**_e], [1.,0,1.]] for _e in [-1,0,1]]
#par_iter=[[[10.**(_e+1.),0,10.**_e], [1.,0,1.]] for _e in range(-4,4)]
par_iter=tuple([tuple([tuple([float(_v) for _v in _pp])  for _pp in _p])  for _p in par_iter]) # transformation in tuple, because arrays cannot be keys for dictionaries.
#par=[[1.,0.,0.001], [1.,0,1000.]]
par=[[1.,0.,1000.], [1.,0,1000.]]
#par=[[1.,0.,0.], [1.,0,1000.]]
#par=[[0., 0., 1.], [1., 0., 1.]]
#par=[[10.**(-8.),0.,1.], [1.,0,1000.]]

#par = [[1., 0., 0.001], [1., 0, 1.]]
#par = [[1., 0., 1000.], [1., 0, 1.]]
#par = [[1., 0., 1000.], [1., 0., 0.001]]
#par=[[1.,0.,0.001]]
par=tuple([tuple(_p)  for _p in par]) 
#par=tuple([tuple([float(_v) for _v in _p])  for _p in par]) 

def func_AC_dict(a, b, c):
    if c[0]==c[1]:
        return {c[0]:int(a*b)}
    else:
        return {c[0]:int(a*b), c[1]:b-int(a*b)}  

AC_dict=func_AC_dict(nA, ACtot, par)                #number of air companies/operators

n_iter=100                 #number of iterations in the main loop 

# ------------------ From M0 to M1 ------------------- #

N_shocks=0
N_shocks_iter=range(0,80,5)

# --------------------System parameter -------------------- #
parallel = True
force = False

# ---------------------------------------------------- #
#paras_to_loop=['Delta_t', 'nA']
#paras_to_loop=['airports', 'Delta_t', 'nA']
#paras_to_loop=['Delta_t', 'par'] #pouic
#paras_to_loop=['par', 'ACtot']
#paras_to_loop=['par', 'Delta_t']
#paras_to_loop=['nA', 'ACsperwave']
#paras_to_loop=['nA', 'ACtot']
#paras_to_loop=['ACsperwave', 'nA']
#paras_to_loop=['par', 'N_shocks']#['nA','ACtot']#['par', 'ACtot']
#paras_to_loop=['ACtot']
#paras_to_loop=['G','ACtot']

#paras_to_loop=['par', 'ACsperwave']#, 'Delta_t'] #pouet
#paras_to_loop=['nA', 'density', 'Delta_t']

#paras_to_loop=['airports', 'par', 'Delta_t']
#paras_to_loop=['airports', 'Delta_t']
#paras_to_loop=['par']#, 'Delta_t']
#paras_to_loop=['nA']#, 'Delta_t']
#paras_to_loop=['N_shocks', 'nA']

#paras_to_loop = ['Nfp', 'ACsperwave']
paras_to_loop = ['Nfp', 'Delta_t']


# if 'G' in paras_to_loop:
#     with open('DEL_C_65_22.pic', 'r') as f:
#         GG=_pickle.load(f)
#     print 'Loading networks...'
#     distance=7
#     pairs_distance=[]
#     for i,n in enumerate(GG.nodes()):
#         for j,m in enumerate(GG.nodes()):
#             if i<j:
#                 if len(nx.shortest_path(GG,n,m))==distance:
#                     pairs_distance.append([n,m])

#     #G_iter=[network_whose_name_is(n) for n in ['DEL_C_4A', 'DEL_C_65_20' , 'DEL_C_4A2']]#, 'DEL_C_6A']]
#     print 'Number of pairs in total:', len(pairs_distance)
#     G_iter=[network_whose_name_is('DEL_C_' + str(n) + '_' + str(m) + '_spatial') for n,m in pairs_distance]#[20:100]

paras=Paras({k:v for k,v in vars().items() if k[:1]!='_' and k!='version' and k!='Paras' and not k in [key for key in locals().keys()
       if isinstance(locals()[key], type(_sys)) and not key.startswith('__')]})

paras.to_update=to_update

paras.to_update['AC']=(func_AC,('nA', 'ACtot'))
paras.to_update['AC_dict']=(func_AC_dict,('nA', 'ACtot', 'par'))

paras.update_priority=update_priority

paras.analyse_dependance()

#if departure_times=='peaks':
#    paras['n_peaks']=n_peaks

