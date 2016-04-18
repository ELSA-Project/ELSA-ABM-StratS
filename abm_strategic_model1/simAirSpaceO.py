#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from paths import path_ksp
sys.path.insert(1, path_ksp)
#sys.path.insert(1,'../Distance')

import networkx as nx
from graph import DiGraph
from algorithms import ksp_yen
from random import sample, uniform, gauss, shuffle
import numpy as np
import matplotlib.delaunay as triang
from triangular_lattice import build_triangular
from math import sqrt
import pickle
import os

from general_tools import delay

version='2.6.3'

class FlightPlan:
    """
    Class FlightPlan. 
    =============
    Keeps in memory its path, time of departure, cost and id of AC.
    """
    def __init__(self,path,time,cost,ac_id):
        self.p=path
        self.t=time
        self.cost=cost
        self.ac_id=ac_id
        self.accepted=True
        self.bottleneck=-1

class Flight:
    """
    Class Flight. 
    =============
    Keeps in memory its id, source, destination, prefered time of departure and id of AC.
    Thanks to AirCompany, keeps also in memory its flight plans (self.FPs).
    """
    def __init__(self,Id,source,destination,pref_time,ac_id,par): # Id is relative to the AC
        self.id=Id
        self.source=source
        self.destination=destination
        self.pref_time=pref_time
#        if self.pref_time>15:
#            print 'poeut'
        #self.FPs=FPs
        self.ac_id=ac_id 
        self.par=par
        
    def make_flags(self):
        try:
            self.flag_first=[fp.accepted for fp in self.FPs].index(True)
        except:
            self.flag_first=len(self.FPs)
            
        self.overloadedFPs=[self.FPs[n].p for n in range(0,self.flag_first)]
        self.bottlenecks=[fp.bottleneck for fp in self.FPs if fp.bottleneck!=-1]
        
    def __repr__(self):
        return 'Flight number ' + str(self.id) + ' from AC number ' + str(self.ac_id) +\
            ' from ' + str(self.source) + ' to ' + str(self.destination)
        
class AirCompany:
    """
    Class AirCompany
    ================
    Keeps in memory the underliying network and several parameters, in particular the 
    coefficients for the utility function and the pairs of airports used.
    """
    def __init__(self, Id, Nfp, na, pairs, par):
        self.Nfp=Nfp
        self.par=par
        self.pairs=pairs
        #self.G=G
        self.na=na
        self.id=Id
        
    def fill_FPs(self,t0spV, tau, G):
        """
        Fills na flight with Nfp flight plans each, between airports given by pairs.
        """
        try:
            assigned=sample(self.pairs,self.na)
        except ValueError:
            print "self.pairs,self.na", self.pairs,self.na
            raise
            
        self.flights=[]
        #self.FPs=[]
        i=0
        #print t0spV
        for (ai,aj) in assigned:
            self.flights.append(Flight(i,ai,aj,t0spV[i],self.id,self.par))
            #print t0spV
            self.flights[-1].FPs=self.add_flightplans(ai,aj,t0spV[i],tau, G)
            i+=1

    def add_flightplans(self,ai,aj,t0sp,tau, G): 
        """
        Add flight plans to a given flight, based on Nfp and the best paths.
        Changed in 2.2: tau introduced.
        """
        shortestPaths=G.short[(ai,aj)]
        uworst=utility(self.par,G.weight_path(shortestPaths[0]),t0sp,G.weight_path(shortestPaths[-1]),t0sp)
      
        u=[[(p,t0sp + i*tau,utility(self.par,G.weight_path(shortestPaths[0]),t0sp,G.weight_path(p),t0sp + i*tau)) for p in shortestPaths] for i in range(self.Nfp)\
            if utility(self.par,G.weight_path(shortestPaths[0]),t0sp,G.weight_path(shortestPaths[0]),t0sp + i*tau)<=uworst]
        fp=[FlightPlan(a[0],a[1],a[2],self.id) for a in sorted([item for sublist in u for item in sublist], key=lambda a: a[2])[:self.Nfp]]

        if not G.weighted:
            # ------------- shuffling of the flight plans with equal utility function ------------ #
            uniq_util=np.unique([item.cost for item in fp])
            sfp=[]
            for i in uniq_util:
                v=[item for item in fp if item.cost==i]
                shuffle(v)
                sfp=sfp+v
            fp=sfp
        #self.FPs.append(fp)

        return fp
        
    def add_dummy_flightplans(self,ai,aj,t0sp): 
        """
        New in 2.5: Add dummy flight plans to a given flight. Used if there is no route between source and destination airports.
        """
        
        fp=[FlightPlan([],t0sp,10**6,self.id) for i in range(self.Nfp)]
        
        return fp
        
    def __repr__(self):
        return 'AC with para ' + str(self.par)
        

class Net(nx.Graph):
    """
    Class Net
    =========
    Derived from nx.Graph. Several methods added to build it, generate, etc...
    """
    def __init__(self):
        super(Net, self).__init__()
        
    def import_from(self,G):
        """
        Used to import the data of an already existing graph (networkx) in a Net obect.
        Weights and directions are conserved.
        """
        print 'Importing network...'
        self.add_nodes_from(G.nodes(data=True))
        self.add_weighted_edges_from([(e[0],e[1],G[e[0]][e[1]]['weight']) for e in G.edges()])
        e1=G.edges()[0]
        e2=G.edges()[1]
        self.weighted=not (self[e1[0]][e1[1]]['weight']==self[e2[0]][e2[1]]['weight']==1.)
        
        if self.weighted:
            print 'Network was found weighted'
        else:
            print 'Network was found NOT weighted'
            print 'Example:', self[e1[0]][e1[1]]['weight']
    
    def build(self,N,nairports,min_dis,Gtype='D', sigma=1.,mean_degree=6):
        """
        Build a graph of the given type. Build also the correspding graph used for ksp_yen.
        """
        print 'Building random network of type ', Gtype
        prob=mean_degree/float(N-1) # <k> = (N-1)p - 6 is the mean degree in Delaunay triangulation.
        
        self.weighted=sigma==0.

        if Gtype=='D' or Gtype=='E':
            for i in range(N):
                    self.add_node(i,coord=[uniform(-1.,1.),uniform(-1.,1.)])           
            x,y =  np.array([self.node[n]['coord'][0] for n in self.nodes()]),np.array([self.node[n]['coord'][1] for n in self.nodes()])   
            cens,edg,tri,neig = triang.delaunay(x,y)
            if Gtype=='D':
                #graphtypeflag='Delaunay'
                for p in tri:
                    self.add_edge(p[0],p[1], weight=max(gauss(1., sigma),0.00001)) # generates weigthed links, centered on 1
                    self.add_edge(p[1],p[2], weight=max(gauss(1., sigma),0.00001))
                    self.add_edge(p[2],p[0], weight=max(gauss(1., sigma),0.00001))
            elif Gtype=='E':  
                #graphtypeflag='Erdos Renyi'
                for n in self.nodes():
                    for m in self.nodes():
                        if n!=m:
                            if np.random.rand()<=prob:
                                self.add_edge(n,m)
        elif Gtype=='T':
            #graphtypeflag=='Triangular Lattice'
            xAxesNodes=np.sqrt(N/float(1.4))
            self=build_triangular(xAxesNodes)  
        self.generate_airports(nairports,min_dis) 
        self.build_H()
            
    def generate_weights(self,typ='gauss',par=[1.,0.01],values=[]):
        """
        Generates weights with different distributions
        """
        assert typ in ['gauss', 'manual', 'coords']
        self.typ_weights, self.par_weights=typ, par
        if typ=='gauss':
            mu=par[0]
            sigma=par[1]
            for e in self.edges():
                self[e[0]][e[1]]['weight']=max(gauss(mu, sigma),0.00001)
        elif typ=='coords':
            for e in self.edges():
                self[e[0]][e[1]]['weight']=sqrt((self.node[e[0]]['coord'][0] - self.node[e[1]]['coord'][0])**2 +(self.node[e[0]]['coord'][1] - self.node[e[1]]['coord'][1])**2)
            avg_weight=np.mean([self[e[0]][e[1]]['weight'] for e in self.edges()])
            for e in self.edges():
                self[e[0]][e[1]]['weight']=self[e[0]][e[1]]['weight']/avg_weight
            
    def generate_capacities(self, C=5, typ='constant', par=[1], func=lambda a:a, file_capacities=None):
        """
        Generates capacities with different distributions
        """
        assert typ in ['constant', 'gauss', 'uniform', 'manual', 'traffic']
        self.C, self.typ_capacities, self.par_capacities = C, typ, par
        if typ=='constant':
            for n in self.nodes():
                self.node[n]['capacity']=C
        elif typ=='gauss':
            for n in self.nodes():
                self.node[n]['capacity']=max(1,int(gauss(C,par[0])))
        elif typ=='uniform':
            for n in self.nodes():
                self.node[n]['capacity']=max(1,int(uniform(C-par[0]/2.,C+par[0]/2.)))
        elif typ=='manual':
            f=open(file_capacities,'r')
            properties=pickle.load(f)
            f.close()
            for n in self.nodes():
                self.node[n]['capacity']=properties[n]['capacity']
        elif typ=='traffic':
            self.extract_capacity_from_traffic()

    def extract_capacity_from_traffic(self):
        """
        New in 2.9.3: Extract the "capacity", the maximum number of flight per hour, based on the traffic.
        Changed in 2.9.4: added paras_real.
        Changed in 2.9.5: fligths are given externally.

        New in 2.6.3: imported and adapted from 2.9 fork. Defines the capacity with the peaks of traffic.
        """
        print 'Extracting the capacities from flights_selected...'

        #flights=get_flights(paras_real)

            
        #loads = {n:[0 for i in range(48)] for n in G.nodes()}
        loads = {n:[[0,0],[10**8,0]] for n in self.nodes()}
        for f in self.flights_selected:
            #print f['sec_path_t']
            path, times = f['sec_path'], [delay(t) for t in f['sec_path_t']]
            for i, n in enumerate(path):
                t1, t2 = times[i],times[i+1]
                ints = np.array([p[0] for p in loads[n]])
                caps = np.array([p[1] for p in loads[n]])
                i1=list(ints>=t1).index(True)
                i2=list(ints>=t2).index(True)
                if ints[i2]!=t2:
                    loads[n].insert(i2,[t2,caps[i2-1]])
                if ints[i1]!=t1:
                    loads[n].insert(i1,[t1,caps[i1-1]])
                    i2+=1
                for k in range(i1,i2):
                    loads[n][k][1]+=1

            # hours = {}
            # r = f['route_m1t']
            # for i in range(len(r)):
            #     if G.G_nav.idx_navs.has_key(r[i][0]):
            #         p1=G.G_nav.idx_navs[r[i][0]]
            #         if G.G_nav.has_node(p1):
            #             s1=G.G_nav.node[p1]['sec']
            #             hours[s1] = hours.get(s1,[]) + [int(float(delay(r[i][1], starting_date = date))/3600.)]

            # for n,v in hours.items():
            #     hours[n] = list(set(v))

            # for n,v in hours.items():
            #     for h in v:
            #         loads[n][h]+=1

        for n,v in loads.items():
            self.node[n]['capacity'] = max([c[1] for c in v])

    def generate_airports(self,nairports,min_dis):
        """
        Generate nairports airports. Build the accessible pairs of airports for this network
        with a  minimum distance min_dis.
        """
        self.airports=sample(self.nodes(),nairports)
        self.pairs=[(ai,aj) for ai in self.airports for aj in self.airports if len(nx.shortest_path(self, ai, aj))-2>=min_dis]
        
    def fix_airports(self,airports, min_dis, pairs=[]):
        """
        Fix airports given by user. The pairs can be given also by the user, 
        or generated automatically, with minimum distance min_dis
        Changed in 2.6.3: enforce the min_dis even when pairs are given.
        """
        self.airports = airports
        if pairs == []:
            self.pairs = [(ai,aj) for ai in self.airports for aj in self.airports if len(nx.shortest_path(self, ai, aj))-2>=min_dis]
        else:
            self.pairs = [(p1, p2) for p1, p2 in pairs if len(nx.shortest_path(self, p1, p2))-2>=min_dis]
        
    def finalize_network(self,Nfp,C,generate_capacities=True,generate_weights=True, typ_capacities='constant', typ_weights='gauss', file_capacities=None):
        """
        Initialize dynamic characteristic for the first time, and compute shortest_path.
        """
        if generate_capacities:
            self.generate_capacities(typ=typ_capacities,C=C, file_capacities=file_capacities)
        if generate_weights:
            self.generate_weights(typ=typ_weights)
        for a in self.airports:
            self.node[a]['capacity']=100000
        self.initialize_load(2*(nx.diameter(self)+Nfp))
        self.compute_shortest_paths(Nfp)
        print 'Number of nodes:', (len(self.nodes()))
        
    def build_H(self): 
        """
        Build the DiGraph obect used in the ksp_yen algorithm.
        """
        self.H=DiGraph()
        self.H._data={}
        for n in self.nodes():
            self.H.add_node(n)
        for e in self.edges():
            self.H.add_edge(e[0],e[1], cost=self[e[0]][e[1]]['weight'])
            self.H.add_edge(e[1],e[0], cost=self[e[1]][e[0]]['weight'])
         
    def initialize_load(self):
        """
        Initialize loads, with length given by t_max.
        Changed in 2.2: keeps in memory only the intervals.
        Changed in 2.5: t_max is set to 10**6.
        """
        for n in self.nodes():
            self.node[n]['load']=[[0,0],[10**9,0]] # the last one should ALWAYS be (t_max,0.)
    
    def clean(self):
        """
        Initialize loads, keeping the same length.
        Changed in 2.2: do only initialization now.
        """
       # for n in self.nodes():
       #     self.node[n]['load']=[0 for i in range(len(self.node[n]['load']))]
        #n=self.nodes()[0]
        #t_max=self.node[n]['load'][-1][0]
        self.initialize_load()
            
    def weight_path(self,p): 
        """
        Return the weight of the given path.
        """
        return sum([self[p[i]][p[i+1]]['weight'] for i in range(len(p)-1)])
        
    def compute_shortest_paths(self, Nfp, repetitions = True, n_tries_duplicates=100):
        """
        Pre-Build Nfp weighted shortest paths between each pair of airports.
        Changed in 2.9.2: added repetitions.
        Modified in 2.6.3: shortest paths were not sorted in output
        """
        #self.short={(a,b):self.kshortestPath(a,b,Nfp) for a in self.airports for b in self.airports if a!=b}
        print 'Computing shortest paths...'
        Nfp_init = Nfp
        #paths_additional=0
        
        deleted_pairs = []
        pairs = self.pairs
        enough_paths = {(a,b):False for (a,b) in pairs if a!=b}
        #self.short = {(a,b):self.kshortestPath(a, b, Nfp) for (a,b) in self.short.keys()}

        # if repetitions:
        #     while False in enough_paths.values():
        #         self.short = {(a,b):self.kshortestPath(a, b, Nfp) for (a,b) in pairs}

        #         for (a,b) in self.short.keys():
        #             if a!=b:
        #                 self.short[(a,b)] = [list(vvv) for vvv in set([tuple(vv) for vv in self.short[(a,b)]])][:Nfp_init]
        #                 self.short[(a,b)] = sorted(self.short[(a,b)], key = lambda p: self.weight_path(p))
        #                 if not (len(self.short[(a,b)]) < Nfp_init):
        #                     #print "For airports", a, b, "I don't find enough paths (only",  len(self.short[(a,b)]), ")"
        #                     enough_paths[(a,b)] = True
        #         Nfp += 1
        #         if False in enough_paths.values():
        #             print 'Not enough paths, doing another round (', Nfp - Nfp_init, 'additional paths).'
            
        #         #self.short={(a,b):self.kshortestPath(a, b, Nfp) for (a,b) in self.short.keys()}                
        # else:
        #     while False in enough_paths.values():
        #         self.short = {}
        #         #pairs = self.pairs
        #         for (a,b) in pairs:
        #             if a!=b and enough_paths[(a,b)]:
        #                 #print 'a, b', a, b
        #                 paths = self.kshortestPath(a, b, Nfp) # self.short[(a,b)] #Initial set of paths
        #                 previous_duplicates=1
        #                 duplicates=[]
        #                 n_counter=0
        #                 while len(duplicates)!=previous_duplicates and n_counter<n_tries_duplicates:
        #                     previous_duplicates=len(duplicates)

        #                     # Shortest paths having duplicates sectors.
        #                     # print "paths:"
        #                     # for sp in paths:
        #                     #     print sp
        #                     # print "weight 0 769:", self[0][769]['weight']
        #                     # print "weight 0 769:", self.H[0][769]
        #                     duplicates = [sp for sp in paths if len(set(sp))<len(sp)]
        #                     print 'Number of paths having duplicated sectors:', len(duplicates), "over", len(paths)
                            
        #                     # If the number of duplicates has changed, compute some more paths.
        #                     if len(duplicates)!=previous_duplicates:
        #                          paths = self.kshortestPath(a, b, Nfp + len(duplicates))
        #                     n_counter+=1

        #                 duplicates = [sp for sp in paths if len(set(sp))<len(sp)]
        #                 #print len(duplicates), len(paths)

        #                 #paths_init = paths[:]
        #                 for path in duplicates:
        #                     paths.remove(path)

        #                 self.short[(a,b)]=paths[:]

        #                 #print self.short[(a,b)]

        #                 try:
        #                     assert len(self.short[(a,b)])==Nfp
        #                 except AssertionError:
        #                     #print 'a:', a, 'b:', b, 'len(self.short[(a,b)]):', len(self.short[(a,b)])
        #                     print "kspyen can't find enough paths (only " + str(len(self.short[(a,b)])) + ')', "for the pair", a, b, "; I delete this pair."
        #                     print 'Number of duplicates:', len(duplicates)
        #                     #print 'Number of paths with duplicates:', len(paths_init)
        #                     deleted_pairs.append((a,b))
        #                     del self.short[(a,b)]
        #                     #self.pairs = list(self.pairs)
        #                 except:
        #                     raise

        #         #enough_paths=True
        #         # Detects if two paths are exactly the same.
        #         for (a,b) in self.short.keys():
        #             if a!=b:
        #                 self.short[(a,b)] = [list(vvv) for vvv in set([tuple(vv) for vv in self.short[(a,b)]])][:Nfp_init]
        #                 self.short[(a,b)] = sorted(self.short[(a,b)], key = lambda p: self.weight_path(p))
        #                 if not (len(self.short[(a,b)]) < Nfp_init):
        #                     #print "For airports", a, b, "I don't find enough paths (only",  len(self.short[(a,b)]), ")"
        #                     enough_paths[(a,b)] = True
        #         #paths_additional += 1
        #         Nfp += 1
        #         if False in enough_paths.values():
        #             print 'Not enough paths, doing another round (', Nfp - Nfp_init, 'additional paths).'

        while False in enough_paths.values():
            if repetitions:
                #self.short={(a,b):self.kshortestPath(a, b, Nfp) for (a,b) in self.short.keys()}
                self.short = {(a,b):self.kshortestPath(a, b, Nfp) for (a,b) in self.pairs}
            else:
                self.short = {}
                pairs = self.pairs
                for (a,b) in pairs:
                    if a!=b and not enough_paths[(a,b)]:
                        #print 'a, b', a, b
                        paths = self.kshortestPath(a, b, Nfp) # self.short[(a,b)] #Initial set of paths
                        previous_duplicates=1
                        duplicates=[]
                        n_counter=0
                        while len(duplicates)!=previous_duplicates and n_counter<n_tries_duplicates:
                            previous_duplicates=len(duplicates)

                            # Shortest paths having duplicates sectors.
                            # print "paths:"
                            # for sp in paths:
                            #     print sp
                            # print "weight 0 769:", self[0][769]['weight']
                            # print "weight 0 769:", self.H[0][769]
                            duplicates = [sp for sp in paths if len(set(sp))<len(sp)]
                            print 'Number of paths having duplicated sectors:', len(duplicates), "over", len(paths)
                            
                            # If the number of duplicates has changed, compute some more paths.
                            if len(duplicates)!=previous_duplicates:
                                 paths = self.kshortestPath(a, b, Nfp + len(duplicates))
                            n_counter+=1

                        duplicates = [sp for sp in paths if len(set(sp))<len(sp)]
                        #print len(duplicates), len(paths)

                        #paths_init = paths[:]
                        for path in duplicates:
                            paths.remove(path)

                        self.short[(a,b)]=paths[:]

                        #print self.short[(a,b)]

                        try:
                            assert len(self.short[(a,b)])==Nfp
                        except AssertionError:
                            #print 'a:', a, 'b:', b, 'len(self.short[(a,b)]):', len(self.short[(a,b)])
                            print "kspyen can't find enough paths (only " + str(len(self.short[(a,b)])) + ')', "for the pair", a, b, "; I delete this pair."
                            print 'Number of duplicates:', len(duplicates)
                            #print 'Number of paths with duplicates:', len(paths_init)
                            deleted_pairs.append((a,b))
                            del self.short[(a,b)]
                            #self.pairs = list(self.pairs)
                        except:
                            raise

            #enough_paths=True
            for (a,b) in self.short.keys():
                if a!=b:
                    self.short[(a,b)] = [list(vvv) for vvv in set([tuple(vv) for vv in self.short[(a,b)]])][:Nfp_init]
                    self.short[(a,b)] = sorted(self.short[(a,b)], key = lambda p: self.weight_path(p))
                    if not (len(self.short[(a,b)]) < Nfp_init):
                        #print "For airports", a, b, "I don't find enough paths (only",  len(self.short[(a,b)]), ")"
                        enough_paths[(a,b)] = True
            #paths_additional += 1
            Nfp += 1
            if False in enough_paths.values():
                print 'Not enough paths, doing another round (', Nfp - Nfp_init, 'additional paths).'

        self.pairs = self.short.keys()

    def kshortestPath(self,i,j,k): 
        """
        Return the k weighted shortest paths on the network. Uses the DiGraph.
        Changed in 2.6.3: the index of the loop was i too!
        """
        spath = [a['path'] for a in  ksp_yen(self.H, i, j, k)]
        spath=sorted(spath, key=lambda a:self.weight_path(a))
        spath_new, ii = [], 0
        while ii<len(spath): # Not sure what this loop is doing...
            w_old=self.weight_path(spath[ii])
            a=[spath[ii][:]]
            ii+=1
            while ii<len(spath) and abs(self.weight_path(spath[ii]) - w_old)<10**(-8.):
                a.append(spath[ii][:])
                ii+=1
            #shuffle(a)
            spath_new+=a[:]

        return spath_new
        
    def overlapSP(self):
        """
        Compute the overkap between the shortest of the network.
        """
        pairs=[(i,j) for i in self.nodes() for j in self.nodes() if i!=j] 
        SP=[]
        for p in pairs:
                SP.append([sp for sp in nx.all_shortest_paths(self,source=p[0],target=p[1])])
        LenOver=[]
        for path in SP:
            if len(path)!=1:
                LenOver.append([(len(a),len(np.intersect1d(a,b))-2) for a in path for b in path if a!=b])
                #For each pair of vertices connected by more than 1 shortest path, a list of tuples is stored in OverLen;
                #the first element of each tuple is the overlap between the paths 
                #(computed as number of common nodes different from the starting and the ending one),
                #the second one is the length of the path.
        FlatLO=np.array(sorted([item for sublist in LenOver for item in sublist]))    
        #this instruction reduce OverLen from a "list of lists of tuples"
        #to a "list of tuples", i.e. from [[(),(),...],[(),(),...],...]
        #to [(),(),...], and sort the list by the first element of the
        #tuples (path length).
        Lengths=[]
        for tu in FlatLO:
            Lengths.append(tu[0])
        Lmax=max(Lengths)
        Lmin=min(Lengths)
        AverO=[]
        StdO=[]
        L=[]
        for i in range(Lmin,Lmax+1):
            r_indices=np.nonzero(FlatLO[:,0]==i)
            selected=FlatLO[r_indices,1]
            AverO.append(np.mean(selected))
            StdO.append(np.std(selected))
            L.append(i)
        return (L,AverO,StdO),Lmin,Lmax
        
    def sector_loading(self,flight,storymode=False):
        """
        Used by netman, tries to fill the flight in the network, with one of 
        its flight plans. The rejection of the flights is kept in memory, as
        well as the first sector overloaded.
        Changed in 2.2: using intervals.
        """
        i=0
        found=False
        while i<len(flight.FPs) and not found:
            fp=flight.FPs[i]
            self.compute_flight_times(fp)
            path, times=fp.p, fp.times

            if storymode:
                print "     FP no", i, "tries to be allocated with trajectory (sectors):"
                print fp.p
                print "and crossing times:"
                print fp.times

            j=0
            try:
                while j<len(path) and not self.overload_capacity(path[j],(times[j],times[j+1])):#and self.node[path[j]]['load'][j+time] + 1 <= self.node[path[j]]['capacity']:
                    j+=1 
            except:
                print path, times
                raise
    
            overloaded=j<len(path)
            fp.accepted=not overloaded

            if storymode:
                print "     FP has been accepted:", fp.accepted
                if not fp.accepted:
                    if overloaded: 
                        print "     because sector", path[j], "was full."
                    # if source_overload:
                    #     print "     because source airport was full."
                    #     print G.node[path[0]]['load_airport']
                    #     print G.node[path[0]]['capacity_airport']
                    # if desetination_overload:
                    #     print "     because destination airport was full."

            
            if not overloaded:
                self.allocate(fp)
                flight.fp_selected=fp
                found=True
            else:
                fp.bottleneck=path[j]
                
            i+=1 
        
        if not found:
            flight.fp_selected=None
  
    def compute_flight_times(self,fp):
        """
        Changed in 2.8: times computed with the navpoint network.
        """
        ints=[0.]*(len(fp.p)+1)
        ints[0]=fp.t
        road=fp.t
        for i in range(1,len(fp.p)):
            w=self[fp.p[i-1]][fp.p[i]]['weight']
            ints[i]=road + w/2.
            road+=w
        ints[len(fp.p)]=road        
        fp.times=ints
        
    def overload_capacity(self,n,(t1,t2)):
        """
        Check if the sector is overloading without actually building the new set
        of intervals.
        """
        ints=np.array([p[0] for p in self.node[n]['load']])
        
        caps=np.array([p[1] for p in self.node[n]['load']])
        i1=max(0,list(ints>=t1).index(True)-1)
        #try:
        i2=list(ints>=t2).index(True)
       # except:
       #     print t2
       #     print ints
       #     raise
        
        pouet=np.array([caps[i]+1 for i in range(i1,i2)])
        
        return len(pouet[pouet>self.node[n]['capacity']])>0
        
    def allocate(self,fp):
        """
        Fill the network with the given flight plan.
        """
        path,times=fp.p,fp.times
        for i,n in enumerate(path):
            t1,t2=times[i],times[i+1]
            ints=np.array([p[0] for p in self.node[n]['load']])
            caps=np.array([p[1] for p in self.node[n]['load']])
            i1=list(ints>=t1).index(True)
            i2=list(ints>=t2).index(True)
            if ints[i2]!=t2:
                self.node[n]['load'].insert(i2,[t2,caps[i2-1]])
            if ints[i1]!=t1:
                self.node[n]['load'].insert(i1,[t1,caps[i1-1]])
                i2+=1
            for k in range(i1,i2):
                self.node[n]['load'][k][1]+=1
                
    def deallocate(self,fp):
        """
        New in 2.5: used to deallocate a flight plan not legit anymore (because one sector has been shutdown).
        """
        
        path,times=fp.p,fp.times
        #print 'path, times', path, times
        for i,n in enumerate(path):
            t1,t2=times[i],times[i+1]
            ints=np.array([p[0] for p in self.node[n]['load']])
            #caps=np.array([p[1] for p in self.node[n]['load']])
            i1=list(ints==t1).index(True)
            i2=list(ints==t2).index(True)
            for k in range(i1,i2):
                self.node[n]['load'][k][1]-=1
                
           # print 't1, t2, i1, i2', t1, t2, i1, i2
           # print [t2,caps[i2]]
           # print [t1,caps[i1]]
           # print 
                
            try:
                if self.node[n]['load'][i2-1][1]==self.node[n]['load'][i2][1]:
                    self.node[n]['load'].remove([t2,self.node[n]['load'][i2][1]])
                if self.node[n]['load'][i1-1][1]==self.node[n]['load'][i1][1]:
                    self.node[n]['load'].remove([t1,self.node[n]['load'][i1][1]])
            except:
               # print fp
               # print fp.p
               # print [t2,caps[i2]], [t1,caps[i1]]
                raise
            
            #print 'load after', self.node[n]['load']
 
    def basic_statistics(self,rep='.'):
        os.system('mkdir -p ' + rep)
        f=open(rep + '/basic_stats_net.txt','w')
        print >>f, 'Mean/std degree:', np.mean([self.degree(n) for n in self.nodes()]), np.std([self.degree(n) for n in self.nodes()])
        print >>f, 'Mean/std weight:', np.mean([self[e[0]][e[1]]['weight'] for e in self.edges()]), np.std([self[e[0]][e[1]]['weight'] for e in self.edges()])
        print >>f, 'Mean/std capacity:', np.mean([self.node[n]['capacity'] for n in self.nodes() if not n in self.airports]),\
            np.std([self.node[n]['capacity'] for n in self.nodes() if not n in self.airports])
        #print >>f, 'Mean/std load:', np.mean([np.mean(self.node[n]['load']) for n in self.nodes()]), np.std([np.mean(self.node[n]['load']) for n in self.nodes()])
        f.close()
        
    def shut_sector(self,n):
        for v in nx.neighbors(self,n):
            self[n][v]['weight']=10**6
        
    def reduce_flights(self):
        print 'Reducing flights...'
        fl = self.flights_selected[:]
        for f in fl:
            idx_s = f['sec_path'][0] # self.G.idx_sectors[source]
            idx_d = f['sec_path'][-1] # self.G.idx_sectors[destination]
            if not (idx_s in self.airports and idx_d in self.airports and (idx_s, idx_d) in self.pairs):    
                print "I do" + (not idx_s in self.airports)*' not', " find", idx_s, ", I do" + (not idx_d in self.airports)*' not', " find", idx_d,\
                 'and the couple is' + (not (idx_s, idx_d) in self.pairs)*' not', ' in pairs.'
                print 'I delete this flight from the flights_selected list.'
                self.flights_selected.remove(f)
        
    def capacities(self):
        return {n:self.node[n]['capacity'] for n in self.nodes()}

def utility(par,Lsp,t0sp,L,t0):

    (alpha,betha1,betha2)=par
    
    #"""
#the imputs of this function are all supposed to be NumPy arrays
#    
#Call: U=UTILITY(ALPHA,BETHA1,BETHA2,LSP,T0SP,L,T0);
#the function utility.m computes the utility function value, comparing two
#paths on a graph;
#
#INPUTS
#
#alpha, betha1, betha2 -> empirically assigned weight parameters,
#ranging from 0 to 1;
#
#Lsp -> length of the shortest path;
#
#t0sp -> departure time of the motion along the shortest path;
#
#L -> length of the path which one wants to compare to the shortest
#one;
#
#t0 -> depature time of the motion on the path used in the
#coparison with the shortes one;
#
#OUTPUT
#
#U -> is the value of the utility function for the given choise of paths;
#
#"""
    
    return np.dot(alpha,L)+np.dot(betha1,np.absolute(t0+L-(t0sp+Lsp)))+np.dot(betha2,np.absolute(t0-t0sp))
    
    

        
    
    

