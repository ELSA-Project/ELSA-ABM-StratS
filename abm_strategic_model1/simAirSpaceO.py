#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(1, '..')
sys.path.insert(1, '../libs/YenKSP')

#import sys
#from paths import path_ksp
#sys.path.insert(1, path_ksp)
#sys.path.insert(1,'../Distance')
import os
from os.path import join

import networkx as nx
from random import sample, uniform, gauss, shuffle
import numpy as np
from numpy import sqrt, exp, log
from numpy.random import lognormal
import matplotlib.delaunay as triang
import pickle

from libs.general_tools import delay, build_triangular
from libs.YenKSP.graph import DiGraph
from libs.YenKSP.algorithms import ksp_yen

version='2.6.4'

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

    Notes
    ----
    Changes in 2.6.4: sorted methods and imported methods from Model 2.

    """
    
    def __init__(self):
        super(Net, self).__init__()
    
    def add_airports(self, airports, min_dis, pairs=[], C_airport=10, singletons=False):
        """
        Add airports given by user. The pairs can be given also by the user, 
        or generated automatically, with minimum distance min_dis.

        Parameters
        ----------
        min_dis : int 
            minimum distance -- in nodes, EXCLUDING THE AIRPORTS -- betwen a pair of airports.
        pairs : list of 2-tuples, optional
            Pairs of nodes for connections. If [], all possible pairs between airports are computed 
            (given min_dis)
        C_airport : int, optional
            Capacity of the sectors which are airports. They are used only by flights which are
            departing or lending in this area. It is different from the standard capacity key
            which is used for flights crossing the area, which is set to 10000.
        singletons : boolean, optional
            If True, pairs in which the source is identical to the destination are authorized 
            (but min_dis has to be smaller or equal to 2.)
        
        Notes
        -----
        New in 2.6.4: taken from Model 2
        (From Model 2)
        Changed in 2.9.8: changed name to add_airports. Now the airports are added
        to the existing airports instead of overwriting the list.

        """

        if not hasattr(self, "airports"):
            self.airports = airports
        else:
            self.airports = np.array(list(set(list(self.airports) + list(airports))))
            
        if not hasattr(self, "short"):
            self.short = {}

        if pairs==[]:
            for ai in self.airports:
                for aj in self.airports:
                    if len(nx.shortest_path(self, ai, aj))-2>=min_dis and ((not singletons and ai!=aj) or singletons):
                        if not self.short.has_key((ai,aj)):
                            self.short[(ai, aj)] = []
        else:
            for (ai,aj) in pairs:
                 if ((not singletons and ai!=aj) or singletons):
                    if not self.short.has_key((ai,aj)):
                        self.short[(ai, aj)] = []

        for a in airports:
            #self.node[a]['capacity']=100000                # TODO: check this.
            self.node[a]['capacity_airport'] = C_airport

    def allocate(self,fp):
        """
        Fill the network with the given flight plan.

        TO REMOVE
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
        Changed in 2.6.4: Taken from Model 2 (unchanged)

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
        Changed in 2.6.4: taken and adapted from Model 2.

        """

        print 'Building random network of type', Gtype
        
        if Gtype=='D' or Gtype=='E':
            self.build_nodes(N, prelist=prelist, put_nodes_at_corners=put_nodes_at_corners)
            self.build_net(Gtype=Gtype, mean_degree=mean_degree)  
        elif Gtype=='T':
            xAxesNodes = int(np.sqrt(N/float(1.4)))
            self.import_from(build_triangular(xAxesNodes))
    
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
        New in 2.6.4: taken from Model 2 (unchanged)
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
        New in 2.6.4: taken from Model 2 (unchanged)

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

    def compute_flight_times(self,fp):
        """
        
        Changed in 2.8: times computed with the navpoint network.
        TO REMOVE

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
        Changed in 2.6.4: taken and adapted from Model 2
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
                #print "Shortest path for", (a, b)
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
                            paths = [list(vvv) for vvv in set([tuple(vv) for vv in paths])][:Nfp_init]
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
                    self.short[(a,b)] = [[a] for i in range(Nfp)]

    def connections(self):
        """
        Notes
        -----
        New in 2.6.2: taken from Model 2 (unchanged)

        (From Model 2)
        New in 2.9.8: returns the possible connections between airports.
        
        """
        
        return self.short.keys()

    def deallocate(self,fp):
        """
        New in 2.5: used to deallocate a flight plan not legit anymore (because one sector has been shutdown).
        
        TO REMOVE
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

    def fix_airports(self, *args, **kwargs):
        """
        Used to reset the airports and then add the new airports.

        Notes
        -----
        Changed in 2.6.4: taken from Model 2 (unchanged).

        """
        
        if hasattr(self, "airports"):
            self.airports = []
            self.short = {}

        self.add_airports(*args, **kwargs)

    def generate_airports(self, nairports, min_dis, C_airport=10):
        """
        Generate nairports airports. Build the accessible pairs of airports for this network
        with a  minimum distance min_dis.

        Notes
        -----
        Changed in 2.6.4: updated with stuff from Model 2
        
        """

        self.airports = sample(self.nodes(),nairports)
        self.short = {(ai,aj):[] for ai in self.airports for aj in self.airports if len(nx.shortest_path(self, ai, aj))-2>=min_dis}
        
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
        Changed in 2.6.4: taken from Model 2 (unchanged).
        Changed in 2.6.4: defulat optinal for typ is now "coords".

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
        Changed in 2.6.4: taken from Model 2 (unchanged).

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
        New in 2.6.4: taken from Model 2(unchanged)
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
        Changed in 2.6.4: taken from Model 2 (unchanged).
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

    def initialize_load(self):
        """
        Initialize loads, with length given by t_max.
        Changed in 2.2: keeps in memory only the intervals.
        Changed in 2.5: t_max is set to 10**6.

        TO REMOVE.
        """
        for n in self.nodes():
            self.node[n]['load']=[[0,0],[10**9,0]] # the last one should ALWAYS be (t_max,0.)

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
        Changed in 2.6.3: the index of the loop was i too!
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
    
    def overload_capacity(self,n,(t1,t2)):
        """
        Check if the sector is overloading without actually building the new set
        of intervals.

        TO REMOVE.
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

    def shut_sector(self,n):
        for v in nx.neighbors(self,n):
            self[n][v]['weight']=10**6

    def weight_path(self,p): 
        """
        Return the weight of the given path.
        """
        return sum([self[p[i]][p[i+1]]['weight'] for i in range(len(p)-1)])
        

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
    
    

        
    
    

