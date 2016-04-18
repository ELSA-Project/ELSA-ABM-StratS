#!/usr/bin/env python

import sys
sys.path.insert(1,'/home/earendil/Documents/ELSA/Modules')

from simAirSpaceO import AirCompany
import networkx as nx
import ABMvars
#from random import getstate, setstate, 
from random import shuffle, uniform,  sample, seed
import pickle
from string import split
import matplotlib.pyplot as plt
import os
import numpy as np
import copy
from utilities import draw_network_map
from math import ceil
from general_tools import delay

#from tools_airports import map_of_net

import warnings

version='2.6.3'
main_version=split(version,'.')[0] + '.' + split(version,'.')[1]

if 0:
    print 'Caution! Seed!'
    seed(2)

def build_path(paras, vers=main_version, in_title=['Nfp', 'tau', 'par', 'ACtot', 'nA', 'departure_times',], only_name = False):
    """
    Used to build a path from a set of paras. 
    Changed 2.2: is only for single simulations.
    Changed 2.4: takes different departure times patterns. Takes names.
    Chnaged in 2.5: added N_shocks and improved the rest.
    """
    
    name = 'Sim_v' + vers + '_' + paras['G'].name
    if not only_name:
        name = '../results/' + name

    in_title=list(np.unique(in_title))

    if paras['departure_times']!='zeros':
        try:
            in_title.remove('ACtot')
        except ValueError:
            pass
        in_title.insert(1,'density')
        in_title.insert(2,'day')
        if paras['departure_times']=='square_waves':
            in_title.insert(1,'Delta_t')
            
    #print in_title
    
    if paras['N_shocks']==0:
        try:
            in_title.remove('N_shocks')
        except ValueError:
            pass      
    else:
        in_title.insert(-1,'N_shocks')
    
    in_title=np.unique(in_title)
    
    for p in in_title:
        if p=='par':
            if len(paras[p])==1 or paras['nA']==1.:
                coin=str(float(paras[p][0][0])) + '_' + str(float(paras[p][0][1])) + '_' + str(float(paras[p][0][2]))
            elif len(paras[p])==2:
                coin=str(float(paras[p][0][0])) + '_' + str(float(paras[p][0][1])) + '_' + str(float(paras[p][0][2])) + '__' +str(float(paras[p][1][0])) + '_' + str(float(paras[p][1][1])) + '_' + str(float(paras[p][1][2])) 
            else:
                coin='_several'
        else:
            coin=str(paras[p])
        name+='_' + p + coin
       
    return name

def check_object(G):
    """
    Use to check if you have an old object.
    """
    
    return hasattr(G, 'comments')
        
#def _read_paras(paras):
#    return  paras['N'],paras['nairports'],paras['n_AC'],paras['Nfp'],paras['na'],paras['C'],paras['tview'], \
#        paras['min_dis'],paras['type_of_net'],paras['tau'], paras['departure_times']

class Simulation:
    def __init__(self,paras,G=None,verbose=False, make_dir=False):
        """
        Initialize the simulation, build the network if none is given in argument, set the verbosity
        """
        self.paras=paras
        

        for k in ['AC', 'Nfp', 'na', 'tau', 'departure_times', 'ACtot', 'N_shocks','Np', \
        'ACsperwave','Delta_t', 'flows', 'nA']:
            if k in paras.keys():
                setattr(self, k, paras[k])
    
        #self.N,self.nairports,self.n_AC,self.Nfp,self.na,self.C,self.tview,self.min_dis,self.net,self.tau, self.departure_times=_read_paras(paras)
        self.make_times(paras['day'])
        #print self.Np, self.ACsperwave
        self.pars=paras['par']
       # if not G:
       #     G=Net()
       #     G.build(self.N,self.nairports,self.C,self.min_dis,Gtype=self.type_of_net)
       #     G.finalize_network(self.Nfp,self.C, generate_capacities=paras['generate_capacities'],generate_weights=paras['generate_weights'], typ_capacities=paras['typ_capacities'],\
       #     typ_weights=paras['typ_weights'], file_capacities=paras['capacities'])
       # else:
       #     G=ensure_object(G,self.nairports,self.min_dis,self.Nfp,self.paras,airports=paras['airports'],pairs=paras['pairs'])
            
       # if save_net_init:
       #     f=open(paras['network_name'])

        assert check_object(G)
        assert G.Nfp==paras['Nfp']
        
        self.G=G.copy()
        self.verb=verbose
        self.rep=build_path(paras)
        if make_dir:
            os.system('mkdir -p ' + self.rep)
        
    def make_simu(self,clean=False, storymode=False):
        """
        Do the simulation, clean afterwards the network (useful for iterations)
        """
        if self.verb:
            print 'Doing simulation...'

        self.G.initialize_load()

        if self.flows == {}:
            self.build_ACs(clean = clean)
        else:
            self.build_ACs_from_flows()

        self.netman(storymode=storymode)
        self.mark_best_of_queue()
        self.M0_to_M1()
        
    def build_ACs(self, clean=True):
        """
        Build all the Air Companies based on the number of Air Companies given in paras.
        If n_AC is an integer and if several sets of parameters are given for the utility 
        function, make the same number of AC sor each set. If n_AC is an array of integers 
        of the same size than the number of set of parameters, populates the different
        types with this array.
        Exs:
            self.n_AC=30, self.pars=[[1,0,0]] 
            gives 30 ACs with parameters [1,0,0].
            
            self.n_AC=30, self.pars=[[1,0,0], [0,0,1]] 
            gives 15 ACs with parameters [1,0,0] and 15 ACs with parameters [0,0,1].
            
            self.n_AC=[10,20], self.pars=[[1,0,0], [0,0,1]] 
            gives 10 ACs with parameters [1,0,0] and 20 ACs with parameters [0,0,1].
            
        Builds all flight plans for all ACs.
        """

        if type(self.AC)==type(1):
            self.AC=[self.AC/len(self.pars) for i in range(len(self.pars))]
        
        try:
            assert len(self.AC)==len(self.pars)
        except:
            print 'n_AC should have the same length than the parameters, or be an integer'
            raise
    
        self.ACs={}
        k=0
        #print len(self.t0sp), sum(self.AC)
        try:
            assert len(self.t0sp)==sum(self.AC)
        except:
            print len(self.t0sp), sum(self.AC)
            raise

        shuffle(self.t0sp)
        for i,par in enumerate(self.pars):
            for j in range(self.AC[i]):
                self.ACs[k]=AirCompany(k,self.Nfp, self.na, self.G.pairs,par)
                self.ACs[k].fill_FPs(self.t0sp[k],self.tau, self.G)
                k+=1
        if clean:   
            self.G.clean()

    def build_ACs_from_flows(self):
        """
        New in 2.9.2: the list of ACs is built from the flows (given by times). 
        (Only the number of flights can be matched, or also the times, which are taken as desired times.)
        New in 2.6.3: imported -- and adapted -- from 2.9 fork. 
        """
        self.ACs={}
        k=0
        #self.flights_taken_from_flow = []
        for ((source, destination), times) in self.flows.items():
            idx_s = source # self.G.idx_sectors[source]
            idx_d = destination # self.G.idx_sectors[destination]
            if idx_s in self.G.airports and idx_d in self.G.airports and self.G.short.has_key((idx_s, idx_d)):    
                n_flights_tot = len(times)
                n_flights_A = int(self.nA*n_flights_tot)
                n_flights_B = n_flights_tot - int(self.nA*n_flights_tot)
                AC= [n_flights_A, n_flights_B]
                l=0
                for i, par in enumerate(self.pars):
                    for j in range(AC[i]):
                        time = times[l]
                        #print times
                        #self.flights_taken_from_flow.append(k_f)
                        self.ACs[k] = AirCompany(k, self.Nfp, self.na, self.G.short.keys(), par)
                        time = int(delay(time, starting_date = [time[0], time[1], time[2], 0., 0., 0.])/(20.*60.))
                        self.ACs[k].fill_FPs([time], self.tau, self.G)
                        k+=1
                        l+=1
            else:
                print "I do " + (not idx_s in self.G.airports)*'not', "find", idx_s, ", I do " + (not idx_d in self.G.airports)*'not', "find", idx_d,\
                 'and the couple is ' + (not self.G.short.has_key((idx_s, idx_d)))*'not', 'in pairs.'
                print 'I skip this flight.'
        # if clean:   
        #     self.Netman.initialize_load(self.G)

    def netman(self, storymode=False):
        """
        Defines a (random) queue of flights to be filled. Loads the sectors with the 
        given flight.
        """
        self.queue=[]
        for ac in self.ACs.values():
            for f in ac.flights:
                self.queue.append(f)
        shuffle(self.queue)
        for i,f in enumerate(self.queue):
            if storymode:
                print "Flight with position", i, "from", f.source, "to", f.destination, "of company", f.ac_id
                print "with parameters", f.par
                print "tries to be allocated."
            f.pos_queue=i
            self.G.sector_loading(f, storymode=storymode)
            if storymode:
                print "flight accepted:", f.fp_selected!=None
                if f.fp_selected==None:
                    print 'because '
                print
                print 
            
    def compute_flags(self):
        """
        Computes flags, bottlenecks and overloadedFPs for each AC and in total.
        The queue is used to sort the flights, even within a given AC.
        """
        for ac in self.ACs.values():
            ac.flag_first,ac.bottlenecks, ac.overloadedFPs = [], [],[]
        self.bottlenecks, self.flag_first, self.overloadedFPs= [], [], []
        for f in self.queue:
            f.make_flags()
        
            self.ACs[f.ac_id].flag_first.append(f.flag_first)
            self.ACs[f.ac_id].bottlenecks.append(f.bottlenecks)
            self.ACs[f.ac_id].overloadedFPs.append(f.overloadedFPs)
            
            self.flag_first.append(f.flag_first)
            self.overloadedFPs.append(f.overloadedFPs)
            self.bottlenecks.append(f.bottlenecks)
        
        self.results={p:{ac.id:{'flags':ac.flag_first,'overloadedFPs':ac.overloadedFPs, 'bottlenecks':ac.bottlenecks}\
                    for ac in self.ACs.values() if ac.par==p} for p in self.pars}
        self.results['all']=self.flag_first, self.overloadedFPs, self.bottlenecks
        
        if self.verb:
            #print 'flags', self.flag_first
            #print 'overloadedFPs', self.overloadedFPs
            #print 'bottlenecks', self.bottlenecks
            print
            print
            
    def save(self, rep='', split=False, only_flags=False):
        """
        Save the network in a pickle file, based on the paras.
        Can be saved in a single file, or different flights to speed up the post-treatment.
        """
        if rep=='':
            rep=build_path(self.paras)
        if only_flags:
            f=open(rep + '_flags.pic','w')
            pickle.dump((self.flag_first, self.bottlenecks, self.overloadedFPs),f)
            f.close()
        else:
            if not split:
                print 'Saving whole object on ', rep
                f=open(rep + '.pic','w')
                pickle.dump(self,f)
                f.close()
            else:
                print 'Saving split object in ', rep
                f=open(rep + '_ACs.pic','w')
                pickle.dump(self.ACs,f)
                f.close()
                
                f=open(rep + '_G.pic','w')
                pickle.dump(self.G,f)
                f.close()
                
                f=open(rep + '_flags.pic','w')
                pickle.dump((self.flag_first, self.bottlenecks, self.overloadedFPs),f)
                f.close()
            
    def load(self, rep=''):
        """
        Load a split Simulation from disk.
        """            
        if rep=='':
            rep=build_path(self.paras)
        print 'Loading split simu from ', rep
        f=open(rep + '_ACs.pic','r')
        self.ACs=pickle.load(f)
        f.close()
        
        f=open(rep + '_G.pic','r')
        self.G=pickle.load(f)
        f.close()
        
        f=open(rep + '_flags.pic','r')
        (self.flag_first, self.bottlenecks, self.overloadedFPs)=pickle.load(f)
        f.close()
            
    def show(self):
        nx.draw(self.G)
        plt.pyplot.show()
        
    def make_times(self,day):
        if self.departure_times=='zeros':
            self.t0sp=[[0 for j in range(self.na)] for i in range(self.ACtot)]     
        elif self.departure_times=='uniform':
            self.t0sp=[[uniform(0,day) for j in range(self.na)] for i in range(self.ACtot)]
        elif self.departure_times=='square_waves':
            self.t0sp=[]
            if self.na==1:
                for i in range(self.Np):
                    for j in range(self.ACsperwave):
                        self.t0sp.append([uniform(i*(self.tau + self.Delta_t),i*(self.tau + self.Delta_t)+self.tau)])
            else:
                print 'Not implemented yet...'
                raise
        elif self.departure_times=='peaks':
            print 'Not implemented yet...'
            raise
            
    def mark_best_of_queue(self):
        for f in self.queue:
            #print len(f.FPs)
            f.best_fp_cost=f.FPs[0].cost
            
    def M0_to_M1(self):
        self.M0_queue=copy.deepcopy(self.queue)
        sectors_to_shut=[n for n in sample(self.G.nodes(), self.N_shocks) if not n in self.G.airports]
        if self.verb:
            print 'Sectors to shut:', sectors_to_shut
        for n in sectors_to_shut:
            flights_to_reallocate=[]
            for f in self.queue:
                if f.fp_selected!=None and n in f.fp_selected.p:
                    flights_to_reallocate.append(f)
                    self.queue.remove(f)
        
            if self.verb:
                print 'Shutting sector', n, ': number of flights to be reallocated: ', len(flights_to_reallocate)
            for f in flights_to_reallocate:
                self.G.deallocate(f.fp_selected)
            
            self.G.shut_sector(n)
            self.G.build_H()
                
            self.G.compute_shortest_paths(self.Nfp)
        
            #shuffle(flights_to_reallocate)
            
            for f in flights_to_reallocate:
                if self.G.short[(f.source, f.destination)]!=[[]]:
                    f.FPs=self.ACs[f.ac_id].add_flightplans(f.source, f.destination, f.pref_time, self.tau, self.G)
                    self.G.sector_loading(f)
                else:
                    f.FPs=self.ACs[f.ac_id].add_dummy_flightplans(f.source, f.destination, f.pref_time)
                    for fp in f.FPs:
                        fp.accepted=False
                        fp.times=[fp.t]
                    f.fp_selected=None
                self.queue.append(f)
                f.pos_queue=len(self.queue)-1
                
                
def post_process_queue(queue):
    """
    Used to post-process results. Every processes between the simulation and 
    the plots should be here.
    Changed in 2.4: add satisfaction, regulated flight & regulated flight plans. On level of iteration added (on par).
    Changed in 2.5: independent function.
    Changed in 2.7: best cost is not the first FP's one.
    """
    for f in queue:   
        # Make flags
        f.make_flags()
        
        #Satisfaction
        bestcost=f.best_fp_cost
        acceptedFPscost=[FP.cost for FP in f.FPs if FP.accepted]
                
        if len(acceptedFPscost)!=0:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    f.satisfaction=bestcost/min(acceptedFPscost)
                except:
                    if min(acceptedFPscost)==0.:
                        assert bestcost==0.
                        f.satisfaction=1.
        else:
            f.satisfaction=0                    
            
        # Regulated flight plans
        f.regulated_FPs=len([FP for FP in f.FPs if not FP.accepted])
        
        # Regulated flights
        if len([FP for FP in f.FPs if FP.accepted])==0:
            f.regulated_F = 1.
        else:
            f.regulated_F = 0.

    return queue

def extract_aggregate_values_on_queue(queue, types_air_companies, mets=['satisfaction', 'regulated_F', 'regulated_FPs']):
    results={}
    for m in mets:
        results[m]={}
        for tac in types_air_companies:
            pouet=[getattr(f,m) for f in queue if tuple(f.par)==tuple(tac)]
            if pouet!=[]:
                results[m][tuple(tac)]=np.mean(pouet)
            else:
                results[m][tuple(tac)]=0.
        
    return results                
            
def extract_aggregate_values_on_network(G):
    coin1=[]
    coin2=[]
    for n in G.nodes():
        if len(G.node[n]['load'])>2:
            weights=[(G.node[n]['load'][i+1][0] - G.node[n]['load'][i][0]) for i in range(len(G.node[n]['load'])-1)]
            coin1.append(np.average([G.node[n]['load'][i][1] for i in range(len(G.node[n]['load'])-1)], weights=weights))
            coin2.append(np.average([G.node[n]['load'][i][1]/float(G.node[n]['capacity']) for i in range(len(G.node[n]['load'])-1)], weights=weights))
        else:
            coin1.append(0.)
            coin2.append(0.)

    return {'loads': np.mean(coin1), 'loads_norm':np.mean(coin2)}     
 
def find_overloaded_sectors(queue):
    """
    Find the sectors at least overloaded once thanks the bottleneck attribute 
    of flights.
    """
    sectors = list(set((fp.bottleneck for f in queue for fp in f.FPs if hasattr(fp, 'bottleneck'))))

    return sectors

def plot_times_departure(queue, rep='.'):
    t_pref=[f.FPs[0].t for f in queue]
    t_real=[f.fp_selected.t for f in queue if f.fp_selected!=None]
    
    plt.figure(1)
    bins=range(int(ceil(max(t_real + t_pref))) + 10)
    plt.hist(t_pref,label='pref',facecolor='green', alpha=0.75, bins=bins)
    #plt.hist(t_real,label='real',facecolor='blue', alpha=0.25, bins=bins)
    #plt.legend()
    plt.ylabel('Number of departing flights')
    plt.xlabel('Time')
    plt.savefig(rep + '/departure_times.png')
    plt.show()
        
if __name__=='__main__': 
    GG=ABMvars.G
    paras=ABMvars.paras

    sim=Simulation(paras, G=GG, make_dir=True, verbose=True)
    sim.make_simu(storymode=False)
    sim.compute_flags()
    queue=post_process_queue(sim.queue)
    M0_queue=post_process_queue(sim.M0_queue)
    print 'Global metrics:'
    res = extract_aggregate_values_on_queue(queue, paras['par'])
    for met, r in res.items():
        print ' -', met, ':'
        for k, rr in r.items():
            print '  --', k, ':', rr

    overloaded_sectors = find_overloaded_sectors(queue)
    print "Overloaded sectors:", overloaded_sectors

    color_overloaded = {}
    for n in GG.nodes():
        if n in overloaded_sectors:
            color_overloaded[n] = 'r'
        else:
            color_overloaded[n] = 'b'

    draw_network_map(GG, title='Network map', 
                         queue=queue, 
                         rep='./',
                         airports=True, 
                         load=False, 
                         generated=True,
                         size=60,
                         colors_nodes=color_overloaded,
                         file_save=sim.rep + '/network_map.png')
    #print 'Average satisfaction:', np.mean(extract_aggregate_values_on_queue(queue, paras['par'])['satisfaction'].values())
    print 
    print 'M0 global metrics:'
    res = extract_aggregate_values_on_queue(M0_queue, paras['par'])
    for met, r in res.items():
        print ' -', met, ':'
        for k, rr in r.items():
            print '  --', k, ':', rr
    print
    print
    #print [f.satisfaction for f in sim.queue]
    
    #sim.save()
    
    #plot_times_departure(sim.queue, rep=sim.rep)
    
    #draw_network_map(sim.G, title=sim.G.name, queue=sim.queue, generated=True, rep='./' + sim.G.name + '_')
    
    # print sim.M0_queue
   
    # print sim.queue

    print 'Done.'
    

        
