#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:06:18 2013

@author: earendil
"""
#from networkx import shortest_path
#import numpy as np
from simAirSpaceO import Net
import pickle
import networkx as nx
import os
from ast import literal_eval

version='2.6.1'


def import_network_pieces(name, save_name):
    """
    New in 2.6.1: import from disk a network from a text file, previously exported 
    from a 2.9 graph.
    """
    #name = 'LF29_RC_FL350_DA0v3_Strong'
    print 'Importing the network called', name, '...'
    G = Net()

    with open(name + '_pieces.txt', 'r') as f:
        lines = f.readlines()

    i = 0
    while lines[i]!='\n':
        n = int(lines[i])
        G.add_node(n)
        i += 1
        dic = literal_eval(lines[i])
        for k, v in dic.items():
            if k!='navs':
                G.node[n][k] = v
        i += 1

    i += 1
    while lines[i]!='\n':
        edge = lines[i]
        e1, e2 = literal_eval(edge)
        G.add_edge(e1, e2)
        i += 1
        dic = literal_eval(lines[i])
        for k, v in dic.items():
            G[e1][e2][k] = float(v)
        i += 1

    with open(name + '_pieces2.pic', 'r') as f:
        dic = pickle.load(f)

    for k, v in dic.items():
        setattr(G, k, v)

    G.pairs = G.short.keys()

    #G.airports = [p[0] for p in G.pairs] + [p[1] for p in G.pairs]

    # G.pairs = G.short.keys()
    # G.build_H()
    # G.short = {k:[] for k in G.short.keys()}
    # G.generate_weights(typ='coords')
    # G.typ_weights='coords'
    # G.initialize_load()
    # G.compute_shortest_paths(G.Nfp)


    with open(save_name + '.pic', 'w') as f:
        pickle.dump(G, f)

def prepare_network(paras_G, generate_new=False, name_init='', save=True, \
    typ = 'D', keep_attributes_G = False):
    """
    New in 2.6.1: the keep_attributes_G
    """


    ####################### Generate/load network #####################
    
    # type_of_net='D'                 #type of graph ('D'=Delaunay triangulation, 'T'=triangular lattice, "E"=Erdos-Renyi random graph)
    # N=30                            #order of the graph (in the case net='T' verify that the order is respected)        
    # #airports=[65,20]                #IDs of the nodes used as airports
    # airports=['LFMME3', 'LFFFTB']   #LFBBN2 (vers Perpignan)
    # nairports=2                     #number of airports
    # pairs=[]#[(22,65)]              #available connections between airports
    #  #[65,20]
    # #[65,22]
    # #[65,62]
    # min_dis=2                       #minimum number of nodes between two airpors
    
    #network_name='generic'
    
    #generate_new=False
    
    #airport_dis=len(shortest_path(G,airports[0],airports[1]))-1  #this is the distance between the airports
    
    if generate_new:
        G=Net()
        G.build(paras_G['N'],paras_G['nairports'],paras_G['min_dis'],Gtype=paras_G['type_of_net'])
        type_of_net = paras_G['type_of_net']
    else:
        if 1:
            #fille='DEL_C_65_22'
            fille=name_init
            type_of_net=typ
            
        if 0:
            fille='test_graph_90_weighted_II'
            type_of_net='D' 
        
        if 0:
            fille='Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0_undirected_threshold'
            type_of_net='R' 
            
        with open('../networks/' + fille + '.pic','r') as _g: 
            if not keep_attributes_G:   
                G=Net()    
                H = pickle.load(_g)
                G.import_from(H)
            else:
                G = pickle.load(_g)

        if generate_new_airports:
            if paras_G['airports']!=[]:
                G.fix_airports(paras_G['airports'], paras_G['min_dis'], pairs=paras_G['pairs'])
            else:
                G.generate_airports(paras_G['nairports'],paras_G['min_dis'])
        else:
            G.fix_airports(G.airports, paras_G['min_dis'], pairs=G.pairs)

        if typ == 'R':
            G.reduce_flights()
        G.build_H()
    
    ####################### Capacities/weights #####################
    
    # generate_weights=True
    # typ_weights='coords'
    # sigma=0.01
    # generate_capacities=True
    # typ_capacities='manual'
    # C=5                             #sector capacity
    
    #file_capacities='capacities_sectors_Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0.pic'
    
    if paras_G['generate_capacities']:
        G.generate_capacities(typ=paras_G['typ_capacities'], C=paras_G['C'], file_capacities=paras_G['file_capacities'])
        G.typ_capacities=paras_G['typ_capacities']
    else:
        G.typ_capacities='constant'
        
    if paras_G['generate_weights']:
        G.generate_weights(typ=paras_G['typ_weights'], par=[1.,paras_G['sigma']])
        G.typ_weights=paras_G['typ_weights']
    else:
        G.typ_weights='gauss'
        
    for a in G.airports:
        G.node[a]['capacity']=100000
    
    ####################### Preprocess stuff ####################
    
    G.Nfp=10   ############################## ATTENTIONNNNNNN ###################
    
    G.initialize_load()#2*(nx.diameter(G)+G.Nfp))
    G.compute_shortest_paths(G.Nfp, repetitions = True)
    
    print 'Number of nodes:', (len(G.nodes()))
    
    ##################### Name ###########################
    
    long_name=type_of_net + '_N' + str(len(G.nodes()))
    
    if paras_G['airports']!=[]:
       long_name+='_airports' +  str(paras_G['airports'][0]) + '_' + str(paras_G['airports'][1])
    if paras_G['pairs']!=[] and len(paras_G['airports'])==2:
        long_name+='_direction_' + str(paras_G['pairs'][0][0]) + '_' + str(paras_G['pairs'][0][1])
    long_name+='_cap_' + G.typ_capacities
    
    if G.typ_capacities!='manual':
        long_name+='_C' + str(paras_G['C'])
    long_name+='_w_' + G.typ_weights
    
    if G.typ_weights=='gauss':
        long_name+='sig' + str(paras_G['sigma'])
    long_name+='_Nfp' + str(G.Nfp)
    
    ##################### Manual name #################
    if paras_G['name']!='':
        name=paras_G['name']
    else:
        name=long_name
        
    G.name=name
    G.comments={'long name':long_name, 'made with version':version}
    G.basic_statistics(rep=name  + '_')
    
    if save:
        with open('../networks/' + name + '.pic','w') as f:
            pickle.dump(G, f)
    
    print 'Done.'

    return G
    
if  __name__=='__main__':
    
    if 0:
        name = 'LF29_RC_FL350_DA0v3_Strong'
        save_name = 'LF26_RC_FL350_DA0v3_Strong_unfinished'
        import_network_pieces(name, save_name)

    if 1:
        type_of_net='D'                 #type of graph ('D'=Delaunay triangulation, 'T'=triangular lattice, "E"=Erdos-Renyi random graph)
        N=1000                            #order of the graph (in the case net='T' verify that the order is respected)        
        airports=[]                #IDs of the nodes used as airports
        #airports=['LFMME3', 'LFFFTB']   #LFBBN2 (vers Perpignan)
        #airports=[65,22,10,45, 30, 16]
        nairports=2                     #number of airports
        pairs=[(22,65)]              #available connections between airports
        #[65,20]
        #[65,22]
        #[65,62]
        min_dis=2                       #minimum number of nodes between two airpors
        
        generate_new_airports = False

        generate_weights=True
        typ_weights='gauss'
        sigma=0.01
        generate_capacities=True
        typ_capacities='constant'
        C=5                             #sector capacity
        
        file_capacities='capacities_sectors_Weak_EXTLF_2010-5-6_15:0:0_2010-5-6_16:59:0.pic'
        
        
        #name='LF_R_NSE'
        
        paras_G={k:v for k,v in vars().items() if k[:1]!='_' and k!='version' and k!='prepare_network'}
        

        if 1:
            #name='NEWDEL1000_C_spatial'
            #name_init = 'LF26_RC_FL350_DA0v3_Strong_unfinished'
            #name_init = 'DEL_C_65_22'
            name_init = 'NEWDEL1000_C_spatial'
            #name = 'DEL_C_65_22_new_spatial'
            name = 'NEWDEL1000_C'
            paras_G['name']=name
            # prepare_network(paras_G, 
            #                 generate_new=False, 
            #                 name_init=name_init, 
            #                 save=True, 
            #                 typ = 'R', 
            #                 keep_attributes_G = True)
            prepare_network(paras_G, 
                            generate_new=False, 
                            name_init=name_init, 
                            save=True, 
                            typ = type_of_net, 
                            keep_attributes_G = True)
        else:
            name_init='NEWDEL1000_C_spatial'
            os.system('mkdir -p ' + name_init + '_SPs')
            with open(name_init + '.pic', 'r') as f:
                GG=pickle.load(f)
            distance=7
            pairs_distance=[]
            
            try:
                with open(name_init + '_SPs/' + name_init + '_pairs.pic', 'r') as _f:
                    pairs_distance = pickle.load(_f)
                print 'Loading pairs from disk...'
            except (IOError, EOFError):
                print 'Computing pairs...'
                for i,n in enumerate(GG.nodes()):
                    for j,m in enumerate(GG.nodes()):
                        if i<j:
                            if len(nx.shortest_path(GG,n,m))==distance:
                                pairs_distance.append([n,m])
                with open(name_init + '_SPs/' + name_init + '_pairs.pic', 'w') as _f:
                    pickle.dump(pairs_distance, _f)

            print 'Number of pairs with distance', distance, ':', len(pairs_distance)
            for n,m in pairs_distance:
                name='NEWDEL1000_C_' + str(n) + '_' + str(m) + '_spatial'
                print 'Doing network with airports', n, m
                #paras_G['airports']=[n,m]
                paras_G['name']=name
                #G=prepare_network(paras_G, name_init = name_init, save = False)
                GG.airports=[n,m]
                GG.compute_shortest_paths(GG.Nfp)
                with open(name_init + '_SPs/' + name + '_SPs.pic','w') as f:
                    pickle.dump(GG.short, f)

