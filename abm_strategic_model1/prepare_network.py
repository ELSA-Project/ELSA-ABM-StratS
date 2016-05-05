#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
===========================================================================
This file is used to build a sector network for model 1.

TODO: Write a builder?
===========================================================================
"""

import sys
from os.path import join
import os

import pickle
import networkx as nx
import numpy as np
from ast import literal_eval
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point, LineString
from random import seed

from simAirSpaceO import Net
from utilities import read_paras

from libs.paths import result_dir
from libs.general_tools import draw_network_and_patches, voronoi_finite_polygons_2d


version='3.0.0'

def area(p):
    """
    Returns the area of a polygon.

    Parameters
    ----------
    p : list of tuples (x, y)
        Coordinates of the boundaries. Last Point must NOT be equal to first point.

    Returns
    -------
    Area based on cartesian distance

    Notes
    -----
    Should be moved to general_tools.

    """
    
    return 0.5 * abs(sum(x0*y1 - x1*y0
                         for ((x0, y0), (x1, y1)) in segments(p)))

def automatic_name(G, paras_G):
    """
    Automatic name based on the parameters of construction.

    Parameters
    ----------
    G : finalized Net object.
    paras_G : dictionary
        The function extract some relevenant keys and values from it
        in order to build the name. 

    Returns
    -------
    long_name : string

    Notes
    -----
    Note all parameters (keys of paras_G) are taken into account.
    New in 3.0.0: Taken and adapted from Model 2.

    """

    long_name = G.type_of_net + '_N' + str(len(G.nodes()))
    
    if G.airports!=[] and len(G.airports)==2:
       long_name+='_airports' +  str(G.airports[0]) + '_' + str(G.airports[1])
    elif len(G.airports)>2:
        long_name+='_nairports' + str(len(G.airports))
    if paras_G['pairs_sec']!=[] and len(G.airports)==2:
        long_name+='_direction_' + str(paras_G['pairs_sec'][0][0]) + '_' + str(paras_G['pairs_sec'][0][1])
    long_name+='_cap_' + G.typ_capacities
    
    if G.typ_capacities!='manual':
        long_name+='_C' + str(paras_G['C'])
    long_name+='_w_' + G.typ_weights
    
    if G.typ_weights=='gauss':
        long_name+='sig' + str(paras_G['sigma'])
    long_name+='_Nfp' + str(G.Nfp)

    return long_name

def check_empty_polygon(G, repair = False):
    """
    Check all sector-polygons of G to see if they are reduced to a
    single point (in which case they have no representative_point).

    Parameters
    ----------
    G : networkx object with attribute polygons
        giving the dictionary of the shapely Polygons to check
    repair : boolean
        if set to True, the empty polygons are removed from the 
        dictionary

    Notes
    -----
    New in 3.0.0: taken from Model 2 (unchanged)
    """

    for n in G.nodes():
        try:
            assert len(G.polygons[n].representative_point().coords) != 0 
        except AssertionError:
            print "Sector", n, "has no surface. I delete it from the network."
            if repair:
                G.remove_node(n)
                del G.polygons[n]

def compute_voronoi(G, xlims=(-1., 1.), ylims=(-1., 1.)):
    """
    Compute the voronoi tesselation of the network G. 
    Tessels are given to G through the polygons attribute,
    which is a list of shapely Polygons. In order to avoid 
    infinite polygons, the polygons are acute with a sqre of 
    size a.

    TODO: there is a problem with the outer polyongs which can be 
    GeomtryCollections (see corresponding unit test).

    Parameters
    ----------
    G : networkx object
        Each node of the network needs to have a 'coord' key
        with a 2-tuple for coordinates.
    a : float, optional
        size of the square which is used to cut the polygons.

    Returns
    -------
    G : same networkx object
        With a (new) attribute 'polygons', which is a list of shapely Polygons
        representing the cells.
    vor : Output of the Voronoi function of scipy.

    Notes
    -----
    New in 3.0.0: taken from Model 2 (no changes).

    """

    polygons = {}
    nodes = G.nodes()

    # Compute the voronoi tesselation with scipy
    #print np.array([G.node[n]['coord'] for n in nodes])
    vor = Voronoi(np.array([G.node[n]['coord'] for n in nodes]))
    # print
    # print dir(vor)
    # print vor.point_region
    # print vor.vertices
    # print vor.ridge_points
    # print
    # print

    new_regions, new_vertices = voronoi_finite_polygons_2d(vor)
    # print new_vertices
    # print 
    # print new_regions

    voronoi_plot_2d(vor)
    #plt.show()

    # Build polygons objects
    #for i, p in enumerate(vor.point_region):
    #    r = vor.regions[p]
    #    coords=[vor.vertices[n] for n in r + [r[0]] if n!=-1]
    for i, p in enumerate(new_regions):
        coords = list(new_vertices[p]) + [new_vertices[p][0]]
        #print "YAYAH", p, coords
        if len(coords)>2:
            G.node[i]['area'] = area(coords)
            polygons[i] = Polygon(coords)
            try:
                assert abs(G.node[i]['area'] - polygons[i].area)<10**(-6.)
            except:
                raise Exception(i, G.node[i]['area'], polygons[i].area)


        else:
            # Case where the polygon is only made of 2 points...
            print "Problem: the sector", i, "has the following coords for its vertices:", coords
            print "I remove the corresponding node from the network."
            G.remove_node(nodes[i])
    
    # raise Exception()
    # eps = 0.1
    # minx, maxx, miny, maxy = min([n[0] for n in vor.vertices])-eps, max([n[0] for n in vor.vertices]) +eps, min([n[1] for n in vor.vertices]) -eps, max([n[1] for n in vor.vertices]) +eps
    
    square = Polygon([[xlims[0],ylims[0]], [xlims[0], ylims[1]], [xlims[1],ylims[1]], [xlims[1], ylims[0]], [xlims[0],ylims[0]]])

    # Cut all polygons with the square.
    for n, pol in polygons.items():
        polygons[n] = pol.intersection(square)
        
        try:
            assert type(polygons[n])==type(Polygon())
        except:
            print "BAM", n, 'coords:', coords
            raise

    G.polygons = polygons
    return G, vor

def extract_capacity_from_traffic(G, flights, date=[2010, 5, 6, 0, 0, 0]):
    """

    Parameters
    ----------
    G: Net object
    flights: list
        of flight from the Distance library.
    date: list, optional
        ?

    Notes
    -----
    New in 3.0.0: taken and adapted from Model 2.
    (From Model 2)
    New in 2.9.3: Extract the "capacity", the maximum number of flight per hour, based on the traffic.
    Changed in 2.9.4: added paras_real.
    Changed in 2.9.5: fligths are given externally.

    TODO: CHECK THIS "48" in loads dic.
    TODO: NOT WORKING NOW: the flights from distance library needs to be converted
    to trajectories of sectors.

    SEE (taken from Net object):

    def extract_capacity_from_traffic(self):
        "
        New in 2.9.3: Extract the "capacity", the maximum number of flight per hour, based on the traffic.
        Changed in 2.9.4: added paras_real.
        Changed in 2.9.5: fligths are given externally.

        New in 2.6.3: imported and adapted from 2.9 fork. Defines the capacity with the peaks of traffic.
        "
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

    """

    print 'Extracting the capacities from traffic...'

    weights, pop = {}, {}
    loads = {n:[0 for i in range(48)] for n in G.nodes()}  # why 48? TODO.
    for f in flights:
        hours = {}
        r = f['route_m1t']
        for i in range(len(r)):
            if G.G_nav.idx_nodes.has_key(r[i][0]):
                p1 = G.G_nav.idx_nodes[r[i][0]]
                if G.G_nav.has_node(p1):
                    s1 = G.G_nav.node[p1]['sec']
                    hours[s1] = hours.get(s1,[]) + [int(float(delay(r[i][1], starting_date = date))/3600.)]

        for n,v in hours.items():
            hours[n] = list(set(v))

        for n,v in hours.items():
            for h in v:
                loads[n][h] += 1

    capacities = {}
    for n,v in loads.items():
        capacities[n] = max(v)
    
    return capacities

def extract_weights_from_traffic(G, flights):
    """
    Compute the average times needed for a filght to go from one node to the other.
    Segments of flights which are not edges of G are not considered.

    Parameters
    ----------
    G : Net or NavpointNet object 
        with matching dictionnary between names and indices of nodes as
        attribute 'idx_nodes'.
    flights : list of Flight Object from the Distance library
        Object needs to have a key 'route_m1t' which is a list of the navpoint label 
        and times (name, time), with time a tuple (y, m, d, h, m s).

    Returns
    -------
    weights : dictionnary
        keys are tuple (node index, node index) and values are the weights, 
        i.e. the average time needed to go from one node to the other, in minutes.

    Notes
    -----
    Changed in 2.9.4: added paras_real.
    Changed in 2.9.5: flights are given externally.
    TODO: use datetime instead of delay. NOT WORKING, NEEDS TO BE 
    ADAPTED TO MODEL 1. SEE extract_capacity_from_traffic FOR THE ISSUE.

    """
    print 'Extracting weights from data...'

    #flights=get_flights(paras_real)
    weights = {}
    pop = {}
    for f in flights:
        r = f['route_m1t']
        for i in range(len(r)-1):
            if G.idx_nodes.has_key(r[i][0]) and G.idx_nodes.has_key(r[i+1][0]):
                p1 = G.idx_nodes[r[i][0]]
                p2 = G.idx_nodes[r[i+1][0]]
                if G.has_edge(p1,p2):
                    weights[(p1,p2)] = weights.get((p1, p2),0.) + delay(np.array(r[i+1][1]),starting_date=np.array(r[i][1]))/60.
                    pop[(p1,p2)] = pop.get((p1, p2),0.) + 1

    for k in weights.keys():
        weights[k] = weights[k]/float(pop[k])

    if len(weights.keys())<len(G.edges()):
        print 'Warning! Some edges do not have a weight!'
    return weights

def give_capacities_and_weights(G, paras_G):
    """
    Gives the capacities and weights (time of travel between nodes)
    to the network.

    Parameters
    ==========
    G: Net object
    paras_G: dicionnary,
        parameters to build the network.

    Notes
    -----
    New in 3.0.0: taken and adapted from model 2

    """

    if paras_G['generate_capacities_from_traffic']:
        raise Exception("The feature 'generate_capacities_from_traffic' is not implemented yet.")
        capacities = extract_capacity_from_traffic(G, paras_G['flights_selected'])
        G.fix_capacities(capacities)
    else:
        if paras_G['capacities']==None:
            G.generate_capacities(typ=paras_G['typ_capacities'], C=paras_G['C'], par=paras_G['suppl_par_capacity'])
            #G.typ_capacities=paras_G['typ_capacities']
        else:
            G.fix_capacities(paras_G['capacities'])

    for n in G.nodes():
        try:
            assert G.node[n].has_key('capacity')
        except AssertionError:
            print "This node did not receive any capacity:", n
            raise

    # THIS PIECE HAS TO BE ADAPTED.
    if paras_G['generate_weights_from_traffic']:
        raise Exception("The feature 'generate_weights_from_traffic' is not implemented yet.")
        weights = extract_weights_from_traffic(G.G_nav, paras_G['flights_selected'])
        G.G_nav.fix_weights(weights, typ='traffic')
        avg_weight = np.mean([G.G_nav[e[0]][e[1]]['weight'] for e in G.G_nav.edges() if G.G_nav[e[0]][e[1]].has_key('weight')])
        avg_length = np.mean([dist_flat_kms(np.array(G.G_nav.node[e[0]]['coord'])*60., np.array(G.G_nav.node[e[1]]['coord'])*60.) for e in G.G_nav.edges() if G.G_nav[e[0]][e[1]].has_key('weight')])
        for e in G.G_nav.edges():
            if not G.G_nav[e[0]][e[1]].has_key('weight'):
                #print G.G_nav.node[e[0]]['coord']
                #raise Exception()
                length = dist_flat_kms(np.array(G.G_nav.node[e[0]]['coord'])*60., np.array(G.G_nav.node[e[1]]['coord'])*60.)
                weight = avg_weight*length/avg_length
                print "This edge did not receive any weight:", e, ", I set it to the average (", weight, "minutes)"
                G.G_nav[e[0]][e[1]]['weight'] = weight
    else:
        if paras_G['weights']==None:
            G.generate_weights(typ=paras_G['typ_weights'], par=paras_G['par_weights'])
        else:
            G.fix_weights(paras_G['weights'], typ='data')
    
    for e in G.edges():
        try:
            assert G[e[0]][e[1]].has_key('weight')
        except AssertionError:
            print "This edge did not receive any weight:", e
            raise 

    G.build_H() #This is the graph used for shortest_path

    return G

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

def recompute_neighbors(G):
    """
    Checks if neighbouring sectors have a common boundary. Disconnects them otherwise.

    Parameters
    ----------
    G : networkx object with attribute polygons
        'polygons' must be a list of shapely Polygons.

    Notes
    -----
    New in 3.0.0: taken from Model 2.
    (from Model2)
    New in 2.9.6

    """

    for n1 in G.nodes():
        neighbors = G.neighbors(n1)[:]
        for n2 in neighbors:
            if n1<n2:
                try:
                    assert G.polygons[n1].touches(G.polygons[n2])
                except AssertionError:
                    print "The two sectors", n1, "and", n2, "are neighbors but do not touch each other. I cut the link."
                    G.remove_edge(n1,n2)

def reduce_airports_to_existing_nodes(G, pairs, airports):
    """
    Remove all nodes in airports list and list of origin-destination 
    which are not in the list of nodes of G. 
    
    Parameters
    ----------
    G : networkx object
    pairs : list of 2-tuples (int, int)
        list of origin-destination pairs. If set to None, the procedure
        is skipped for this list.
    airports : list of integers
        list of possible airports. If set to None, the procedure is 
        skipped for this list.

    Returns
    -------
    pairs : same as input without the tuples including nodes absent of G
    airports : same as input without nodes absent of G

    Notes
    -----
    New in 3.0.0: taken from model 2.  

    """

    if pairs!=None:
        for e1, e2 in pairs[:]:
            if not G.has_node(e1) or not G.has_node(e2):
                pairs.remove((e1,e2))

    if airports!=None:
        for a in airports[:]:
            if not G.has_node(a):
                airports.remove(a)    

    return pairs, airports

def segments(p):
    """
    Compute the list of segments given a list of coordinates. Attach
    the end point to the first one.

    Parameters
    ----------
    p : list of tuples (x, y)
        Coordinates of the boundaries.

    Returns
    l : list of 2-tuples of 2-tuples.

    Notes
    -----
    TODO: should be put in general_tools (alos for model 2)
    """

    return zip(p, p[1:] + [p[0]])

"""
===========================================================================
"""
def hard_infrastructure(G, paras_G):
    """
    Defines the "hard" infrastructure: position of nodes and definition of edges.
    """

    # Import the network from the paras if given, build a new one otherwise
    if paras_G['net_sec']!=None:
        G.import_from(paras_G['net_sec'], numberize=not type(paras_G['net_sec'].nodes()[0]) in [int, float])
    else:
        #G.build(paras_G['N'],paras_G['nairports'],paras_G['min_dis'],Gtype=paras_G['type_of_net'],put_nodes_at_corners = True)
        if paras_G['N']<=4:
            print "WARNING: you requested less than 4 sectors, so I do not put any sectors on the corners of the square."
            paras_G['N']+=4
        G.build(paras_G['N']-4, Gtype=paras_G['type_of_net'], put_nodes_at_corners=paras_G['N']>4)

    # Give the pre-built polygons to the network.
    if paras_G['polygons']!=None:
        G.polygons={}
        for name, shape in paras_G['polygons'].items():
            G.polygons[G.idx_nodes[name]]=shape
    else:
        G, vor = compute_voronoi(G)
    
    # Check if every sector has a polygon
    for n in G.nodes():
        try:
            assert G.polygons.has_key(n)
        except:
            print "Sector", n, "doesn't have any polygon associated!"
            raise

    # Check if there are polygons reduced to a single point.
    check_empty_polygon(G, repair=True)
    check_empty_polygon(G, repair=False)

    # Make sure that neighbors have a common boundary
    recompute_neighbors(G)

    return G

def soft_infrastructure(G, paras_G):
    """
    Defines the "soft" infrastructure: capacity, weights, airports, shortest paths.
    """

    ########### Choose the airports #############
    # `Airports' means all entry and exit points here, not only physical airports.
    print "Choosing the airports..."

    paras_G['pairs_sec'], paras_G['airports_sec'] = reduce_airports_to_existing_nodes(G, paras_G['pairs_sec'], paras_G['airports_sec'])

    if paras_G['airports_sec']!=None:
        G.add_airports(paras_G['airports_sec'], paras_G['min_dis'], pairs=paras_G['pairs_sec'], C_airport=paras_G['C_airport'])
    else:
        # If none of them are specified, draw at random some entry/exits for the navpoint network and
        # infer the sector airports.
        G.generate_airports(paras_G['nairports_sec'], paras_G['min_dis'], C_airport=100000)

    print 'Number of airports (sectors) at this point:', len(G.airports)
    print 'Number of connections (sectors) at this point:', len(G.connections())
    print 'Airports at this point:', G.airports
    
    ########## Generate Capacities and weights ###########
    print "Choosing capacities and weights..."
    G = give_capacities_and_weights(G, paras_G)
    print
    
    ############# Computing shortest paths ###########
    G.Nfp = paras_G['Nfp']
    print 'Computing shortest_paths ...'
    pairs_deleted = G.compute_shortest_paths(G.Nfp, repetitions=False, delete_pairs=False)   

    G.stamp_airports()
    # CHECK IF AIRPORTS ARE ALRIGHT!

    return G

def prepare_network(paras_G, rep=None, save_name=None, show=True):
    """
    Prepare the network of sectors. If no file 
    containing a networkx object is given via paras_G['net_sec'], it builds a new sector network
    with paras_G['N'] number of sectors, including one sector at each corner of the square 
    [0, 1] x [0, 1]. The latter is done to facilitate the voronoi tesselation and the subsequent
    computation of the boundaries of the sectors.

    Parameters
    ----------
    paras_G : dictionary
        of the parameters for the construction of the network

    Returns
    -------
    G : Net object.

    Notes
    -----
    New in 2.6.1: the keep_attributes_G
    Changed in 3.0.0: Some parts taken from model 2.
    
    """

    G = Net()
    G.type = 'sec' #for sectors
    G.type_of_net = paras_G['type_of_net']

    # --------------------- 'Hard' infrastructure ----------------------- #

    G = hard_infrastructure(G, paras_G)

    # --------------------- 'Soft' infrastructure ----------------------- #

    G = soft_infrastructure(G, paras_G)    

    # ---------------------------- Summary ------------------------------ #

    if paras_G['flights_selected']!=None:
        print 'Selected finally', len(paras_G['flights_selected']), "flights."

        #G.check_all_real_flights_are_legitimate(flights_selected) # no action taken here

    print 'Final number of sectors:', len(G.nodes())
    print 'Final number of airports:', len(G.get_airports())
    print 'Final number of connections:', len(G.connections())

    # Possibly add a cconsistency check between traffic and airports,
    # See Model 2 line around line 1526.

    # ------------------------------ Finish ------------------------------ #

    # Automatic Name
    long_name = automatic_name(G, paras_G)

    # Manual name
    if paras_G['name']!='':
        name = paras_G['name']
    else:
        name = long_name
        
    G.name = name
    G.comments = {'long name':long_name, 'made with version':version}

    if save_name==None:
        save_name = name

    rep = join(rep, save_name)
    os.system('mkdir -p ' + rep)
    G.rep = rep

    # Save 
    if rep!=None:
        with open(join(rep, save_name) + '.pic','w') as f:
            pickle.dump(G, f)
        if paras_G['flights_selected']!=None:
            with open(join(rep, save_name + '_flights_selected.pic'),'w') as f:
                pickle.dump(flights_selected, f)

    # Stats
    G.basic_statistics(rep=rep)

    # Draw network
    if show:
        #if paras_G['flights_selected']==None:
        draw_network_and_patches(G, None, G.polygons,
                                     name=save_name, 
                                     show=True, 
                                     flip_axes=True, 
                                     trajectories=[sp for paths in G.short.values() for sp in paths], 
                                     rep=rep,
                                     trajectories_type='sectors')
        #else:
        #    trajectories = [[G.G_nav.idx_nodes[p[0]] for p in f['route_m1']] for f in paras_G['flights_selected']]
        #    draw_network_and_patches(G, G.G_nav, G.polygons, name=save_name, show=True, flip_axes=True, trajectories=trajectories, rep=rep)

    print 'Network saved in', rep 
    print 'Done.'
    
    return G

if  __name__=='__main__':
    # Canonical run:
    # python prepare_network.py [path to paras_G] [path for saving files]
    # If the path of saveing files is omitted, the path is [results_dir]/networks/
    # Ex:
    # python prepare_network.py my_paras/my_paras_G_DiskWorld_medium.py /home/earendil/Documents/ELSA/ABM/results_new/networks
    
    if 1:
        # Manual seed
        see_ = 2
        print "===================================="
        print "USING SEED", see_
        print "===================================="
        seed(see_)

    if len(sys.argv)==1:
        paras_file = 'paras_G.py' 
        rep = join(result_dir, 'networks')
    elif len(sys.argv)==2:
        paras_file = sys.argv[1]
        rep = join(result_dir, 'networks')
    elif len(sys.argv)==3:
        paras_file = sys.argv[1]
        rep = sys.argv[2]
    else:
        raise Exception("You should put 0, 1 or 2 arguments.")

    paras_G = read_paras(paras_file=paras_file, post_process=False)
    
    print 'Building network with paras_G:'
    print paras_G
    print
    #rep = join(result_dir, "networks")
    
    G = prepare_network(paras_G, rep=rep, show=True)

