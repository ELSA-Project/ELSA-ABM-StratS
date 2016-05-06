# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:24:00 2013

@author: earendil

Utilies for the ABM.
"""

from mpl_toolkits.basemap import Basemap
from math import sqrt, cos, sin, pi, ceil
import numpy as np
from numpy import *
import matplotlib.gridspec as gridspec
#from descartes import PolygonPatch
import matplotlib.pyplot as plt
import pickle
import imp

from general_tools import split_coords, map_of_net, nice_colors

version='3.0.0'

def draw_network_map(G, title='Network map', queue=[], rep='./',airports=True, load=True,
    generated=False, file_save=None, colors_nodes='b', size=10, dpi=300):
    x_min=min([G.node[n]['coord'][0]/60. for n in G.nodes()])-0.5
    x_max=max([G.node[n]['coord'][0]/60. for n in G.nodes()])+0.5
    y_min=min([G.node[n]['coord'][1]/60. for n in G.nodes()])-0.5
    y_max=max([G.node[n]['coord'][1]/60. for n in G.nodes()])+0.5
    
    #(x_min,y_min,x_max,y_max),G,airports,max_wei,zone_geo = rest
    fig=plt.figure(900,figsize=(9,6))#*(y_max-y_min)/(x_max-x_min)))#,dpi=600)
    gs = gridspec.GridSpec(1, 2,width_ratios=[6.,1.])
    ax = plt.subplot(gs[0])
    ax.set_aspect(1./0.8)
    
    if generated:
        def m(a,b):
            return a,b
        x,y=[G.node[n]['coord'][0] for n in G.nodes()], [G.node[n]['coord'][1] for n in G.nodes()]
    else:
        m=draw_zonemap(x_min,y_min,x_max,y_max,'i')
        x,y=split_coords(G,G.nodes(),r=0.08)
    
    #     for n in self.nodes():
    #         print self.node[n]['load']
    #         print [self.node[n]['load'][i][1] for i in range(len(self.node[n]['load'])-1)]
    # if load:
    #     sze = [(np.average([G.node[n]['load'][i][1] for i in range(len(G.node[n]['load'])-1)],\
    #     weights=[(G.node[n]['load'][i+1][0] - G.node[n]['load'][i][0]) for i in range(len(G.node[n]['load'])-1)])
    #     /float(G.node[n]['capacity'])*800 + 5) for n in G.nodes()]
    # else:
    #     sze = size
        
    # coords={n:m(y[i],x[i]) for i,n in enumerate(G.nodes())}
    
    # ax.set_title(title)
    # #sca = ax.scatter([self.node[n]['coord'][0] for n in self.nodes()],[self.node[n]['coord'][0] for n in self.nodes()],marker='o',zorder=6,s=sze)#,s=snf,lw=0,c=[0.,0.45,0.,1])
    # nodes = G.nodes()[:]
    # if type(colors_nodes)==type({1:1}):
    #     colors_nodes = [colors_nodes[n] for n in nodes]
    # sca=ax.scatter([coords[n][0] for n in nodes], [coords[n][1] for n in nodes], marker='o', zorder=6, s=sze, c=colors_nodes)#,s=snf,lw=0,c=[0.,0.45,0.,1])
    # if airports:
    #     scairports=ax.scatter([coords[n][0] for n in G.airports],[coords[n][1] for n in G.airports], marker='s', zorder=7, s=sze, c='g')#,s=snf,lw=0,c=[0.,0.45,0.,1])
    #     scaa=ax.scatter(x_a,y_a,marker='s',zorder=5,s=sna,c=[0.7,0.133,0.133,1],edgecolor=[0,0,0,1],lw=0.7)
    #     scat = ax.scatter(x_m1t,y_m1t,marker='d',zorder=6,s=snt,lw=0,c=[0.,0.45,0.,1])

    # if 1:
    #     for e in G.edges():
    #         #if G.node[e[0]]['m1'] and G.node[e[1]]['m1']:
    #             #print e,width(G[e[0]][e[1]]['weight'])
    #                 xe1,ye1=m(self.node[e[0]]['coord'][1]/60.,self.node[e[0]]['coord'][0]/60.)
    #                 xe2,ye2=m(self.node[e[1]]['coord'][1]/60.,self.node[e[1]]['coord'][0]/60.)
    #             plt.plot([coords[e[0]][0],coords[e[1]][0]],[coords[e[0]][1],coords[e[1]][1]],'k-',lw=0.5)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)
          
    # weights={n:{v:0. for v in G.neighbors(n)} for n in G.nodes()}
    # for f in queue:
    #     try:
    #         path=f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p
    #         for i in range(0,len(path)-1):
    #             weights[path[i]][path[i+1]]+=1.
    #     except ValueError:
    #         pass
    #     except:
    #         raise
    
    # max_w=np.max([w for vois in weights.values() for w in vois.values()])
    
    # print 'max_w', max_w
    
    # for n,vois in weights.items():
    #     for v,w in vois.items():
    #         if G.node[n]['m1'] and G.node[v]['m1']:
    #plt.plot([coords[n][0],coords[v][0]],[coords[n][1],coords[v][1]],'r-',lw=w/max_w*4.)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)

    #     if 0:
    #         patch=PolygonPatch(adapt_shape_to_map(zone_geo,m),facecolor='grey', edgecolor='grey', alpha=0.08,zorder=3)#edgecolor='grey', alpha=0.08,zorder=3)
    #         ax.add_patch(patch)
            
    #     if 0:
    #         patch=PolygonPatch(adapt_shape_to_map(expand(zone_geo,0.005),m),facecolor='brown', edgecolor='black', alpha=0.1,zorder=3)#edgecolor='grey', alpha=0.08,zorder=3)
    #         ax.add_patch(patch)
    # if file_save!=None:
    #     plt.savefig(file_save, dpi=dpi)
    #     print "Graph saved as",file_save
    # plt.show()

def draw_network_map_bis(G, title='Network map', trajectories=[], rep='./',
    airports=True, load=True, generated=False, add_to_title='', polygons=[], 
    numbers=False, show=True, colors='b'):
    print "Drawing network..."
    x_min=min([G.node[n]['coord'][0]/60. for n in G.nodes()])-0.5
    x_max=max([G.node[n]['coord'][0]/60. for n in G.nodes()])+0.5
    y_min=min([G.node[n]['coord'][1]/60. for n in G.nodes()])-0.5
    y_max=max([G.node[n]['coord'][1]/60. for n in G.nodes()])+0.5
    

    #(x_min,y_min,x_max,y_max),G,airports,max_wei,zone_geo = rest
    fig=plt.figure(figsize=(9,6))#*(y_max-y_min)/(x_max-x_min)))#,dpi=600)
    gs = gridspec.GridSpec(1, 2,width_ratios=[6.,1.])
    ax = plt.subplot(gs[0])
    ax.set_aspect(1./0.8)
    
    if generated:
        def m(a,b):
            return a,b
        y,x=[G.node[n]['coord'][0] for n in G.nodes()], [G.node[n]['coord'][1] for n in G.nodes()]
    else:
        m=draw_zonemap(x_min,y_min,x_max,y_max,'i')
        x,y=split_coords(G,G.nodes(),r=0.08)
    
    for i,pol in enumerate(polygons):
        patch = PolygonPatch(pol,alpha=0.5, zorder=2, color=_colors[i%len(_colors)])
        ax.add_patch(patch) 

    if load:
        sze=[(np.average([G.node[n]['load'][i][1] for i in range(len(G.node[n]['load'])-1)],\
        weights=[(G.node[n]['load'][i+1][0] - G.node[n]['load'][i][0]) for i in range(len(G.node[n]['load'])-1)])
        /float(G.node[n]['capacity'])*800 + 5) for n in G.nodes()]
    else:
        sze=10
        
    coords={n:m(y[i],x[i]) for i,n in enumerate(G.nodes())}
    
    ax.set_title(title)
    sca=ax.scatter([coords[n][0] for n in G.nodes()],[coords[n][1] for n in G.nodes()],marker='o',zorder=6,s=sze,c=colors)#,s=snf,lw=0,c=[0.,0.45,0.,1])
    if airports:
        scairports=ax.scatter([coords[n][0] for n in G.airports],[coords[n][1] for n in G.airports],marker='o',zorder=6,s=20,c='r')#,s=snf,lw=0,c=[0.,0.45,0.,1])

    if 1:
        for e in G.edges():
            plt.plot([coords[e[0]][0],coords[e[1]][0]],[coords[e[0]][1],coords[e[1]][1]],'k-',lw=0.5)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)
          
    #weights={n:{v:0. for v in G.neighbors(n)} for n in G.nodes()}
    weights={n:{} for n in G.nodes()}
    for path in trajectories:
        try:
            #path=f.FPs[[fpp.accepted for fpp in f.FPs].index(True)].p
            for i in range(0,len(path)-1):
                #print path[i], path[i+1]
                #weights[path[i]][path[i+1]]+=1.
                weights[path[i]][path[i+1]] = weights[path[i]].get(path[i+1], 0.) + 1.
        except ValueError: # Why?
            pass
    
    max_w=np.max([w for vois in weights.values() for w in vois.values()])
     
    for n,vois in weights.items():
        for v,w in vois.items():
           # if G.node[n]['m1'] and G.node[v]['m1']:
                plt.plot([coords[n][0],coords[v][0]],[coords[n][1],coords[v][1]],'r-',lw=w/max_w*4.)#,lw=width(G[e[0]][e[1]]['weight'],max_wei),zorder=4)

    if numbers:
        for n in G.nodes():
            plt.text(G.node[n]['coord'][0], G.node[n]['coord'][1], ster(n))
       # if 0:
       #     patch=PolygonPatch(adapt_shape_to_map(zone_geo,m),facecolor='grey', edgecolor='grey', alpha=0.08,zorder=3)#edgecolor='grey', alpha=0.08,zorder=3)
       #     ax.add_patch(patch)
           
       # if 0:
       #     patch=PolygonPatch(adapt_shape_to_map(expand(zone_geo,0.005),m),facecolor='brown', edgecolor='black', alpha=0.1,zorder=3)#edgecolor='grey', alpha=0.08,zorder=3)
       #     ax.add_patch(patch)
    plt.savefig(rep + 'network_flights' + add_to_title + '.png',dpi=300)
    if show:
        plt.show()

def draw_sector_map(G, ax=None, save_file=None, fmt='png', load=False, airports=False, \
    polygons=False, show=False, size_airports=30, color_airports=nice_colors[6], \
    shift_numbers=(0., 0.), numbers=False, size_numbers=10, **kwargs):
    if ax==None:
        fig=plt.figure(figsize=(9,6))#*(y_max-y_min)/(x_max-x_min)))#,dpi=600)
        gs = gridspec.GridSpec(1, 2, width_ratios=[6.,1.])
        ax = plt.subplot(gs[0])
        ax.set_aspect(1./0.8)

    if load:
        kwargs['size_nodes'] = 'load'
    
    kwargs['save_file'] = None
    kwargs['show'] = False

    ax = map_of_net(G, ax=ax, **kwargs)

    if airports:
        lis = [G.node[n]['coord'] for n in G.get_airports()]
        scairports = ax.scatter(*zip(*lis), marker='s', zorder=6, s=size_airports, c=color_airports, edgecolor='w')#,s=snf,lw=0,c=[0.,0.45,0.,1])

    if polygons:
        for i, pol in enumerate(G.polygons):
            patch = PolygonPatch(pol, alpha=0.5, zorder=2, color=_colors[i%len(_colors)])
            ax.add_patch(patch)
    if numbers:
        for n in G.nodes():
            pos_point = array(G.node[n]['coord'])
            pos_text = pos_point + array(shift_numbers)
            ax.annotate(str(n), pos_point, size=size_numbers, xytext=pos_text)

    if save_file!=None:
        plt.savefig(save_file + '.' + fmt, dpi = dpi)
        print 'Figure saved as', save_file + '.' + fmt
    if show:
        plt.show()

    return ax

def draw_zonemap(x_min,y_min,x_max,y_max,res):
    m = Basemap(projection='gall',lon_0=0.,llcrnrlon=y_min,llcrnrlat=x_min,urcrnrlon=y_max,urcrnrlat=x_max,resolution=res)
    m.drawmapboundary(fill_color='white') #set a background colour
    m.fillcontinents(color='white',lake_color='white')  # #85A6D9')
    m.drawcoastlines(color='#6D5F47', linewidth=0.8)
    m.drawcountries(color='#6D5F47', linewidth=0.8)
    m.drawmeridians(np.arange(-180, 180, 5), color='#bbbbbb')
    m.drawparallels(np.arange(-90, 90, 5), color='#bbbbbb')
    return 

def extract_data_from_files(paras_G):
    """
    Used to import the different files from disk, like network, polygons, etc.
    """
    for item in ['net_sec', 'polygons', 'capacities', 'weights', 'airports_sec', 'flights_selected']:
        if paras_G['file_' + item]!=None:
            try:
                with open( paras_G['file_' + item]) as f:
                    try:
                        paras_G[item] = pickle.load(f)
                        print "Loaded file", paras_G['file_' + item], "for", item
                    except:
                        print "Could not load the file", paras_G['file_' + item], " as a pickle file."
                        print "I skipped it and the", item, "will be generated or ignored."

            except:
                print "Could not find file",  paras_G['file_' + item]
                print "I skipped it and the", item, "will be generated or ignored."

            if item == 'airports_sec':
                paras_G['nairports_sec'] = len(paras_G['airports_sec'])

        else:
            paras_G[item]=None

    # Check consistency of pairs and airports.

    for p1, p2 in paras_G['pairs_sec']:
        try:
            assert p1 in paras_G['airports_sec'] and p2 in paras_G['airports_sec']
        except:
            print "You asked a connection for which one of the airport does not exist:", (p1, p2)
            raise

    return paras_G

# ============================================================================ #
# =============================== Parameters ================================= #
# ============================================================================ #

class Paras(dict):
    """
    Class Paras
    ===========
    Custom dictionnary used to update parameters in a controlled way.
    This class is useful in case of multiple iterations of simulations
    with sweeping parameters and more or less complex interdependances
    between variables.
    In case of simple utilisation with a single iteration or no sweeping,
    a simple dictionary is enough.

    The update process is based on the attribute 'update_priority', 'to_update'.

    The first one is a list of keys. First entries should be updated before updating 
    later ones.

    The second is a dictionary. Each value is a tuple (f, args) where f is function
    and args is a list of keys that the function takes as arguments. The function
    returns the value of the corresponding key. 

    Notes
    -----
    'update_priority' and 'to_update' could be merged in an sorted dictionary.
    New in 3.0.0: taken from Model 2 (unchanged).
    Changed in 3.0.0: initialized with self.update_priority = {}.

    """
    
    def __init__(self, dic):
        for k,v in dic.items():
            self[k] = v
        self.to_update = {}
        self.update_priority = {}

    def update(self, name_para, new_value):
        """
        Updates the value with key name_para to new_value.

        Parameters
        ----------
        name_para : string
            label of the parameter to be updated
        new_value : object
            new value of entry name_para of the dictionary.

        Notes
        -----
        Changed in 2.9.4: self.update_priority instead of update_priority.

        """
        
        self[name_para] = new_value
        # Everything before level_of_priority_required should not be updated, given the para being updated.
        lvl = self.levels.get(name_para, len(self.update_priority)) #level_of_priority_required
        #print name_para, 'being updated'
        #print 'level of priority:', lvl, (lvl==len(update_priority))*'(no update)'
        for j in range(lvl, len(self.update_priority)):
            k = self.update_priority[j]
            (f, args) = self.to_update[k]
            vals = [self[a] for a in args] 
            self[k] = f(*vals)

    def analyse_dependance(self):
        """
        Detect the first level of priority hit by a dependance in each parameter.
        Those who don't need any kind of update are not in the dictionnary.

        This should be used once when the 'update_priority' and 'to_update' are 
        finished.

        It computes the attribute 'levels', which is a dictionnary, whose values are 
        the parameters. The values are indices relative to update_priority at which 
        the update should begin when the parameter corresponding to key is changed. 

        """

        # print 'Analysing dependances of the parameter with priorities', self.update_priority
        self.levels = {}
        for i, k in enumerate(self.update_priority):
            (f, args) = self.to_update[k]
            for arg in args:
                if arg not in self.levels.keys():
                    self.levels[arg] = i
 
def read_paras(paras_file=None, post_process=True):
    """
    Reads parameter file for a single simulation.

    Notes
    -----
    New in 3.0.0: taken from Model 2
    
    """

    if paras_file==None:
        import my_paras as paras_mod
    else:
        paras_mod = imp.load_source("paras", paras_file)
    paras = paras_mod.paras

    if post_process:
        paras = post_process_paras(paras)

    return paras

def read_paras_iter(paras_file=None):
    """
    Reads parameter file for a iterated simulations.
    """
    if paras_file==None:
        import my_paras_iter as paras_mod
    else:
        paras_mod = imp.load_source("paras_iter", paras_file)
    paras = paras_mod.paras

    return paras

def control_density_ACtot(paras, to_update, update_priority):
    if paras['control_density']:
        # ACtot is not an independent variable and is computed thanks to density
        paras['ACtot']=_func_ACtot_vs_density_day_na(paras['density'], paras['day'], paras['na'])
        to_update['ACtot']=(_func_ACtot_vs_density_day_na, ('density', 'day', 'na'))
        update_priority.append('ACtot')
    else:
        # Density is not an independent variables and is computed thanks to ACtot.
        paras['density']=_func_density_vs_ACtot_na_day(paras['ACtot'], paras['na'], paras['day'])
        to_update['density']=(_func_density_vs_ACtot_na_day,('ACtot','na','day'))
        update_priority.append('density')

    return paras, to_update, update_priority

def post_process_paras(paras):
    ##################################################################################
    ################################# Post processing ################################
    ##################################################################################
    # This is useful in case of change of parameters (in particular using iter_sim) in
    # order to record the dependencies between variables.
    update_priority = []
    to_update = {}

    # -------------------- Post-processing -------------------- #

    paras['par']=tuple([tuple([float(_v) for _v in _p])  for _p in paras['par']]) # This is to ensure hashable type for keys.

    # Load network
    if paras['file_net']!=None:
        with open(paras['file_net']) as f:
            paras['G'] = pickle.load(f)
    
    if not 'G' in paras.keys():
        paras['G'] = None

    if paras['file_traffic']!=None:
        with open(paras['file_traffic'], 'r') as _f:
            flights = pickle.load(_f)
        paras['traffic'] = flights
        paras['flows'] = {}
        for f in flights:
            # _entry = G.G_nav.idx_navs[f['route_m1t'][0][0]]
            # _exit = G.G_nav.idx_navs[f['route_m1t'][-1][0]]
            if paras['G']!=None: 
                # # Find the first node in trajectory which is in airports
                # idx_entry = 0
                # while idx_entry<len(f['route_m1t']) and not paras['G'].G_nav.idx_nodes[f['route_m1t'][idx_entry][0]]:# in paras['G'].G_nav.airports:
                #     idx_entry += 1
                # if idx_entry==len(f['route_m1t']): idx_entry = 0
                
                # # Find the first node in trajectory which is in airports (backwards).
                # idx_exit = -1
                # while abs(idx_exit)<len(f['route_m1t']) and not paras['G'].G_nav.idx_nodes[f['route_m1t'][idx_exit][0]]:# in paras['G'].G_nav.airports:
                #     idx_exit -= 1
                # if idx_exit==len(f['route_m1t']): idx_exit = -1

                _entry, _exit = find_entry_exit(paras['G'].G_nav, f, names=True)
            else:
                idx_entry = 0
                idx_exit = -1
                _entry = f['route_m1t'][idx_entry][0]
                _exit = f['route_m1t'][idx_exit][0]

            paras['flows'][(_entry, _exit)] = paras['flows'].get((_entry, _exit),[]) + [f['route_m1t'][0][1]]

        if not paras['bootstrap_mode']:
            #paras['departure_times'] = 'exterior'
            paras['ACtot'] = sum([len(v) for v in paras['flows'].values()])
            paras['control_density'] = False
        else:
            if not 'ACtot' in paras.keys():
                paras['ACtot'] = sum([len(v) for v in paras['flows'].values()])
           
        #print 'pouet' 
        #print paras['ACtot']
        density=_func_density_vs_ACtot_na_day(paras['ACtot'], paras['na'], paras['day'])

        # There is no update requisites here, because the traffic should not be changed
        # when it is extracted from data.

    else:
        paras['flows'] = {}
        paras['times'] = []
        if paras['file_times'] != None:
            if paras['departure_times']=='from_data': #TODO
                with open('times_2010_5_6.pic', 'r') as f:
                    paras['times']=pickle.load(f)
        else:
            assert paras['departure_times'] in ['zeros','from_data','uniform','square_waves']

            if paras['departure_times']=='square_waves':
                paras['Np'] = _func_Np(paras['day'], paras['width_peak'], paras['Delta_t'])
                to_update['Np']=(_func_Np,('day', 'width_peak', 'Delta_t'))
                update_priority.append('Np')

                if paras['control_ACsperwave']:
                    # density/ACtot based on ACsperwave
                    # Could also swap computation of density and ACtot here.
                    paras['density'] = _func_density_vs_ACsperwave_Np_na_day(paras['ACsperwave'], paras['Np'], paras['ACtot'], paras['na'], paras['day'])
                    to_update['density'] = (_func_density_vs_ACsperwave_Np_na_day,('ACsperwave', 'Np', 'ACtot', 'na', 'day'))
                    update_priority.append('density')   
                    paras['ACtot']=_func_ACtot_vs_density_day_na(paras['density'], paras['day'], paras['na'])
                    to_update['ACtot']=(_func_ACtot_vs_density_day_na, ('density', 'day', 'na'))
                    update_priority.append('ACtot')
                else:
                    # ACperwave based on density/ACtot
                    paras, to_update, update_priority = control_density_ACtot(paras, to_update, update_priority)
                    # The following could also be computed based on ACtot. At this stage, both ACtot and density
                    # should be up to date.
                    paras['ACsperwave'] =_func_ACsperwave_vs_density_day_Np(paras['density'], paras['day'], paras['Np'])
                    to_update['ACsperwave'] = (_func_ACsperwave_vs_density_day_Np,('density', 'day','Np'))
                    update_priority.append('ACsperwave')
            else:
                paras, to_update, update_priority = control_density_ACtot(paras, to_update, update_priority)

    # --------------- Network stuff --------------#
    # if paras['G']!=None:
    #     paras['G'].choose_short(paras['Nsp_nav'])

    # Expand or reduce capacities:
    if paras['capacity_factor']!=1.:
        for n in paras['G'].nodes():
            paras['G'].node[n]['capacity'] = int(paras['G'].node[n]['capacity']*paras['capacity_factor'])
            #print "Capacity sector", n, ":", paras['G'].node[n]['capacity']

    # ------------------- From M0 to M1 ----------------------- #
    if paras['mode_M1'] == 'standard':
        paras['STS'] = None
    else: 
        paras['N_shocks'] = 0

    paras['N_shocks'] = int(paras['N_shocks'])

    # ------------ Building of AC --------------- #

    def _func_AC(a, b):
        return [int(a*b),b-int(a*b)]  

    paras['AC']=_func_AC(paras['nA'], paras['ACtot'])               #number of air companies/operators

    def _func_AC_dict(a, b, c):
        if c[0]==c[1]:
            return {c[0]:int(a*b)}
        else:
            return {c[0]:int(a*b), c[1]:b-int(a*b)}  

    paras['AC_dict']=_func_AC_dict(paras['nA'], paras['ACtot'], paras['par'])                #number of air companies/operators


    # ------------ Building paras dictionary ---------- #

    paras.to_update = to_update

    paras.to_update['AC'] = (_func_AC,('nA', 'ACtot'))
    paras.to_update['AC_dict'] = (_func_AC_dict,('nA', 'ACtot', 'par'))

    # Add update priority here

    update_priority.append('AC')
    update_priority.append('AC_dict')

    paras.update_priority = update_priority

    paras.analyse_dependance()

    return paras

# ============================================================================ #

"""
Functions of dependance between variables.
"""
def _func_density_vs_ACtot_na_day(ACtot, na, day):
    """
    Used to compute density when ACtot, na or day are variables.
    """
    return ACtot*na/float(day/60.)

def _func_density_vs_ACsperwave_Np_na_day(ACsperwave, Np, ACtot, na, day):
    ACtot = _func_ACtot_vs_ACsperwave_Np(ACsperwave, Np)
    return _func_density_vs_ACtot_na_day(ACtot, na, day)

def _func_ACtot_vs_ACsperwave_Np(ACsperwave, Np):
    """
    Used to compute ACtot when ACsperwave or Np are variables.
    """
    return int(ACsperwave*Np)

def _func_ACsperwave_vs_density_day_Np(density, day, Np):
    """
    Used to compute ACsperwave when density, day or Np are variables.
    """
    return int(float(density*day/60.)/float(Np))

def _func_ACtot_vs_density_day_na(density, day, na):
    """
    Used to compute ACtot when density, day or na are variables.
    """
    return int(density*(day/60.)/float(na))

def _func_Np(day, width_peak, Delta_t):
    """
    Used to compute Np based on width of waves, duration of day and 
    time between the end of a wave and the beginning of the next wave.
    """
    return int(ceil(day/float(width_peak+Delta_t)))

# ============================================================================ #

def total_static_overlap(paths):
    """
    Compute the overlap between a list of paths.

    Parameters
    ----------
    paths : list
        of pretty much anything which can be put in a set.

    Returns
    -------
    mean_overlap : float,
        average of (number of common elements/ all elements) between pairs
        of paths.

    Notes
    -----
    Probably equal to the (mean) Jacquard index between pairs.

    """
    
    overlaps = []
    for i, p1 in enumerate(paths):
        for j, p2 in enumerate(paths):
            if i<j:
                common_elements = float(len(set(p1).intersection(set(p2))))
                all_elements = float(len(set(p1).union(set(p2))))
                overlaps.append(common_elements/all_elements)

    return np.mean(overlaps)