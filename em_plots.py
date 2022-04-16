#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:47:22 2021

@author: kyobikakaria
"""

import sys
import pickle
import matplotlib.pyplot as plt
import plotlibrary as emplt

#% Plot the graph figures
if sys.platform == 'linux1':
    path = '/home/kyobi/em_analysis/meshes/'
    input_path = '/home/kyobi/em_analysis/data/'
    figure_path = '/home/kyobi/em_analysis/images/'
elif sys.platform == 'darwin':
    path = '/Users/kyobikakaria/em_analysis/meshes/'
    input_path = '/Users/kyobikakaria/em_analysis/data/'
    figure_path = '/Users/kyobikakaria/em_analysis/images/'
fname = '2022_02_02_04_11_41_2step_analysis.pickle'

fpath = input_path+fname
save_datetime = fname[:19]
with open(fpath,'rb') as f:
    analysis = pickle.load(f)
    

dpi = 600

plt.figure(dpi=dpi)
title = 'Adjacency Matrix'
colorbar_label = 'Connection'
emplt.generate_graph_theory_heatmaps(analysis['A'], analysis['edges'], \
                                     title, colorbar_label, \
                                     figure_path+save_datetime+'_'+title+'.svg')
plt.figure(dpi=dpi)
title = 'Connectivity Matrix'
colorbar_label = '# of synapses'
emplt.generate_graph_theory_heatmaps(analysis['G'], analysis['edges'], \
                                     title, colorbar_label, \
                                     figure_path+save_datetime+'_'+title+'.svg')
plt.figure(dpi=dpi)
title = 'Matching Input Matrix'
colorbar_label = 'Proportion of Shared\nInput Neurons'
emplt.generate_graph_theory_heatmaps(analysis['M_in'], analysis['edges'], \
                                     title, colorbar_label, \
                                     figure_path+save_datetime+'_'+title+'.svg')
plt.figure(dpi=dpi)
title = 'Matching Output Matrix'
colorbar_label = 'Proportion of Shared\nOutput Neurons'
emplt.generate_graph_theory_heatmaps(analysis['M_out'], analysis['edges'], \
                                     title, colorbar_label, \
                                     figure_path+save_datetime+'_'+title+'.svg')
plt.figure(dpi=dpi)
title = 'Reachability Matrix'
colorbar_label = 'Connected'
emplt.generate_graph_theory_heatmaps(analysis['R'], analysis['edges'], \
                                     title, colorbar_label, \
                                     figure_path+save_datetime+'_'+title+'.svg')
plt.figure(dpi=dpi)
title = 'Reciprocal Matrix'
colorbar_label = 'Connection'
emplt.generate_graph_theory_heatmaps(analysis['reciprocal'], analysis['edges'], \
                                     title, colorbar_label, \
                                     figure_path+save_datetime+'_'+title+'.svg')


emplt.make_hop_connectivity_plots(analysis['G'],analysis['D'],analysis['input_data_sheet'],\
                                  analysis['output_data_sheet'],analysis['number_of_steps'])
fig_title = 'all_dns_hop_analysis.svg'
plt.tight_layout()
plt.savefig(figure_path+save_datetime+'_'+fig_title,backend='cairo',format='svg')

emplt.generate_neuron_bargraph(analysis)
fig_title = 'dn_bargraph.svg'
plt.tight_layout()
plt.savefig(figure_path+save_datetime+'_'+fig_title,backend='cairo',format='svg')

emplt.generate_convergence_bargraph(analysis)
fig_title = 'mn_convergence_bargraph.svg'
plt.tight_layout()
plt.savefig(figure_path+save_datetime+'_'+fig_title,backend='cairo',format='svg')

emplt.generate_divergence_bargraph(analysis)
fig_title = 'dn_divergence_bargraph.svg'
plt.tight_layout()
plt.savefig(figure_path+save_datetime+'_'+fig_title,backend='cairo',format='svg')


plt.figure(dpi=dpi)
title = 'inter_clustering_out'
colorbar_label = 'Proportion of Shared\nOutput Neurons'
ind = analysis['edges'] == 2
M = analysis['M_out'].loc[ind,ind]
emplt.generate_clustered_inter_graph(M, \
                                     title, \
                                     figure_path+save_datetime+'_'+title+'.svg',\
                                         analysis)
    
plt.figure(dpi=dpi)
title = 'inter_clustering_in'
colorbar_label = 'Proportion of Shared\nOutput Neurons'
ind = analysis['edges'] == 2
M = analysis['M_in'].loc[ind,ind]
emplt.generate_clustered_inter_graph(M, \
                                     title, \
                                     figure_path+save_datetime+'_'+title+'.svg',\
                                         analysis)
    
    

emplt.DN_activation_MN_plots(analysis,'G')
fig_title = 'dn_activation_maps_weighted.svg'
plt.tight_layout()
plt.savefig(figure_path+save_datetime+'_'+fig_title,backend='cairo',format='svg')

emplt.DN_activation_MN_plots(analysis,'A')
fig_title = 'dn_activation_maps_noweights.svg'
plt.tight_layout()
plt.savefig(figure_path+save_datetime+'_'+fig_title,backend='cairo',format='svg')