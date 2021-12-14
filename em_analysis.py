#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:38:23 2021

@author: kyobi
"""
from caveclient import CAVEclient
import numpy as np
import pandas as pd
import datetime
import emlibrary as lib
import plotlibrary as emplt
import matplotlib.pyplot as plt
import pickle

synapse_cutoff = 5 # minimum number of synapses in a connection
score_bounds = [30,100] # minimum distance between pre and postsynaptic membrane
frac_connected = 0.05 # minimum fraction of synapses that can be attributed to proper connections
number_of_steps = 3 # order of connections into brain (e.g. 1 is direct connections from DNs)

path = '/home/kyobi/em_analysis/meshes/'
output_path = '/home/kyobi/em_analysis/data/'
datastack_name = 'fanc_production_mar2021'
client = CAVEclient(datastack_name)
soma_table = client.materialize.live_query('soma_aug2021',datetime.datetime.now())

sheet_id = "12Iyt-jARNPMLGWooUxRSn5XmWY9n6X6S7eU4oTNcPxg"
sheet_name = "dDNs"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data_sheet = pd.read_csv(url)
which_input_cells = data_sheet['Cell type'].str.contains('') & \
    data_sheet['Cell type'].str.contains('')
data_sheet = data_sheet.loc[which_input_cells,:]
data_sheet['SegIDs'] = data_sheet['SegIDs'].apply(lib.get_latest_roots)
input_ids = np.array(data_sheet['SegIDs'])

sheet_id = "12Iyt-jARNPMLGWooUxRSn5XmWY9n6X6S7eU4oTNcPxg"
sheet_name = "Motor_Neurons"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
second_data_sheet = pd.read_csv(url)
second_data_sheet['id'] = second_data_sheet['id'].fillna(0).astype(int)
second_data_sheet['SegIDs'] = second_data_sheet['SegIDs'].apply(lib.get_latest_roots)
sorted_ds = second_data_sheet.sort_values(by=['Nerve Side','Tract','Cell type'])
which_cells = sorted_ds['Cell type'].str.contains('')
output_ids = np.array(sorted_ds[which_cells]['SegIDs'])

df,fraction_connected = lib.get_connections(input_ids,output_ids,synapse_cutoff,score_bounds,frac_connected,number_of_steps)

matrix = lib.get_matrix_from_df(df,input_ids,output_ids)

analysis = lib.get_graphs(matrix)

fname = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.datetime.utcnow())+"_graphs.pickle"
with open(output_path+fname,'wb') as fid:
    pickle.dump(analysis,fid)

#%% Plot the figures

fname = fname
analysis.load(fname,'rb')

edges = np.zeros(len(analysis['G']))
edges[np.isin(analysis['G'].index,input_ids)] = 0
edges[np.isin(analysis['G'].index,output_ids)] = 2
edges[~np.isin(analysis['G'].index,input_ids) & ~np.isin(analysis['G'].index,output_ids)] = 1    
    
fig_title = 'all_dns_L'
fig,axs = plt.subplots()
plt.subplot(231)
title = 'Connectivity Matrix'
colorbar_label = '# of synapses'
emplt.generate_graph_theory_heatmaps(analysis['G'],edges,title,colorbar_label,False)

plt.subplot(232)
title = 'Adjacency Matrix'
colorbar_label = 'Connection'
emplt.generate_graph_theory_heatmaps(analysis['A'], edges, title, colorbar_label,False)

plt.subplot(233)
title = 'Matching Output Matrix'
colorbar_label = 'Proportion of Shared\nOutput Neurons'
emplt.generate_graph_theory_heatmaps(analysis['M_out'], edges, title, colorbar_label,False)

plt.subplot(234)
title = 'Matching Input Matrix'
colorbar_label = 'Proportion of Shared\nInput Neurons'
emplt.generate_graph_theory_heatmaps(analysis['M_in'], edges, title, colorbar_label,False)

plt.subplot(236)
title = 'Reachability Matrix'
colorbar_label = 'Connected'
emplt.generate_graph_theory_heatmaps(analysis['R'], edges, title, colorbar_label,False)

plt.subplot(235)
title = 'Reciprocal Matrix'
colorbar_label = 'Connection'
emplt.generate_graph_theory_heatmaps(analysis['reciprocal'], edges, title, colorbar_label,False)

plt.tight_layout()
fig.suptitle(fig_title,fontsize=10)
plt.savefig(fig_title,backend='cairo',format='png')
    
emplt.make_hop_connectivity_plots(analysis['G'],analysis['D'],data_sheet,sorted_ds,number_of_steps)
