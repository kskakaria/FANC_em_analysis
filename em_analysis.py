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
number_of_steps = 2 # order of connections into brain (e.g. 1 is direct connections from DNs)

path = '/home/kyobi/em_analysis/meshes/'
output_path = '/home/kyobi/em_analysis/data/'
figure_path = '/home/kyobi/em_analysis/images/'
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
data_sheet['needs_update'] = ~client.chunkedgraph.is_latest_roots(np.array(data_sheet['SegIDs']))
data_sheet['SegIDs'].loc[data_sheet['needs_update']] = \
    data_sheet['SegIDs'].loc[data_sheet['needs_update']].apply(lib.get_latest_roots)
input_ids = np.array(data_sheet['SegIDs'])

sheet_id = "12Iyt-jARNPMLGWooUxRSn5XmWY9n6X6S7eU4oTNcPxg"
sheet_name = "Motor_Neurons"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
second_data_sheet = pd.read_csv(url)
second_data_sheet['id'] = second_data_sheet['id'].fillna(0).astype(int)
second_data_sheet['needs_update'] = ~client.chunkedgraph.is_latest_roots(np.array(second_data_sheet['SegIDs']))
second_data_sheet['SegIDs'].loc[second_data_sheet['needs_update']] = \
    second_data_sheet['SegIDs'].loc[second_data_sheet['needs_update']].apply(lib.get_latest_roots)
sorted_ds = second_data_sheet.sort_values(by=['Nerve Side','Tract','Cell type'])
which_cells = sorted_ds['Cell type'].str.contains('')
output_ids = np.array(sorted_ds[which_cells]['SegIDs'])


#%% Run analysis

analysis = {
        'input_ids'             :   input_ids,
        'output_ids'            :   output_ids,
        'input_data_sheet'      :   data_sheet,
        'output_data_sheet'     :   sorted_ds,
        'number_of_steps'       :   number_of_steps,
        'synapse_cutoff'        :   synapse_cutoff,
        'score_bounds'          :   score_bounds,
        'frac_connected'        :   frac_connected
        }

analysis.update(lib.get_connections(analysis['input_ids'],analysis['output_ids'],\
                                    analysis['synapse_cutoff'],analysis['score_bounds'],\
                                        analysis['frac_connected'],analysis['number_of_steps']))
analysis.update(lib.get_matrix_from_df(analysis['df'],analysis['input_ids'],analysis['output_ids']))
analysis.update(lib.get_graphs(analysis['matrix']))
    
fname = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.datetime.utcnow())+'_'+str(analysis['number_of_steps'])+"step_analysis.pickle"
with open(output_path+fname,'wb') as fid:
    pickle.dump(analysis,fid)

#%% Plot the graph figures

fname = '2021_12_16_23_51_26_1step_analysis.pickle'
fpath = output_path+fname
save_datetime = fname[:19]
with open(fpath,'rb') as f:
    analysis = pickle.load(f)
    

fig,axs = plt.subplots()
plt.subplot(231)
title = 'Connectivity Matrix'
colorbar_label = '# of synapses'
emplt.generate_graph_theory_heatmaps(analysis['G'], analysis['edges'], title, colorbar_label, False)

plt.subplot(232)
title = 'Adjacency Matrix'
colorbar_label = 'Connection'
emplt.generate_graph_theory_heatmaps(analysis['A'], analysis['edges'], title, colorbar_label, False)

plt.subplot(233)
title = 'Matching Output Matrix'
colorbar_label = 'Proportion of Shared\nOutput Neurons'
emplt.generate_graph_theory_heatmaps(analysis['M_out'], analysis['edges'], title, colorbar_label, False)

plt.subplot(234)
title = 'Matching Input Matrix'
colorbar_label = 'Proportion of Shared\nInput Neurons'
emplt.generate_graph_theory_heatmaps(analysis['M_in'], analysis['edges'], title, colorbar_label,False)

plt.subplot(236)
title = 'Reachability Matrix'
colorbar_label = 'Connected'
emplt.generate_graph_theory_heatmaps(analysis['R'], analysis['edges'], title, colorbar_label,False)

plt.subplot(235)
title = 'Reciprocal Matrix'
colorbar_label = 'Connection'
emplt.generate_graph_theory_heatmaps(analysis['reciprocal'], analysis['edges'], title, colorbar_label,False)

plt.tight_layout()
fig_title = 'all_dns_matrices'
fig.suptitle(fig_title,fontsize=10)
plt.savefig(figure_path+save_datetime+'_'+fig_title,backend='cairo',format='png')

emplt.make_hop_connectivity_plots(analysis['G'],analysis['D'],analysis['input_data_sheet'],\
                                  analysis['output_data_sheet'],analysis['number_of_steps'])
fig_title = 'all_dns_hop_analysis'
plt.savefig(figure_path+save_datetime+'_'+fig_title,backend='cairo',format='png')

emplt.generate_neuron_bargraph(analysis['input_data_sheet']['Cell type'])
fig_title = 'dn_bargraph'
plt.savefig(figure_path+save_datetime+'_'+fig_title,backend='cairo',format='png')



#%% Draw mesh images of interneurons

canonical_segIDs = analysis['input_data_sheet'][['SegIDs','Cell type']].\
    loc[analysis['input_data_sheet']['Cell type'].str.contains('DNg02_canonical')]
    
canonical_segIDs = canonical_segIDs['SegIDs'].values

emplt.generate_neuron_mesh(canonical_segIDs,figure_path+save_datetime+'_canonical')

#%%
segID = canonical_segIDs['SegIDs'].values[1]
idx = analysis['M_out'].loc[segID,:] > 0.5
segIDs = analysis['A'].index[idx]

emplt.generate_neuron_mesh(segIDs,figure_path+save_datetime+'_'+str(segID))



