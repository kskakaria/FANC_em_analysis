#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:38:23 2021

@author: kyobi
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pickle
import sys
from caveclient import CAVEclient
import emlibrary as lib
import plotlibrary as emplt

#%% Hard coded section, all parameters to vary belong here.
synapse_cutoff = 5 # minimum number of synapses in a connection
score_bounds = [30,100] # minimum distance between pre and postsynaptic membrane
frac_connected = 0.05 # minimum fraction of synapses that can be attributed to proper connections
number_of_steps = 2 # order of connections into brain (e.g. 1 is direct connections from DNs)

if sys.platform == 'linux1':
    path = '/home/kyobi/em_analysis/meshes/'
    output_path = '/home/kyobi/em_analysis/data/'
    figure_path = '/home/kyobi/em_analysis/images/'
elif sys.platform == 'darwin':
    path = '/Users/kyobikakaria/em_analysis/meshes/'
    output_path = '/Users/kyobikakaria/em_analysis/data/'
    figure_path = '/Users/kyobikakaria/em_analysis/images/'


datastack_name = 'fanc_production_mar2021'
client = CAVEclient(datastack_name)
soma_table = client.materialize.live_query('soma_aug2021',datetime.datetime.now())

sheet_id = "12Iyt-jARNPMLGWooUxRSn5XmWY9n6X6S7eU4oTNcPxg"
sheet_name = "dDNs"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
data_sheet = pd.read_csv(url)
which_input_cells = data_sheet['Cell type'].str.contains('DNg02') & \
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
second_data_sheet['needs_update'] = ~client.chunkedgraph.\
    is_latest_roots(np.array(second_data_sheet['SegIDs']))
second_data_sheet['SegIDs'].loc[second_data_sheet['needs_update']] = \
    second_data_sheet['SegIDs'].loc[second_data_sheet\
                                    ['needs_update']].apply(lib.get_latest_roots)
sorted_ds = second_data_sheet.sort_values(by=['side','Tract','specific'])
which_cells = sorted_ds['specific'].str.contains('')
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
        'frac_connected'        :   frac_connected,
        'output_path'           :   output_path,
        'figure_path'           :   figure_path
        }

analysis.update(lib.get_connections(analysis['input_ids'],analysis['output_ids'],\
                                    analysis['synapse_cutoff'],analysis['score_bounds'],\
                                        analysis['frac_connected'],analysis['number_of_steps']))
analysis.update(lib.get_matrix_from_df(analysis['df'],analysis['input_ids'],analysis['output_ids']))
analysis.update(lib.get_graphs(analysis['matrix']))
analysis = lib.separate_datasheet_label(analysis)

sheet_id = "12Iyt-jARNPMLGWooUxRSn5XmWY9n6X6S7eU4oTNcPxg"
sheet_name = "Interneurons"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
inter_data_sheet = pd.read_csv(url)
updated_interneurons = analysis['A'].index[analysis['edges']==2]
in_ds = updated_interneurons.isin(inter_data_sheet.SegIDs)

inter_data_sheet['needs_update'] = ~client.chunkedgraph.\
    is_latest_roots(np.array(inter_data_sheet['SegIDs']))
inter_data_sheet['SegIDs'].loc[inter_data_sheet['needs_update']] = \
    inter_data_sheet['SegIDs'].loc[inter_data_sheet\
                                    ['needs_update']].apply(lib.get_latest_roots)
inter_ds = inter_data_sheet.sort_values(by=['side','specific','subtype'])

analysis['inter_data_sheet'] = inter_ds
    
fname = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.datetime.utcnow())+'_'+\
    str(analysis['number_of_steps'])+"step_analysis.pickle"
with open(output_path+fname,'wb') as fid:
    pickle.dump(analysis,fid)


    
