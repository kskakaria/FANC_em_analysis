#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:48:07 2022

@author: kyobikakaria
"""

import sys
import pickle
import plotlibrary as emplt

#%% Plot the graph figures
if sys.platform == 'linux1':
    path = '/home/kyobi/em_analysis/meshes/'
    input_path = '/home/kyobi/em_analysis/data/'
    figure_path = '/home/kyobi/em_analysis/meshes/'
elif sys.platform == 'darwin':
    path = '/Users/kyobikakaria/em_analysis/meshes/'
    input_path = '/Users/kyobikakaria/em_analysis/data/'
    figure_path = '/Users/kyobikakaria/em_analysis/meshes/'
fname = '2022_02_02_04_11_41_2step_analysis.pickle'

fpath = input_path+fname
save_datetime = fname[:19]
with open(fpath,'rb') as f:
    analysis = pickle.load(f)
    

#%% Draw each class of DN
        
# ds = analysis['input_data_sheet']
# classes = ds['specific'].drop_duplicates().values
# for ii,value in enumerate(classes):
#     # idx = (ds['specific']==value) & (ds['side']=='L')
#     # emplt.generate_neuron_mesh(ds['SegIDs'][idx],figure_path+save_datetime+\
#     #                            '_'+value+'_L',ds)
#     # idx = (ds['specific']==value) & (ds['side']=='R')
#     # emplt.generate_neuron_mesh(ds['SegIDs'][idx],figure_path+save_datetime+\
#     #                            '_'+value+'_R',ds)
#     idx = (ds['specific']==value)
#     emplt.generate_neuron_mesh(ds['SegIDs'][idx],figure_path+save_datetime+\
#                                '_'+value,ds)
        

#%% Draw each class of IN

ds = analysis['inter_data_sheet']
classes = ds['specific'].drop_duplicates().values
for ii,value in enumerate(classes):
    # idx = (ds['Hemilineage']==value) & (ds['side']=='L')
    # emplt.generate_neuron_mesh(ds['SegIDs'][idx],figure_path+save_datetime+\
    #                            '_'+value+'_L',ds)
    # idx = (ds['Hemilineage']==value) & (ds['side']=='R')
    # emplt.generate_neuron_mesh(ds['SegIDs'][idx],figure_path+save_datetime+\
    #                            '_'+value+'_R',ds)
    idx = (ds['specific']==value)
    emplt.generate_neuron_mesh(ds['SegIDs'][idx],figure_path+save_datetime+\
                                '_'+value,ds)
    

# #%% Draw each class of MN

# ds = analysis['output_data_sheet']
# classes = ds['specific'].drop_duplicates().values
# for ii,value in enumerate(classes):
#     if ii > 13:
#         # idx = (ds['specific']==value) & (ds['side']=='L')
#         # emplt.generate_neuron_mesh(ds['SegIDs'][idx],figure_path+save_datetime+\
#         #                            '_'+value+'_L',ds)
#         # idx = (ds['specific']==value) & (ds['side']=='R')
#         # emplt.generate_neuron_mesh(ds['SegIDs'][idx],figure_path+save_datetime+\
#         #                            '_'+value+'_R',ds)
#         idx = (ds['specific']==value)
#         emplt.generate_neuron_mesh(ds['SegIDs'][idx],figure_path+save_datetime+\
#                                     '_'+value+'png',ds)