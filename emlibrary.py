#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:10:11 2021

@author: kyobi
"""

import numpy as np
import datetime
import pandas as pd
import time
from meshparty import meshwork, skeletonize, skeleton_io, skeleton, trimesh_io, trimesh_vtk

def get_matched_inter_datasheet(segIDs,ds):
    client = load_fanc_client()
    for ii,val in enumerate(segIDs):
        print(ii)
        cond_1 = client.chunkedgraph.is_latest_roots([val])[0]
        cond_2 = ds.SegIDs.isin(client.chunkedgraph.get_past_ids\
                             ([val])['past_id_map'][val])
        if cond_1 & ~np.any(cond_2):
            print('Seg not found in spreadsheet -- '+str(val))
        elif np.any(cond_2):
            print('Updated segment ID from '+str(ds.loc[cond_2,:]['SegIDs'])+\
                  'to '+str(val))
            ds.loc[cond_2,:]['SegIDs'] = val
        ds['specific'] = ds['specific'].astype(str)
    return ds

def get_graphs(matrix):
    G = matrix
    
    A = (G > 0).astype(int)   
    max_steps = 5
    D, R = distance_and_reachability_matrices(A, max_steps)   
    
    k_den = A.unstack().sum() / A.size   
    J = joint_degree_matrix(A,30)
    
    M_in = matching_matrix(A,1)
    M_out = matching_matrix(A,0) 
    M_all = M_out - M_in  
    rec = reciprocal_matrix(A)   
    analysis = {
        'G'             : G,
        'A'             : A,
        'D'             : D,
        'J'             : J,
        'M_in'          : M_in,
        'M_out'         : M_out,
        'M_all'         : M_all,
        'reciprocal'    : rec,   
        'k_den'         : k_den,
        'R'             : R
        }

    return analysis

def load_fanc_client():
    
    from caveclient import CAVEclient
    datastack_name = 'fanc_production_mar2021'
    client = CAVEclient(datastack_name)
    return client

def get_matrix_from_df(df,input_ids,output_ids):
    
    a = df.pre_pt_root_id[df.order==0]
    b = pd.Series(input_ids)
    c = df.pre_pt_root_id[df.order==2]
    d = pd.Series(output_ids)
    
    ids = pd.concat([a,b,c,d]).drop_duplicates().reset_index(drop=True)
    
    edges = np.zeros(len(ids))
    edges[np.isin(ids,a)] = 0
    edges[np.isin(ids,b)] = 1
    edges[np.isin(ids,c)] = 2
    edges[np.isin(ids,d)] = 3
    
    matrix = pd.DataFrame(np.zeros([len(ids),len(ids)]),index=ids,\
                               columns=ids)
    for ii in range(len(df)):
        x = df.iloc[ii,:]
        if (sum(x.post_pt_root_id == ids) == 1):
            matrix.loc[x.pre_pt_root_id,x.post_pt_root_id] = x.syn_in_conn
    
    analysis = {
        'matrix' : matrix,
        'edges' : edges
        }
    return analysis


def get_latest_roots(neuron_x):
    client = load_fanc_client()
    if ~client.chunkedgraph.is_latest_roots([neuron_x])[0]:
        self_leaves = client.chunkedgraph.get_leaves(neuron_x)
        new_roots = client.chunkedgraph.get_latest_roots(neuron_x)
        def func(x):
            return len(np.intersect1d(self_leaves,client.chunkedgraph.get_leaves(x)))
        updated_root = new_roots[np.argmax(list(map(func,new_roots)))]
        return updated_root
    else:
        return neuron_x
        

def get_connections(input_ids,output_ids,synapse_cutoff,score_bounds,frac_connected,num_steps):
    
    zeroeth_connections = get_input_connections(input_ids,output_ids,\
                                                synapse_cutoff,score_bounds,\
                                                    frac_connected,num_steps)
    client = load_fanc_client()
    soma_table = client.materialize.live_query('soma_aug2021',datetime.datetime.now())
    all_connections = []
    all_connections.append(zeroeth_connections['df'])
    all_fractions_connected = []
    all_fractions_connected.append(zeroeth_connections['fraction_connected'])
    all_total_synapses = []
    all_total_synapses.append(zeroeth_connections['total_output_synapses'])
    ids = input_ids
    for ii in range(num_steps):
        time.sleep(5)
        connections = []
        chunk_size = 30
        for count in range(1,np.ceil(ids.shape[0]/chunk_size).astype(int)+1):
            connections.append(client.materialize.synapse_query(\
                pre_ids=ids[((count-1)*chunk_size):(count*chunk_size)],\
                    timestamp=datetime.datetime.utcnow())[['pre_pt_root_id',\
                                                                    'post_pt_root_id','score']])
                                                           
        connections = pd.concat(connections)                                                                             
        connections = apply_score_cutoff(connections,score_bounds)
        total_synapses = get_total_synapses(connections)
        connections = get_syn_in_conn(connections)
        connections = apply_synapse_cutoff(connections,synapse_cutoff)                                                               

        fraction_connected,has_soma_or_MN = get_fraction_connected(connections,\
                                                    soma_table,output_ids,\
                                                        total_synapses)

        # has_connections = connections.pre_pt_root_id.\
        #                                           isin(fraction_connected.\
        #                                                index[fraction_connected\
        #                                                      > frac_connected])
        connections['order'] = ii+1      
        print('proportion of connections passing soma/MN filter:' + \
              str(has_soma_or_MN.sum()/len(connections)))                                                                      
        connections = connections.loc[has_soma_or_MN,:]
        all_connections.append(connections)
        all_fractions_connected.append(fraction_connected)
        all_total_synapses.append(total_synapses)                  
        ids = connections['post_pt_root_id'].values
        ids = ids[~np.isin(ids,output_ids)]
   
    df = pd.concat(all_connections).drop_duplicates().reset_index(drop=True)
    all_fracs = pd.concat(all_fractions_connected)
    all_synapses = pd.concat(all_total_synapses)
    analysis = {
        'df'    :  df,
        'fraction_connected'   :   all_fracs,
        'total_output_synapses' : all_synapses
        }
    return analysis                                                                                                             


def get_syn_in_conn(df):
    syn_in_conn = df.groupby(['pre_pt_root_id','post_pt_root_id']).\
        transform(len)
    df['syn_in_conn'] = syn_in_conn
    return df

def apply_synapse_cutoff(df,synapse_cutoff):
    df = df[['pre_pt_root_id','post_pt_root_id','syn_in_conn']].\
        loc[(df['syn_in_conn'] > synapse_cutoff),:].\
        drop_duplicates().reset_index(drop=True)
    return df

def apply_score_cutoff(df,score_bounds):
    df = df.loc[((df.score>=score_bounds[0]) & (df.score<=score_bounds[1])),:]
    return df

def get_total_synapses(df):
    df = df.groupby('pre_pt_root_id').\
        aggregate(len).rename(columns={'post_pt_root_id':'num_synapses'})\
            ['num_synapses']       
    return df                                    

def get_fraction_connected(df,soma_table,output_ids,total_synapses):
    has_soma_or_MN = df['post_pt_root_id'].isin(soma_table['pt_root_id']) | \
        df['post_pt_root_id'].isin(output_ids)
    df['has_soma_or_MN'] = has_soma_or_MN
    synapses_with_soma_or_MN = df.loc[has_soma_or_MN,:].groupby('pre_pt_root_id').\
        aggregate(sum)['syn_in_conn']
    fraction_connected = (synapses_with_soma_or_MN/total_synapses).fillna(0)
    return fraction_connected,has_soma_or_MN

def get_input_connections(input_ids,output_ids,synapse_cutoff,score_bounds,frac_connected,num_steps):
    client = load_fanc_client()
    soma_table = client.materialize.live_query('soma_aug2021',datetime.datetime.now())

    ids = input_ids

    time.sleep(5)
    connections = []
    chunk_size = 30
    for count in range(1,np.ceil(ids.shape[0]/chunk_size).astype(int)+1):
        connections.append(client.materialize.synapse_query(\
            post_ids=ids[((count-1)*chunk_size):(count*chunk_size)],\
                timestamp=datetime.datetime.utcnow())[['pre_pt_root_id',\
                                                                'post_pt_root_id','score']])
                                                       
    connections = pd.concat(connections)                                                                    
    connections = apply_score_cutoff(connections,score_bounds)
    connections = get_syn_in_conn(connections)
    connections = apply_synapse_cutoff(connections,synapse_cutoff)

    zeroeth_ids = np.array(connections.pre_pt_root_id.drop_duplicates().reset_index(drop=True))
    zeroeth_connections = client.materialize.synapse_query(\
        pre_ids=zeroeth_ids,\
            timestamp=datetime.datetime.utcnow())[['pre_pt_root_id',\
                                                            'post_pt_root_id','score']]
    zeroeth_connections = apply_score_cutoff(zeroeth_connections,score_bounds)
    total_zeroeth_synapses = get_total_synapses(zeroeth_connections)
    zeroeth_connections = get_syn_in_conn(zeroeth_connections)
    zeroeth_connections = apply_synapse_cutoff(zeroeth_connections,synapse_cutoff)
     
    fraction_connected,has_soma_or_MN = get_fraction_connected(zeroeth_connections,\
                                                soma_table,output_ids,\
                                                    total_zeroeth_synapses)

    # has_connections = connections.pre_pt_root_id.\
    #                                           isin(fraction_connected.index[fraction_connected > \
    #                                                                         frac_connected])
  
    connections = connections.loc[has_soma_or_MN,:]       
    connections['order'] = 0             
        
    df = connections.drop_duplicates().reset_index(drop=True)
    analysis = {
        'df'    :  df,
        'fraction_connected'   :   fraction_connected,
        'total_output_synapses' : total_zeroeth_synapses
        }
    return analysis                                            

    
def joint_degree_matrix(A,max_steps):
    in_deg = A.sum(axis=1)
    out_deg = A.sum(axis=0)
    df = pd.concat([in_deg,out_deg],axis=1).fillna(0)
    matrix = pd.DataFrame(np.zeros([max_steps,max_steps]))
    for t in range(matrix.shape[0]):
        for u in range(matrix.shape[1]):
            matrix.iloc[t,u] = ((df[0] == t) & (df[1] == u)).sum()
    return matrix

def matching_matrix(A,direction):
    if direction == 1:
        axis_on = 1
        A = A.T
    else:
        axis_on = 0
    matrix = pd.DataFrame(np.zeros([A.shape[axis_on],A.shape[axis_on]]),index=A.index,columns=A.columns)
    for i in range(matrix.shape[axis_on]):
        for j in range(matrix.shape[axis_on]):
            if (A.iloc[i,:] | A.iloc[j,:]).sum() > 0:
                matrix.iloc[i,j] = (A.iloc[i,:] & A.iloc[j,:]).sum() / (A.iloc[i,:] | A.iloc[j,:]).sum()
    return matrix.fillna(0)
            
def reciprocal_matrix(A):

    matrix = pd.DataFrame(np.zeros(A.shape),index = A.index,columns = A.columns)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix.iloc[i,j] = int((A.iloc[i,j] == 1) & (A.iloc[j,i]==1))
    return matrix.fillna(0)


def distance_and_reachability_matrices(A,max_steps):
    matrix = pd.DataFrame(np.zeros(A.shape),index = A.index,columns = A.columns)
    for tt in range(max_steps):
        new_mat = np.linalg.matrix_power(A,tt)
        for ii in range(matrix.shape[0]):
            for jj in range(matrix.shape[1]):
                if (matrix.iloc[ii,jj] == 0) & new_mat[ii,jj] == 1:
                    matrix.iloc[ii,jj] = tt
                    
    return matrix, (matrix > 0).astype(int)

def number_of_cells_matrix(df,A,max_steps):  # thjis is not a function right now
    matrix = pd.DataFrame(np.zeros(A.shape),index = A.index,columns = A.columns)
    for tt in range(max_steps):
        new_mat = np.linalg.matrix_power(A,tt)
        for ii in range(matrix.shape[0]):
            for jj in range(matrix.shape[1]):
                if (matrix.iloc[ii,jj] == 0) & new_mat[ii,jj] == 1:
                    matrix.iloc[ii,jj] = tt
                    
    return matrix, (matrix > 0).astype(int)
    
def get_top_connections(input_id,synapse_cutoff):
    client = load_fanc_client()
    x = client.materialize.synapse_query(pre_ids=input_id).groupby('post_pt_root_id').count()['score']
    y = x[x>synapse_cutoff]
    return y

def separate_datasheet_label(analysis):
    analysis['input_data_sheet'][['broad','specific','side']] = \
        analysis['input_data_sheet']['Cell type'].str.split('_',expand=True)
    return analysis
            

    
    
    
    
    
        