#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:10:11 2021

@author: kyobi
"""

import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_fanc_client():
    
    from caveclient import CAVEclient
    datastack_name = 'fanc_production_mar2021'
    client = CAVEclient(datastack_name)
    return client

def get_matrix_from_df(df,input_ids,output_ids):
    
    ids = pd.concat([df.pre_pt_root_id.drop_duplicates(),pd.Series(output_ids)])
    matrix = pd.DataFrame(np.zeros([len(ids),len(ids)]),index=ids,\
                               columns=ids)
    for ii in range(len(df)):
        x = df.iloc[ii,:]
        if (sum(x.post_pt_root_id == ids) == 1):
            matrix.loc[x.pre_pt_root_id,x.post_pt_root_id] = x.syn_in_conn
        
    return matrix
        

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
        

def get_connections(input_ids,output_ids,synapse_cutoff,min_score,frac_connected,num_steps):
    client = load_fanc_client()
    soma_table = client.materialize.live_query('soma_aug2021',datetime.datetime.now())
    all_connections = []
    all_fractions_connected = []
    ids = input_ids
    for ii in range(num_steps):
        connections = []
        chunk_size = 30
        for count in range(1,np.ceil(ids.shape[0]/chunk_size).astype(int)+1):
            connections.append(client.materialize.synapse_query(\
                pre_ids=ids[((count-1)*chunk_size):(count*chunk_size)],\
                    timestamp=datetime.datetime.utcnow())[['pre_pt_root_id',\
                                                                    'post_pt_root_id','score']])
        connections = pd.concat(connections)                                                                    
        connections = connections.loc[connections.score>min_score,:]
        total_synapses = connections.groupby('pre_pt_root_id').\
            aggregate(len).rename(columns={'post_pt_root_id':'num_synapses'})['num_synapses']        
        syn_in_conn = connections.groupby(['pre_pt_root_id','post_pt_root_id']).\
            transform(len)           
        connections['syn_in_conn'] = syn_in_conn
        connections = connections[['pre_pt_root_id','post_pt_root_id','syn_in_conn']].\
            loc[(syn_in_conn > synapse_cutoff).iloc[:,0],:].\
            drop_duplicates().reset_index(drop=True)        
        has_soma_or_MN = connections['post_pt_root_id'].isin(soma_table['pt_root_id']) | \
            connections['post_pt_root_id'].isin(output_ids)
        connections['has_soma_or_MN'] = has_soma_or_MN
        connections['order'] = ii        
        synapses_with_soma_or_MN = connections.loc[has_soma_or_MN,:].groupby('pre_pt_root_id').\
            aggregate(sum)['syn_in_conn']   
        fraction_connected = (synapses_with_soma_or_MN/total_synapses).fillna(0)
        has_connections = connections.pre_pt_root_id.\
                                                  isin(fraction_connected.index[fraction_connected > \
                                                                                frac_connected])
        connections = connections.loc[has_soma_or_MN & has_connections,:]       
        all_connections.append(connections)
        all_fractions_connected.append(fraction_connected)                  
        ids = connections['post_pt_root_id'].values
        ids = ids[~np.isin(ids,output_ids)]
        
    df = pd.concat(all_connections).drop_duplicates().reset_index(drop=True)
    return df                                                                                                                 

    
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
    matrix = pd.DataFrame(np.zeros([A.shape[axis_on],A.shape[axis_on]]))
    for i in range(matrix.shape[axis_on]):
        for j in range(matrix.shape[axis_on]):
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
    
def get_top_connections(input_id,synapse_cutoff):
    client = load_fanc_client()
    x = client.materialize.synapse_query(pre_ids=input_id).groupby('post_pt_root_id').count()['score']
    y = x[x>synapse_cutoff]
    return y

def generate_graph_theory_heatmaps(M,edges,title,colorbar_label,cluster):
    if cluster:
        clus_bool = True
    else:
        clus_bool = False
    if title == 'Distance Matrix':
        cmap = 'Greys'
    elif title == 'Net Matching Matrix':
        cmap='PiYG'
    else:
        cmap = 'Greys_r'
        
    ax = plt.imshow(M,cmap=cmap)
    cbar = plt.colorbar(shrink=0.35,aspect=5,use_gridspec=True,location='left')   
    cbar.set_label(label=colorbar_label,size=6)       
    cbar.ax.tick_params(labelsize=3)   
    plt.title(title,{'fontsize':8})
    # ax.axes.set_xlabel('Cell types')
    # ax.axes.set_ylabel('Cell types')
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])   
    x_ticks = [np.mean(edges[0:2]),np.mean(edges[2:4]),np.mean(edges[4:6])]
    y_ticks = x_ticks  
    colors = ['red','green','blue']   
    for ii in range(len(edges)):      
        col = colors[int(edges[ii])]       
        ax.axes.add_patch(patches.Rectangle((-10,ii), width=10, height=1,color=col,\
                                            clip_on=False))      
        ax.axes.add_patch(patches.Rectangle((ii,len(M)), height=10, width=1,color=col,\
                                            clip_on=False))   

def make_hop_connectivity_plots(G,D,input_ds,output_ds):
    import seaborn as sns
    fig_name = 'Hop analysis'
    def make_heatmap(data,idx):
        sns.heatmap(data,xticklabels=output_ds['Cell type'][idx],\
                    yticklabels=input_ds['Cell type'],vmax=vmax,cmap=cmap,cbar_kws=cbar_params)
        plt.gca().set_aspect('equal')
        plt.rc('font',size=4)
        plt.tight_layout()
    
    cmap = 'gray_r'
    cbar_params = {'shrink':0.05}
    fig = plt.figure()
    vmax = 1
    
    def get_sided_data(D,side,order):
    
        idx = (output_ds['Cell type'] != 'PSI') & (output_ds['Nerve Side'] == side)
        M = (D==order).astype(int)
        data = M.loc[input_ds['SegIDs'],output_ds['SegIDs'][idx]]     
        make_heatmap(data,idx)
        plt.title(side+' nerves')
        plt.rc('font',size=4)
        return data
    
    plt.subplot(331)
    first_L = get_sided_data(D,'L',1)
    plt.subplot(332)
    first_R = get_sided_data(D,'R',1) 
    plt.subplot(334)
    second_L = get_sided_data(D,'L',2)
    plt.subplot(335)
    second_R = get_sided_data(D,'R',2)
    plt.subplot(337)
    third_L = get_sided_data(D,'L',3)
    plt.subplot(338)
    third_R = get_sided_data(D,'R',3)
    
    # def get_LR_comparison_plots(L,R):
    #     # for ii in range(L.shape[0]):
    #     #     plt.scatter(R.iloc[ii,:],L.iloc[ii,:],s=5,color='black')
            
    #     # plt.ylim([-0.1*vmax,vmax*1.1])
    #     # plt.xlim([-0.1*vmax,vmax*1.1])
    #     # plt.xlabel('Right nerve (synapse #)')
    #     # plt.ylabel('Left nerve (synapses #)')
    #     # plt.gca().set_aspect('equal')
    #     # plt.rc('font',size=4)
        
    #     for ii in range(L.shape[0]):
    #         plt.scatter(R.iloc[ii,:],L.iloc[ii,:],s=5,color='black')
    
    
    # plt.subplot(333)
    # get_LR_comparison_plots(first_L,first_R)
    # plt.subplot(336)
    # get_LR_comparison_plots(second_L,second_R)
    # plt.subplot(339)
    # get_LR_comparison_plots(third_L,third_R)
    
    
    plt.savefig(fig_name) 
    

