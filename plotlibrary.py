#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:49:41 2021

@author: kyobi
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
    
