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
    colors = ['orange','red','green','blue']   
    for ii in range(len(edges)):      
        col = colors[int(edges[ii])]       
        ax.axes.add_patch(patches.Rectangle((-10,ii), width=10, height=1,color=col,\
                                            clip_on=False))      
        ax.axes.add_patch(patches.Rectangle((ii,len(M)), height=10, width=1,color=col,\
                                            clip_on=False))   

def make_hop_connectivity_plots(G,D,input_ds,output_ds,number_of_steps):
    import seaborn as sns
    fig_name = 'Hop analysis'
    def make_heatmap(data,idx):
        sns.heatmap(data,xticklabels=output_ds['Cell type'][idx],\
                    yticklabels=input_ds['Cell type'],vmax=vmax,cmap=cmap,cbar_kws=cbar_params)
        plt.gca().set_aspect('equal')  
    
    def plot_sided(D,side,order):
        idx = (output_ds['Cell type'] != 'PSI') & (output_ds['Nerve Side'] == side)
        M = (D==order).astype(int)
        data = M.loc[input_ds['SegIDs'],output_ds['SegIDs'][idx]]     
        make_heatmap(data,idx)
        plt.title(side+' nerves')

    plt.subplots()
    cmap = 'gray_r'
    cbar_params = {'shrink':0.15}
    vmax = 1
    plt.rcParams['font.size'] = '4'
    plt.rcParams['font.weight'] = 'bold'
    for ii in range(number_of_steps):
        plt.subplot(number_of_steps,2,ii*2+1)
        
        plot_sided(D,'L',ii+1)
        plt.subplot(number_of_steps,2,ii*2+2)
        plot_sided(D,'R',ii+1)

    plt.tight_layout()
    plt.draw()


def generate_neuron_bargraph(vals):
    plt.figure()
    L_ind = vals.str.contains('_L')
    R_ind = vals.str.contains('_R')
    vals = vals.apply(lambda x: x[:-2])
    val_list = [vals.loc[L_ind],vals.loc[R_ind]]
    plt.rcParams['font.size'] = '24'
    plt.hist(val_list,bins=range(0,6))
    # plt.rc('font',size=24)

    plt.xticks(rotation=90)
    # plt.xticks(np.arange(0.1,1,0.2))
    plt.ylabel('Neuron Count')
    plt.tight_layout()
    plt.legend(['Left','Right'])
    
    
def generate_neuron_mesh(segIDs,figure_path):
    import emlibrary as lib
    from itkwidgets import view
    from meshparty import meshwork, skeletonize, skeleton_io, skeleton, trimesh_io, trimesh_vtk
    import datetime
    client = lib.load_fanc_client()
    seg_source = client.info.segmentation_source()
    mm = trimesh_io.MeshMeta(cv_path = seg_source, cache_size = 0,
                             disk_cache_path='~/em_analysis/meshes',map_gs_to_https=True)
    
    mesh_actors = []
    for ii, segID in enumerate(segIDs):
        neuron_mesh = mm.mesh(seg_id = segID, remove_duplicate_vertices=True, merge_large_components=True)
    
        voxel_resolution = [4.3, 4.3, 45]
        neuron_mw = meshwork.Meshwork(
            neuron_mesh, seg_id=segID, voxel_resolution=voxel_resolution)
        colors = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
        mesh_actor = trimesh_vtk.mesh_actor(
            neuron_mw.mesh, color=colors,opacity = 1, line_width = 3, calc_normals = True)
        mesh_actors.append(mesh_actor)
        
    camera = trimesh_vtk.oriented_camera(
        [37788, 129510, 462]*np.array(voxel_resolution), # focus point
        backoff=800, # put camera 1000 units back from focus
        # backoff_vector=[0, 0, 1], # back off in negative z
        up_vector = [0, -1, 0] # make up negative y
    )
    trimesh_vtk.render_actors(mesh_actors,camera=camera,
                              filename=figure_path+'_mesh.png',
                              do_save=True,back_color=(0,0,0),scale=10)
    