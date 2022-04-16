#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:49:41 2021

@author: kyobi
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

def generate_graph_theory_heatmaps(M,edges,title,colorbar_label,figure_title):

    if title == 'Distance Matrix':
        cmap = 'Greys'
    elif title == 'Net Matching Matrix':
        cmap='PiYG'
    else:
        cmap = 'Greys_r'
    
    fig = plt.figure()
    ax = plt.imshow(M,cmap=cmap,interpolation='none')
    cbar = plt.colorbar(shrink=0.35,aspect=5,use_gridspec=True,location='left')   
    cbar.set_label(label=colorbar_label,size=12)       
    cbar.ax.tick_params(labelsize=12)   
    plt.title(title,{'fontsize':16,'weight':'bold'})
    # ax.axes.set_xlabel('Cell types')
    # ax.axes.set_ylabel('Cell types')
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])   
    x_ticks = [np.mean(edges[0:2]),np.mean(edges[2:4]),np.mean(edges[4:6])]
    y_ticks = x_ticks  
    colors = ['orange','red','green','blue']   
    for ii in range(len(edges)):      
        col = colors[int(edges[ii])]       
        ax.axes.add_patch(patches.Rectangle((-15,ii), width=10, height=1,color=col,\
                                            clip_on=False))      
        ax.axes.add_patch(patches.Rectangle((ii,len(M)+5), height=10, width=1,color=col,\
                                            clip_on=False))
    plt.tight_layout()
    plt.savefig(figure_title,backend='cairo',format='svg')

def generate_clustered_input_graph(M,title,figure_title,analysis):
    import seaborn as sns
    
    segIDs = M.index
    ds = analysis['input_data_sheet']
    ds.index = ds['SegIDs']
    labels = ds.specific[segIDs]
    lut = dict(zip(labels.unique(), "rbgk"))
    row_colors = labels.map(lut)
    fig = plt.figure()
    ax = sns.clustermap(M,row_colors=row_colors)
    plt.title(title,{'fontsize':16,'weight':'bold'})
    plt.tight_layout()
    plt.savefig(figure_title,backend='cairo',format='svg')

    
def generate_clustered_inter_graph(M,title,figure_title,analysis):
    import seaborn as sns
    
    segIDs = M.index
    ds = analysis['inter_data_sheet']
    ds.index = ds['SegIDs']
    segIDs = segIDs[segIDs.isin(ds.index)]
    labels = ds.specific[segIDs]
    colors = plt.cm.Spectral(range(0,labels.unique().shape[0]*30,30))[:,0:3]
    lut = dict(zip(labels.unique(),colors))
    row_colors = labels.map(lut)
    fig = plt.figure()
    ax = sns.clustermap(M,row_colors=row_colors)
    plt.title(title,{'fontsize':16,'weight':'bold'})
    plt.tight_layout()
    plt.savefig(figure_title,backend='cairo',format='svg')

    
def generate_clustered_output_graph(M,title,colorbar_label,figure_title,analysis):
    import seaborn as sns
    
    segIDs = M.index
    ds = analysis['output_data_sheet']
    ds.index = ds['SegIDs']
    labels = ds.specific[segIDs]
    lut = dict(zip(labels.unique(), "rbgk"))
    row_colors = labels.map(lut)
    fig = plt.figure()
    ax = sns.clustermap(M,row_colors=row_colors)
    plt.title(title,{'fontsize':16,'weight':'bold'})
    plt.tight_layout()
    plt.savefig(figure_title,backend='cairo',format='svg')


def make_hop_connectivity_plots(G,D,input_ds,output_ds,number_of_steps):
    import seaborn as sns
    fig_name = 'Hop analysis'
    def make_heatmap(data,idx):
        sns.heatmap(data,xticklabels=output_ds['specific'][idx],\
                    yticklabels=input_ds['specific'],vmax=vmax,cmap=cmap,cbar_kws=cbar_params)
        plt.gca().set_aspect('equal')  
    
    def plot_sided(D,side,order):
        idx = (output_ds['specific'] != 'PSI') & (output_ds['side'] == side)
        M = (D==order).astype(int)
        data = M.loc[input_ds['SegIDs'],output_ds['SegIDs'][idx]]     
        make_heatmap(data,idx)
        plt.title(side+' nerves')

    plt.subplots(dpi=600)
    cmap = 'gray_r'
    cbar_params = {'shrink':0.15}
    vmax = 1
    plt.rcParams['font.size'] = '5'
    plt.rcParams['font.weight'] = 'bold'
    for ii in range(number_of_steps):
        plt.subplot(number_of_steps,2,ii*2+1)
        plot_sided(D,'L',ii+1)
        plt.subplot(number_of_steps,2,ii*2+2)
        plot_sided(D,'R',ii+1)

    plt.tight_layout()

def generate_neuron_bargraph(analysis):
    plt.figure(dpi=600)
    plt.rcParams['font.size'] = '12'
    ylabel = '# of Neurons'
    analysis['input_data_sheet'].groupby(['specific','side']).count().\
        iloc[:,0].unstack().plot(kind='bar',ylim=[0,10],ylabel=ylabel,\
                                 xlabel='')
    plt.draw()

    

def generate_convergence_bargraph(analysis):
    plt.figure(dpi=600)
    plt.rcParams['font.size'] = '12'
    ylabel = '# of Upstream Neurons'
    inter_data = analysis['A'].loc[analysis['edges']==2,:].sum(axis=0)
    dn_data = analysis['A'].loc[analysis['edges']==1,:].sum(axis=0)
    df = pd.concat([inter_data,dn_data],axis=1)
    df.columns = ['interneurons','dns']
    df.loc[analysis['output_ids'],:].groupby\
        (np.array(analysis['output_data_sheet']['specific'])).mean().\
            plot(kind='bar',ylim=[0,25],ylabel=ylabel,\
                                 xlabel='')
    plt.draw()
  
def generate_divergence_bargraph(analysis):
    plt.figure(dpi=600)
    plt.rcParams['font.size'] = '12'
    ylabel = '# of Downstream Neurons'
    inter_data = analysis['A'].loc[:,analysis['edges']==2].sum(axis=1)
    mn_data = analysis['A'].loc[:,analysis['edges']==3].sum(axis=1)
    df = pd.concat([inter_data,mn_data],axis=1)
    df.columns = ['interneurons','motorneurons']
    df.loc[analysis['input_ids'],:].groupby\
        (np.array(analysis['input_data_sheet']['specific'])).mean().\
            plot(kind='bar',ylim=[0,10],ylabel=ylabel,\
                                 xlabel='')
    plt.draw()
    
# def compare_inter_to_datasheet(analysis):
      

    
def generate_neuron_mesh(segIDs,figure_path,ds):
    import emlibrary as lib
    from itkwidgets import view
    from meshparty import meshwork, skeletonize, skeleton_io, skeleton, trimesh_io, trimesh_vtk
    import datetime
    client = lib.load_fanc_client()
    seg_source = client.info.segmentation_source()
    mm = trimesh_io.MeshMeta(cv_path = seg_source, cache_size = 0,
                             disk_cache_path='~/em_analysis/meshes',map_gs_to_https=True)
    voxel_resolution = [4.3, 4.3, 45]
    
    mesh_actors = []
    for ii, segID in enumerate(segIDs):
        neuron_mesh = mm.mesh(seg_id = segID, remove_duplicate_vertices=True, merge_large_components=True)
    
        neuron_mw = meshwork.Meshwork(
            neuron_mesh, seg_id=segID, voxel_resolution=voxel_resolution)
        
        if (ds.loc[ds.SegIDs==segID]['side'] == 'R').iloc[0]:
            colors = np.array([0, np.random.rand(), 1])
        elif (ds.loc[ds.SegIDs==segID]['side'] =='L').iloc[0]:
            colors = np.array([1, np.random.rand(), 0])
        else:
            colors = np.array([np.random.rand(), 1, np.random.rand()])
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
    # trimesh_vtk.render_actors_360(mesh_actors,camera_start=camera,
    #                           directory=figure_path,
    #                           do_save=True,back_color=(0,0,0),scale=10,
    #                           nframes=3)

def DN_activation_MN_plots(analysis,connect_mat):
    #%%
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    params = {
        }
    params['test_conditions'] = [
        ['canonical','L'],
        ['canonical','R'],
        ['truncate','L'],
        ['truncate','R'],
        ['nowrap','L'],
        ['nowrap','R'],
        ['lateral','L'],
        ['lateral','R']
        
        ]
    
    connect = analysis[connect_mat].copy()
    if connect_mat == 'G':
        ind = analysis['fraction_connected'].index.isin(connect.index)
        connect.loc[analysis['fraction_connected'][ind].index,:] = \
            connect.loc[analysis['fraction_connected'][ind].index,:].divide(\
                analysis['fraction_connected'][ind].values,axis=0)
    valence = pd.Series(np.ones(connect.shape[0]),index = analysis['G'].index)
    to_keep = analysis['inter_data_sheet'].SegIDs.isin(valence.index)
    valence[analysis['inter_data_sheet'].SegIDs[to_keep]] = analysis['inter_data_sheet'].\
        Sign[to_keep].fillna(1)
    valence = valence.values
    
    connect = connect.multiply(valence,axis=0)
    
    input_neurons = analysis['input_data_sheet'][['SegIDs','specific','side']]
    blocks = len(params['test_conditions'])
    stimulus_matrix=np.zeros([blocks,input_neurons.shape[0]])

    for ii in range(blocks):
        test_conditions = params['test_conditions'][ii]
        columns = input_neurons['specific'].str.contains(test_conditions[0]) & \
            input_neurons['side'].str.contains(test_conditions[1])
        stimulus_matrix[ii,columns] = 1
    
    full_vect = np.zeros([blocks,connect.shape[0]])
    full_vect[:,analysis['edges']==1]=stimulus_matrix
    
    layer_1 = full_vect@connect
    layer_2 = layer_1@connect
    full_activation = layer_1+layer_2
    
    MN_vect = full_activation.loc[:,analysis['edges']==3]
            
    def make_heatmap(data,idx,vmax,vmin):
        sns.heatmap(data,xticklabels=output_ds['specific'][idx],\
                    vmax=vmax,cmap=cmap,cbar_kws=cbar_params,vmin=vmin)
        plt.gca().set_aspect('equal')  
    
    input_ds = analysis['input_data_sheet']
    output_ds = analysis['output_data_sheet']
    def plot_sided(M,side):
        idx = (output_ds['specific'] != 'PSI') & (output_ds['side'].str.contains(side))
        data = M.loc[:,output_ds['SegIDs'][idx]]    
        vmax = np.max(data.values)
        vmin = np.min(data.values)
        make_heatmap(data,idx,vmax,-vmax)
        plt.title(side+' nerves')
        
    def plot_subtraction(M):
        idx_1 = (output_ds['specific'] != 'PSI') & (output_ds['side'].str.contains('L'))
        idx_2 = (output_ds['specific'] != 'PSI') & (output_ds['side'].str.contains('R'))
        data = M.loc[:,output_ds['SegIDs'][idx_1]].values - \
            M.loc[:,output_ds['SegIDs'][idx_2]].values
        vmax = np.max(data)
        vmin = np.min(data)
        make_heatmap(data,idx_1,vmax,-vmax)
        plt.title('L - R nerves')


    plt.subplots(dpi=600)
    cmap = 'PiYG'
    cbar_params = {'shrink':0.05}
    
    plt.rcParams['font.size'] = '8'
    plt.rcParams['font.weight'] = 'bold'

    M = MN_vect
    plt.subplot(3,1,1)
    
    plt.rcParams['font.size'] = '8'
    plt.rcParams['font.weight'] = 'bold'
    plot_sided(M,'L')
    plt.subplot(3,1,2)
    
    plt.rcParams['font.size'] = '8'
    plt.rcParams['font.weight'] = 'bold'
    plot_sided(M,'R')
    
    plt.subplot(3,1,3)
    plt.rcParams['font.size'] = '8'
    plt.rcParams['font.weight'] = 'bold'
    plot_subtraction(M)


    plt.tight_layout()
    plt.savefig

#%%

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # principalComponents = pca.fit_transform(full_activation)
    # principalDf = pd.DataFrame(data = principalComponents
    #              , columns = ['principal component 1', 'principal component 2'])
    
    # labels = pd.DataFrame(params['test_conditions'],columns=['type','']).iloc[:,0]
    # lut = dict(zip(labels.unique(), "yrbgk"))
    # row_colors = labels.map(lut)
    # pd.concat([principalDf,labels.reset_index(drop=True)],axis=1)
    # principalDf.plot(kind='scatter',x = 'principal component 1',\
    #                  y = 'principal component 2',legend=True,colormap='Spectral',\
    #                      c=row_colors)

def generate_interneuron_counts(analysis):
    segs = analysis['inter_data_sheet']['SegIDs']
    ind = segs.loc[segs.isin(analysis['A'].index)]
    counts = analysis['A'].loc[:,ind].sum(axis=0)

    
