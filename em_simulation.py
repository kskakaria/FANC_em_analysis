#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 15:12:39 2021

@author: kyobikakaria
"""
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Plot the graph figures
if sys.platform == 'linux1':
    path = '/home/kyobi/em_analysis/meshes/'
    input_path = '/home/kyobi/em_analysis/data/'
    figure_path = '/home/kyobi/em_analysis/images/'
elif sys.platform == 'darwin':
    path = '/Users/kyobikakaria/em_analysis/meshes/'
    input_path = '/Users/kyobikakaria/em_analysis/data/'
    figure_path = '/Users/kyobikakaria/em_analysis/images/'
fname = '2022_01_01_03_42_25_2step_analysis.pickle'

fpath = input_path+fname
save_datetime = fname[:19]
with open(fpath,'rb') as f:
    analysis = pickle.load(f)
    
#%% 

# time_steps = 100
# input_cols = (analysis['edges']==1) | (analysis['edges']==0) 
# readout_cols = analysis['edges']==3
# initial_activity_vector = np.zeros((len(analysis['A']),10000))
# initial_activity_vector[input_cols,] = np.random.rand(np.sum(input_cols),10000)

# output_matrix = (analysis['G']@initial_activity_vector).iloc[readout_cols,:]
# plt.bar(range(sum(readout_cols)),(initial_activity_vector[:,0]@analysis['G'].\
#                                   values)[readout_cols])
# plt.ylim([0,200])



def flyLI(analysis):
    #%%
    params = {
        'cell_params' : [],
        'stimulus_params' : []
        }    
    params['cell_params'] = {
        'V0' : -52e-3, # resting voltage -52e-3 (or -40e-3, -60e-3) V
        'Cmem' : 2e-7, # membrane capacitance 2e-7 (or 0.5e-7, 8e-7) F
        'Rmem' : 1e6 # 1e6 Ohm
        }
    params['stimulus_params'] = {
        'time' : 180, # time in seconds
        'dt' : 1e-2, # time per step
        'ap_duration' : 2, # 2ms AP duration
        'pscRiseT' : 0.002, 
        'pscFallT' : 0.005,
        'pscMag' : 5e-9,
        'pscWeights' : 1,
        }
    params['test_conditions'] = [
        ['',''],
        ['','L'],
        ['','R'],
        ['canonical',''],
        ['canonical','L'],
        ['canonical','R'],
        ['truncate',''],
        ['truncate','L'],
        ['truncate','R'],
        ['nowrap',''],
        ['nowrap','L'],
        ['nowrap','R'],
        ['lateral',''],
        ['lateral','L'],
        ['lateral','R']
        
        ]
    
    cell_params = params['cell_params']
    stimulus_params = params['stimulus_params']
    connect_mat = 'G'
    connect = analysis[connect_mat].copy()
    if connect_mat == 'G':
        ind = analysis['fraction_connected'].index.isin(connect.index)
        connect.loc[analysis['fraction_connected'][ind].index,:] = \
            connect.loc[analysis['fraction_connected'][ind].index,:].divide(\
                analysis['fraction_connected'][ind].values,axis=0)
    numNeurons=connect.shape[0]
    N = int(np.ceil(stimulus_params['time']/stimulus_params['dt']))
    valence = pd.Series(np.ones(connect.shape[0]),index = analysis['G'].index)
    to_keep = analysis['inter_ds'].SegIDs.isin(valence.index)
    valence[analysis['inter_ds'].SegIDs[to_keep]] = analysis['inter_ds'].\
        Sign[to_keep].fillna(1)
    valence = valence.values
    
    V=np.zeros([N,numNeurons])+cell_params['V0'];
   
    I=np.zeros([N,numNeurons]);
    
    input_neurons = analysis['input_data_sheet'][['SegIDs','specific','side']]
    stimulus_matrix=np.zeros([N,input_neurons.shape[0]]) # BROKEN
    break_length=5
    break_length=break_length/stimulus_params['dt'] #s
    blocks = len(params['test_conditions'])
    stim_block_length = np.floor((N-break_length*(blocks+1))/blocks)
    for ii in range(blocks):
        start_I = int(ii*(stim_block_length+break_length)+break_length)
        end_I = int(start_I + stim_block_length)
        test_conditions = params['test_conditions'][ii]
        columns = input_neurons['specific'].str.contains(test_conditions[0]) & \
            input_neurons['side'].str.contains(test_conditions[1])
        stimulus_matrix[start_I:end_I,columns] = 1
            
    
    
    # spikeMax=20e-3;     %purely cosmetic parameter
    
    # apRiseTime=np.round((apDuration*10)/2);
    # apRise=normpdf(-1:1/apRiseTime:0);
    # apRise=(apRise-min(apRise))/(max(apRise)-min(apRise));
    # apRise=apRise*(spikeMax-spikeThr(1))+spikeThr(1);
    # apFallTime=round((apDuration*9)/2);
    # apFall=sin(pi()/2:pi()/apFallTime:3*pi()/2);
    # apFall=(apFall-min(apFall))/(max(apFall)-min(apFall));
    # apFall=apFall*(spikeMax-spikeMin)+spikeMin-.0001;
    # AP=[apRise apFall]';
    # AP(1)=[];
    # AP(end+1)=AP(end)+0.0001;
    # AP(end+1)=AP(end)+0.0001;
    # apLength=length(AP);
    
    # pscRise=sin(linspace(-pi()/2,pi()/2,pscRiseT/dt));
    # pscRise=(pscRise-min(pscRise))/(max(pscRise)-min(pscRise));
    # pscFall=2.^(-(0:(7*pscFallT/dt))*dt/pscFallT);
    # pscFall=(pscFall-min(pscFall))/(max(pscFall)-min(pscFall));
    # PSC=[pscRise pscFall]';
    # PSC=PSC*23.6305/sum(PSC); %normalize PSC magnitude to counteract changes in duration or time-constants
    # PSC=PSC*pscMag/max(PSC);
    
    # tBuff=length(PSC);
    
    # %prepare input current vectors
    # Iin=zeros(N+tBuff,numNeurons);
    # for i=1:numInputs
    #     if ~ iscell(inputPSCs)
    #         if min(inputPSCs(:))==0
    #         pscFallTimes=find(inputPSCs(:,i)==1);
    #         else
    #             pscFallTimes=inputPSCs(:,i);
    #         end
    #     else
    #         pscFallTimes=inputPSCs{i};
    #     end
    #     for j=1:length(pscFallTimes)
    #         PSCrange=pscFallTimes(j):(pscFallTimes(j)+tBuff-1);
    #         Iin(PSCrange,inputs(i))=Iin(PSCrange,inputs(i))+PSC*inputWeights(i);
    #     end
    # end
    
    # V=zeros(N+tBuff,numNeurons)+V0;
    # V(1,:)=V0;
    # I=zeros(N+tBuff,numNeurons);
    # APMask=zeros(N+tBuff,numNeurons);

    
    Iin=np.zeros(V.shape);
    dcCurrentScalingFactor=0.01
    inCurrent=5e-9*dcCurrentScalingFactor
    Iin[:,analysis['edges']==1]=inCurrent*stimulus_matrix
    Iin[:,analysis['edges']==1]=stimulus_matrix
    
    currentNoise=0#0.15*2e-9; #optional current noise
    
    Cmem = cell_params['Cmem']
    Rmem = cell_params['Rmem']
    V0 = cell_params['V0']
    dt = stimulus_params['dt']
    con = connect.multiply(valence,axis=0)
    for t in range(1,503):
        VtoI = V[t-1,:] - V0
        # VtoI=inCurrent*np.tanh(100*VtoI)
        VtoI = (V[t-1,:] - V0)/Rmem
        # for ii in range(connect.shape[0]):
            # %         size(connect(:,i))
            # %         size((VtoI*(V(t-1,:)'-V0)))
            # %         size(V)
            # currentTemp=Iin[t-1,ii] + np.sum(connect[:,ii]*VtoI) +\
            #     currentNoise*np.random.normal()
            # dV=(1/Cmem)*( (V0-V[t-1,ii])/Rmem + currentTemp )
            # V[t,ii]=V[t-1,ii]+dV*dt
        Iout = VtoI + Iin[t-1,:]
        #* valence
        
        I = Iin[t-1,:] + (Iout@np.array(con)) \
            + currentNoise*np.random.normal(size=connect.shape[1])

        dV=(1/Cmem)*( (V0-V[t-1,:])/Rmem + I )
        # print(np.sum(I))
        V[t,:]=V[t-1,:]+dV*dt
        # V[t,:] = V[t-1,:]+Rmem*I*(1-np.exp(-dt/(Rmem*Cmem)))

    Vpd = pd.DataFrame(V,columns = analysis['G'].index)
    plt.plot(Vpd.loc[:,analysis['edges']==3])
    plt.xlim([500,505])
    plt.show()
    # plt.plot(V)
    ind = analysis['output_data_sheet']['Cell type'].isin(['hg1','hg2','b2','i1','iii3'])
    segs = analysis['output_data_sheet']['SegIDs'][ind]
    # plt.plot(Vpd[segs])

    # plt.show()
    
        
        