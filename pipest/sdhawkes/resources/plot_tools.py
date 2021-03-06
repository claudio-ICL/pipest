import os
path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<6):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    raise ValueError("path_pipest not found. Instead: {}".format(path_pipest))
path_sdhawkes=path_pipest+'/sdhawkes_powerlaw'
path_lobster=path_pipest+'/lobster_for_sdhawkes'
path_lobster_pyscripts=path_lobster+'/py_scripts'
import sys
sys.path.append(path_sdhawkes+'/')
sys.path.append(path_sdhawkes+'/resources/')
sys.path.append(path_sdhawkes+'/modelling/')
sys.path.append(path_lobster_pyscripts+'/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
#import seaborn
#import statsmodels.tsa.stattools as stattools
from matplotlib.colors import ListedColormap
import copy
import bisect
import computation




def plot_queue_imbalance(model, t0=0.0, t1=100.0,
        figsizex=10, figsizey=8,
        plot_st2=False,
        save_fig=False, path='./', name='queueimb', plot=True, return_ax=False):
    def idx_timewindow(times):
        times-=times[0]
        times+=earliest_time
        return np.logical_and(times>=t0, times<=t1)
    def select_(history):
        h=np.array(history, copy=True)
        h[:,0]-=h[0,0]
        h[:,0]+=earliest_time
        return computation.select_interval(h, t0,t1)
    fig=plt.figure(figsize=(figsizex,figsizey))
    ax=fig.add_subplot(111)
    try:
        earliest_time=model.simulated_price[0,0]
    except:
        earliest_time=0.0
    if plot_st2:
        try:
            idx=idx_timewindow(model.simulated_times)
            ax.step(model.simulated_times[idx], model.simulated_2Dstates[idx,1], label='st2 simulation', linewidth=0.5)
        except:
            print("I could not plot simulation")
        try:
            idx=idx_timewindow(model.data.observed_times)
            ax.step(model.data.observed_times[idx], model.data.observed_2Dstates[idx,1], label='st2 data', linewidth=0.5)
        except:
            print("I could not plot data")
        ax.legend(loc=3)
        ax.set_ylabel('discretised queue imbalance')
    ax.set_ylim([-2.5, 2.5])
    ax2=ax.twinx()
    try:
        h_sim=select_(model.simulated_history_weighted_queueimb)
        ax2.plot(h_sim[:,0], h_sim[:,1], label='conv-st2 simulation', linewidth=3)
    except:
        print("I could not plot simulated history of weighted queue imbalance")
    try:
        h_emp=select_(model.data.observed_history_weighted_queueimb)
        ax2.plot(h_emp[:,0], h_emp[:,1], label='conv-st2 data', linewidth=3)
    except:
        print("I could not plot observed history of weighted queue imbalance")
    ax2.legend(loc=2)
    ax2.set_ylim([-2.5, 2.5])
    if save_fig:    
        fname=path+name
        plt.savefig(fname)
    if plot:
        plt.show()   
    if return_ax:
        return ax

def plot_price_trajectories(model,  t0 =0.0,  t1 = 100.0, 
        figsizex=10, figsizey=8,
        save_fig=False, path=None, name='prices', plot=True, return_ax=False):
    def prepare_traj(x):
        price = np.array(x, copy=True)
        price[:,0]-=price[0,0]
        price[:,0]+=earliest_time
        return computation.select_interval(price,t0,t1)
    fig=plt.figure(figsize=(figsizex,figsizey))
    ax=fig.add_subplot(111)
    try:
        earliest_time=model.simulated_price[0,0]
    except:
        earliest_time=0.0
    try:
        p=prepare_traj(model.simulated_price)
        ax.plot(p[:,0],p[:,1], label='simulation')
    except:
        print("I could not plot simulated price")
        pass
    try:
        p=prepare_traj(model.data.mid_price.values)
        ax.plot(p[:,0],p[:,1], label='data')
    except:
        print("I could not plot data")
        pass
    try:
        p=prepare_traj(model.reconstructed_empirical_price)
        ax.plot(p[:,0],p[:,1], label='reconstruction', linestyle='--')
    except:
        print("I could not plot reconstructed_empirical_price")
        pass
    ax.set_ylabel('price')
    ax.set_xlabel('time')
    ax.legend()
    fig.suptitle('Price trajectories')
    if save_fig:    
        if path==None:
            path='/home/claudio/Desktop/'
        fname=path+name
        plt.savefig(fname)
    if plot:
        plt.show()   
    if return_ax:
        return ax


def plot_events(events,time=None,n_event_types=0):
    fig = plt.figure(figsize=(10,5))
    if np.any(time==None):
        time=np.arange(events.shape[0])
    if n_event_types==0:
        n_event_types=1+np.amax(events)
    ax=fig.add_subplot(111)
    x=time
    y=events
    ax.scatter(x,y)
    ax.set_xlabel('time')
    ax.set_ylabel('event')
    plt.yticks(np.arange(n_event_types))
    plt.show()
    
def plot_3D_state_trajectories(times,states_2D,):
    "it is expected that states.shape[1]=2"
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(times,states_2D[:,0],states_2D[:,1])
    ax.set_ylim([-5,5])
    ax.set_yticks(np.arange(3)-1)
    ax.set_ylabel('X_1')
    ax.set_zlim([-7,7])
    ax.set_zticks(np.arange(5)-2)
    ax.set_zlabel('X_2')
    ax.set_xlabel('time')
    ax.set_title('State trajectory')
    plt.show()
   
    
def plot_events_colour(events,time=None,n_event_types=0):
    fig = plt.figure(figsize=(10,5))
    if np.any(time==None):
        time=np.arange(events.shape[0])
    if n_event_types==0:
        n_event_types=1+np.amax(events)
    cmap=plt.get_cmap('Set1')
    colours=map(cmap,list(np.linspace(0,1,n_event_types)))
    colours=list(colours)
    colours=np.vstack(colours)    
    ax=fig.add_subplot(111)
    for e in range(n_event_types):
        idx=events==e
        x=time[idx]
        y=events[idx]
        ax.scatter(x,y,c=colours[e].reshape(1,-1))
    ax.set_xlabel('time')
    ax.set_ylabel('event')
    plt.yticks(np.arange(n_event_types))
    plt.show()

    
def plot_bm_impact(bm_profile,
                           bm_intensity=None,
                           time_start=0.0,time_end=-1.0,
                           save_fig=False,path=path_pipest,name='bm_impact_profile',plot=True
                          ):
    time_start = max(time_start,bm_profile[0,0])
    if time_end<=time_start:
        time_end = bm_profile[-1,0]
    else:
        time_end = max(time_start,min(time_end,bm_profile[-1,0]))
    idx_start=bisect.bisect_left(bm_profile[:,0], time_start)
    idx_end=bisect.bisect(bm_profile[:,0],time_end)
    time=np.array(bm_profile[idx_start:idx_end,0],copy=True)
    impact_profile=np.array(bm_profile[idx_start:idx_end,1],copy=True)
    fig = plt.figure(figsize=(8,4))
    ax_profile=fig.add_subplot(111)
    if not len(bm_intensity) <=1:
        idx_start_intensity=bisect.bisect_left(bm_intensity[:,0], time_start)
        idx_end_intensity=bisect.bisect(bm_intensity[:,0], time_end)
        time_intensity = np.array(bm_intensity[idx_start_intensity:idx_end_intensity,0],copy=True)
        impact_intensity = np.array(bm_intensity[idx_start_intensity:idx_end_intensity,1],copy=True)
        ax_intensity=ax_profile.twinx()
#         ax_intensity.plot(time_intensity,impact_intensity,color=[0,0.4,0.4,0.4],label='profile_rate')
        ax_intensity.set_yticks([])
    ax_profile.plot(time,impact_profile,color='green',label='impact_profile',linewidth=2.0)
    if not len(bm_intensity) <=1:
        ax_profile.plot(time_intensity,impact_intensity,color=[0,0.4,0.4,0.4],label='profile_rate')
    ax_profile.set_xlabel('time')
    ax_profile.set_ylabel('impact')
#     ax_profile.set_yticks([])
    ax_profile.legend()
    fig.suptitle('Impact profile a` la Bacry-Muzy')
    if save_fig:    
        fname=path+'/'+name
        plt.savefig(fname)
    if plot:
        plt.show()   
        
def select_interval(arr, t0, t1):
    arr=np.atleast_2d(arr)
    if arr.shape[0]<=1:
        arr=arr.T
    idx0=bisect.bisect_left(arr[:,0], t0)
    idx1=bisect.bisect_right(arr[:,0], t1)
    return np.array(arr[idx0:idx1,:],copy=True)


def plot_bm_impact_profile(
        times, events, price, inventory,
        history_of_intensity,
        bm_profile, bm_intensity,
        time_start, time_end,
        plot_bm_intensity=False,
        save_fig=False, path=None,name='bm_impact_profile', plot=True
        ):
    bm_profile=select_interval(bm_profile,time_start, time_end)
    bm_intensity=select_interval(bm_intensity,time_start, time_end)
    history_of_intensity=select_interval(history_of_intensity, time_start, time_end)
    price=select_interval(price, time_start, time_end)
    inventory=select_interval(inventory, time_start, time_end)
    idx=np.logical_and(times>=time_start, times<=time_end)
    times=np.array(times[idx], copy=True)
    events=np.array(events[idx], copy=True)
    fig = plt.figure(figsize=(10,8))
    ax_profile=fig.add_subplot(211)
    ax_events=ax_profile.twinx()
    idx_liquidator=(events==0)
    ax_1,ax_2 = plot_events_and_intensities(
        events[idx_liquidator],times[idx_liquidator],history_of_intensity[:,:2],
        axes=ax_events,return_axes=True,plot=False)
    ax_1.set_yticks([])
    ax_1.set_ylim([-1,1])
    ax_1.set_ylabel('')
    ax_2.set_yticks(
        np.linspace(
            -np.amax(history_of_intensity[:,1]),
            np.amax(history_of_intensity[:,1]),num=3))
    ax_2.set_ylim([-4,4])
    ax_2.set_ylabel('liquidator_intensity')
    ax_profile.plot(bm_profile[:,0],bm_profile[:,1],
            color='green',label='impact_profile',linewidth=2.0)
    if plot_bm_intensity:
        ax_profile.plot(
                bm_intensity[:,0],bm_intensity[:,1],color=[0,0.4,0.4,0.4],label='profile intensity')
    ax_profile.set_xlabel('time')
    ax_profile.set_ylabel('impact')
#     ax_profile.set_yticks([])
    ax_profile.legend()
    
    ax_inventory=fig.add_subplot(212)
    ax_inventory.plot(inventory[:,0],inventory[:,1],color='red',label='inventory')
    ax_price=ax_inventory.twinx()
    ax_price.step(price[:,0],price[:,1],where='post',color='blue',label='price')
    ax_inventory.set_xlabel('time')
    ax_inventory.set_ylabel('inventory')
    ax_inventory.set_ylim([-0.1,1.1*inventory[0,1]])
    ax_inventory.set_yticks(np.linspace(0,inventory[0,1],num=10))
    ax_price.set_ylabel('price')
    ax_price.legend(loc=1)
    ax_inventory.legend(loc=5)
    fig.suptitle('Impact profile a` la Bacry-Muzy')
    if save_fig:    
        if path==None:
            path='/home/claudio/Desktop/'
        fname=path+name
        plt.savefig(fname)
    if plot:
        plt.show()   
        
        
        
def plot_impact_profile(
        times, events, price, inventory,
        history_of_intensity,
        profile, 
        time_start, time_end,
        save_fig=False, path=None,name='onesided_impact_profile', plot=True
        ):
    profile=select_interval(profile,time_start, time_end)
    history_of_intensity=select_interval(history_of_intensity, time_start, time_end)
    price=select_interval(price, time_start, time_end)
    inventory=select_interval(inventory, time_start, time_end)
    idx=np.logical_and(times>=time_start, times<=time_end)
    times=np.array(times[idx], copy=True)
    events=np.array(events[idx], copy=True)
    fig = plt.figure(figsize=(10,8))
    ax_profile=fig.add_subplot(211)
    ax_events=ax_profile.twinx()
    idx_liquidator=(events==0)
    ax_1,ax_2 = plot_events_and_intensities(
        events[idx_liquidator],times[idx_liquidator],history_of_intensity[:,:2],
        axes=ax_events,return_axes=True,plot=False)
    ax_1.set_yticks([])
    ax_1.set_ylim([-1,1])
    ax_1.set_ylabel('')
    ax_2.set_yticks(
        np.linspace(
            -np.amax(history_of_intensity[:,1]),
            np.amax(history_of_intensity[:,1]),num=3))
    ax_2.set_ylim([-4,4])
    ax_2.set_ylabel('liquidator_intensity')
    ax_profile.plot(profile[:, 0], profile[:,1], color='green',label='impact_profile',linewidth=2.0)
    ax_profile.set_xlabel('time')
    ax_profile.set_ylabel('impact')
#     ax_profile.set_yticks([])
    ax_profile.legend()
    
    ax_inventory=fig.add_subplot(212)
    ax_inventory.plot(inventory[:,0], inventory[:,1], color='red', label='inventory')
    ax_price=ax_inventory.twinx()
    ax_price.step(price[:,0], price[:,1],  where='post',color='blue',label='price')
    ax_inventory.set_xlabel('time')
    ax_inventory.set_ylabel('inventory')
    ax_inventory.set_ylim([-0.1,1.1*inventory[0,1]])
    ax_inventory.set_yticks(np.linspace(0,inventory[0,1],num=10))
    ax_price.set_ylabel('price')
    ax_price.legend(loc=1)
    ax_inventory.legend(loc=5)
    fig.suptitle('One-sided impact profile')
    if save_fig:    
        if path==None:
            path='/home/claudio/Desktop/'
        fname=path+name
        plt.savefig(fname)
    if plot:
        plt.show()      

def plot_liquidation(times,events,inventory,history_of_intensity,
                     start_index=0,end_index=100,
                     save_fig=False,path=None,name='liquidation',plot=True
                    ):
    n_event_types=history_of_intensity.shape[1]-1
    fig = plt.figure(figsize=(10,8))
    ax_events=fig.add_subplot(211)
    ax_inventory=fig.add_subplot(212)
    start_index=max(0,start_index)
    end_index=min(len(times)-1,end_index)
    time=times[start_index:end_index]
    events=events[start_index:end_index]
    inventory=inventory[start_index:end_index]
    idx_history=np.logical_and(history_of_intensity[:,0]>=time[0],history_of_intensity[:,0]<=time[-1])
    history_of_intensity=history_of_intensity[idx_history,:]
    cmap=plt.get_cmap('Set1')
    colours=np.vstack(list(map(cmap,list(np.linspace(0,1,n_event_types)))))
    colour_inventory=colours[0]
    ax_1,ax_2 = plot_events_and_intensities(
        events,time,history_of_intensity,axes=ax_events,return_axes=True,plot=False)
    ax_inventory.plot(time,inventory,color=colour_inventory,label='inventory')
    ax_inventory.set_xlabel('time')
    ax_inventory.set_ylabel('inventory')
    ax_inventory.set_ylim([-0.1,1.1*inventory[0]])
    ax_inventory.set_yticks(np.linspace(0,inventory[0],num=10))
    ax_inventory.legend()
    if save_fig:    
        if path==None:
            path='/home/claudio/Desktop/'
        fname=path+name
        plt.savefig(fname)
    if plot:
        plt.show()    

def plot_liquidation_with_price(times,events,inventory,history_of_intensity,price,
                     start_index=0,end_index=100,
                     save_fig=False,path=None,name='liquidation_with_price',plot=True
                    ):
    n_event_types=history_of_intensity.shape[1]-1
    fig = plt.figure(figsize=(10,8))
    ax_events=fig.add_subplot(211)
    ax_inventory=fig.add_subplot(212)
    start_index=max(0,start_index)
    end_index=min(len(times)-1,end_index)
    time=times[start_index:end_index]
    events=events[start_index:end_index]
    inventory=inventory[start_index:end_index]
    price=price[start_index:end_index]
    idx_history=np.logical_and(history_of_intensity[:,0]>=time[0],history_of_intensity[:,0]<=time[-1])
    history_of_intensity=history_of_intensity[idx_history,:]
    cmap=plt.get_cmap('Set1')
    colours=np.vstack(list(map(cmap,list(np.linspace(0,1,n_event_types)))))
    colour_inventory=colours[0]
    ax_1,ax_2 = plot_events_and_intensities(
        events,time,history_of_intensity,axes=ax_events,return_axes=True,plot=False)
    ax_inventory.plot(time,inventory,color=colour_inventory,label='inventory')
    ax_price=ax_inventory.twinx()
    ax_price.step(time,price,where='post',color='blue',label='price')
    ax_inventory.set_xlabel('time')
    ax_inventory.set_ylabel('inventory')
    ax_inventory.set_ylim([-0.1,1.1*inventory[0]])
    ax_inventory.set_yticks(np.linspace(0,inventory[0],num=10))
    ax_price.set_ylabel('price')
    ax_price.legend(loc=1)
    ax_inventory.legend(loc=5)
    if save_fig:  
        if path==None:
                path='/home/claudio/Desktop/'
        fname=path+name
        plt.savefig(fname)
    if plot:
        plt.show()
        
def plot_liquidator_only(times,events,inventory,history_of_intensity,price,
                     start_index=0,end_index=100,
                     save_fig=False,path=None,name='liquidator_only',plot=True,
                    ):
    fig = plt.figure(figsize=(8,6))
    ax_events=fig.add_subplot(211)
    ax_inventory=fig.add_subplot(212)
    start_index=max(0,start_index)
    end_index=min(len(times)-1,end_index)
    idx_liquidator=(events[start_index:end_index]==0)
    time=times[start_index:end_index]
    events=events[start_index:end_index]
    inventory=inventory[start_index:end_index]
    price=price[start_index:end_index]
    idx_history=np.logical_and(history_of_intensity[:,0]>=time[0],history_of_intensity[:,0]<=time[-1])
    history_of_intensity=history_of_intensity[idx_history,:]
    cmap=plt.get_cmap('Set1')
    colours=np.vstack(list(map(cmap,list(np.linspace(0,1,2)))))
    colour_inventory=colours[0]
    ax_1,ax_2 = plot_events_and_intensities(
        events[idx_liquidator],time[idx_liquidator],history_of_intensity[:,:2],
        axes=ax_events,return_axes=True,plot=False)
    ax_2.set_yticks(np.arange(0,1,step=0.2))
    ax_inventory.plot(time,inventory,color=colour_inventory,label='inventory')
    ax_price=ax_inventory.twinx()
    ax_price.step(time,price,where='post',color='blue',label='price')
    ax_inventory.set_xlabel('time')
    ax_inventory.set_ylabel('inventory')
    ax_inventory.set_ylim([-0.1,1.1*inventory[0]])
    ax_inventory.set_yticks(np.linspace(0,inventory[0],num=10))
    ax_price.set_ylabel('price')
    ax_price.legend(loc=1)
    ax_inventory.legend(loc=5)
    if save_fig:
        if path==None:
            path='/home/claudio/Desktop/'
        fname=path+name
        plt.savefig(fname)
    if plot:
        plt.show()
    
def plot_intensities(history_of_intensities,first_event_index=1,transparency=0.85,plot=True,save_fig=False,path=None,name='intensities'):
    """
    it is assumed that history_of_intensities[:,0] represents the time of evaluation, whereas history_of_intensities[:,i], i=1,2,... represents the intensity of the i-th event type
    """
    n_event_types=max(1,history_of_intensities.shape[1]-1)
    lowest_event_type=first_event_index
    lambdas=np.array(history_of_intensities[:,1:],copy=True)
    cmap=plt.get_cmap('Set1')
    colours=map(cmap,list(np.linspace(0,1,lowest_event_type+n_event_types)))
    colours=list(colours)
    colours=np.vstack(colours) 
    fig = plt.figure(figsize=(10,5))
    ax=fig.add_subplot(111)
    ax.set_xlabel('time')
    ax.set_ylabel('intensity')
    for e in range(n_event_types):
        col_1=colours[lowest_event_type+e].reshape(1,-1)
        col_2=np.array(col_1[0,:],copy=True)
        col_2[-1]=transparency
        ax.plot(history_of_intensities[:,0],lambdas[:,e],color=col_2)
    if save_fig:
        if path==None:
            path='/home/claudio/Desktop/'
        fname=path+name
        plt.savefig(fname)
    if plot:
        plt.show()

def plot_events_and_intensities(events,time,history_of_intensities,
                                transparency=0.5,
                                return_axes=False,
                                plot=True,
                                axes=None,
                                save_fig=False,path=None,name='events_and_intensities'
                               ):
    """
    it is assumed that history_of_intensities[:,0] represents the time of evaluation, whereas history_of_intensities[:,i], i=1,2,... represents the intensity of the i-th event type
    """
    n_event_types=max(1,history_of_intensities.shape[1]-1)
    lowest_event_type=np.amin(events).astype(int)
    lambdas=np.array(history_of_intensities[:,1:],copy=True)
    lambdas=lambdas/(0.1+np.amax(lambdas))
    cmap=plt.get_cmap('Set1')
    colours=map(cmap,list(np.linspace(0,1,lowest_event_type+n_event_types)))
    colours=list(colours)
    colours=np.vstack(colours) 
    if plot:
        fig = plt.figure(figsize=(10,5))
        ax_1=fig.add_subplot(111)
    else:
        ax_1=axes
    ax_1.set_xlabel('time')
    ax_1.set_ylabel('event')
    ax_1.set_yticks(np.arange(n_event_types+1))
    y_limits=[min(0,-0.25+np.amin(events)),0.5+np.amax(events)+1.01*np.amax(lambdas)]
    ax_1.set_ylim(y_limits)
    ax_2=ax_1.twinx()
    ax_2.set_ylabel('intensities')
    ax_2.set_yticks([])
    ax_2.set_ylim(y_limits)
    for e in range(n_event_types):
        y0=e+lowest_event_type
        col_1=colours[lowest_event_type+e].reshape(1,-1)
        col_2=np.array(col_1[0,:],copy=True)
        col_2[-1]=transparency
        idx=(events==(e+lowest_event_type))
        if np.any(idx):
            x=time[idx]
            y=events[idx]
            ax_1.scatter(x,y,c=col_1)
        ax_2.plot(history_of_intensities[:,0],y0+lambdas[:,e],color=col_2)
    if save_fig:
        if path==None:
            path='/home/claudio/Desktop/'
        fname=path+name
        plt.savefig(fname)
    if return_axes:
        return ax_1,ax_2
    elif plot:
        plt.show()
        return fig
    
def plot_events_and_states(events,times,history_of_intensities,states_2D,
                                transparency=0.5,
                                plot=True,
                                save_fig=False,path='/home/claudio/Desktop/',name='events_and_intensities'
                               ):
    """
    it is assumed that history_of_intensities[:,0] represents the time of evaluation, whereas history_of_intensities[:,i], i=1,2,... represents the intensity of the i-th event type
    """
    assert len(times)==len(states_2D)
    n_event_types=max(1,history_of_intensities.shape[1]-1)
    lowest_event_type=np.amin(events).astype(int)
    lambdas=np.array(history_of_intensities[:,1:],copy=True)
    lambdas=lambdas/(0.1+np.amax(lambdas))
    cmap=plt.get_cmap('Set1')
    colours=map(cmap,list(np.linspace(0,1,lowest_event_type+n_event_types)))
    colours=list(colours)
    colours=np.vstack(colours) 
    fig = plt.figure(figsize=(14,5))
    ax_1=fig.add_subplot(121)
    ax_1.set_xlabel('time')
    ax_1.set_ylabel('event')
    ax_1.set_yticks(np.arange(n_event_types+1))
    y_limits=[min(0,-0.25+np.amin(events)),0.5+np.amax(events)+1.01*np.amax(lambdas)]
    ax_1.set_ylim(y_limits)
    ax_2=ax_1.twinx()
    ax_2.set_ylabel('intensities')
    ax_2.set_yticks([])
    ax_2.set_ylim(y_limits)
    for e in range(n_event_types):
        y0=e+lowest_event_type
        col_1=colours[lowest_event_type+e].reshape(1,-1)
        col_2=np.array(col_1[0,:],copy=True)
        col_2[-1]=transparency
        idx=(events==(e+lowest_event_type))
        if np.any(idx):
            x=times[idx]
            y=events[idx]
            ax_1.scatter(x,y,c=col_1)
        ax_2.plot(history_of_intensities[:,0],y0+lambdas[:,e],color=col_2)
    ax3D=fig.add_subplot(122, projection='3d')
    ax3D.step(times,states_2D[:,0],states_2D[:,1])
    ax3D.set_ylim([-5,5])
    ax3D.set_yticks(np.arange(3)-1)
    ax3D.set_ylabel('X_1')
    ax3D.set_zlim([-7,7])
    ax3D.set_zticks(np.arange(5)-2)
    ax3D.set_zlabel('X_2')
    ax3D.set_xlabel('time')
    ax3D.set_title('State trajectory')
    if save_fig:
        fname=path+name
        plt.savefig(fname)
    if plot:
        plt.show()
    

def qq_plot(residuals, shape=None, path='', fig_name='qq_plot.pdf', log=False, q_min=0.01, q_max=0.99,
            number_of_quantiles=100, title=None, labels=None, model_labels=None, palette=None, figsize=(12, 6),
            size_labels=16, size_ticks=14, legend_size=16, bottom=0.12, top=0.93, left=0.08, right=0.92, savefig=False,
            leg_pos=0):
    """
    Qq-plot of residuals.

    :type residuals: list
    :param residuals: list of lists (one list of residuals per event type) or list of lists of lists when multiple models are compared (one list of lists per model).
    :type shape: (int, int)
    :param shape: 2D-tuple (number of rows, number of columns), shape of the array of figures.
    :type path: string
    :param path: where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :type log: boolean
    :param log: set to True for qq-plots with log-scale.
    :type q_min: float
    :param q_min: smallest quantile to plot (e.g., 0.01 for 1%).
    :type q_max: float
    :param q_max: largest quantile to plot.
    :type number_of_quantiles: int
    :param number_of_quantiles: number of points used to plot.
    :type title: string
    :param title: suptitle.
    :type labels: list of strings
    :param labels: labels of the event types.
    :type model_labels: list of strings
    :param model_labels: names of the different considered models.
    :type palette: list of colours
    :param palette: color palette, one color per model.
    :type figsize: (int, int)
    :param figsize: tuple (width, height).
    :type size_labels: int
    :param size_labels: fontsize of labels.
    :type size_ticks: int
    :param size_ticks: fontsize of tick labels.
    :type legend_size: int
    :param legend_size: fontsize of the legend.
    :type bottom: float
    :param bottom: between 0 and 1, adjusts the bottom margin, see matplotlib subplots_adjust.
    :type top: float
    :param top: between 0 and 1, adjusts the top margin, see matplotlib subplots_adjust.
    :type left: float
    :param left: between 0 and 1, adjusts the left margin, see matplotlib subplots_adjust.
    :type right: float
    :param right: between 0 and 1, adjusts the right margin, see matplotlib subplots_adjust.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :type leg_pos: int
    :param leg_pos: position of the legend in the array of figures.
    :rtype: Figure, array of Axes
    :return: the figure and array of figures (see matplotlib).
    """
    quantile_levels = np.linspace(q_min, q_max, number_of_quantiles)
    quantiles_theoretical = np.zeros(number_of_quantiles)
    for i in range(number_of_quantiles):
        q = quantile_levels[i]
        x = - np.log(1 - q)  # standard exponential distribution
        quantiles_theoretical[i] = x
    # find number of models given and number of event types (dim)
    n_models = 1
    dim = len(residuals)
    if type(residuals[0][0]) in [list, np.ndarray]:  # case when there is more than one model
        n_models = len(residuals)
        dim = len(residuals[0])
    # set empty model labels if no labels provided
    if model_labels==None:
        model_labels = [None]*n_models
    if shape is None:
        shape = (1, dim)
    v_size = shape[0]
    h_size = shape[1]
    if palette==None:
        palette = seaborn.color_palette('husl', n_models)
    f, fig_array = plt.subplots(v_size, h_size, figsize=figsize, sharex='col', sharey='row')
    if title is not None:
        f.suptitle(title)
    for i in range(v_size):
        for j in range(h_size):
            n = j + h_size * i
            if n < dim:  # the shape of the subplots might be bigger than dim, i.e. 3 plots on a 2x2 grid.
                axes = None
                if v_size == 1 and h_size == 1:
                    axes = fig_array
                elif v_size == 1:
                    axes = fig_array[j]
                elif h_size == 1:
                    axes = fig_array[i]
                else:
                    axes = fig_array[i, j]
                axes.tick_params(axis='both', which='major', labelsize=size_ticks)  # font size for tick labels
                if n_models == 1:
                    quantiles_empirical = np.zeros(number_of_quantiles)
                    for k in range(number_of_quantiles):
                        q = quantile_levels[k]
                        x = np.percentile(residuals[n], q * 100)
                        quantiles_empirical[k] = x
                    axes.plot(quantiles_theoretical, quantiles_empirical, color=palette[0])
                    axes.plot(quantiles_theoretical, quantiles_theoretical, color='k', linewidth=0.8, ls='--')
                else:
                    for m in range(n_models):
                        quantiles_empirical = np.zeros(number_of_quantiles)
                        for k in range(number_of_quantiles):
                            q = quantile_levels[k]
                            x = np.percentile(residuals[m][n], q * 100)
                            quantiles_empirical[k] = x
                        axes.plot(quantiles_theoretical, quantiles_empirical, color=palette[m],
                                     label=model_labels[m])
                        if m == 0:
                            axes.plot(quantiles_theoretical, quantiles_theoretical, color='k', linewidth=0.8,
                                      ls='--')
                    if n == leg_pos :  # add legend in the specified subplot
                        legend = axes.legend(frameon=1, fontsize=legend_size)
                        legend.get_frame().set_facecolor('white')
                if log:
                    axes.set_xscale('log')
                    axes.set_yscale('log')
                if labels is not None:
                    axes.set_title( labels[n], fontsize=size_labels)
    plt.tight_layout()
    if bottom!=None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
    f.text(0.5, 0.02, 'Quantile (standard exponential distribution)', ha='center', fontsize=size_labels)
    f.text(0.02, 0.5, 'Quantile (empirical)', va='center', rotation='vertical', fontsize=size_labels)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return f, fig_array

def correlogram(residuals, path='', fig_name='correlogram.pdf', title=None, labels=None, model_labels=None,
                palette=None, n_lags=50, figsize=(8, 6), size_labels=16, size_ticks=14, size_legend=16, bottom=None,
                top=None, left=None, right=None,savefig=False):
    """
    Correlogram of residuals.

    :type residuals: list
    :param residuals: list of lists (one list of residuals per event type) or list of lists of lists when multiple models are compared (one list of lists per model).
    :type path: string
    :param path: where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :type title: string
    :param title: suptitle.
    :type labels: list of strings
    :param labels: labels of the event types.
    :type model_labels: list of strings
    :param model_labels: names of the different considered models.
    :type palette: list of colours
    :param palette: color palette, one color per model.
    :type n_lags: int
    :param n_lags: number of lags to plot.
    :type figsize: (int, int)
    :param figsize: tuple (width, height).
    :type size_labels: int
    :param size_labels: fontsize of labels.
    :type size_ticks: int
    :param size_ticks: fontsize of tick labels.
    :type legend_size: int
    :param legend_size: fontsize of the legend.
    :type bottom: float
    :param bottom: between 0 and 1, adjusts the bottom margin, see matplotlib subplots_adjust.
    :type top: float
    :param top: between 0 and 1, adjusts the top margin, see matplotlib subplots_adjust.
    :type left: float
    :param left: between 0 and 1, adjusts the left margin, see matplotlib subplots_adjust.
    :type right: float
    :param right: between 0 and 1, adjusts the right margin, see matplotlib subplots_adjust.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :rtype: Figure, array of Axes
    :return: the figure and array of figures (see matplotlib).
    """
    # find number of models given and number of event types (dim)
    n_models = 1
    dim = len(residuals)
    if type(residuals[0][0]) in [list, np.ndarray]:  # case when there is more than one model
        n_models = len(residuals)
        dim = len(residuals[0])
    # set empty model labels if no labels provided
    if model_labels is None:
        model_labels = [None] * n_models
    v_size = dim
    h_size = dim
    if palette is None:
        palette = seaborn.color_palette('husl', n_models)
    f, fig_array = plt.subplots(v_size, h_size, figsize=figsize, sharex='col', sharey='row')
    if title is not None:
        f.suptitle(title)
    for i in range(v_size):
        for j in range(h_size):
            axes = None
            if v_size == 1 and h_size == 1:
                axes = fig_array
            elif v_size == 1:
                axes = fig_array[j]
            elif h_size == 1:
                axes = fig_array[i]
            else:
                axes = fig_array[i, j]
            axes.tick_params(axis='both', which='major', labelsize=size_ticks)  # font size for tick labels
            if n_models == 1:
                max_length = min(len(residuals[i]), len(residuals[j]))
                ccf = stattools.ccf(np.array(residuals[i][0:max_length]),
                                    np.array(residuals[j][0:max_length]),
                                    unbiased=True)
                axes.plot(ccf[0:n_lags+1], color=palette[0])
                axes.set_xlim(xmin=0, xmax=n_lags)
            else:
                for m in range(n_models):
                    max_length = min(len(residuals[m][i]), len(residuals[m][j]))
                    ccf = stattools.ccf(np.array(residuals[m][i][0:max_length]),
                                        np.array(residuals[m][j][0:max_length]),
                                        unbiased=True)
                    axes.plot(ccf[0:n_lags + 1], color=palette[m], label=model_labels[m])
                    axes.set_xlim(xmin=0, xmax=n_lags)
                if i+j==0:  # only add legend in the first subplot
                    legend = axes.legend(frameon=1, fontsize=size_legend)
                    legend.get_frame().set_facecolor('white')
            if labels is not None:
                axes.set_title(labels[i] + r'$\rightarrow$' + labels[j], fontsize=size_labels)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if bottom!=None:
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    f.text(0.5, 0.025, 'Lag', ha='center', fontsize=size_labels)
    f.text(0.015, 0.5, 'Correlation', va='center', rotation='vertical', fontsize=size_labels)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return f, fig_array

def transition_probabilities(probabilities, shape=None, path='', fig_name='transition_probabilities.pdf',
                             events_labels=None, states_labels=None, title=None, color_map=None, figsize=(12, 6),
                             size_labels=16, size_values=14, bottom=0.1, top=0.95, left=0.08, right=0.92,
                             wspace=0.2, hspace=0.2,
                             savefig=False, usetex=False):
    """
    Annotated heatmap of the transition probabilities of a state-dependent Hawkes process.

    :type probabilities: 3D array
    :param probabilities: the transition probabilities.
    :type shape: (int, int)
    :param shape: 2D-tuple (number of rows, number of columns), shape of the array of figures.
    :type path: string
    :param path: where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :type events_labels: list of strings
    :param events_labels: labels of the event types.
    :type states_labels: list of strings
    :param states_labels: labels of the states.
    :type title: string
    :param title: suptitle.
    :param color_map: color map for the heatmap, see seaborn documentation.
    :type figsize: (int, int)
    :param figsize: tuple (width, height).
    :type size_labels: int
    :param size_labels: fontsize of labels.
    :type size_values: int
    :param size_values: fontsize of the annotations on top of the heatmap.
    :type bottom: float
    :param bottom: between 0 and 1, adjusts the bottom margin, see matplotlib subplots_adjust.
    :type top: float
    :param top: between 0 and 1, adjusts the top margin, see matplotlib subplots_adjust.
    :type left: float
    :param left: between 0 and 1, adjusts the left margin, see matplotlib subplots_adjust.
    :type right: float
    :param right: between 0 and 1, adjusts the right margin, see matplotlib subplots_adjust.
    :type wspace: float
    :param wspace: horizontal spacing between the subplots, see matplotlib subplots_adjust.
    :type hspace: float
    :param hspace: vertical spacing between the subplots, see matplotlib subplots_adjust.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :type usetex: boolean
    :param usetex: set to True if matplolib figure is rendered with TeX.
    :rtype: Figure, array of Axes
    :return: the figure and array of figures (see matplotlib).
    """
    if color_map is None:
        color_map = seaborn.cubehelix_palette(as_cmap=True, reverse=False, start=0.5, rot=-.75)
    number_of_states = np.shape(probabilities)[0]
    number_of_event_types = np.shape(probabilities)[1]
    if shape is None:
        v_size = 1
        h_size = number_of_event_types
    else:
        v_size = shape[0]
        h_size = shape[1]
    f, fig_array = plt.subplots(v_size, h_size, figsize=figsize)
    if title is not None:
        f.suptitle(title)
    for i in range(v_size):
        for j in range(h_size):
            n = i*h_size + j
            if n < number_of_event_types:  # we could have more subplots than event types
                axes = None
                if v_size == 1 and h_size == 1:
                    axes = fig_array
                elif v_size == 1:
                    axes = fig_array[j]
                elif h_size == 1:
                    axes = fig_array[i]
                else:
                    axes = fig_array[i, j]
                axes.tick_params(axis='both', which='major', labelsize=size_labels)  # font size for tick labels
                # Create annotation matrix
                annot = np.ndarray((number_of_states, number_of_states), dtype=object)
                for x1 in range(number_of_states):
                    for x2 in range(number_of_states):
                        p = probabilities[x1, n, x2]
                        if p == 0:
                            if usetex:
                                annot[x1, x2] = r'$0$\%'
                            else:
                                annot[x1, x2] = r'0%'
                        elif p < 0.01:
                            if usetex:
                                annot[x1, x2] = r'$<1$\%'
                            else:
                                annot[x1, x2] = r'<1%'
                        else:
                            a = str(int(np.floor(100 * p)))
                            if usetex:
                                annot[x1, x2] = r'$' + a + r'$\%'
                            else:
                                annot[x1, x2] = a + r'%'
                seaborn.heatmap(probabilities[:, n, :], ax=axes,
                                xticklabels=states_labels, yticklabels=states_labels, annot=annot, cbar=False,
                                cmap=color_map, fmt='s', square=True, annot_kws={'size': size_values})
                axes.set_yticklabels(states_labels, va='center')
                if not usetex:
                    axes.set_title(r'$\phi_{' + events_labels[n] + '}$', fontsize=size_labels)
                else:
                    axes.set_title(r'$\bm{\phi}_{' + events_labels[n] + '}$', fontsize=size_labels)
    if bottom!=None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right, wspace=wspace, hspace=hspace)
    f.text(0.5, 0.02, 'Next state', ha='center', fontsize=size_labels)
    f.text(0.02, 0.5, 'Previous state', va='center', rotation='vertical', fontsize=size_labels)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return f, fig_array

def discrete_distribution(probabilities, path='', fig_name='distribution_events_states.pdf', v_labels=None,
                          h_labels=None, title=None, color_map=None, figsize=(12, 6), size_labels=16, size_values=14,
                          bottom=None, top=None, left=None, right=None, savefig=False, usetex=False):
    """
    Annotated heatmap of a given discrete distribution with 2 dimensions.

    :type probabilities: 2D array
    :param probabilities: the 2D discrete distribution.
    :type path: string
    :param path: where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :type v_labels: list of strings
    :param v_labels: labels for the first dimension (vertical).
    :type h_labels: list of strings
    :param h_labels: labels for the second dimension (horizontal).
    :type title: string
    :param title: suptitle.
    :param color_map: color map for the heatmap, see seaborn documentation.
    :type figsize: (int, int)
    :param figsize: tuple (width, height).
    :type size_labels: int
    :param size_labels: fontsize of labels.
    :type size_values: int
    :param size_values: fontsize of the annotations on top of the heatmap.
    :type bottom: float
    :param bottom: between 0 and 1, adjusts the bottom margin, see matplotlib subplots_adjust.
    :type top: float
    :param top: between 0 and 1, adjusts the top margin, see matplotlib subplots_adjust.
    :type left: float
    :param left: between 0 and 1, adjusts the left margin, see matplotlib subplots_adjust.
    :type right: float
    :param right: between 0 and 1, adjusts the right margin, see matplotlib subplots_adjust.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :type usetex: boolean
    :param usetex: set to True if matplolib figure is rendered with TeX.
    :rtype: Figure
    :return: the figure (see matplotlib).
    """
    if color_map is None:
        color_map = seaborn.cubehelix_palette(as_cmap=True, reverse=False, start=0.5, rot=-.75)
    v_size = np.shape(probabilities)[0]
    h_size = np.shape(probabilities)[1]
    # Create annotation matrix
    annot = np.ndarray((v_size, h_size), dtype=object)
    for x1 in range(v_size):
        for x2 in range(h_size):
            p = probabilities[x1, x2]
            if p == 0:
                if usetex:
                    annot[x1, x2] = r'$0$\%'
                else:
                    annot[x1, x2] = r'0%'
            elif p < 0.01:
                if usetex:
                    annot[x1, x2] = r'$<1$\%'
                else:
                    annot[x1, x2] = r'<1%'
            else:
                a = str(int(np.floor(100 * p)))
                if usetex:
                    annot[x1, x2] = r'$' + a + r'$\%'
                else:
                    annot[x1, x2] = a + r'%'
    f = plt.figure(figsize=figsize)
    ax = seaborn.heatmap(probabilities, xticklabels=h_labels, yticklabels=v_labels, annot=annot, cbar=False,
                    cmap=color_map, fmt='s', square=True, annot_kws={'size': size_values})
    ax.tick_params(axis='both', which='major', labelsize=size_labels)  # font size for tick labels
    ax.set_yticklabels(v_labels, va='center')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if bottom is not None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return f

def kernels_exp(impact_coefficients, decay_coefficients, events_labels=None, states_labels=None, path='',
                fig_name='kernels.pdf', title=None, palette=None, figsize=(9, 7), size_labels=16,
                size_values=14, size_legend=16, bottom=None, top=None, left=None, right=None, savefig=False,
                fig_array=None, fig=None,
                tmin=None, tmax=None, npoints=500, ymax=None, alpha=1, legend_pos=0, log_timescale=True,
                ls='-'):
    r"""
    Plots the kernels of a state-dependent Hawkes process.
    Here the kernels are assumed to be exponential, that is, :math:`k_{e'e}(t,x)=\alpha_{e'xe}\exp(-\beta_{e'xe}t)`.
    We plot the functions

    .. math::
        t\mapsto ||k_{e'e}(\cdot,x)||_{1,t} := \int _{0}^{t} k_{e'e}(s,x)ds.

    The quantity :math:`||k_{e'e}(\cdot,x)||_{1,t}` can be interpreted as the average number of events of type :math:`e`
    that are directly precipitated by an event of type :math:`e'` within :math:`t` units of time, under state :math:`x`.
    There is a subplot for each couple of event types :math:`(e',e)`.
    In each subplot, there is a curve for each possible state :math:`x`.

    :type impact_coefficients: 3D array
    :param impact_coefficients: the alphas :math:`\alpha_{e'xe}`.
    :type decay_coefficients: 3D array
    :param decay_coefficients: the betas :math:`\beta_{e'xe}`.
    :type events_labels: list of strings
    :param events_labels: labels of the event types.
    :type states_labels: list of strings
    :param states_labels: labels of the states.
    :type path: string
    :param path: where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :type events_labels: list of strings
    :type title: string
    :param title: suptitle.
    :type palette: list of colours
    :param palette: color palette, one color per state :math:`x`.
    :type figsize: (int, int)
    :param figsize: tuple (width, height).
    :type size_labels: int
    :param size_labels: fontsize of labels.
    :type size_values: int
    :param size_values: fontsize of tick labels.
    :type size_legend: int
    :param size_legend: fontsize of the legend.
    :type bottom: float
    :param bottom: between 0 and 1, adjusts the bottom margin, see matplotlib subplots_adjust.
    :type top: float
    :param top: between 0 and 1, adjusts the top margin, see matplotlib subplots_adjust.
    :type left: float
    :param left: between 0 and 1, adjusts the left margin, see matplotlib subplots_adjust.
    :type right: float
    :param right: between 0 and 1, adjusts the right margin, see matplotlib subplots_adjust.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :type fig_array: array of Axes
    :param fig_array: fig_array, where to plot the kernels (see matplotlib).
    :type fig: Figure
    :param fig: figure, where to plot the figure (see matplotlib).
    :type tmin: float
    :param tmin: we plot over the time interval [`tmin`, `tmax`].
    :type tmax: float
    :param tmax: we plot over the time interval [`tmin`, `tmax`].
    :type npoints: int
    :param npoints: number of points used to plot.
    :type ymax: float
    :param ymax: upper limit of the y axis.
    :type alpha: float
    :param alpha: between 0 and 1, transparency of the curves.
    :type legend_pos: int
    :param legend_pos: position of the legend in the array of figures.
    :type log_timescale: boolean
    :param log_timescale: set to False to plot with a linear timescale.g
    :type ls: string
    :param ls: the linestyle (see matplotlib).
    :rtype: Figure, array of Axes
    :return: the figure and array of figures (see matplotlib).
    """
    s = np.shape(impact_coefficients)
    number_of_event_types = s[0]
    number_of_states = s[1]
    beta_min = np.min(decay_coefficients)
    beta_max = np.max(decay_coefficients)
    t_max = tmax
    if tmax is None:
        t_max = -np.log(0.1) / beta_min
    t_min = tmin
    if tmin is None:
        t_min = -np.log(0.9) / beta_max
    tt = np.zeros(1)
    if log_timescale:
        order_min = np.floor(np.log10(t_min))
        order_max = np.ceil(np.log10(t_max))
        tt = np.logspace(order_min, order_max, num=npoints)
    else:
        tt = np.linspace(t_min, t_max, num=npoints)
    norm_max = ymax
    if ymax is None:
        norm_max = np.max(np.divide(impact_coefficients, decay_coefficients)) * 1.05
    if palette is None:
        palette = seaborn.color_palette('husl', n_colors=number_of_states)
    if fig_array is None:
        fig, fig_array = plt.subplots(number_of_event_types, number_of_event_types, sharex='col', sharey='row',
                                figsize=figsize)
    for e1 in range(number_of_event_types):
        for e2 in range(number_of_event_types):
            axes = None
            if number_of_event_types == 1:
                axes = fig_array
            else:
                axes = fig_array[e1, e2]
            for x in range(number_of_states):  # mean
                a = impact_coefficients[e1, x, e2]
                b = decay_coefficients[e1, x, e2]
                yy = a / b * (1 - np.exp(-b * tt))
                l = None
                if np.shape(states_labels) != ():
                    l = states_labels[x]
                axes.plot(tt, yy, color=palette[x], label=l, alpha=alpha, ls=ls)
            axes.tick_params(axis='both', which='major', labelsize=size_values)  # font size for tick labels
            if log_timescale:
                axes.set_xscale('log')
            axes.set_ylim(ymin=0, ymax=norm_max)
            axes.set_xlim(xmin=t_min, xmax=t_max)
            if np.shape(events_labels) != ():
                axes.set_title(events_labels[e1] + r' $\rightarrow$ ' + events_labels[e2], fontsize=size_labels)
            pos = e2 + number_of_event_types*e1
            if pos == legend_pos and np.shape(states_labels) != () :
                legend = axes.legend(frameon=1, fontsize=size_legend)
                legend.get_frame().set_facecolor('white')
    if title is not None:
        fig.suptitle(title, fontsize=size_labels)
    plt.tight_layout()
    if bottom is not None:
        plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return fig, fig_array

def sample_path(times, events, states, model, time_start, time_end, color_palette=None, labelsize=16, ticksize=14,
                legendsize=16, num=1000, s=12, savefig=False, path='', fig_name='sample_path.pdf'):
    r"""
    Plots a sample path along with the intensities.

    :type times: array of floats
    :param times: times when the events occur.
    :type events: array of int
    :param events: type of the event at each event time.
    :type states: array of int
    :param states: state process after each event time.
    :type model: :py:class:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp`
    :param model: the model that is used to compute the intensities.
    :type time_start: float
    :param time_start: time at which the plot starts.
    :type time_end: float
    :param time_end: time at which the plot ends.
    :type color_palette: list of colours
    :param color_palette: one colour per event type.
    :type labelsize: int
    :param labelsize: fontsize of labels.
    :type ticksize: int
    :param ticksize: fontsize of tick labels.
    :type legendsize: int
    :param legendsize: fontsize of the legend.
    :type num: int
    :param num: number of points used to plot.
    :type s: int
    :param s: size of the dots in the scatter plot of the events.
    :type savefig: boolean
    :param savefig: set to True to save the figure.
    :type path: string
    :param path:  where the figure is saved.
    :type fig_name: string
    :param fig_name: name of the file.
    :rtype: Figure, array of Axes
    :return: the figure and array of figures (see matplotlib).
    """
    if color_palette is None:
        color_palette = seaborn.color_palette('husl', n_colors=model.number_of_event_types)
    'Compute the intensities - this may require all the event times prior to start_time'
    compute_times = np.linspace(time_start, time_end, num=num)
    aggregated_times, intensities = model.intensities_of_events_at_times(compute_times, times, events, states)
    'We can now discard the times outside the desired time period'
    index_start = bisect.bisect_left(times, time_start)
    index_end = bisect.bisect_right(times, time_end)
    initial_state = 0
    if index_start > 0:
        initial_state = states[index_start-1]
    times = list(copy.copy(times[index_start:index_end]))
    events = list(copy.copy(events[index_start:index_end]))
    states = list(copy.copy(states[index_start:index_end]))
    f, fig_array = plt.subplots(2, 1, sharex='col')
    'Plot the intensities'
    ax = fig_array[1]
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # intensity_max = intensities.max() * 1.01
    for n in range(model.number_of_event_types):
        ax.plot(aggregated_times, intensities[n], linewidth=1, color=color_palette[n], label=model.events_labels[n])
    ax.set_ylim(ymin=0)
    ax.set_ylabel('Intensity', fontsize=labelsize)
    ax.set_xlabel('Time', fontsize=labelsize)
    legend = ax.legend(frameon=1, fontsize=legendsize)
    legend.get_frame().set_facecolor('white')
    'Plot the state process and the events'
    ax = fig_array[0]
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    # Plot the event times and types, one color per event type, y-coordinate corresponds to new state of the system
    color_map = ListedColormap(color_palette)
    ax.scatter(times, states, c=events, cmap=color_map, s=s, alpha=1, edgecolors='face',
               zorder=10)
    ax.set_xlim(xmin=time_start, xmax=time_end)
    ax.set_ylim(ymin=-0.1, ymax=model.number_of_states - 0.9)
    ax.set_yticks(range(model.number_of_states))
    ax.set_yticklabels(model.states_labels, fontsize=ticksize)
    ax.set_ylabel('State', fontsize=labelsize)
    # Plot the state process
    times.insert(0, time_start)
    states.insert(0, initial_state)
    times.append(time_end)
    states.append(states[-1])  # these two appends are required to plot until `time_end'
    ax.step(times, states, where='post', linewidth=1, color='grey', zorder=1)
    # Save the figure
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9)
    if savefig:
        entire_path = os.path.join(path, fig_name)
        plt.savefig(entire_path)
    return f, fig_array
