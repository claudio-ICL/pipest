#!/usr/bin/env python
# coding: utf-8

import os
path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<4):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    print("path_pipest not found. Instead: {}".format(path_pipest))
    raise ValueError("path_pipest not found.")
path_models=path_pipest+'/models'    
path_sdhawkes=path_pipest+'/sdhawkes_powerlaw'
path_lobster=path_pipest+'/lobster_for_sdhawkes'
path_lobster_data=path_lobster+'/data'
path_lobster_pyscripts=path_lobster+'/py_scripts'


import numpy as np
import pandas as pd
import pickle
import datetime
import time
import datetime

import sys
sys.path.append(path_lobster_pyscripts+'/')
sys.path.append(path_sdhawkes+'/modelling/')
sys.path.append(path_sdhawkes+'/resources/')

import model as sd_hawkes_model
import lob_model
import prepare_from_lobster as from_lobster


number_of_event_types=4
#Optional prameters for "read_lobster"
first_read_fromLOBSTER=True
dump_after_reading=False
add_level_to_messagefile=True
#Optional parameters for "calibrate"
type_of_preestim='nonparam' #'ordinary_hawkes' or 'nonparam'
max_imp_coef = 20.0
learning_rate = 0.0001
maxiter = 40
num_guesses = 8
parallel=False
use_prange=True
num_processes = 8
batch_size = 15000
num_run_per_minibatch = 3  
#Optional parameters for "nonparam_estim"
num_quadpnts = 80
quad_tmax = 1.0
quad_tmin = 1.0e-1
num_gridpnts = 100
grid_tmax = 1.1
grid_tmin = 1.5e-1

def redirect_stdout(direction= 'from', # or 'to'
                    message= '',
                    path='',
                    fout=None,saveout=None):
    if direction=='from':
        print(message)
        print("stdout is being redirected to {}".format(path))
        saveout=sys.stdout
        fout=open(path,'w')
        sys.stdout = fout
        print(message)
        return fout, saveout
    elif direction=='to':
        print(message)
        fout.close()
        sys.stdout=saveout
        print(message)
    else:
        print("WARNINNG: redirect_stdout failed! direction={} not recognised".format(direction))
        print(message)        

def read_lobster():
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='I am reading from lobster\n'
    message+='symbol={}, date={}, time_window={}'.format(symbol,date,time_window)
    path_readout=path_mdata+'_readout.txt'
    fout,saveout=redirect_stdout(direction='from',path=path_readout,message=message)  

    if (first_read_fromLOBSTER):
        LOB,messagefile=from_lobster.read_from_LOBSTER(symbol,date,
                                          dump_after_reading=dump_after_reading,
                                          add_level_to_messagefile=add_level_to_messagefile)
    else:
        LOB,messagefile=from_lobster.load_from_pickleFiles(symbol,date)

    LOB,messagefile=from_lobster.select_subset(LOB,messagefile,
                                  initial_time=initial_time,
                                  final_time=final_time)

    print('\nDATA CLEANING\n')
    aggregate_time_stamp=True
    eventTypes_to_aggregate=[1,2,3,4,5]
    eventTypes_to_drop_with_nonunique_time=[]
    eventTypes_to_drop_after_aggregation=[3]
    only_4_events=False
    separate_directions=True
    separate_13_events=False
    separate_31_events=False
    separate_41_events=True
    separate_34_events=True
    separate_43_events=True
    equiparate_45_events_with_same_time_stamp=True
    drop_all_type3_events_with_nonunique_time=False
    drop_5_events_with_same_time_stamp_as_4=True
    drop_5_events_after_aggregation=True
    tolerance_when_dropping=1.0e-7
    add_hawkes_marks=True
    clear_same_time_stamp=True
    num_iter=4

    man_mf=from_lobster.ManipulateMessageFile(
         LOB,messagefile,
         symbol=symbol,
         date=date,
         aggregate_time_stamp=aggregate_time_stamp,
         eventTypes_to_aggregate=eventTypes_to_aggregate,  
         only_4_events=only_4_events,
         separate_directions=separate_directions,
         separate_13_events=separate_13_events,
         separate_31_events=separate_31_events,
         separate_41_events=separate_41_events,
         separate_34_events=separate_34_events,
         separate_43_events=separate_43_events,
         equiparate_45_events_with_same_time_stamp=
         equiparate_45_events_with_same_time_stamp,
         eventTypes_to_drop_with_nonunique_time=eventTypes_to_drop_with_nonunique_time,
         eventTypes_to_drop_after_aggregation=eventTypes_to_drop_after_aggregation,
         drop_all_type3_events_with_nonunique_time=drop_all_type3_events_with_nonunique_time,  
         drop_5_events_with_same_time_stamp_as_4=drop_5_events_with_same_time_stamp_as_4,
         drop_5_events_after_aggregation=drop_5_events_after_aggregation,
         tolerance_when_dropping=tolerance_when_dropping,
         clear_same_time_stamp=clear_same_time_stamp,
         add_hawkes_marks=add_hawkes_marks,
         num_iter=num_iter
    )

    man_ob=from_lobster.ManipulateOrderBook(
        man_mf.LOB_sdhawkes,symbol=symbol,date=date,
        ticksize=man_mf.ticksize,n_levels=man_mf.n_levels,volume_imbalance_upto_level=2)

    data=from_lobster.DataToStore(man_ob,man_mf,time_origin=initial_time)
    sym = data.symbol
    d = data.date
    t_0 = data.initial_time
    t_1 = data.final_time
    assert sym==symbol
    assert date==d
    assert number_of_event_types==data.number_of_event_types
    if not np.isclose(t_0,initial_time,rtol=1.0,atol=1.0):
        message="data.initial_time={}, whereas global initial_time={}".format(t_0,initial_time)
        print(message)
        raise ValueError(message)
    if not np.isclose(t_1,final_time,rtol=1.0,atol=1.0):
        message="data.final_time={}, whereas global final_time={}".format(t_1,final_time)
        print(message)
        raise ValueError(message)

    message='Data is being stored in {}'.format(path_mdata)
    with open(path_mdata, 'wb') as outfile:
        pickle.dump(data,outfile)
    redirect_stdout(direction='to',message=message,fout=fout,saveout=saveout)
    

def nonparam_preestim():
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='I am pre-estimating the model using non-parametric procedure\n'
    message+='symbol={}, date={}, time_window={}'.format(symbol,date,time_window)
    name_of_model_nonp=name_of_model+'_nonp'
    path_mnonp=path_mmodel+'_nonp'
    path_readout=path_mnonp+'_readout.txt'
    fout,saveout=redirect_stdout(direction='from',path=path_readout,message=message)    
    with open(path_mdata,'rb') as source:
        data=pickle.load(source)
    assert symbol==data.symbol    
    assert date==data.date
   
    model=sd_hawkes_model.SDHawkes(
        number_of_event_types=data.number_of_event_types,  number_of_states = data.number_of_states,
        number_of_lob_levels=data.n_levels, volume_imbalance_upto_level = data.volume_enc.volume_imbalance_upto_level,
        list_of_n_states=data.state_enc.list_of_n_states, st1_deflationary=data.state_enc.st1_deflationary,
        st1_inflationary=data.state_enc.st1_inflationary, st1_stationary=data.state_enc.st1_stationary
    )
    model.get_input_data(data)
    model.create_nonparam_estim(type_of_input='empirical',
                                num_quadpnts = num_quadpnts,
                                quad_tmax = quad_tmax,
                                quad_tmin = quad_tmin,
                                num_gridpnts = num_gridpnts,
                                grid_tmax = grid_tmax,
                                grid_tmin = grid_tmin,
                                two_scales=True,
                                tol=1.0e-7
                               ) 
    run_time = -time.time()
    model.nonparam_estim.estimate_hawkes_kernel(
        parallel=True,
        store_L1_norm=False, use_filter=True, enforce_positive_g_hat=True,
        filter_cutoff=50.0, filter_scale=30.0, num_addpnts_filter=3000)
    model.nonparam_estim.fit_powerlaw(compute_L1_norm=True,ridge_param=1.0e-02, tol=1.0e-7)
    model.nonparam_estim.store_base_rates()
    run_time+=time.time()
    model.nonparam_estim.store_runtime(run_time)
    model.nonparam_estim.create_goodness_of_fit()
    model.dump(name=name_of_model_nonp,path=path_models+'/'+symbol)
    n=datetime.datetime.now()
    message='\nNon-parametric pre-estimation terminates on {}-{:02d}-{:02d} at {}:{:02d}\n'\
    .format(n.year, n.month, n.day, n.hour, n.minute)
    redirect_stdout(direction='to', message=message, fout=fout, saveout=saveout)


    
def calibrate(event_type = 0):
    if type_of_preestim == 'nonparam':
        with open(path_mmodel+'_nonp', 'rb') as source:
            model=pickle.load(source)
    elif type_of_preestim == 'ordinary_hawkes':        
        with open(path_mdata,'rb') as source:
            data=pickle.load(source)
        assert symbol==data.symbol    
        assert date==data.date
        model=sd_hawkes_model.SDHawkes(
            number_of_event_types=data.number_of_event_types,  number_of_states = data.number_of_states,
            number_of_lob_levels=data.n_levels, volume_imbalance_upto_level = data.volume_enc.volume_imbalance_upto_level,
            list_of_n_states=data.state_enc.list_of_n_states, st1_deflationary=data.state_enc.st1_deflationary,
            st1_inflationary=data.state_enc.st1_inflationary, st1_stationary=data.state_enc.st1_stationary
        )
        model.get_input_data(data)
    else:
        print("type_of_preestim={}".format(type_of_preestim))
        raise ValueError("type_of_preestim not recognised")

    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='I am calibrating on lobster\n'
    message+='symbol={}, date={}, time_window={}\n'.format(symbol,date,time_window)
    message+='event type: {}'.format(event_type)
    name_of_model_partial=name_of_model+'_partial{}'.format(event_type)
    path_mpartial=path_mmodel+'_partial{}'.format(event_type)
    path_readout=path_mpartial+'_readout.txt'
    fout,saveout=redirect_stdout(direction='from',message=message,path=path_readout)
    model.calibrate_on_input_data(
        e=event_type,
        name_of_model=name_of_model_partial,
        type_of_preestim=type_of_preestim,
        max_imp_coef = max_imp_coef,
        learning_rate = learning_rate, maxiter = maxiter, num_of_random_guesses=num_guesses,
        parallel=parallel,
        use_prange=use_prange,
        number_of_attempts = 2, num_processes = num_processes,
        batch_size = batch_size, num_run_per_minibatch = num_run_per_minibatch,
        store_trans_prob=False, store_dirichlet_param=False,
        dump_after_calibration=True
    )
    n=datetime.datetime.now()
    message='\nCalibration of event_type {} terminates on {}-{:02d}-{:02d} at {}:{:02d}\n'\
    .format(event_type, n.year, n.month, n.day, n.hour, n.minute)
    redirect_stdout(direction='to',message=message, fout=fout, saveout=saveout)
            
def merge_from_partial():
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='I am merging from partial models\n'
    message+='symbol={}, date={}, time_window={}'.format(symbol,date,time_window)
    path_readout=path_mmodel+'_readout'
    fout,saveout=redirect_stdout(direction='from',path=path_readout,message=message)    
    list_of_partial_names=[name_of_model+'_partial{}'.format(e) for e in range(number_of_event_types)]
    partial_models=[]
    with open(path_mdata,'rb') as source:
        data=pickle.load(source)
        assert data.symbol == symbol
        assert data.date == date
        assert data.number_of_event_types==number_of_event_types
    MODEL=sd_hawkes_model.SDHawkes(
        number_of_event_types=data.number_of_event_types, number_of_states=data.number_of_states,
        number_of_lob_levels=data.n_levels,volume_imbalance_upto_level = data.volume_enc.volume_imbalance_upto_level,
        list_of_n_states=data.state_enc.list_of_n_states, st1_deflationary=data.state_enc.st1_deflationary,
        st1_inflationary=data.state_enc.st1_inflationary, st1_stationary=data.state_enc.st1_stationary
    )
    MODEL.get_input_data(data)    
    for mn in list_of_partial_names:
        with open(path_models+'/{}/'.format(symbol)+mn,'rb') as source:
            model=pickle.load(source)
            assert model.data.symbol == symbol
            assert model.data.date == date
            assert model.calibration.type_of_preestim == type_of_preestim
            partial_models.append(model)
    if type_of_preestim == 'nonparam':
        MODEL.store_nonparam_estim_class(model.nonparam_estim)
    MODEL.initialise_from_partial_calibration(partial_models, set_parameters=True, dump_after_merging=True, name_of_model=name_of_model)  
    n=datetime.datetime.now()
    message='\nMerging has been completed  on {}-{:02d}-{:02d} at {}:{:02d}'.format(n.year,n.month,n.day,n.hour,n.minute)
    redirect_stdout(direction='to',message=message,fout=fout, saveout=saveout) 


def main():  
    global symbol
    symbol=str(sys.argv[1])
    global date
    date=str(sys.argv[2])
    global initial_time
    initial_time=float(sys.argv[3])
    global final_time
    final_time=float(sys.argv[4])
    global action
    action=str(sys.argv[5])
    global time_window
    time_window=str('{}-{}'.format(int(initial_time),int(final_time)))
    print("{} {} {} {} {} {}".format(sys.argv[0],symbol,date,initial_time,final_time,action))
    global path_mdata
    path_mdata=path_lobster_data+'/{}/{}_{}_{}'.format(symbol,symbol,date,time_window)
    global name_of_model
    name_of_model=symbol+'_'+date+'_'+time_window
    global path_mmodel
    path_mmodel=path_models+'/'+symbol+'/'+name_of_model        
    if action=='-r' or action=='--read':
        read_lobster()
    elif action=='-p' or (action=='--preestim' or action=='--nonparam_preestim'):
        if type_of_preestim=='nonparam':
            nonparam_preestim()
    elif action=='-c' or action=='--calibrate':
        if not str(sys.argv[6])=='-e':
            print("Example of usage for event_type 0:\n  {} {} {} {} {} {} -e 0".format(sys.argv[0],symbol,date,initial_time,final_time,action))
            raise ValueError("User need to specify the event type for partial calibration")
        event_type = int(sys.argv[7])    
        calibrate(event_type)
    elif action=='-m' or action=='--merge':
        merge_from_partial()
    else:
        print("action: {}".format(action))
        raise ValueError("action not recognised")
    
if __name__=='__main__':
    main()              
