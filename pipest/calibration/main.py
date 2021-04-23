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
path_sdhawkes=path_pipest+'/sdhawkes'
path_lobster=path_pipest+'/lobster'
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
#Optional parameters for "calibrate"
type_of_preestim='ordinary_hawkes' 
max_imp_coef = 20.0
learning_rate = 0.00005
maxiter = 50
num_guesses = 1
parallel=False
use_prange=True
num_processes = 8
batch_size = 40000
num_run_per_minibatch = 2
number_of_attempts = 3  

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
                                          )
    else:
        LOB,messagefile=from_lobster.load_from_pickleFiles(symbol,date)

    LOB,messagefile=from_lobster.select_subset(LOB,messagefile,
                                  initial_time=initial_time,
                                  final_time=final_time)

    print('\nDATA CLEANING\n')

    man_mf=from_lobster.ManipulateMessageFile(
         LOB=LOB, 
         mf=messagefile,
         symbol=symbol,
         date=date,
         )     

    man_ob=from_lobster.ManipulateOrderBook(
        LOB=man_mf.LOB,symbol=symbol,date=date,
        ticksize=man_mf.ticksize,n_levels=man_mf.n_levels,volume_imbalance_upto_level=2)
    man_ob.set_states(midprice_changes = np.array(man_mf.messagefile['sign_delta_mid'].values, dtype=np.int))

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
    



    
def calibrate(event_type = 0):
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
            
def merge_from_partial(adjust_base_rates = False, leading_coef = 0.66):
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='I am merging from partial models\n'
    message+='symbol={}, date={}, time_window={}'.format(symbol,date,time_window)
    path_readout=path_mmodel+'_merge_readout.txt'
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
        try:
            with open(path_models+'/{}/{}_{}/'.format(symbol,symbol,date)+mn,'rb') as source:
                model=pickle.load(source)
        except:
            with open(path_models+'/{}/'.format(symbol,)+mn,'rb') as source:
                model=pickle.load(source)
            assert model.data.symbol == symbol
            assert model.data.date == date
            assert model.calibration.type_of_preestim == type_of_preestim
            partial_models.append(model)
    if type_of_preestim == 'nonparam':
        MODEL.store_nonparam_estim_class(model.nonparam_estim)
    MODEL.initialise_from_partial_calibration(partial_models, set_parameters=True, adjust_base_rates = True, dump_after_merging=True, name_of_model=name_of_model)       
    n=datetime.datetime.now()
    message='\nMerging has been completed  on {}-{:02d}-{:02d} at {}:{:02d}'.format(n.year,n.month,n.day,n.hour,n.minute)
    redirect_stdout(direction='to',message=message,fout=fout, saveout=saveout) 
def uncertainty_quantification(input_model_name=''):
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='I am performing uncertainty quantification\n'
    message+='symbol={}, date={}, time_window={}'.format(symbol,date,time_window)
    path_readout=path_mmodel+'_uq_readout.txt'
    fout,saveout=redirect_stdout(direction='from',path=path_readout,message=message)    
    if input_model_name=='':
        input_model_name='{}_sdhawkes_{}_{}'.format(symbol, date, time_window)
    try:
        with open(path_models+'/{}/{}_{}/'.format(symbol,symbol,date)+input_model_name,'rb') as source:
            input_model=pickle.load(source)
    except:
        with open(path_models+'/{}/'.format(symbol,)+input_model_name,'rb') as source:
            input_model=pickle.load(source)
    assert input_model.data.symbol == symbol
    assert input_model.data.date == date

    model=sd_hawkes_model.SDHawkes(
        name_of_model = input_model_name+'_uq',
        number_of_event_types=input_model.number_of_event_types,
        number_of_states=input_model.number_of_states,
        number_of_lob_levels=input_model.n_levels,
        volume_imbalance_upto_level = input_model.volume_enc.volume_imbalance_upto_level,
        list_of_n_states=input_model.state_enc.list_of_n_states,
        st1_deflationary=input_model.state_enc.st1_deflationary,
        st1_inflationary=input_model.state_enc.st1_inflationary,
        st1_stationary=input_model.state_enc.st1_stationary
    )
    model.get_input_data(input_model.data)
    try:
        model.store_nonparam_estim_class(input_model.nonparam_estim)
    except:
        pass
    model.store_mle_estim(input_model.mle_estim)
    model.store_calibration(input_model.calibration)
    model.store_goodness_of_fit(input_model.goodness_of_fit)
    model.set_hawkes_parameters(input_model.base_rates,
                                input_model.impact_coefficients,
                                input_model.decay_coefficients)
    model.set_transition_probabilities(input_model.transition_probabilities)
    model.set_dirichlet_parameters(input_model.dirichlet_param)
    model.create_uq()
    time_start=0.0
    time_end=3600.0
    max_events=40000
    model.uncertainty_quantification.simulate(
            time_start, time_end, max_number_of_events=max_events)
    model.uncertainty_quantification.create_goodness_of_fit()
    model.uncertainty_quantification.calibrate_on_simulated_data(
        type_of_preestim = type_of_preestim,
        max_imp_coef = max_imp_coef,
        learning_rate = learning_rate,
        maxiter = maxiter,
        num_of_random_guesses = num_guesses,
        parallel=parallel, use_prange = use_prange,
        number_of_attempts = number_of_attempts,
        num_processes = num_processes,
        batch_size = batch_size,
        num_run_per_minibatch = num_run_per_minibatch,
    )
    model.dump(path=path_models+'/{}'.format(symbol))  
    n=datetime.datetime.now()
    message='\nUncertainty quantification completed  on {}-{:02d}-{:02d} at {}:{:02d}'.format(n.year,n.month,n.day,n.hour,n.minute)
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
    print("$python {} {} {} {} {} {}".format(sys.argv[0],symbol,date,int(initial_time),int(final_time),action))
    global path_mdata
    path_mdata=path_lobster_data+'/{}/{}_{}_{}'.format(symbol,symbol,date,time_window)
    global name_of_model
    name_of_model=symbol+'_'+date+'_'+time_window
    global path_mmodel
    path_mmodel=path_models+'/'+symbol+'/'+symbol+'_'+date+'/'+name_of_model        
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
    elif action=='-uq' or action=='--uncertainty_quant':
        input_model_name="{}_sdhawkes_{}_{}".format(symbol,date,time_window)
        uncertainty_quantification(input_model_name)
    else:
        print("action: {}".format(action))
        raise ValueError("action not recognised")
    
if __name__=='__main__':
    main()              
