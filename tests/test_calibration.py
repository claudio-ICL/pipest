#!/usr/bin/env python
# coding: utf-8




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
path_lobster_data=path_lobster+'/data'
path_lobster_pyscripts=path_lobster+'/py_scripts'
path_models=path_pipest+'/models'
path_tests=path_pipest+'/tests'
path_saved_tests=path_tests+'/saved_tests'


import time
import datetime
import sys
sys.path.append(path_sdhawkes+'/resources/')
sys.path.append(path_sdhawkes+'/modelling/')
sys.path.append(path_lobster_pyscripts+'/')

import pickle
import numpy as np
import pandas as pd
import bisect
import copy


import model as sd_hawkes_model
import lob_model
import computation
import simulation
import goodness_of_fit
import mle_estimation as mle_estim
import prepare_from_lobster as from_lobster
import nonparam_estimation as nonparam_estim


symbol='INTC'
date='2019-01-04'
initial_time=float(36000.0)
final_time=float(36250.0)
event_type=int(2)

#Optional parameters for "calibrate"
type_of_preestim='ordinary_hawkes' #'ordinary_hawkes' or 'nonparam'
max_imp_coef = 15.0
learning_rate = 0.0001
maxiter = 10
num_guesses = 6
num_processes = 8
#Optional parameters for "nonparam_estim"
num_quadpnts = 50
quad_tmax = 1.0
quad_tmin = 1.0e-1
num_gridpnts = 60
grid_tmax = 1.1
grid_tmin = 1.5e-1
    
def main():    
    global now
    name_of_model_nonp=name_of_model+'_nonp'
    path_mnonp=path_mmodel+'_nonp'
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
    if type_of_preestim == 'nonparam':
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
        model.nonparam_estim.estimate_hawkes_kernel(store_L1_norm=False,
                                   use_filter=True, enforce_positive_g_hat=True,
                                   filter_cutoff=50.0, filter_scale=30.0, num_addpnts_filter=3000)
        model.nonparam_estim.fit_powerlaw(compute_L1_norm=True,ridge_param=1.0e-02, tol=1.0e-7)
        model.nonparam_estim.store_base_rates()
        run_time+=time.time()
        model.nonparam_estim.store_runtime(run_time)
        model.nonparam_estim.create_goodness_of_fit()
#         model.dump(name=name_of_model_nonp,path=path_models+'/'+symbol)
        n=datetime.datetime.now()
        message='\nNon-parametric pre-estimation terminates on {}-{:02d}-{:02d} at {}:{:02d}\n'\
        .format(n.year, n.month, n.day, n.hour, n.minute)
        print(message)
    
        
    
    print("\nMLE ESTIMATION\n")
    name_of_model_partial=name_of_model+'_partial{}'.format(event_type)
    path_mpartial=path_mmodel+'_partial{}'.format(event_type)
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='I am calibrating on lobster\n'
    message+='symbol={}, date={}, time_window={}\n'.format(symbol,date,time_window)
    message+='event type: {}'.format(event_type)
    print(message)
    model.calibrate_on_input_data(
        partial=True, e=event_type, name_of_model=name_of_model_partial,
        type_of_preestim=type_of_preestim,
        max_imp_coef = max_imp_coef,
        learning_rate = learning_rate, maxiter = maxiter, num_of_random_guesses=num_guesses,
        parallel=True,
        number_of_attempts = 2, num_processes = num_processes,
        skip_estim_of_state_processes=True,
        dump_after_calibration=True
    )
    n=datetime.datetime.now()
    message='\nCalibration of event_type {} terminates on {}-{:02d}-{:02d} at {}:{:02d}\n'\
    .format(event_type, n.year, n.month, n.day, n.hour, n.minute)
    print(message)    











if __name__=='__main__':
    print("I am executing test_calibration.py")
    now=datetime.datetime.now()
    print('\ndate of run: {}-{:02d}-{:02d} at {:02d}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute))
    this_test_readout=path_saved_tests+'/nonparam_test_{}-{:02d}-{:02d}_{:02d}{:02d}_readout'.format(now.year,now.month,now.day, now.hour, now.minute)
    this_test_model=path_saved_tests+'/test_model_{}-{:02d}-{:02d}_{:02d}{:02d}'.format(
        now.year,now.month,now.day,now.hour,now.minute
    )
    global time_window
    time_window=str('{}-{}'.format(int(initial_time),int(final_time)))
    global path_mdata
    path_mdata=path_lobster_data+'/{}/{}_{}_{}'.format(symbol,symbol,date,time_window)
    global name_of_model
    name_of_model=symbol+'_'+date+'_'+time_window
    global path_mmodel
    path_mmodel=path_models+'/'+symbol+'/'+name_of_model        
    main()
    now=datetime.datetime.now()
    print("Test terminates on {}-{:02d}-{:02d} at {:02d}:{:02d}".format(
        now.year,now.month,now.day,now.hour,now.minute
    ))
    print("\nEND OF TEST")
