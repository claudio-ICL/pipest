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
path_tests = path_pipest+'/tests'
path_saved_tests = path_tests+'/saved_tests'


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

#parameters of model
n_states=[3,5]
number_of_event_types = 4 
n_events = number_of_event_types  # number of event types, $d_e$
n_levels = 2
upto_level = 2
time_start = 0.0
time_end = time_start + 1.0*60*60
#Optional parameters for "estimate"
max_imp_coef = 25.0
learning_rate = 0.0001
maxiter = 120
num_guesses = 6
num_processes = 20
batch_size = 5000
num_run_per_minibatch = 4
parallel=False
use_prange = True
type_of_preestim = 'ordinary_hawkes' # 'ordinary_hawkes' or 'nonparam'
#Optional parameters for "nonparam_estim"
num_quadpnts = 85
quad_tmax = 1.0
quad_tmin = 1.0e-1
num_gridpnts = 80
grid_tmax = 1.1
grid_tmin = 1.5e-1

def redirect_stdout(direction= 'from', # 'from' or 'to'
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

def instantiate_and_simulate():
    path_readout=path_saved_tests+'/'+this_test_model_name+'_simulation_readout.txt'
    now=datetime.datetime.now()
    message="I am executing {} --simulation".format(sys.argv[0])
    message+="\nDate of run: {}-{:02d}-{:02d} at {:02d}:{:02d}".format(now.year, now.month, now.day, now.hour, now.minute)
    fout,saveout=redirect_stdout(direction="from", message=message, path=path_readout)
    model = sd_hawkes_model.SDHawkes(
        number_of_event_types=n_events,
        number_of_lob_levels=n_levels,
        list_of_n_states=n_states,
        volume_imbalance_upto_level=upto_level,
        name_of_model=this_test_model_name)
    tot_n_states=model.state_enc.tot_n_states
    phis = model.state_enc.generate_random_transition_prob(n_events=n_events).astype(np.float)
    nus = 0.1*np.random.randint(low=15,high=20,size=n_events).astype(np.float)
    alphas = np.power(10,-np.random.uniform(low=1.0, high=1.2))*np.random.randint(low=0,high=4,size=(n_events, tot_n_states, n_events)).astype(np.float)
    betas = np.random.uniform(1.25025,2.1,size=(n_events, tot_n_states, n_events)).astype(np.float)
    gammas = np.random.uniform(low=1.25, high = 5.6,size=(tot_n_states,2*n_levels))

    model.set_hawkes_parameters(nus,alphas,betas)
    model.set_dirichlet_parameters(gammas)
    model.set_transition_probabilities(phis)

    print("\nSIMULATION\n")
    global time_start
    global time_end
    max_number_of_events = np.random.randint(low=7900, high=8010)
    times, events, states, volumes = model.simulate(
        time_start, time_end, max_number_of_events=max_number_of_events,
        add_initial_cond=True,
        store_results=True, report_full_volumes=False)
    time_end=float(times[-1])
    model.create_goodness_of_fit(type_of_input='simulated', parallel=True)
    model.goodness_of_fit.ks_test_on_residuals()
    model.goodness_of_fit.ad_test_on_residuals()
    model.dump(path=path_saved_tests)
    model.dump(name=this_test_model_name+"_simulation", path=path_saved_tests)
    now=datetime.datetime.now()
    message='\nSimulation terminates on {}-{:02d}-{:02d} at {:02d}:{:02d}\n'\
    .format(now.year, now.month, now.day, now.hour, now.minute)
    redirect_stdout(direction="to", message=message, fout=fout, saveout=saveout) 

def nonparam_preestim():
    path_readout=path_saved_tests+'/'+this_test_model_name+'_nonp_readout.txt'
    now=datetime.datetime.now()
    message="I am executing {} --nonparam_preestim".format(sys.argv[0])
    message+="\nDate of run: {}-{:02d}-{:02d} at {:02d}:{:02d}".format(now.year, now.month, now.day, now.hour, now.minute)
    fout,saveout=redirect_stdout(direction="from", message=message, path=path_readout)
     
    with open(path_saved_tests+'/'+this_test_model_name,'rb') as source:
        model=pickle.load(source)
    
    model.create_nonparam_estim(type_of_input='simulated',
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
    model.nonparam_estim.create_goodness_of_fit(parallel=False)
    model.dump(path=path_saved_tests)
    now=datetime.datetime.now()
    message='\nNon-parametric pre-estimation terminates on {}-{:02d}-{:02d} at {}:{:02d}\n'\
    .format(now.year, now.month, now.day, now.hour, now.minute)
    redirect_stdout(direction='to', message=message, fout=fout, saveout=saveout)
    
def estimate():
    with open(path_saved_tests+'/'+this_test_model_name, 'rb') as source:
        model=pickle.load(source)
    time_start = float(model.simulated_times[0])
    time_end = float(model.simulated_times[-1])
    for event_type in range(model.number_of_event_types):
        try:
            array_index=int(os.environ['PBS_ARRAY_INDEX'])
            switch = (event_type==array_index)
            print("array_index={}; event_type={}; switch={}".format(array_index, event_type,switch))
        except:
            print("No PBS_ARRAY_INDEX was found; 'switch' set to 'True'")
            switch = True
        if switch:
            model.set_name_of_model(type_of_paral+this_test_model_name+'_partial{}'.format(event_type))
            path_readout=path_saved_tests+'/'+type_of_paral+this_test_model_name+'_mle_readout_partial{}.txt'.format(event_type)
            now=datetime.datetime.now()
            message="I am executing {} --mle_estimation --{}".format(sys.argv[0], type_of_paral)
            message+="\nDate of run: {}-{:02d}-{:02d} at {:02d}:{:02d}".format(now.year, now.month, now.day, now.hour, now.minute)
            fout,saveout=redirect_stdout(direction="from", message=message, path=path_readout)
            "Initialise the class"
            model.create_mle_estim(type_of_input='simulated', store_trans_prob=False)
            "Set the estimation"    
            if type_of_preestim == 'nonparam':
                list_init_guesses = model.nonparam_estim.produce_list_init_guesses_for_mle_estimation(
                    num_additional_random_guesses = max(1,num_guesses//2), max_imp_coef=max_imp_coef) 
                pre_estim_ord_hawkes = False
            else:
                list_init_guesses = []
                pre_estim_ord_hawkes = False
                if type_of_preestim == 'ordinary_hawkes':
                    pre_estim_ord_hawkes = True
            model.mle_estim.set_estimation_of_hawkes_param(
                time_start, time_end,
                list_of_init_guesses = list_init_guesses,
                max_imp_coef = max_imp_coef,
                learning_rate = learning_rate,
                maxiter=maxiter,
                number_of_additional_guesses = num_guesses,
                parallel=parallel,
                pre_estim_ord_hawkes=pre_estim_ord_hawkes,
                pre_estim_parallel=parallel,
                use_prange = use_prange,
                number_of_attempts = 3,
                num_processes = num_processes,
                batch_size = batch_size,
                num_run_per_minibatch = num_run_per_minibatch,
            )
            "Launch estimation"
            model.mle_estim.launch_estimation_of_hawkes_param(e=event_type)
            model.dump(path=path_saved_tests) 
            now=datetime.datetime.now()
            message='\nEstimation of component_e {} terminates on {}-{:02d}-{:02d} at {}:{:02d}\n'\
            .format(event_type, now.year, now.month, now.day, now.hour, now.minute)
            redirect_stdout(direction='to',message=message, fout=fout, saveout=saveout)
            
def merge_from_partial():
    path_readout=path_saved_tests+'/'+type_of_paral+this_test_model_name+'_merge_readout.txt'
    now=datetime.datetime.now()
    message="I am executing {} --merge".format(sys.argv[0])
    message+="\nDate of run: {}-{:02d}-{:02d} at {:02d}:{:02d}".format(now.year, now.month, now.day, now.hour, now.minute)
    fout,saveout=redirect_stdout(direction="from", message=message, path=path_readout)    
    list_of_partial_names=[type_of_paral+this_test_model_name+'_partial{}'.format(e) for e in range(number_of_event_types)]
    partial_models=[]
    with open(path_saved_tests+'/'+this_test_model_name,'rb') as source:
        print("I am loading "+path_saved_tests+'/'+this_test_model_name)
        model=pickle.load(source)
    model.set_name_of_model(type_of_paral+this_test_model_name)    
    for mn in list_of_partial_names:
        with open(path_saved_tests+'/'+mn,'rb') as source:
            partial_model=pickle.load(source)
            partial_models.append(partial_model)
    model.initialise_from_partial(partial_models,
                                  dump_after_merging=True,
                                  path=path_saved_tests)  
    now=datetime.datetime.now()
    message='\nMerging has been completed  on {}-{:02d}-{:02d} at {:02d}:{:02d}\n'\
    .format(now.year,now.month,now.day,now.hour,now.minute)
    redirect_stdout(direction='to',message=message,fout=fout, saveout=saveout)
        
def main():  
    global action
    action=str(sys.argv[1])
    print("\n$python {} {}".format(sys.argv[0],action))
    global this_test_model_name     
    if action=='-s' or action=='--simulate':
        now = datetime.datetime.now()
        this_test_model_name = 'test_model_{}-{:02d}-{:02d}_{:02d}{:02d}'\
        .format(now.year,now.month,now.day,now.hour,now.minute)
        instantiate_and_simulate()
        with open(path_saved_tests+'/name_test_estim_', 'wb') as outfile:
            pickle.dump(this_test_model_name, outfile)
    else:
        with open(path_saved_tests+'/name_test_estim_', 'rb') as source:
            this_test_model_name = pickle.load(source)
        print("this_test_model_name: {}".format(this_test_model_name))
        global type_of_paral
        try:
            type_of_paral = str(sys.argv[2])
            global parallel
            global use_prange
            if type_of_paral =="pool_" or type_of_paral =="pool":
                parallel = True
                use_prange = False
            elif type_of_paral =="prange_" or type_of_paral =="prange":
                parallel = False
                use_prange = True
            elif type_of_paral =="plain_" or type_of_paral =="plain":
                parallel = False
                use_prange = False
            else:
                print("type_of_paral: {}".format(type_of_paral))
                raise ValueError("type_of_paral not recognised")
        except:
            if parallel:
                type_of_paral = "pool_"
            elif use_prange:
                type_of_paral = "prange_"
            else:
                type_of_paral = "plain_"
        print("type_of_paral: {}".format(type_of_paral))
        if action=='-p' or (action=='--preestim' or action=='--nonparam_preestim'):
            if type_of_preestim == 'nonparam':
                nonparam_preestim()
            else:
                print("type_of_preestim: {}. Non-parametric estimation is being skipped".format(type_of_preestim))
        elif action=='-e' or (action=='--estimate' or action=='--mle'):
            estimate()
        elif action=='-m' or action=='--merge':
            merge_from_partial()
            #os.remove(path_saved_tests+'/name_test_estim_')
        else:
            print("action: {}".format(action))
            raise ValueError("action not recognised")
    print("{}: main() end of file\n\n".format(str(sys.argv[0])))     

        
        
if __name__=='__main__':
    main()              
