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
path_perf = path_tests + '/performance_measurements'


import numpy as np
import pandas as pd
import pickle
import datetime
import time
import datetime
import timeit

import sys
sys.path.append(path_lobster_pyscripts+'/')
sys.path.append(path_sdhawkes+'/modelling/')
sys.path.append(path_sdhawkes+'/resources/')
sys.path.append(path_perf+'/')

import model as sd_hawkes_model
import lob_model
import measure_exectime

#parameters of model
n_states=[3,5]
number_of_event_types = 4 
n_events = number_of_event_types  # number of event types, $d_e$
n_levels = 2
upto_level = 2
time_start = 0.0
time_end = time_start + 1.0*60*60

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
    path_readout=path_perf+'/'+this_test_model_name+'_simulation_readout.txt'
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
    max_number_of_events = np.random.randint(low=4900, high=5050)
    times, events, states, volumes = model.simulate(
        time_start, time_end, max_number_of_events=max_number_of_events,
        add_initial_cond=True,
        store_results=True, report_full_volumes=False)
    time_end=float(times[-1])
    model.create_goodness_of_fit(type_of_input='simulated', parallel=True)
    model.goodness_of_fit.ks_test_on_residuals()
    model.goodness_of_fit.ad_test_on_residuals()
    model.dump(path=path_saved_tests)
    model.dump(path=path_saved_tests)
    now=datetime.datetime.now()
    message='\nSimulation terminates on {}-{:02d}-{:02d} at {:02d}:{:02d}\n'\
    .format(now.year, now.month, now.day, now.hour, now.minute)
    redirect_stdout(direction="to", message=message, fout=fout, saveout=saveout) 

def measure_ESSE(timeit_number = 100):
    timeit_number = max(1,int(timeit_number)) 
    path_readout=path_perf+'/'+this_test_model_name+'_ESSE_readout.txt'
    now=datetime.datetime.now()
    message="I am executing {} --ESSE".format(sys.argv[0])
    message+="\nDate of run: {}-{:02d}-{:02d} at {:02d}:{:02d}".format(now.year, now.month, now.day, now.hour, now.minute)
#     fout,saveout=redirect_stdout(direction="from", message=message, path=path_readout)
    print(message)
    with open(path_saved_tests+'/'+this_test_model_name, 'rb') as source:
        model = pickle.load(source)
    global meas    
    meas = measure_exectime.PerformanceMeasure(model)
    exectimes_prange = []
    exectimes_plain = []
    for e in range(model.number_of_event_types):
        exectime_prange = timeit.timeit('meas.target_ESSE({})'.format(e), globals=globals(), number = timeit_number)
        exectimes_prange.append(exectime_prange)
        exectime_plain = timeit.timeit('meas.target_plain_ESSE({})'.format(e), globals=globals(), number = timeit_number)
        exectimes_plain.append(exectime_plain)
    print("exectimes_plain (no prange):\n{}".format(exectimes_plain))    
    print("exectimes_prange:\n{}".format(exectimes_prange))
    now=datetime.datetime.now()
    message='\nmeasure_ESSE() terminates on {}-{:02d}-{:02d} at {:02d}:{:02d}\n'\
    .format(now.year, now.month, now.day, now.hour, now.minute)
    print(message)
#     redirect_stdout(direction="to", message=message, fout=fout, saveout=saveout)    
    
def measure_loglikelihood(timeit_number = 20):
    path_readout=path_perf+'/'+this_test_model_name+'_loglikelihood_readout.txt'
    now=datetime.datetime.now()
    message="I am executing {} --loglikelihood".format(sys.argv[0])
    message+="\nDate of run: {}-{:02d}-{:02d} at {:02d}:{:02d}".format(now.year, now.month, now.day, now.hour, now.minute)
#     fout,saveout=redirect_stdout(direction="from", message=message, path=path_readout)
    print(message)
    with open(path_saved_tests+'/'+this_test_model_name, 'rb') as source:
        model = pickle.load(source)
    global meas    
    meas = measure_exectime.PerformanceMeasure(model)
    exectimes_prange = []
    exectimes_plain = []
    for e in range(model.number_of_event_types):
        exectime_plain = timeit.timeit("meas.target_loglikelihood({},False)".format(e),
                                       globals = globals(), number = timeit_number)
        exectimes_plain.append(exectime_plain)
        exectime_prange = timeit.timeit("meas.target_loglikelihood({},True)".format(e),
                                        globals=globals(), number = timeit_number)
        exectimes_prange.append(exectime_prange)
    print("exectimes_plain (no prange):\n{}".format(exectimes_plain))    
    print("exectimes_prange:\n{}".format(exectimes_prange))
    now=datetime.datetime.now()
    message='\nmeasure_loglikelihood() terminates on {}-{:02d}-{:02d} at {:02d}:{:02d}\n'\
    .format(now.year, now.month, now.day, now.hour, now.minute)
    print(message)
#     redirect_stdout(direction="to", message=message, fout=fout, saveout=saveout)     
    
def main():  
    global action
    action=str(sys.argv[1])
    print("\n$python {} {}".format(sys.argv[0],action))
    global this_test_model_name     
    if action=='s' or action=='--simulate':
        now = datetime.datetime.now()
        this_test_model_name = 'performance_test_{}-{:02d}-{:02d}_{:02d}{:02d}'\
        .format(now.year,now.month,now.day,now.hour,now.minute)
        instantiate_and_simulate()
        with open(path_saved_tests+'/name_test_perf_', 'wb') as outfile:
            pickle.dump(this_test_model_name, outfile)
    else:
        with open(path_saved_tests+'/name_test_perf_', 'rb') as source:
            this_test_model_name = pickle.load(source)
        print("this_test_model_name: {}".format(this_test_model_name))
        if action == '--loglikelihood':
            measure_loglikelihood()
        elif action=='--ESSE':
            measure_ESSE()
        else:
            print("action: {}".format(action))
            raise ValueError("action not recognised")
    print("{}: main() end of file\n\n".format(str(sys.argv[0])))     

        
        
if __name__=='__main__':
    main()              
