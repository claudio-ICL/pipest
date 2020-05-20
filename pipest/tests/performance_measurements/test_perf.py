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
#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput
#from pycallgraph import Config
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
#parameters of measurement
num_meas = 2
draw_graph=True

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
  
    
def measure_loglikelihood(num_meas = 1, first_event_index = 1, max_depth=15, grouped=False):
    nodename=os.uname().nodename
    path_readout=path_perf+'/'+nodename+'_loglikelihood_readoutgraph_'+date_time+'.txt'
    now=datetime.datetime.now()
    message="I am executing {} --loglikelihood".format(sys.argv[0])
    message+="\nDate of run: {}-{:02d}-{:02d} at {:02d}:{:02d}".format(now.year, now.month, now.day, now.hour, now.minute)
    fout,saveout=redirect_stdout(direction="from", message=message, path=path_readout)
#     print(message)
    try:
        with open(path_saved_tests+'/'+this_test_model_name, 'rb') as source:
            model = pickle.load(source)
    except:
        with open(path_saved_tests+'/perf-test_'+date_time+'/'+this_test_model_name, 'rb') as source:
            model = pickle.load(source)
    global meas    
    if draw_graph:
        e=0
        print("\nI am drawing a graph using pycallgraph")
        print("grouped: {}".format(grouped))
        print("e: {}".format(e))
        meas = measure_exectime.PerformanceMeasure(model)
        meas.target_loglikelihood(e,use_prange=False,draw_graph=True)
#        config=Config(grouped=grouped)
#        graphviz=GraphvizOutput(output_file='graph{}_'.format(e)+this_test_model_name+'.png')
#        with PyCallGraph(config=config,output=graphviz):
#            meas = measure_exectime.PerformanceMeasure(model)
#            meas.target_loglikelihood(e,use_prange=False)
#    print("Number of cpus: {}".format(os.cpu_count()))
#    print("\nModel's key features:")
#    print("d_E={}; d_S={}".format(meas.model.number_of_event_types, meas.model.number_of_states))
#    print("Number of simulated LOB events: {}\n".format(len(meas.model.simulated_events)))
#    measurement = {}
#    exectimes_prange = []
#    exectimes_plain = []
#    for e in range(model.number_of_event_types):
#        exectime_plain = min(
#            timeit.repeat("meas.target_loglikelihood({},use_prange=False)".format(e),
#            globals = globals(), number = 1, repeat = num_meas)
#        )
#        exectimes_plain.append(exectime_plain)
#        exectime_prange = min(
#            timeit.repeat("meas.target_loglikelihood({},use_prange=True)".format(e),
#            globals=globals(), number = 1, repeat = num_meas)
#        )
#        exectimes_prange.append(exectime_prange)
#        measurement.update({'e={}'.format(e+first_event_index): {'plain':exectime_plain, 'prange':exectime_prange}})
#    print("Execution times for the function 'computation.compute_event_loglikelihood_partial_and_gradient_partial' with 'plain' for-loops (no prange):\n{}".format(exectimes_plain))    
#    print("Execution times for the function 'computation.compute_event_loglikelihood_partial_and_gradient_partial' with 'prange' in outermost for-loop:\n{}".format(exectimes_prange))
#    print("\nSummary of measurements:\n{}".format(measurement))
    now=datetime.datetime.now()
    message='\nmeasure_loglikelihood() terminates on {}-{:02d}-{:02d} at {:02d}:{:02d}\n'\
    .format(now.year, now.month, now.day, now.hour, now.minute)
#     print(message)
    redirect_stdout(direction="to", message=message, fout=fout, saveout=saveout)     


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
        with open(path_saved_tests+'/name_test_perf_', 'wb') as outfile:
            pickle.dump(this_test_model_name, outfile)
    else:
        with open(path_saved_tests+'/name_test_estim_', 'rb') as source:
            this_test_model_name = pickle.load(source)
        global date_time
        date_time = this_test_model_name[-15:]
        print("this_test_model_name: {}".format(this_test_model_name))
        print("date_time: {}".format(date_time))
        if action=='-l' or action == '--loglikelihood':
            global num_meas
            try:
                num_meas = int(sys.argv[2])
            except:
                pass
            measure_loglikelihood(num_meas=num_meas)
        else:
            print("action: {}".format(action))
            raise ValueError("action not recognised")
    print("{}: main() end of file\n\n".format(str(sys.argv[0])))     

        
        
if __name__=='__main__':
    main()              
