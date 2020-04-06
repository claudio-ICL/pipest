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

# In[2]:


print(path_sdhawkes)
print(os.path.basename(path_sdhawkes))
print(os.path.dirname(path_sdhawkes+'/'))
print(os.path.basename(os.path.dirname(path_sdhawkes)))


# In[3]:


import time
import datetime
import sys
sys.path.append(path_sdhawkes+'/resources/')
sys.path.append(path_sdhawkes+'/modelling/')
sys.path.append(path_lobster_pyscripts+'/')


# In[4]:


import pickle
import numpy as np
import pandas as pd
import bisect
import copy


# In[5]:


import model as sd_hawkes_model
import lob_model
import computation
import simulation
import goodness_of_fit
import mle_estimation as mle_estim
import prepare_from_lobster as from_lobster
import nonparam_estimation as nonparam_estim



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
maxiter = 15
num_guesses = 4
num_processes = 10
batch_size = 3000
num_run_per_minibatch = 2
parallel=False
type_of_preestim = 'nonparam' # 'ordinary_hawkes' or 'nonparam'
#Optional parameters for "nonparam_estim"
num_quadpnts = 40
quad_tmax = 1.0
quad_tmin = 1.0e-1
num_gridpnts = 50
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

        
def main(model_name= ''):    
    global time_start
    global time_end
    model = sd_hawkes_model.SDHawkes(
        number_of_event_types=n_events,
        number_of_lob_levels=n_levels,
        list_of_n_states=n_states,
        volume_imbalance_upto_level=upto_level,
        name_of_model=model_name)
    tot_n_states=model.state_enc.tot_n_states
    phis = model.state_enc.generate_random_transition_prob(n_events=n_events).astype(np.float)
    nus = 0.1*np.random.randint(low=15,high=20,size=n_events)
    alphas = np.power(10,-np.random.uniform(low=1.0, high=1.2))*np.random.randint(low=0,high=4,size=(n_events, tot_n_states, n_events)).astype(np.float)
    betas = np.random.uniform(1.25025,2.1,size=(n_events, tot_n_states, n_events)).astype(np.float)
    gammas = np.random.uniform(low=1.25, high = 5.6,size=(tot_n_states,2*n_levels))

    model.set_hawkes_parameters(nus,alphas,betas)
    model.set_dirichlet_parameters(gammas)
    model.set_transition_probabilities(phis)

    print("\nSIMULATION\n")
    max_number_of_events = np.random.randint(low=7050, high=9000)
    times, events, states, volumes = model.simulate(
        time_start, time_end, max_number_of_events=max_number_of_events,
        add_initial_cond=True,
        store_results=True, report_full_volumes=False)
    time_end=float(times[-1])
    model.create_goodness_of_fit(type_of_input='simulated', parallel=False)
    model.goodness_of_fit.ks_test_on_residuals()
    model.goodness_of_fit.ad_test_on_residuals()
    if type_of_preestim == 'nonparam':
        print("\nNON-PARAMETRIC ESTIMATION\n")
        run_time = -time.time()
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
        model.nonparam_estim.estimate_hawkes_kernel(store_L1_norm=False,
                                   use_filter=True, enforce_positive_g_hat=True,
                                   filter_cutoff=20.0, filter_scale=30.0, num_addpnts_filter=2000,
                                   parallel=False, parallel_prep=True
                                   )
        model.nonparam_estim.fit_powerlaw(compute_L1_norm=True,ridge_param=1.0e-01, tol=1.0e-7)
        model.nonparam_estim.store_base_rates()
        run_time+=time.time()
        model.nonparam_estim.store_runtime(run_time)
        model.nonparam_estim.create_goodness_of_fit(parallel=False)
    
    print("\nMLE ESTIMATION\n")
    "Initialise the class"
    model.create_mle_estim(type_of_input = 'simulated', store_trans_prob = True)
    "Set the estimation"
    if type_of_preestim == 'nonparam':
        list_init_guesses = model.nonparam_estim.produce_list_init_guesses_for_mle_estimation(
            num_additional_random_guesses = 2, max_imp_coef = 50.0)
    else:
        list_init_guesses = []
    model.mle_estim.set_estimation_of_hawkes_param(
                time_start, time_end,
                list_of_init_guesses = list_init_guesses,
                max_imp_coef = max_imp_coef,
                learning_rate = learning_rate,
                maxiter=maxiter,
                number_of_additional_guesses = num_guesses,
                parallel=parallel,
                pre_estim_ord_hawkes=True,
                pre_estim_parallel=parallel,
                number_of_attempts = 3,
                num_processes = num_processes,
                batch_size = batch_size,
                num_run_per_minibatch = num_run_per_minibatch,
    )
#     exit()
    "Launch estimation"
    model.mle_estim.launch_estimation_of_hawkes_param(partial=False)
    model.mle_estim.store_hawkes_parameters()
    model.mle_estim.create_goodness_of_fit(parallel=False)
    model.dump(path=path_saved_tests) 
    


if __name__=='__main__':
    now=datetime.datetime.now()
    this_test_readout=path_saved_tests\
    +'/test_estim_readout_{}-{:02d}-{:02d}_{:02d}{:02d}'.format(now.year,now.month,now.day, now.hour, now.minute)
    this_test_model_name = 'test_estim_model_{}-{:02d}-{:02d}_{:02d}{:02d}'\
    .format(now.year,now.month,now.day,now.hour,now.minute)
    this_test_model_path=path_saved_tests+this_test_model_name
    message="I am executing {}".format(str(sys.argv[0]))
    message+='\ndate of run: {}-{:02d}-{:02d} at {:02d}:{:02d}\n'.format(now.year,now.month,now.day,now.hour,now.minute)
    fout, saveout = redirect_stdout(direction='from', message=message, path=this_test_readout)
#     print(message)
    main(model_name = this_test_model_name)
    
    now=datetime.datetime.now()
    message="\nTest terminates on {}-{:02d}-{:02d} at {:02d}:{:02d}".format(
        now.year,now.month,now.day,now.hour,now.minute)
    message+="\nEND OF TEST"
#     print(message)
    redirect_stdout(direction='to',message=message,fout=fout,saveout=saveout)
