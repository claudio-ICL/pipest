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


import model
import lob_model
import minimisation_algo as minim_algo
import mle_estimation as mle_estim

def main():
    list_of_n_states=[3,5]
    n_events = 4 
    n_levels = 2

    sd_model = model.SDHawkes(number_of_event_types=n_events,
                     number_of_lob_levels=n_levels,
                     volume_imbalance_upto_level=2,
                     list_of_n_states=list_of_n_states
                    )

    tot_n_states=sd_model.state_enc.tot_n_states

    # The base rates $\nu$
    nus = 0.002*np.random.randint(low=2,high=6,size=n_events)
    # The impact coefficients $\alpha$
    alphas = np.random.uniform(0.0002,0.2435,size=(n_events, tot_n_states, n_events)).astype(np.float)
    # The decay coefficients $\beta$
    betas = np.random.uniform(1.265,1.805,size=(n_events, tot_n_states, n_events)).astype(np.float)
    sd_model.set_hawkes_parameters(nus,alphas,betas)
    # The transition probabilities $\phi$
    phis = sd_model.state_enc.generate_random_transition_prob(n_events=n_events).astype(np.float)
    sd_model.set_transition_probabilities(phis)
#     sd_model.enforce_symmetry_in_transition_probabilities()
    # The Dirichlet parameters $\kappa$
    kappas = np.random.lognormal(size=(tot_n_states,2*n_levels))
    sd_model.set_dirichlet_parameters(kappas)


    time_start = 0.0
    time_end = time_start + 2*60*60
    max_number_of_events = 4000



    print("\nSIMULATION\n")

    times, events, states, volumes = sd_model.simulate(
        time_start, time_end,max_number_of_events=max_number_of_events,
        add_initial_cond=True,
        store_results=True, report_full_volumes=False)
    time_end=float(times[-1])

    sd_model.create_goodness_of_fit(type_of_input='simulated')
    sd_model.goodness_of_fit.ks_test_on_residuals()
    sd_model.goodness_of_fit.ad_test_on_residuals()
    
    



    print("\nMINIMISATION PROCEDURE\n")
    event_type=0
    num_init_guesses = 5
    list_init_guesses = []
    for n in range(num_init_guesses):
        list_init_guesses.append(
            1.0 + np.random.lognormal(size=(1+2*sd_model.number_of_event_types*sd_model.number_of_states,))
        )
    MinimProc = minim_algo.MinimisationProcedure(
        sd_model.labelled_times, sd_model.count, time_start, time_end,
        sd_model.number_of_event_types,sd_model.number_of_states,
        event_type,
        list_init_guesses = list_init_guesses,
        learning_rate = 0.0004,
        maxiter = 5,
        tol = 1.0e-7,
        copy_lt = True,
    )
    run_time = -time.time()
    MinimProc.launch_minimisation(parallel=True, return_results = False)
    run_time+=time.time()
    print("Minimisation terminates. run_time = {}".format(run_time))

    

    print("\nESTIMATION HAWKES PARAMETERS\n")
    del MinimProc
    times=copy.copy(sd_model.sampled_times)
    events=copy.copy(sd_model.sampled_events)
    d_E = copy.copy(sd_model.number_of_event_types)
    d_S = copy.copy(sd_model.number_of_states)
    e = copy.copy(event_type)
    t_start=copy.copy(time_start)
    t_end=copy.copy(time_end)
    res=mle_estim.pre_estimate_ordinary_hawkes(
        e,
        d_E, 
        times,
        events,
        t_start,
        t_end,
        num_init_guesses = 5,
        parallel = True,
        learning_rate = 0.0001,
        maxiter = 4,
        tol = 1.0e-07,
        n_states = d_S, #used to reshape the results
        reshape_to_sd = True,
        return_as_flat_array = True
    )
#     print("I am printing result of pre_estim_ord: \n {}".format(res))
#     exit()
#     list_init_guesses.append(res)
    new_list_init_guesses = mle_estim.produce_list_init_guesses(
        sd_model.number_of_event_types, sd_model.number_of_states,
        num_additional_guesses = 4,
        given_list_of_guesses = list_init_guesses,
        print_list = False
    )
    del res
    run_time = -time.time()
    base_rate,imp_coef,dec_coef=mle_estim.estimate_hawkes_param_partial(
        event_type,
        sd_model.number_of_event_types, sd_model.number_of_states,
        time_start,
        time_end,
        sd_model.labelled_times,
        sd_model.count,
        list_init_guesses = new_list_init_guesses,
        num_additional_guesses = 4,
        maxiter = 5,
        learning_rate = 0.0005,
        pre_estim_ord_hawkes = True,
        return_minim_proc = 0,
        parallel= True,
        pre_estim_parallel = True,
        copy_lt = True,
        prepare_list_of_guesses = False,
        print_list = False,
        timeout= 20.0
    )
    run_time+=time.time()
    print("Estimation of hawkes parameters terminates. run_time={}".format(run_time))
#     print("base_rate={}".format(base_rate))
#     print("imp_coef={}".format(imp_coef))
#     print("dec_coef={}".format(dec_coef))
    

    
if __name__=="__main__":
    print("I am executing 'test_minim_algo.py'")
    now=datetime.datetime.now()
    print("Test launched on {}-{:02d}-{:02d} at {:02d}:{:02d}".format(
        now.year,now.month,now.day,now.hour,now.minute
    ))
    main()
    now=datetime.datetime.now()
    print("Test terminates on {}-{:02d}-{:02d} at {:02d}:{:02d}".format(
        now.year,now.month,now.day,now.hour,now.minute
    ))
