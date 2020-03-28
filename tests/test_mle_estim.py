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


# In[4]:


import pickle
import numpy as np
import pandas as pd
import bisect
import copy


# In[ ]:


import model
import lob_model
import computation
# import simulation
# import prepare_from_lobster as from_lobster
# from computation import parameters_to_array_partial
# import goodness_of_fit




# import mle_estimation as mle_estim



# In[ ]:

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
    nus = 0.0018025*np.random.randint(low=2,high=6,size=n_events)
    # The impact coefficients $\alpha$
    alphas = np.random.uniform(0.0002,0.2435,size=(n_events, tot_n_states, n_events)).astype(np.float)
    # The decay coefficients $\beta$
    betas = np.random.uniform(1.265,1.805,size=(n_events, tot_n_states, n_events)).astype(np.float)
    sd_model.set_hawkes_parameters(nus,alphas,betas)
    # The transition probabilities $\phi$
    phis = sd_model.state_enc.generate_random_transition_prob(n_events=n_events).astype(np.float)
    sd_model.set_transition_probabilities(phis)
    sd_model.enforce_symmetry_in_transition_probabilities()
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
    time_end=np.array(times[-1],copy=True)

    sd_model.create_goodness_of_fit(type_of_input='simulated')
    sd_model.goodness_of_fit.ks_test_on_residuals()
    sd_model.goodness_of_fit.ad_test_on_residuals()

#     exit()
    
    print("\nMLE ESTIMATION\n")
    "Initialise the class"
    sd_model.create_mle_estim(type_of_input = 'simulated')
    d_E = sd_model.number_of_event_types
    d_S = sd_model.number_of_states 
    "Fictious initial guess"
    nus = np.random.uniform(low=0.0, high = 1.0, size=(d_E,))
    alphas = np.random.uniform(low=0.0, high = 2.0, size=(d_E,d_S,d_E))
    betas = np.random.uniform(low=1.1, high = 5.0, size=(d_E,d_S,d_E))
    guess = computation.parameters_to_array(nus, alphas, betas)
    list_init_guesses = [guess]
    "Set the estimation"    
    sd_model.mle_estim.set_estimation_of_hawkes_param(
        time_start, time_end,
        list_of_init_guesses = list_init_guesses,
        learning_rate = 0.0005,
        maxiter=10,
        number_of_additional_guesses=3,
        parallel=True,
        pre_estim_ord_hawkes=True,
        pre_estim_parallel=True,
        number_of_attempts = 2
    )
    
#     exit()
    
    "Launch estimation"
    run_time = -time.time()
    sd_model.mle_estim.launch_estimation_of_hawkes_param(all_components=True)
    run_time+=time.time()
    sd_model.mle_estim.store_runtime(run_time)
#     model.mle_estim.create_goodness_of_fit()

                           
                           
if __name__=="__main__":
    print("I am executing 'test_mle_estim.py'")
    now=datetime.datetime.now()
    print("Test launched on {}-{:02d}-{:02d} at {:02d}:{:02d}".format(
        now.year,now.month,now.day,now.hour,now.minute
    ))
    main()
    now=datetime.datetime.now()
    print("Test terminates on {}-{:02d}-{:02d} at {:02d}:{:02d}".format(
        now.year,now.month,now.day,now.hour,now.minute
    ))
                           

# print('\n\nI will let the model estimate the parameters from the sample just produced')
# br,ic,dc=model.estimate_hawkes_parameters(times, events, states,
#                                           time_start, time_end,
#                                           maximum_number_of_iterations=90)
# print('\n')
# print('Estimate transition probabilities')
# phis_hat = mle_estim.estimate_transition_probabilities(number_of_event_types, number_of_states,
#     events, states)
# rel_error=np.linalg.norm(phis_hat-phis)/np.linalg.norm(phis)
# print('relative error: {}'.format(rel_error))
# # print('phis_hat=\n {}'.format(phis_hat))
# # print('The above result needs to be compared to:')
# # print('phis = \n {}'.format(phis))
# print('\nNow, with the estimated parameters I perform the goodness_of_fit again')
# good_fit = goodness_of_fit.good_fit(number_of_event_types,
#                             number_of_states,
#                             br,
#                             ic,
#                             dc,
#                             phis_hat,        
#                             times,
#                             events,
#                             states)
# good_fit.ks_test_on_residuals()
# good_fit.ks_test_on_total_residuals()
# good_fit.ad_test_on_residuals(distr='expon')
# print('good_fit.adtest_residuals for exponential distribution: \n {}'.format(good_fit.adtest_residuals))
# with open('./goodfit_mle', 'wb') as outfile:
#     pickle.dump(good_fit, outfile)



        
        
# print('\n')
# print('Estimate dirichlet parameters')
# kappa_hat=mle_estim.estimate_dirichlet_parameters(number_of_states, n_levels,states,volumes,verbose=True)
# rel_error=np.linalg.norm(kappa_hat-kappas)/np.linalg.norm(kappas)
# print('relative error: {}'.format(rel_error))
# print('kappa_hat = \n {}'.format(kappa_hat))
# print('The above result needs to be compared to:')
# print('kappa = \n {}'.format(kappas))


# guess_imp_coef = mle_estim.pre_guess_impact_coefficients(
#     number_of_event_types, number_of_states,events)
# print('pre-guess of imp coef : \n {}'.format(guess_imp_coef))
# print(' Actual imp coef: \n {}'.format(alphas))

# e=0
# for e in range(number_of_event_types-2):
#     print('\n')  
#     print('Event type of focus: e={}'.format(e))
#     br=np.atleast_1d(np.array(nus[e],dtype=np.float)) 
#     print('I will perform the estimation of sd-hawkes process')
#     x_target=parameters_to_array_partial(br,alphas[:,:,e],betas[:,:,e])
#     minim,base_rate,imp_coef,dec_coef = mle_estim.estimate_hawkes_power_partial(
#         e,
#         number_of_event_types, number_of_states,
#         times,
#         events,
#         states,
#         time_start,
#         time_end,
#         labelled_times,
#         count,
#         maxiter = 80,
#         return_minim_proc = 1
#     )
#     y_target,_=minim.compute_f_and_grad(x_target)
#     print('Estimation of sd-hawkes has terminated')
#     error_base_rate=np.linalg.norm(nus[e]-base_rate)/max(np.linalg.norm(nus[e]),1.0e-8)
#     error_imp_coef = np.linalg.norm(imp_coef - alphas[:,:,e])/np.linalg.norm(alphas[:,:,e])
#     error_dec_coef = np.linalg.norm(dec_coef - betas[:,:,e])/np.linalg.norm(betas[:,:,e])
# #     print('Result of estimation:')
# #     print('base_rate=\n{}'.format(base_rate))
# #     print('imp_coef=\n{}'.format(imp_coef))
# #     print('dec_coef=\n{}'.format(dec_coef))
# #     print('The above results are to be compared with:')
# #     print('nus[e]=\n{}'.format(nus[e]))
# #     print('alphas[:,:,e]=\n{}'.format(alphas[:,:,e]))
# #     print('betas[:,:,e]=\n{}'.format(betas[:,:,e]))
#     print('realtive error in base_rate: {}'.format(error_base_rate))
#     print('realtive error in imp_coef: {}'.format(error_imp_coef))
#     print('realtive error in dec_coef: {}'.format(error_dec_coef))
#     print('Objective funtion evaluated at estimated parameter: {}'.format(minim.minimum))
#     print('Objective function evaluated at target: {}'.format(y_target))

#     print('\n')
#     print('GOODNESS OF FIT, event_type: {}'.format(e))
#     print('Estimated parameters')
#     ratio = imp_coef / (dec_coef -1)
#     residual_hat =  computation.compute_event_residual(e, 
#                                base_rate,
#                                dec_coef,
#                                ratio,
#                                labelled_times,
#                                count
#                               )
#     verdict_expon=stats.kstest(residual_hat,'expon')
#     verdict_uniform=stats.kstest(residual_hat/np.amax(residual_hat),'uniform')
#     print('verdict_expon: {}'.format(verdict_expon))
#     print('verdict_uniform: {}'.format(verdict_uniform))
#     print('\nTrue parameters')
#     ratio = alphas[:,:,e] / (betas[:,:,e] -1)
#     residual =  computation.compute_event_residual(e, 
#                                nus[e],
#                                betas[:,:,e],
#                                ratio,
#                                labelled_times,
#                                count
#                               )
#     check=np.all(np.isclose(residual,good_fit.residuals[e]))
#     if not check:
#         print('check of residual failed')      
#     verdict_expon=stats.kstest(residual,'expon')
#     verdict_uniform=stats.kstest(residual/np.amax(residual),'uniform')
#     print('verdict_expon: {}'.format(verdict_expon))
#     print('verdict_uniform: {}'.format(verdict_uniform))




# for e in range(number_of_event_types):
#     print('\n')  
#     print('Event type of focus: e={}'.format(e))
#     br=np.atleast_1d(np.array(nus[e],dtype=np.float))
#     x_target=parameters_to_array_partial(br,alphas[:,:,e],betas[:,:,e])
#     print('I will perform the pre_estimation of ordinary hawkes process')
#     minim,base_rate,imp_coef,dec_coef =  mle_estim.pre_estimate_ordinary_hawkes(
#         e,
#         number_of_event_types, 
#         times,
#         events,
#         time_start,time_end,
#         maxiter = 60,
#         return_minim_proc = 1
#     )
#     y_target,_=minim.compute_f_and_grad(x_target)
    
#     print('Estimation terminated')
#     print('Result of estimation:')
#     print('base_rate=\n{}'.format(base_rate))
#     print('imp_coef=\n{}'.format(imp_coef))
#     print('dec_coef=\n{}'.format(dec_coef))
#     print('The above results are to be compared with:')
#     print('nus[e]=\n{}'.format(nus[e]))
#     print('alphas[:,:,e]=\n{}'.format(alphas[:,:,e]))
#     print('betas[:,:,e]=\n{}'.format(betas[:,:,e]))
#     print('Objective funtion evaluated at estimated parameter: {}'.format(minim.minimum))
#     print('Objective function evaluated at target: {}'.format(y_target))


