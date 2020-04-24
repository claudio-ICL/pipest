#cython: boundscheck=False, wraparound=False, nonecheck=False 


import os
cdef str path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<6):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    raise ValueError("path_pipest not found. Instead: {}".format(path_pipest))
cdef str path_sdhawkes=path_pipest+'/sdhawkes_powerlaw'
cdef str path_lobster=path_pipest+'/lobster_for_sdhawkes'
cdef str path_lobster_pyscripts=path_lobster+'/py_scripts'
import sys
sys.path.append(path_sdhawkes+'/')
sys.path.append(path_sdhawkes+'/resources/')
sys.path.append(path_sdhawkes+'/modelling/')
sys.path.append(path_lobster_pyscripts+'/')

import time as clock
import numpy as np
cimport numpy as np
import pandas as pd
from scipy.stats import dirichlet as scipy_dirichlet 
from scipy.ndimage.interpolation import shift as array_shift
import bisect
import copy
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport ceil

import lob_model
from lob_model import RejectSampling
import computation

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t


def random_choice(np.ndarray[DTYPEf_t, ndim=1] weights):
    cdef np.ndarray[DTYPEf_t,ndim=1] prob = weights/np.sum(weights)
    cdef np.ndarray[DTYPEi_t,ndim=1] arr = np.random.multinomial(1,prob).astype(np.int)
    cdef int choice = arr.argmax().astype(np.int)
    return choice

def compute_volume_imbalance_vector(np.ndarray[DTYPEf_t, ndim=2] volumes, int uplim=5):
    """
    it is assumed that volumes are already normalised, so that the sum of every row is equal to 1
    """          
    cdef np.ndarray[DTYPEf_t, ndim=1] vol_ask = np.sum(volumes[:,0:uplim:2], axis=1)
    cdef np.ndarray[DTYPEf_t, ndim=1] vol_bid = np.sum(volumes[:,1:uplim:2], axis=1)
    return (vol_bid-vol_ask).reshape(-1,1)

def is_volimb_constraint_satisfied(DTYPEf_t vol_imb, DTYPEf_t low, DTYPEf_t high):
    if (low <= vol_imb) & (vol_imb  <= high):
        return 1
    else:
        return 0    
    

def sample_volumes(long state, double [:,:] proposal_dir_param,
    double [:,:] difference_of_dir_params, double [:] inverse_bound,
    long [:] is_target_equal_to_proposal,
    int num_of_st2, double [:] volimb_limits, int upto_lim = 5, long maxiter= 999999
):
    cdef int st_2 = state%num_of_st2                      
    cdef DTYPEf_t lower_bound = volimb_limits[st_2]
    cdef DTYPEf_t upper_bound = volimb_limits[1+st_2]
    cdef double [:] gamma_tilde = proposal_dir_param[state,:]
    cdef double [:] delta_gamma = difference_of_dir_params[state,:]
    cdef DTYPEf_t K = inverse_bound[state]
    cdef np.ndarray[DTYPEf_t, ndim=1] sample = np.zeros((len(gamma_tilde),), dtype=DTYPEf)
    cdef DTYPEf_t val=0.0, u = 0.0, vol_imb = 0.0
    cdef int reject = 1
    cdef long count = 0

    if is_target_equal_to_proposal[state]:
        while (reject) & (count<=maxiter):
            count += 1
            sample = scipy_dirichlet.rvs(gamma_tilde)[0,:]
            vol_imb = np.sum(sample[1:upto_lim:2] - sample[0:upto_lim:2])
            if (lower_bound <= vol_imb)&(vol_imb<=upper_bound):
                reject = 0
            else:
                pass
    else:
        while (reject) & (count<=maxiter):
            count += 1
            sample = scipy_dirichlet.rvs(gamma_tilde)[0,:]
            vol_imb = np.sum(sample[1:upto_lim:2] - sample[0:upto_lim:2])
            if (lower_bound <= vol_imb)&(vol_imb<=upper_bound):
                u = np.random.uniform(low=0.0, high=1.0)
                val = np.prod(np.power(sample,delta_gamma))
                if u < K*val:
                    reject = 0                 
    return sample        
    

# def sample_volumes_vectorized(long state, double [:,:] proposal_dir_param,
#     double [:,:] difference_of_dir_params, double [:] prob_constraint, double [:] inverse_bound,
#     long [:] is_target_equal_to_proposal,
#     int num_of_st2, double [:] volimb_limits, int upto_lim = 5
# ):
#     cdef int st_2 = state%num_of_st2                      
#     cdef list args = [volimb_limits[st_2], volimb_limits[1+st_2]]
#     cdef double [:] gamma_tilde = proposal_dir_param[state,:]
#     cdef double [:] delta_gamma = difference_of_dir_params[state,:]
#     cdef DTYPEf_t K = inverse_bound[state]
#     cdef long M = long(ceil(K/max(1.0e-5,prob_constraint[state])))
#     cdef np.ndarray[DTYPEi_t, ndim=1] idx_constraint = np.zeros(M, dtype=DTYPEi)
#     cdef np.ndarray[DTYPEf_t, ndim=2] sample = np.zeros((M,len(gamma_tilde)), dtype=DTYPEf)
#     cdef np.ndarray[DTYPEf_t, ndim=1] u = np.zeros((M,), dtype=DTYPEf)
#     cdef np.ndarray[DTYPEf_t, ndim=2] vol_imb = np.zeros((M,1),dtype=DTYPEf)
#     cdef DTYPEf_t val=0.0, 
#     cdef np.ndarray[DTYPEf_t, ndim=1] result = np.zeros(len(gamma_tilde),dtype=DTYPEf)
#     cdef int reject = 1, n=0

#     if is_target_equal_to_proposal[state]:
#         while reject:
#             sample = scipy_dirichlet.rvs(gamma_tilde, size=M)
#             vol_imb = compute_volume_imbalance_vector(sample, upto_lim)
#             idx_constraint = np.apply_along_axis(is_volimb_constraint_satisfied, 1, vol_imb, *args)
#             if np.any(idx_constraint):
#                 n = np.argmax(idx_constraint)
#                 result = sample[n,:]
#                 reject = 0
#             else:
#                 pass
#     else:
#         while reject:    
#             sample = scipy_dirichlet.rvs(gamma_tilde, size=M )
#             u = np.random.uniform(low=0.0, high=1.0, size=(M,))          
#             vol_imb = compute_volume_imbalance_vector(sample, upto_lim)
#             n=0
#             while ((reject)&(n<M)):
#                 if (args[0] <= vol_imb[n,0]) & (vol_imb[n,0]  <= args[1]):
#                     val = np.prod(np.power(sample[n,:],delta_gamma))
#                     if u[n] < K*val:
#                         result = sample[n,:]
#                         reject = 0
#                 n+=1                     
#     return result               
        
    
    
                   

def prepare_initial_conditions(int num_event_types,int num_states,int num_levels,
                               np.ndarray[DTYPEf_t, ndim=1] base_rates,
                               np.ndarray[DTYPEf_t, ndim=2] dirichlet_param,
                               int num_preconditions = 10,
                               double largest_negative_time = -1.0,
                               double tol = 1.0e-6):
    tot_num_preconditions = num_preconditions*num_event_types*num_states
    largest_negative_time -= np.sum(base_rates)
    cdef np.ndarray[DTYPEf_t, ndim=1] \
    times = largest_negative_time*np.ones(tot_num_preconditions,dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=1] \
    events = np.zeros(tot_num_preconditions,dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] \
    states = np.zeros(tot_num_preconditions,dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=2]\
    labels = np.zeros((tot_num_preconditions,2),dtype=DTYPEi)
    cdef int e,x
    cdef int m,n
    cdef int M = num_event_types*num_states
    for m in range(num_preconditions):
        n=0
        for e in range(num_event_types):
            for x in range(num_states):
                times[m*M + n] -= np.random.exponential(1/np.maximum(tol,base_rates[e]))
                labels[m*M + n,0] = e
                labels[m*M + n,1] = x
                n+=1
    times_and_labels=pd.DataFrame({'times': times, 'events': labels[:,0], 'states':labels[:,1]})
#     times_and_labels.drop_duplicates(inplace=True)
    times_and_labels.sort_values(by='times',inplace=True)
    times = times_and_labels['times'].values
    events = times_and_labels['events'].values
    states = times_and_labels['states'].values
    
    cdef np.ndarray[DTYPEf_t, ndim=2] history_of_dirichlet_param = np.zeros(
        (tot_num_preconditions,2*num_levels),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] volumes = np.zeros_like(history_of_dirichlet_param)
    for n in range(states.shape[0]):
        history_of_dirichlet_param[n,:]=dirichlet_param[states[n],:]
    
    volumes=np.apply_along_axis(
        np.random.dirichlet,1,
        history_of_dirichlet_param)
    
    return tot_num_preconditions,times,events,states,volumes
    
    

def launch_simulation(num_event_types,
                      num_states,
                      num_levels,
                      nus,
                      alphas,
                      betas,
                      phis,
                      kappas,
                      init_condition_times,
                      init_condition_events,
                      init_condition_states,
                      init_volumes,
                      rejection_sampling,
                      t_start,
                      t_end,
                      max_num_of_events,
                      add_initial_cond=False,
                      int num_preconditions = 10,
                      double largest_negative_time = -1.0,
                      int initialise_intensity_on_history = 1,
                      int report_full_volumes = 0
                     ):
    
    cdef int number_of_event_types = num_event_types
    cdef int number_of_states = num_states
    cdef int n_levels = num_levels
    cdef np.ndarray[DTYPEf_t, ndim=1] base_rates = np.array(nus,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=3] impact_coefficients = np.array(alphas,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=2] reshaped_imp_coef = np.array(
        impact_coefficients.reshape(-1,number_of_event_types), copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=3] decay_coefficients = np.array(betas,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=3] transition_probabilities = np.array(phis,copy=True)
    assert not np.any(np.isnan(transition_probabilities))
    assert not np.any(np.isinf(transition_probabilities))
    assert np.all(np.sum(transition_probabilities,axis=2)<=1.0)
    assert np.all(transition_probabilities>=0.0)
    cdef np.ndarray[DTYPEf_t, ndim=2] dirichlet_param = np.array(kappas,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=1] initial_condition_times 
    cdef np.ndarray[DTYPEi_t, ndim=1] initial_condition_events
    cdef np.ndarray[DTYPEi_t, ndim=1] initial_condition_states
    cdef np.ndarray[DTYPEf_t, ndim=2] initial_volumes
    cdef np.ndarray[DTYPEf_t, ndim=2] proposal_dir_param = rejection_sampling.proposal_dir_param
    cdef np.ndarray[DTYPEf_t, ndim=2] difference_of_dir_params = rejection_sampling.difference_of_dir_params
    cdef np.ndarray[DTYPEf_t, ndim=1] inverse_bound = rejection_sampling.inverse_bound
    cdef np.ndarray[DTYPEi_t, ndim=1] is_target_equal_to_proposal = rejection_sampling.is_target_equal_to_proposal
    cdef int num_of_st2 = rejection_sampling.num_of_st2,
    cdef np.ndarray[DTYPEf_t, ndim=1] volimb_limits = rejection_sampling.volimb_limits
    cdef int volimb_upto_level = rejection_sampling.volimb_upto_level
    if add_initial_cond:
        print('I am adding initial conditions on the negative time axis')
        num_preconditions,times,events,states,volumes = prepare_initial_conditions(
            num_event_types, num_states, num_levels,
            base_rates,dirichlet_param,
            num_preconditions,
            largest_negative_time)
        initial_condition_times =\
        np.concatenate([times,init_condition_times],axis=0)
        initial_condition_events=\
        np.concatenate([events,init_condition_events],axis=0)
        initial_condition_states=\
        np.concatenate([states,init_condition_states],axis=0)
        initial_volumes =\
        np.concatenate([volumes,init_volumes],axis=0)
    else:
        initial_condition_times =\
        init_condition_times
        initial_condition_events=\
        init_condition_events
        initial_condition_states=\
        init_condition_states
        initial_volumes =\
        init_volumes
    cdef DTYPEf_t time_start = t_start
    cdef DTYPEf_t time_end = t_end
    cdef int max_number_of_events = max_num_of_events
    return simulate(number_of_event_types,
                    number_of_states,
                    n_levels,
                    num_preconditions,
                    base_rates,
                    impact_coefficients,
                    reshaped_imp_coef,
                    decay_coefficients,
                    transition_probabilities,
                    dirichlet_param,
                    initial_condition_times,
                    initial_condition_events,
                    initial_condition_states,
                    initial_volumes,
                    proposal_dir_param,
                    difference_of_dir_params,
                    inverse_bound,
                    is_target_equal_to_proposal,
                    num_of_st2, volimb_limits, volimb_upto_level,
                    time_start,
                    time_end,
                    max_number_of_events,
                    initialise_intensity_on_history,
                    report_full_volumes = report_full_volumes)





def simulate(int number_of_event_types,
              int number_of_states,
              int n_levels,
              int num_preconditions,
              np.ndarray[DTYPEf_t, ndim=1] base_rates,
              np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
              np.ndarray[DTYPEf_t, ndim=2] reshaped_imp_coef, 
              np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
              np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
              np.ndarray[DTYPEf_t, ndim=2] dirichlet_param,
              np.ndarray[DTYPEf_t, ndim=1] initial_condition_times,
              np.ndarray[DTYPEi_t, ndim=1] initial_condition_events,
              np.ndarray[DTYPEi_t, ndim=1] initial_condition_states,
              np.ndarray[DTYPEf_t, ndim=2] initial_volumes,
              np.ndarray[DTYPEf_t, ndim=2] proposal_dir_param,
              np.ndarray[DTYPEf_t, ndim=2] difference_of_dir_params,
              np.ndarray[DTYPEf_t, ndim=1] inverse_bound,
              np.ndarray[DTYPEi_t, ndim=1] is_target_equal_to_proposal,
              int num_of_st2, np.ndarray[DTYPEf_t, ndim=1] volimb_limits, int volimb_upto_level,
              DTYPEf_t time_start,
              DTYPEf_t time_end,
              int max_number_of_events,
              int initialise_intensity_on_history = 1,
              int report_full_volumes = 0
            ):
    """
    Simulates a state-dependent Hawkes process with power-law kernels.
    :param number_of_event_types:
    :param number_of_states:
    :return:
    """
    print('sd_hawkes_powerlaw_simulation.simulate: start of initialisation')
    print('   Number of levels in the order book: {}'.format(n_levels))
    cdef int number_of_initial_events = initial_condition_times.shape[0]
    print('   number_of_initial_events={}'.format(number_of_initial_events))
    cdef int max_size = number_of_initial_events + max_number_of_events
    cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times = -np.ones(
        (number_of_event_types,number_of_states,max_size),dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] count = np.zeros(
        (number_of_event_types,number_of_states),dtype=DTYPEi)
    cdef np.ndarray[DTYPEf_t,ndim=1] imp_coeff =np.zeros((number_of_event_types,),dtype=DTYPEf)
    cdef DTYPEf_t time, 
    cdef int n
    cdef int state=copy.copy(initial_condition_states[len(initial_condition_states)-1])
    cdef int event=copy.copy(initial_condition_events[len(initial_condition_events)-1])
    cdef DTYPEf_t run_time
    run_time = -clock.time()
    'Initialise labelled_times and count from the initial conditions'
    labelled_times,count = computation.distribute_times_per_event_state(
        number_of_event_types,number_of_states,
        initial_condition_times, 
        initial_condition_events, 
        initial_condition_states,
        len_labelled_times = max_size
    )
    print('sd_hawkes_powerlaw_simulation: simulate: labelled_times and count have been initialised.')
#     print(' number_of_initial_events={}'.format(number_of_initial_events))
#     print(' np.sum(count,axis=None)={}'.format(np.sum(count,axis=None)))
    'Compute the initial intensities of events and the total intensity'
    cdef np.ndarray[DTYPEf_t, ndim=1] intensities = np.ones(number_of_event_types, dtype=DTYPEf)
    cdef DTYPEf_t intensity_overall = 0.0
    if initialise_intensity_on_history:
        intensities=computation.fast_compute_intensities_given_lt(
            time_start,labelled_times, count,
            base_rates, impact_coefficients, decay_coefficients, reshaped_imp_coef,
            number_of_event_types, number_of_states,
        )
    else:
        intensities = np.random.uniform(low=0,high=1,size=(number_of_event_types,))
    intensity_overall=np.sum(intensities)
    print('sd_hawkes_powerlaw_simulation: simulate: intensities have been initialised.')
    print('  intensities at start: {}'.format(intensities))
    print('  intensity_overall at start: {}'.format(intensity_overall))

    'Simulate the state-dependent Hawkes process'
    cdef np.ndarray[DTYPEf_t, ndim=1] result_times = -np.ones(max_size, dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=1] result_events = -np.ones(max_size, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] result_states = -np.ones(max_size, dtype=DTYPEi)
    
    
    'Initialise history of volumes'
    cdef np.ndarray[DTYPEf_t, ndim=2] volume = -np.ones((max_size,2*n_levels),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] history_of_dirichlet_param = np.ones((max_size,2*n_levels),dtype=DTYPEf)
    volume[0:number_of_initial_events,:]=initial_volumes
    'Initialise results'
    result_times[0:number_of_initial_events] = initial_condition_times
    result_events[0:number_of_initial_events] = initial_condition_events
    result_states[0:number_of_initial_events] = initial_condition_states
    time = np.array(time_start,copy=True).astype(np.float)
    cdef DTYPEf_t random_exponential, random_uniform, intensity_total
    cdef np.ndarray[DTYPEf_t, ndim=1] probabilities_state
    cdef DTYPEf_t r
    n = copy.copy(number_of_initial_events)
    print('sd_hawkes_powerlaw_simulation.simulate: start of simulation')
    print('  time_start={},  time at start ={}'.format(time_start,time))
    while time < time_end and n < max_size:
        'Generate an exponential random variable with rate parameter intensity_overall'
        random_exponential = np.random.exponential(1 / intensity_overall)
        'Increase the time'
        time += random_exponential
        if time <= time_end:  # if we are not out of the considered time window
            
            'Update the intensities of events and compute the total intensity'
            intensities=computation.fast_compute_intensities_given_lt(
                time, labelled_times, count,
                base_rates, impact_coefficients, decay_coefficients, reshaped_imp_coef,
                number_of_event_types, number_of_states,           
            )
            intensity_total = np.sum(intensities)

            'Determine if this is an event time'
            random_uniform =  np.random.uniform(0, intensity_overall)
            if random_uniform < intensity_total:  # then yes, it is an event time
                'Determine what event occurs'
                event = random_choice(intensities)
                'Determine the new state of the system'
                previous_state=copy.copy(state)
                probabilities_state = transition_probabilities[previous_state, event, :]
                state = random_choice(probabilities_state)
                'Store volume parameter for dirichlet sampling'
                history_of_dirichlet_param[n,:] = dirichlet_param[state,:]
                'Update the result'
                result_times[n] = copy.copy(time)  # add the event time to the result
                result_events[n] = copy.copy(event)  # add the new event to the result
                result_states[n] = copy.copy(state) # add the new state to the result
                n += 1  # increment counter of number of events
                'update labelled_times and count'
                labelled_times[event,state,count[event,state]] = copy.copy(time)
                count[event,state]+=1
                'update intensities, and intensity_total'
                imp_coeff=impact_coefficients[event,state,:]
                intensities+=imp_coeff
                intensity_total+=np.sum(imp_coeff)
            intensity_overall = intensity_total  # the maximum total intensity until the next event
    
    if report_full_volumes:
        print('sd_hawkes_powerlaw_simulation: I am sampling lob volumes for every state')
        args_volume_sampling=[proposal_dir_param, difference_of_dir_params, inverse_bound, is_target_equal_to_proposal,
              num_of_st2, volimb_limits, 1+2*volimb_upto_level]
        volume[number_of_initial_events:n,:]=np.apply_along_axis(
            sample_volumes,1,
            np.expand_dims(result_states[number_of_initial_events:n],axis=1), *args_volume_sampling)
    cdef int t_0 = num_preconditions
    run_time += clock.time()
    print(' Simulation terminates. time at end ={},  num_of_event={}'.format(time,n))
    print('sd_hawkes_powerlaw_simulation: simulate. run_time={:.1f} seconds'.format(run_time))
    return result_times[t_0:n], result_events[t_0:n], result_states[t_0:n], volume[t_0:n,:]





def launch_liquidation(state_encoding, volume_encoding,
                      num_event_types,
                      num_states,
                      num_levels,
                      initial_inventory,
                      array_of_n_states,       
                      nus,
                      alphas,
                      betas,
                      phis,
                      kappas,
                      init_condition_times,
                      init_condition_events,
                      init_condition_states,
                      init_volumes,
                      rejection_sampling, 
                      t_start,
                      t_end,
                      max_num_of_events,
                      add_initial_cond=False,
                      int num_preconditions = 10,
                      double largest_negative_time = -1.0,
                      double liquidator_control = 0.25,         
                      str liquidator_control_type = 'fraction_of_inventory',
                      int verbose = 0 , 
                      int initialise_intensity_on_history = 1,
                      int report_history_of_intensities = 0, 
                      int report_full_volumes = 0 
                     ):
    
    cdef int number_of_event_types = num_event_types
    cdef int number_of_states = num_states
    cdef int n_levels = num_levels
    cdef np.ndarray[DTYPEf_t, ndim=1] base_rates = np.array(nus,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=3] impact_coefficients = np.array(alphas,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=2] reshaped_imp_coef = np.array(
        impact_coefficients.reshape(-1,number_of_event_types), copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=3] decay_coefficients = np.array(betas,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=3] transition_probabilities = np.array(phis,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=2] dirichlet_param = np.array(kappas,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=1] initial_condition_times 
    cdef np.ndarray[DTYPEi_t, ndim=1] initial_condition_events
    cdef np.ndarray[DTYPEi_t, ndim=1] initial_condition_states
    cdef np.ndarray[DTYPEf_t, ndim=2] initial_volumes 
    cdef np.ndarray[DTYPEf_t, ndim=2] proposal_dir_param = rejection_sampling.proposal_dir_param
    cdef np.ndarray[DTYPEf_t, ndim=2] difference_of_dir_params = rejection_sampling.difference_of_dir_params
    cdef np.ndarray[DTYPEf_t, ndim=1] inverse_bound = rejection_sampling.inverse_bound
    cdef np.ndarray[DTYPEi_t, ndim=1] is_target_equal_to_proposal = rejection_sampling.is_target_equal_to_proposal
    cdef int num_of_st2 = rejection_sampling.num_of_st2,
    cdef np.ndarray[DTYPEf_t, ndim=1] volimb_limits = rejection_sampling.volimb_limits
    cdef int volimb_upto_level = rejection_sampling.volimb_upto_level
    if add_initial_cond:
        print('I am adding initial conditions on the negative time axis')
        num_preconditions,times,events,states,volumes = prepare_initial_conditions(
            num_event_types, num_states, num_levels,
            base_rates,dirichlet_param,
            num_preconditions,
            largest_negative_time)
        initial_condition_times =\
        np.concatenate([times,init_condition_times],axis=0)
        initial_condition_events=\
        np.concatenate([events,init_condition_events],axis=0)
        initial_condition_states=\
        np.concatenate([states,init_condition_states],axis=0)
        initial_volumes =\
        np.concatenate([volumes,init_volumes],axis=0)
    else:
        initial_condition_times =\
        init_condition_times
        initial_condition_events=\
        init_condition_events
        initial_condition_states=\
        init_condition_states
        initial_volumes =\
        init_volumes
    cdef DTYPEf_t time_start = t_start
    cdef DTYPEf_t time_end = t_end
    cdef int max_number_of_events = max_num_of_events
    return simulate_liquidation(state_encoding.arr_state_enc,
                    number_of_event_types,
                    number_of_states,
                    n_levels,
                    num_preconditions,
                    initial_inventory,
                    array_of_n_states,            
                    base_rates,
                    impact_coefficients,
                    reshaped_imp_coef,            
                    decay_coefficients,
                    transition_probabilities,
                    dirichlet_param,
                    initial_condition_times,
                    initial_condition_events,
                    initial_condition_states,
                    initial_volumes,
                    proposal_dir_param,
                    difference_of_dir_params,            
                    inverse_bound,
                    is_target_equal_to_proposal,     
                    num_of_st2, volimb_limits, volimb_upto_level,            
                    time_start,
                    time_end,
                    max_number_of_events,
                    liquidator_control,
                    liquidator_control_type,
                    verbose=verbose,            
                    initialise_intensity_on_history = initialise_intensity_on_history,
                    report_history_of_intensities =  report_history_of_intensities,
                    report_full_volumes = report_full_volumes            
                    )




def simulate_liquidation(
              np.ndarray[DTYPEi_t, ndim=2] arr_state_enc, 
              int number_of_event_types,
              int number_of_states,    
              int n_levels,
              int num_preconditions,           
              DTYPEf_t initial_inventory,           
              np.ndarray[DTYPEi_t, ndim=1] array_of_n_states,           
              np.ndarray[DTYPEf_t, ndim=1] base_rates,
              np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
              np.ndarray[DTYPEf_t, ndim=2] reshaped_imp_coef,           
              np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
              np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
              np.ndarray[DTYPEf_t, ndim=2] dirichlet_param,
              np.ndarray[DTYPEf_t, ndim=1] initial_condition_times,
              np.ndarray[DTYPEi_t, ndim=1] initial_condition_events,
              np.ndarray[DTYPEi_t, ndim=1] initial_condition_states,
              np.ndarray[DTYPEf_t, ndim=2] initial_volumes,
              np.ndarray[DTYPEf_t, ndim=2] proposal_dir_param,
              np.ndarray[DTYPEf_t, ndim=2] difference_of_dir_params,          
              np.ndarray[DTYPEf_t, ndim=1] inverse_bound,
              np.ndarray[DTYPEi_t, ndim=1] is_target_equal_to_proposal, 
              int num_of_st2, DTYPEf_t [:] volimb_limits, int volimb_upto_level,           
              DTYPEf_t time_start,
              DTYPEf_t time_end,
              int max_number_of_events,         
              DTYPEf_t liquidator_control,         
              str liquidator_control_type,
              int verbose = 0 ,
              int initialise_intensity_on_history = 1,
              int report_history_of_intensities = 0,
              int report_full_volumes = 0           
              ):
    """
    Simulates a liquidation in a orderbook modelled as state-dependent Hawkes process with powerlaw kernels. The variables pertaining to the liquidation are indexed at the zero-th entry.
    :param number_of_event_types:
    :param number_of_states:
    :return:
    """
    liquidator_control=max(0.0,liquidator_control)
    print(('simulate_liquidation. liquidator_control_type: {}\n  initial_invetory:{}').format(liquidator_control_type, initial_inventory))
    cdef int number_of_initial_events = initial_condition_times.shape[0]
    print('   number_of_initial_events={}'.format(number_of_initial_events))
    cdef int upto_lim = 1+2*volimb_upto_level
    cdef list args_volume_sampling=[proposal_dir_param, difference_of_dir_params, inverse_bound,
                                    is_target_equal_to_proposal,num_of_st2, volimb_limits, upto_lim]
    cdef int max_size = number_of_initial_events + max_number_of_events
    cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times = -np.ones(
        (number_of_event_types,number_of_states,max_size),dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] count = np.zeros(
        (number_of_event_types,number_of_states),dtype=DTYPEi)
    cdef np.ndarray[DTYPEf_t,ndim=1] coef_of_impact =np.zeros((number_of_event_types,),dtype=DTYPEf)
    cdef DTYPEf_t time = copy.copy(time_start)
    cdef int n = number_of_initial_events
    cdef int state = copy.copy(initial_condition_states[len(initial_condition_states)-1])
    cdef int event = copy.copy(initial_condition_events[len(initial_condition_events)-1])
    cdef DTYPEf_t run_time = -clock.time()
    print('initial state:{}, last event:{}'.format(state,event))
    print('initialisation of run_time: {}'.format(run_time))
    'Initialise labelled_times and count from the initial conditions'
    labelled_times,count = computation.distribute_times_per_event_state(
        number_of_event_types,number_of_states,
        initial_condition_times, 
        initial_condition_events, 
        initial_condition_states,
        len_labelled_times = max_size
    )
    print('sd_hawkes_powerlaw_simulation: simulate: labelled_times and count have been initialised.')
    print(' labelled_times.shape=({},{},{}), count.shape=({},{})'
          .format(labelled_times.shape[0],labelled_times.shape[1],labelled_times.shape[2],
                  count.shape[0],count.shape[1]))
#     print(' number_of_initial_events={}'.format(number_of_initial_events))
#     print(' np.sum(count,axis=None)={}'.format(np.sum(count,axis=None)))
    'Compute the initial intensities of events and the total intensity'
    cdef np.ndarray[DTYPEf_t, ndim=1] intensities = np.ones(number_of_event_types, dtype=DTYPEf)
    cdef DTYPEf_t intensity_overall = 0.0
    if initialise_intensity_on_history:
        intensities=computation.fast_compute_intensities_given_lt(
            time_start, labelled_times, count,
            base_rates, impact_coefficients, decay_coefficients, reshaped_imp_coef,
            number_of_event_types, number_of_states,
        )
    else:
        intensities = np.random.uniform(low=0,high=1,size=(number_of_event_types,))
    intensity_overall=np.sum(intensities)
    cdef np.ndarray[DTYPEf_t, ndim=2] history_of_intensities = np.zeros((max_size,1+number_of_event_types),dtype=DTYPEf)
    print('sd_hawkes_powerlaw_simulation: simulate: intensities have been initialised.')
    print('  intensities at start: {}'.format(intensities))
    print('  intensity_overall at start: {}'.format(intensity_overall))
    
    'Initialise times, events and states that will be produced by the simulation'
    cdef np.ndarray[DTYPEf_t, ndim=1] result_times = -np.ones(max_size, dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=1] result_events = -np.ones(max_size, dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] result_states = -np.ones(max_size, dtype=DTYPEi)
    
    
    'Initialise history of volumes'
    cdef np.ndarray[DTYPEf_t, ndim=2] volume = -np.ones((max_size,2*n_levels),dtype=DTYPEf)
    volume[0:number_of_initial_events,:]=np.array(initial_volumes,dtype=DTYPEf,copy=True)
    cdef DTYPEf_t [:,:] volume_memview = volume
    'Initialise inventory'
    cdef np.ndarray[DTYPEf_t, ndim=1] inventory=initial_inventory*np.ones(max_size, dtype=DTYPEf)
    cdef DTYPEf_t [:] inventory_memview = inventory
    cdef int is_liquidation_terminated = (initial_inventory<=0)
    'Initialise results'
    result_times[0:number_of_initial_events] = copy.copy(initial_condition_times)
    result_events[0:number_of_initial_events] = copy.copy(initial_condition_events)
    result_states[0:number_of_initial_events] = copy.copy(initial_condition_states)
    'memory allocation of variables in process update'
    cdef DTYPEf_t random_exponential, random_uniform, intensity_total
    cdef np.ndarray[DTYPEf_t, ndim=1] probabilities_state = np.ones(number_of_states, dtype=DTYPEf) / number_of_states
    cdef DTYPEf_t r
    cdef int idx_of_history = 0, l=0
    'memory allocation for variables in liquidation update'
    cdef int previous_state = 0
    cdef np.ndarray[DTYPEf_t, ndim=1] previous_volume = np.zeros(2*n_levels, dtype=DTYPEf)
    cdef DTYPEf_t tot_ask, tot_bid, bid_level_1, vol_imb, new_tot_bid, 
    cdef DTYPEf_t quantity_to_liquidate=0.0, quantity_liquidated=0.0
    cdef np.ndarray[DTYPEf_t, ndim=1] to_subtract = np.empty(n_levels,dtype=DTYPEf)
    cdef DTYPEf_t [:] to_subtract_memview = to_subtract
    cdef int indicator_walk_the_book=0, delta_price=0
    cdef np.ndarray[DTYPEi_t, ndim=1] multidim_previous_state = np.zeros(2,dtype=DTYPEi)
    cdef np.ndarray[DTYPEi_t, ndim=1] multidim_new_state = np.zeros(2,dtype=DTYPEi)
    cdef str message=''
    
    print('sd_hawkes_powerlaw_simulation.simulate: start of simulation')
    print('  time_start={},  time at start ={}'.format(time_start,time))
    while time < time_end and n < max_size:
        'Generate an exponential random variable with rate parameter intensity_overall'
        random_exponential = np.random.exponential(1 / intensity_overall)
        'Increase the time'
        time += random_exponential
        if time <= time_end:              
            
            'Update the intensities of events and compute the total intensity'
            if is_liquidation_terminated:
                intensities=computation.fast_compute_intensities_given_lt(
                    time,labelled_times, count,
                    base_rates, impact_coefficients, decay_coefficients, reshaped_imp_coef,
                    number_of_event_types, number_of_states, first_event_index=1
                )
            else:
                intensities=computation.fast_compute_intensities_given_lt(
                    time, labelled_times, count,
                    base_rates, impact_coefficients, decay_coefficients, reshaped_imp_coef,
                    number_of_event_types, number_of_states, first_event_index=0
                )
            intensity_total = np.sum(intensities)
            if report_history_of_intensities:
                history_of_intensities[idx_of_history,0]=copy.copy(time)
                history_of_intensities[idx_of_history,1:]=copy.copy(intensities)
                idx_of_history+=1

            'Determine if this is an event time'
            random_uniform =  np.random.uniform(0, intensity_overall)
            if random_uniform < intensity_total:  # then yes, it is an event time
                'Determine what event occurs'
                if is_liquidation_terminated:
                    event = 1+random_choice(intensities[1:])
                else:
                    event = random_choice(intensities)
                
                if (event == 0) & (inventory[n-1]>0):
                    if verbose:
                        message='simulate_liquidation. At time {}, liquidator intervenes:'.format(time)
                    previous_state=copy.copy(state)
                    previous_volume=sample_volumes(    
                        previous_state,proposal_dir_param, difference_of_dir_params, inverse_bound,
                        is_target_equal_to_proposal,num_of_st2, volimb_limits, upto_lim
                    )
                    volume[n-1,:]=copy.copy(previous_volume)
                    tot_ask=0.0
                    tot_bid=0.0
                    new_tot_bid=0.0
                    bid_level_1 = volume[n-1,1]
                    if verbose:
                        multidim_previous_state=arr_state_enc[state,1:]
                        message+='\n  previous state={}, previous_volume={}'.format(multidim_previous_state,previous_volume)
                    if liquidator_control_type=='fraction_of_bid_side':
                        quantity_to_liquidate=min(inventory[n-1],tot_bid*liquidator_control)
                    elif liquidator_control_type=='fraction_of_inventory':
                        quantity_to_liquidate=min(inventory[n-1],liquidator_control)
                    if verbose:
                        message+='\n  quantity_to_liquidate={},'.format(quantity_to_liquidate)
                    with nogil:
                        if (quantity_to_liquidate>=bid_level_1):
                            indicator_walk_the_book = 1
                        else:
                            indicator_walk_the_book = 0
                        quantity_liquidated=0.0
                        for l in range(n_levels):
                            tot_ask+=volume_memview[n-1,2*l]
                            tot_bid+=volume_memview[n-1,1+2*l]
                            to_subtract_memview[l] = min(volume_memview[n-1,1+2*l],quantity_to_liquidate)
                            quantity_to_liquidate += -to_subtract_memview[l]
                            quantity_liquidated += to_subtract_memview[l]
                            volume_memview[n,1+2*l]=volume_memview[n-1,1+2*l]-to_subtract_memview[l]
                            new_tot_bid+=volume_memview[n,1+2*l]
                        inventory_memview[n:]=max(0.0,inventory_memview[n-1]-quantity_liquidated)
                    if verbose:
                        message+=' to_subtract={}, quantity_liquidated={}'.format(to_subtract,quantity_liquidated)    
                    volume[n,0::2]=(1.0-new_tot_bid)*volume[n-1,0::2]/tot_ask                 
                    with nogil:
                        st_2=compute_and_classify_volimb_scalar(
                            volume_memview[n,:], volimb_limits, volimb_upto_level, num_of_st2)
                        delta_price = -indicator_walk_the_book
                        st_1 = 1+delta_price
                    multidim_new_state=np.array([st_1,st_2],dtype=DTYPEi)
                    state=convert_multidim_state_code(number_of_states,arr_state_enc,multidim_new_state)
                    if verbose:
                        message+='\n  new_state={}, new_volume={}, new_tot_bid={}.'.format(multidim_new_state,volume[n,:],new_tot_bid)
                        print(message)
                    with nogil:    
                        if (inventory_memview[n]<=0.0):
                            is_liquidation_terminated=1
                        else:
                            is_liquidation_terminated=0   
                else:
                    'Determine the new state of the system'
                    previous_state=copy.copy(state)
                    probabilities_state = copy.copy(transition_probabilities[previous_state, event, :])
                    state = random_choice(probabilities_state)
                'Update the result'
                result_times[n] = copy.copy(time)  # add the event time to the result
                result_events[n] = copy.copy(event)  # add the new event to the result
                result_states[n] = copy.copy(state)  # add the new state to the result
                n += 1  # increment counter of number of events
                'update labelled_times and count'
                labelled_times[event,state,count[event,state]] = copy.copy(time)
                count[event,state]+=1
                'update intensities, and intensity_total'
                coef_of_impact=copy.copy(impact_coefficients[event,state,:])
                if is_liquidation_terminated:
                    coef_of_impact[0]=0.0
                intensities+=coef_of_impact
                intensity_total+=np.sum(coef_of_impact)
            intensity_overall = copy.copy(intensity_total)  # the maximum total intensity until the next event
    
    cdef int t_0 = num_preconditions
    idx_event=(result_events[number_of_initial_events:n]==0)
    idx=np.array(array_shift(idx_event,-1,cval=0),dtype=np.bool)
    idx=np.logical_not(np.logical_or(idx_event,idx))
    cdef np.ndarray[DTYPEf_t, ndim=2] result_volumes = volume[number_of_initial_events:n,:]
    if report_full_volumes:
        print('sd_hawkes_powerlaw_simulation: I am sampling lob volumes for every state')
        try:
            result_volumes[idx,:] = np.apply_along_axis(
            sample_volumes,1,
            np.expand_dims(result_states[number_of_initial_events:n][idx],axis=1), *args_volume_sampling)
        except:
            print('lob_hawkes_simulation. I could not sample volumes from dirichlet distribution')
    cdef int idx_liquid_termination = np.argmin(inventory)
    cdef DTYPEf_t liquid_termination_time = copy.copy(result_times[n-1])
    if is_liquidation_terminated:
        liquid_termination_time = copy.copy(result_times[idx_liquid_termination])
    run_time += clock.time()
    cdef str termination_message = "Simulation terminates. time at end ={},  num_of_event={}".format(time,n)
    if is_liquidation_terminated:
        termination_message += "\n liquidation terminates at time: {}".format(liquid_termination_time)
    else:
        termination_message += "\n liquidation did not terminate"
    print(termination_message)
    print('sd_hawkes_powerlaw_simulation: simulate. run_time={:.1f} seconds'.format(run_time))
    if report_history_of_intensities:
        return result_times[t_0:n], result_events[t_0:n], result_states[t_0:n], result_volumes, inventory[t_0:n],liquid_termination_time,history_of_intensities[:idx_of_history,:]
    else:
        return result_times[t_0:n], result_events[t_0:n], result_states[t_0:n], result_volumes, inventory[t_0:n],liquid_termination_time, history_of_intensities[t_0:n,:]





cdef long convert_multidim_state_code(
    int num_of_states, DTYPEi_t [:,:] arr_state_enc, DTYPEi_t [:] state) nogil:
    cdef int i=0
    for i in range(num_of_states):
        if ( (arr_state_enc[i,1] == state[0]) & (arr_state_enc[i,2]==state[1]) ):
            return arr_state_enc[i,0]
    return 0         
        
cdef int compute_and_classify_volimb_scalar(
    DTYPEf_t [:] volumes, DTYPEf_t [:] volimb_limits, int upto_level, int num_of_st2) nogil:
    cdef int n=0
    cdef DTYPEf_t vol_ask=0.0, vol_bid=0.0
    for n in range(upto_level):
        vol_ask+=volumes[2*n]
        vol_bid+=volumes[1+2*n]
    cdef DTYPEf_t volimb = ((vol_bid-vol_ask)/max(1.0e-10,vol_bid+vol_ask))
    for n in range(num_of_st2):
        if (volimb>=volimb_limits[n])&(volimb<=volimb_limits[n+1]):
            return n
    return n    
    
        
cdef double compute_volume_imbalance_scalar(DTYPEf_t [:] volumes, int upto_level=2) nogil:
    cdef int n=0
    cdef DTYPEf_t vol_ask=0.0, vol_bid=0.0
    while n<upto_level:
        vol_ask+=volumes[2*n]
        vol_bid+=volumes[1+2*n]
        n+=1
    return ((vol_bid-vol_ask)/max(1.0e-10,vol_bid+vol_ask))

cdef int classify_vol_imb_scalar(DTYPEf_t vol_imb, DTYPEf_t [:] volimb_limits):
    """
    volume imbalance is expected as a scalar with value between -1 and 1
    categories are sorted from the most negative volume imbalance to the most positive
    """
    return int(max(0,-1+bisect.bisect_left(volimb_limits, vol_imb)))
