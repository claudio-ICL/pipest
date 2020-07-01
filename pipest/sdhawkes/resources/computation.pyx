#cython: boundscheck=False, wraparound=False, nonecheck=False
import os
from cython.parallel import prange
cimport openmp
#openmp.omp_set_num_threads(min(16,os.cpu_count()))
print("openmp.omp_get_max_threads(): {}".format(openmp.omp_get_max_threads()))
"""
This version of the resource "computation.pyx" contains functions called by oher resources. Some of these functions have multiple implementations. This was done to test performances, and choose the best possible implementation in different cases.
"""
import numpy as np
cimport numpy as np
from scipy.stats import dirichlet as scipy_dirichlet
from scipy.optimize import minimize as scipy_minimize
import bisect
import copy
from libc.math cimport pow
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport isnan
from scipy.special import gamma as scipy_gamma_fun
from libc.stdlib cimport rand, RAND_MAX, srand

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t
DTYPEfd = np.longdouble
DTYPEil = np.int64
ctypedef np.longdouble_t DTYPEfd_t
ctypedef np.int64_t DTYPEil_t

cdef DTYPEf_t fRANDMAX = float(RAND_MAX)

def compute_avgrate(int e,
        np.ndarray[DTYPEf_t, ndim=1] times,
        np.ndarray[DTYPEi_t, ndim=1] events,
    ):
    assert len(times)==len(events)
    cdef np.ndarray[DTYPEf_t, ndim=1] t = np.array(times, copy=True)
    t-=t[0]
    return np.mean(np.cumsum(np.array(events[1:]==e,dtype=DTYPEi))/t[1:])
def avg_rates(int d_E,
        np.ndarray[DTYPEf_t, ndim=1] times,
        np.ndarray[DTYPEi_t, ndim=1] events,
        int e=0, partial=True
    ):
    assert len(times)==len(events)
    cdef np.ndarray[DTYPEf_t, ndim=1] t = np.array(times, copy=True)
    t-=t[0]
    if partial:
        return np.mean(np.cumsum(np.array(events[1:]==e,dtype=DTYPEi))/t[1:])
    else:
        return np.array(
            [np.mean(np.cumsum(np.array(events[1:]==e1,dtype=DTYPEi))/t[1:]) 
                for e1 in range(d_E)],
            dtype=DTYPEf)

def distribute_times_per_event_state(
    int n_event_types,
    int n_states,
    double [:] times,
    long[:] events,
    long[:] states,
    int len_labelled_times = 0):
    cdef int len_times = len(times) 
    len_labelled_times = max(len_labelled_times,len_times)
    cdef np.ndarray[DTYPEi_t, ndim=2] count =  np.zeros((n_event_types,n_states),dtype=DTYPEi)
    cdef long [:,:] count_memview = count
    cdef np.ndarray[DTYPEf_t,ndim=3] t = -np.ones(
        (n_event_types,n_states,len_labelled_times),
        dtype=DTYPEf)
    cdef double [:,:,:] t_memview = t
    cdef Py_ssize_t e,x,i
    with nogil:
        for e in range(n_event_types):
            for x in range(n_states):
                for i in range(len_times):
                    if ((events[i]==e)&(states[i]==x)):
                        t_memview[e,x,count_memview[e,x]]=times[i]
                        count_memview[e,x] +=1
    return t, count

def update_labelled_times(double time, long event, long state,
                          np.ndarray[DTYPEf_t, ndim=3] labelled_times,
                          np.ndarray[DTYPEi_t, ndim=2] count):
    cdef np.ndarray[DTYPEf_t, ndim=3] new_lt = np.array(labelled_times,copy=True)
    cdef np.ndarray[DTYPEi_t, ndim=2] new_count = np.array(count,copy=True)
    new_lt[event,state,count[event,state]] = time
    new_count[event,state]+=1
    return new_lt, new_count 

"FUNCTIONS RELATED TO COMPUTATION OF INTENSITIES"

def plain_compute_ESSE_partial(double t, double s,
                         int num_event_types, int num_states,
                         double [:,:] decay_coefficients,
                         double [:,:,:] labelled_times,
                         long [:,:] count,
                        ):
    "This is compute_ESSE_partial implemented without using prange"
    cdef np.ndarray[DTYPEf_t,ndim=2] ESSE = np.zeros((num_event_types, num_states), dtype=DTYPEf)
    cdef double [:,:] ESSE_memview = ESSE
    cdef int e1, x, i
    cdef double eval_time,power_comp
    with nogil:
        for e1 in range(num_event_types):
            for x in range(num_states):
                for i in range(count[e1,x]):
                    if ((labelled_times[e1,x,i]>=s) & (labelled_times[e1,x,i]<t)):
                        eval_time = t-labelled_times[e1,x,i]+1.0
                        power_comp = pow(eval_time,-decay_coefficients[e1,x])
                        ESSE_memview[e1,x] += power_comp        
    return ESSE 

def compute_ESSE_partial(double t, double s,
                         int num_event_types, int num_states,
                         double [:,:] decay_coefficients,
                         double [:,:,:] labelled_times,
                         long [:,:] count,
                        ):
    cdef np.ndarray[DTYPEf_t,ndim=2] ESSE = np.zeros((num_event_types, num_states), dtype=DTYPEf)
    cdef double [:,:] ESSE_memview = ESSE
    cdef int e1, x, i
    cdef double eval_time,power_comp
    for e1 in prange(num_event_types, nogil=True,  ):
        for x in range(num_states):
            for i in range(count[e1,x]):
                if ((labelled_times[e1,x,i]>=s) & (labelled_times[e1,x,i]<t)):
                    eval_time = t-labelled_times[e1,x,i]+1.0
                    power_comp = pow(eval_time,-decay_coefficients[e1,x])
                    ESSE_memview[e1,x] += power_comp        
    return ESSE    

def compute_intensity_partial(
    DTYPEf_t t, int num_event_types, int num_states,
    DTYPEf_t base_rate,
    np.ndarray[DTYPEf_t, ndim=2] impact_coefficients,
    np.ndarray[DTYPEf_t, ndim=2] decay_coefficients,
    np.ndarray[DTYPEf_t, ndim=3] labelled_times,
    np.ndarray[DTYPEi_t, ndim=2] count
):
    cdef np.ndarray[DTYPEf_t,ndim=1] alpha = impact_coefficients.flatten()
    cdef DTYPEf_t s = -1.0 # the input s=-1.0 accounts for summing over all times smaller than t
    cdef np.ndarray[DTYPEf_t,ndim=1] S = compute_ESSE_partial(
        t,s, num_event_types, num_states, decay_coefficients,labelled_times,count).flatten()
    return base_rate+np.dot(alpha,S)

def compute_intensity(double t,
                      np.ndarray[DTYPEf_t, ndim=1] times,
                      np.ndarray[DTYPEi_t, ndim=1] events,
                      np.ndarray[DTYPEi_t, ndim=1] states,
                      np.ndarray[DTYPEf_t, ndim=1] base_rates,
                      np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
                      np.ndarray[DTYPEf_t, ndim=3] decay_coefficients
                      ):
    cdef int num_event_types = base_rates.shape[0]
    cdef int num_states = impact_coefficients.shape[1]
    cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times = np.zeros((num_event_types,num_states,len(times)),dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] count = np.zeros((num_event_types,num_states),dtype=DTYPEi)
    labelled_times, count = distribute_times_per_event_state(
        num_event_types,num_states,
        times,events,states)
    cdef np.ndarray[DTYPEf_t,ndim=2] alpha = impact_coefficients.reshape(-1,num_event_types)
    cdef float s=np.array(-1,dtype=float)# the input s=-1.0 accounts for summing over all times smaller than t
    cdef np.ndarray[DTYPEf_t,ndim=2] S = np.zeros_like(alpha)
    cdef double [:] base_rates_memview = base_rates
    cdef double [:,:,:] dec_coef_memview = decay_coefficients
    cdef double [:,:,:] labelled_times_memview = labelled_times
    cdef long [:,:] count_memview = count
    cdef double [:,:] alpha_memview = alpha
    cdef double [:,:] S_memview = S
    cdef int e = 0
    cdef double result = 0.0
    for e in range(num_event_types):
        S[:,e] = compute_ESSE_partial(
            t,s, num_event_types, num_states, dec_coef_memview[:,:,e],labelled_times_memview,count_memview).flatten()    
        result+=base_rates_memview[e]+np.dot(alpha_memview[:,e],S_memview[:,e])
    return result

"This newer compute_intensities_given_lt improves performance by utilising less loops"
def compute_intensities_given_lt(
    double t,
    np.ndarray[DTYPEf_t, ndim=3] labelled_times,
    np.ndarray[DTYPEi_t, ndim=2] count,           
    np.ndarray[DTYPEf_t, ndim=1] base_rates,
    np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
    np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
    np.ndarray[DTYPEf_t, ndim=2] reshaped_imp_coef,           
    int num_event_types,
    int num_states,           
    int first_event_index = 0           
):
    cdef np.ndarray[DTYPEf_t, ndim=1] intensities = np.zeros(num_event_types,dtype=DTYPEf)
    cdef DTYPEf_t [:] intensities_memview = intensities
    cdef DTYPEf_t [:,:,:] lt_memview = labelled_times
    cdef DTYPEi_t [:,:] count_memview = count
    cdef DTYPEf_t [:] base_rates_memview = base_rates
    cdef DTYPEf_t [:,:,:] imp_coef_memview = impact_coefficients
    cdef DTYPEf_t [:,:,:] dec_coef_memview = decay_coefficients
    cdef np.ndarray[DTYPEf_t,ndim=3] ESSE = np.zeros((num_event_types, num_states, num_event_types), dtype=DTYPEf)
    cdef double [:,:,:] ESSE_memview = ESSE
    cdef int e, e1, x, i
    cdef double eval_time,power_comp 
    for e in prange(first_event_index, num_event_types, nogil=True,):
        for e1 in range(num_event_types):
            for x in range(num_states):
                for i in range(count_memview[e1,x]):
                    if (lt_memview[e1,x,i]<t):
                        eval_time = t-lt_memview[e1,x,i]+1.0
                        power_comp = pow(eval_time,-dec_coef_memview[e1,x,e])
                        ESSE_memview[e1,x,e] += power_comp
                intensities_memview[e]+=imp_coef_memview[e1,x,e]*ESSE_memview[e1,x,e]
        intensities_memview[e]+= base_rates_memview[e]        
    return intensities    

#Older compute_intensities_given_lt:
#def compute_intensities_given_lt(double t,
#                      np.ndarray[DTYPEf_t, ndim=3] labelled_times,
#                      np.ndarray[DTYPEi_t, ndim=2] count,           
#                      np.ndarray[DTYPEf_t, ndim=1] base_rates,
#                      np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
#                      np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
#                      np.ndarray[DTYPEf_t, ndim=2] reshaped_imp_coef,           
#                      int num_event_types,
#                      int num_states,           
#                      int first_event_index = 0           
#                      ):
#    cdef np.ndarray[DTYPEf_t, ndim=1] intensities = np.zeros(num_event_types,dtype=DTYPEf)
#    cdef float s = -1.0 # the input s=-1.0 accounts for summing over all times smaller than t
#    cdef np.ndarray[DTYPEf_t,ndim=2] S = np.ones_like(reshaped_imp_coef)
#    cdef int e = 0
#    for e in range(first_event_index,num_event_types):
#        S[:,e] = compute_ESSE_partial(
#            t,s, num_event_types, num_states, decay_coefficients[:,:,e], labelled_times, count).flatten()
#        intensities[e]=base_rates[e]+np.dot(reshaped_imp_coef[:,e],S[:,e])
#    return intensities 

def compute_intensities(double t,
                      np.ndarray[DTYPEf_t, ndim=1] times,
                      np.ndarray[DTYPEi_t, ndim=1] events,
                      np.ndarray[DTYPEi_t, ndim=1] states,
                      np.ndarray[DTYPEf_t, ndim=1] base_rates,
                      np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
                      np.ndarray[DTYPEf_t, ndim=3] decay_coefficients
                      ):
    cdef int num_event_types = base_rates.shape[0]
    cdef int num_states = impact_coefficients.shape[1]
    cdef np.ndarray[DTYPEf_t, ndim=1] intensities = np.zeros(num_event_types,dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times = -np.ones((num_event_types,num_states,len(times)),dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] count = np.zeros((num_event_types,num_states),dtype=DTYPEi)
    labelled_times, count = distribute_times_per_event_state(
        num_event_types,num_states,
        times,events,states)
    cdef np.ndarray[DTYPEf_t,ndim=2] alpha = impact_coefficients.reshape(-1,num_event_types)
    cdef float s=np.array(-1,dtype=float)# the input s=-1.0 accounts for summing over all times smaller than t
    cdef np.ndarray[DTYPEf_t,ndim=2] S = np.ones_like(alpha)
    cdef int e = 0
    for e in range(num_event_types):
        S[:,e] = compute_ESSE_partial(
            t,s, num_event_types, num_states, decay_coefficients[:,:,e],labelled_times,count).flatten()
        intensities[e]=base_rates[e]+np.dot(alpha[:,e],S[:,e])
    return intensities 

def compute_tilda_intensities(double t,
                      np.ndarray[DTYPEf_t, ndim=1] times,
                      np.ndarray[DTYPEi_t, ndim=1] events,
                      np.ndarray[DTYPEi_t, ndim=1] states,
                      np.ndarray[DTYPEf_t, ndim=1] base_rates,
                      np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
                      np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
                      np.ndarray[DTYPEf_t, ndim=3] transition_probabilities        
                      ):
    cdef int num_event_types = base_rates.shape[0]
    cdef int num_states = impact_coefficients.shape[1]
    cdef np.ndarray[DTYPEf_t, ndim=1] intensities = compute_intensities(
        t,times, events, states, base_rates,impact_coefficients, decay_coefficients)
    cdef np.ndarray[DTYPEf_t, ndim=1] tilda_intensities = np.zeros((num_event_types,num_states),dtype=DTYPEf)
    cdef DTYPEi_t current_state = find_current_state(t,times,states)
    cdef double[:,:] current_phi = transition_probabilities[current_state,:,:]
    cdef double[:] intensities_memview = intensities
    cdef double[:,:] tilda_intensities_memview = tilda_intensities
    with nogil:
        for e in range(num_event_types):
            for x in range(num_states):
                tilda_intensities_memview[e,x]=current_phi[e,x]*intensities_memview[e]
    return tilda_intensities
    
def compute_tilda_intensities_given_lt(double t,
                      np.ndarray[DTYPEf_t, ndim=1] times,
                      np.ndarray[DTYPEi_t, ndim=1] events,
                      np.ndarray[DTYPEi_t, ndim=1] states,
                      np.ndarray[DTYPEf_t, ndim=1] base_rates,
                      np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
                      np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
                      np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
                      np.ndarray[DTYPEf_t, ndim=3] labelled_times,
                      np.ndarray[DTYPEi_t, ndim=2] count,
                      int first_event_index = 0                  
                      ):
    cdef int num_event_types = base_rates.shape[0]
    cdef int num_states = impact_coefficients.shape[1]
    cdef np.ndarray[DTYPEf_t, ndim=2] reshaped_imp_coef = impact_coefficients.reshape((-1,num_event_types))
    cdef np.ndarray[DTYPEf_t, ndim=1] intensities = compute_intensities_given_lt(
        t, labelled_times, count, base_rates, impact_coefficients,decay_coefficients,reshaped_imp_coef,
        num_event_types, num_states, first_event_index)
    cdef np.ndarray[DTYPEf_t, ndim=2] tilda_intensities = np.zeros((num_event_types,num_states),dtype=DTYPEf)
    cdef DTYPEi_t current_state = find_current_state(t,times,states)
    cdef double[:,:] current_phi = transition_probabilities[current_state,:,:]
    cdef double[:] intensities_memview = intensities
    cdef double[:,:] tilda_intensities_memview = tilda_intensities
    with nogil:
        for e in range(first_event_index,num_event_types):
            for x in range(num_states):
                tilda_intensities_memview[e,x]=current_phi[e,x]*intensities_memview[e]
    return tilda_intensities        

def compute_tilda_intensity_partial(double t, int num_event_types, int num_states,
                      np.ndarray[DTYPEf_t, ndim=1] times,              
                      np.ndarray[DTYPEi_t, ndim=1] states,               
                      double base_rate,
                      np.ndarray[DTYPEf_t, ndim=2] impact_coefficients,
                      np.ndarray[DTYPEf_t, ndim=2] decay_coefficients,
                      np.ndarray[DTYPEf_t, ndim=1] trans_prob,              
                      np.ndarray[DTYPEf_t, ndim=3] labelled_times,
                      np.ndarray[DTYPEi_t, ndim=2] count):
    cdef DTYPEf_t intensity = compute_intensity_partial(
        t, num_event_types, num_states, base_rate,impact_coefficients,decay_coefficients,labelled_times,count)
    cdef DTYPEi_t current_state = find_current_state(t,times,states)
    return trans_prob[current_state]*intensity

def compute_history_of_intensities(int n_event_types,
                                   int n_states,
                                   np.ndarray[DTYPEf_t, ndim=1] times,
                                   np.ndarray[DTYPEi_t, ndim=1] events,
                                   np.ndarray[DTYPEi_t, ndim=1] states,
                                   np.ndarray[DTYPEf_t, ndim=1] base_rates,
                                   np.ndarray[DTYPEf_t, ndim=3] imp_coef,
                                   np.ndarray[DTYPEf_t, ndim=3] dec_coef,
                                   liquidator_present=False,
                                   DTYPEf_t start_time_zero_event=-1.0,
                                   DTYPEf_t end_time_zero_event=-2.0,
                                   int density_of_eval_points=100,
                                  ):
    if not liquidator_present:
        start_time_zero_event = float(times[0])
        end_time_zero_event=float(times[len(times)-1])
    cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times =np.zeros((n_event_types,n_states,len(times)),dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] count=np.zeros((n_event_types,n_states),dtype=DTYPEi)
    labelled_times,count=distribute_times_per_event_state(
        n_event_types,
        n_states,
        times,
        events,
        states,
        len_labelled_times = len(times)
    )
    cdef np.ndarray[DTYPEf_t, ndim=1] tt = np.sort(
        np.concatenate([times,
                        np.linspace(times[0],times[len(times)-1],num=density_of_eval_points,dtype=DTYPEf)],
                       axis=0),
        axis=0)
    idx=np.concatenate([[1],(np.diff(tt)>1.0e-8)]).astype(np.bool)
    cdef np.ndarray[DTYPEf_t, ndim=2] eval_points =tt[idx].reshape(-1,1)
    cdef int idx_start_zero_event=bisect.bisect_left(np.squeeze(eval_points),start_time_zero_event)
    cdef int idx_end_zero_event=min(eval_points.shape[0],
                                    bisect.bisect_right(np.squeeze(eval_points),end_time_zero_event)
                                   )
#     print('idx_start_zero_event={}'.format(idx_start_zero_event))
#     print('idx_end_zero_event={}'.format(idx_end_zero_event))
    cdef np.ndarray[DTYPEf_t, ndim=2] reshaped_imp_coef = imp_coef.reshape((-1,n_event_types))
    cdef np.ndarray[DTYPEf_t, ndim=2] history_intensities = np.zeros((len(eval_points),n_event_types),dtype=DTYPEf)
    args_0=[labelled_times,count,base_rates,imp_coef,dec_coef,reshaped_imp_coef, n_event_types,n_states,0]
    args_1=[labelled_times,count,base_rates,imp_coef,dec_coef,reshaped_imp_coef, n_event_types,n_states,1]
    if idx_start_zero_event>0:
        history_intensities[:idx_start_zero_event,:]=np.apply_along_axis(
            compute_intensities_given_lt,1,
            eval_points[:idx_start_zero_event,:],
            *args_1)
    if idx_start_zero_event<idx_end_zero_event:
        history_intensities[idx_start_zero_event:idx_end_zero_event,:]=\
        np.apply_along_axis(
            compute_intensities_given_lt,1,
            eval_points[idx_start_zero_event:idx_end_zero_event,:],
            *args_0)
    if idx_end_zero_event<len(eval_points):    
        history_intensities[idx_end_zero_event:,:]=np.apply_along_axis(
            compute_intensities_given_lt,1,
            eval_points[idx_end_zero_event:,:],
            *args_1)
    cdef np.ndarray[DTYPEf_t, ndim=2] result =\
    np.concatenate([eval_points,history_intensities],axis=1)
    return result 

def compute_history_of_tilda_intensities(int n_event_types,
                                   int n_states,
                                   np.ndarray[DTYPEf_t, ndim=1] times,
                                   np.ndarray[DTYPEi_t, ndim=1] events,
                                   np.ndarray[DTYPEi_t, ndim=1] states,
                                   np.ndarray[DTYPEf_t, ndim=1] base_rates,
                                   np.ndarray[DTYPEf_t, ndim=3] imp_coef,
                                   np.ndarray[DTYPEf_t, ndim=3] dec_coef,
                                   np.ndarray[DTYPEf_t, ndim=3] trans_prob,      
                                   DTYPEf_t start_time_zero_event=-1.0,
                                   DTYPEf_t end_time_zero_event=-2.0,
                                   int density_of_eval_points=100,
                                  ):
    if (start_time_zero_event < 0):
        start_time_zero_event = np.array(times[0],dtype=DTYPEf,copy=True)
    if end_time_zero_event < start_time_zero_event:
        end_time_zero_event=np.array(times[len(times)-1],dtype=DTYPEf,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times =np.zeros((n_event_types,n_states,len(times)),dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] count=np.zeros((n_event_types,n_states),dtype=DTYPEi)
    labelled_times,count=distribute_times_per_event_state(
        n_event_types,
        n_states,
        times,
        events,
        states,
        len_labelled_times = len(times)
    )
    cdef np.ndarray[DTYPEf_t, ndim=1] tt = np.sort(
        np.concatenate([times,
                        np.linspace(times[0],times[len(times)-1],num=density_of_eval_points,dtype=DTYPEf)],
                       axis=0),
        axis=0)
    idx=np.concatenate([[1],(np.diff(tt)>1.0e-8)]).astype(np.bool)
    cdef np.ndarray[DTYPEf_t, ndim=2] eval_points =np.array(tt[idx].reshape(-1,1),copy=True)
    cdef int idx_start_zero_event=bisect.bisect_left(np.squeeze(eval_points),start_time_zero_event)
    cdef int idx_end_zero_event=min(eval_points.shape[0],
                                    bisect.bisect_right(np.squeeze(eval_points),end_time_zero_event))
#     print('idx_start_zero_event={}'.format(idx_start_zero_event))
#     print('idx_end_zero_event={}'.format(idx_end_zero_event))
    cdef np.ndarray[DTYPEf_t, ndim=3] history_tilda_intensities =\
    np.zeros((len(eval_points),n_event_types,n_states),dtype=DTYPEf)
    args_0=[times,events,states,base_rates,imp_coef,dec_coef,trans_prob,labelled_times,count,0]
    args_1=[times,events,states,base_rates,imp_coef,dec_coef,trans_prob,labelled_times,count,1]
    if idx_start_zero_event>0:
        history_tilda_intensities[:idx_start_zero_event,:,:]=\
        np.apply_along_axis(compute_tilda_intensities_given_lt,1,
                            eval_points[:idx_start_zero_event,:],
                            *args_1)
    if idx_start_zero_event<idx_end_zero_event:
        history_tilda_intensities[idx_start_zero_event:idx_end_zero_event,:,:]=\
        np.apply_along_axis(compute_tilda_intensities_given_lt,1,
                            eval_points[idx_start_zero_event:idx_end_zero_event,:],
                            *args_0)
    if idx_end_zero_event<len(eval_points):
        history_tilda_intensities[idx_end_zero_event:,:,:]=\
        np.apply_along_axis(compute_tilda_intensities_given_lt,1,
                            eval_points[idx_end_zero_event:,:],
                            *args_1)
    cdef np.ndarray[DTYPEf_t, ndim=3] eval_point_matrix = np.repeat(
        np.expand_dims(eval_points,axis=2),
        n_states,axis=2)
    cdef np.ndarray[DTYPEf_t, ndim=3] result =\
    np.concatenate([eval_point_matrix,history_tilda_intensities],axis=1)
    return result 

"FUNCTIONS RELATED TO COMPUTATION OF LOGLIKELIHOOD"

def compute_ESSE_two_partial(double t, double s,
                             int num_event_types, int num_states,
                             double [:,:] decay_coefficients,
                             double [:,:,:] labelled_times,
                             long [:,:] count
                            ):
    cdef np.ndarray[DTYPEf_t,ndim=2] result = np.zeros((num_event_types, num_states), dtype=DTYPEf)
    cdef double [:,:] result_memview = result
    cdef int e1, x, i 
    cdef double  eval_time = 1.0
    cdef double  power_comp = 1.0
    cdef double  log_comp = 0.0
    with nogil:
        for e1 in range(num_event_types):
            for x in range(num_states):
                for i in range(count[e1,x]):
                    if ((labelled_times[e1,x,i]>=s) & (labelled_times[e1,x,i]<t)):
                        eval_time = t-labelled_times[e1,x,i]+1.0
                        power_comp = pow(eval_time,1-decay_coefficients[e1,x])
                        log_comp = log(eval_time)
                        result_memview[e1,x] += power_comp*log_comp                
    return result

def prange_compute_ESSE_two_partial(double t, double s,
                             int num_event_types, int num_states,
                             double [:,:] decay_coefficients,
                             double [:,:,:] labelled_times,
                             long [:,:] count
                            ):
    cdef np.ndarray[DTYPEf_t,ndim=2] result = np.zeros((num_event_types, num_states), dtype=DTYPEf)
    cdef double [:,:] result_memview = result
    cdef int e1, x, i 
    cdef double  eval_time = 1.0
    cdef double  power_comp = 1.0
    cdef double  log_comp = 0.0
    for e1 in prange(num_event_types, nogil=True,  ):
        for x in range(num_states):
            for i in range(count[e1,x]):
                if ((labelled_times[e1,x,i]>=s) & (labelled_times[e1,x,i]<t)):
                    eval_time = t-labelled_times[e1,x,i]+1.0
                    power_comp = pow(eval_time,1.0-decay_coefficients[e1,x])
                    log_comp = log(eval_time)
                    result_memview[e1,x] += power_comp*log_comp                
    return result

def compute_ESSE_three_partial(double u, double t, double s,
                               int num_event_types, int num_states,
                               double [:,:] decay_coefficients,
                               double [:,:,:] labelled_times,
                               long [:,:] count):
    cdef np.ndarray[DTYPEf_t,ndim=2] result = np.zeros_like(decay_coefficients)
    cdef double [:,:] result_memview = result
    cdef int e1, x, i
    cdef double eval_time = 0.0
    cdef double  power_comp = 0.0
    with nogil:
        for e1 in range(num_event_types):
            for x in range(num_states):
                for i in range(count[e1,x]):
                    if ((labelled_times[e1,x,i]>=s) & (labelled_times[e1,x,i]<t)):
                        eval_time = u-labelled_times[e1,x,i]+1.0
                        power_comp = pow(eval_time,1-decay_coefficients[e1,x])
                        result_memview[e1,x] += power_comp

    return result

def prange_compute_ESSE_three_partial(double u, double t, double s,
                               int num_event_types, int num_states,
                               double [:,:] decay_coefficients,
                               double [:,:,:] labelled_times,
                               long [:,:] count):
    cdef np.ndarray[DTYPEf_t,ndim=2] result = np.zeros((num_event_types, num_states), dtype=DTYPEf)
    cdef double [:,:] result_memview = result
    cdef int e1, x, i
    cdef double eval_time = 0.0
    cdef double  power_comp = 0.0
    for e1 in prange(num_event_types, nogil=True,  ):
        for x in range(num_states):
            for i in range(count[e1,x]):
                if ((labelled_times[e1,x,i]>=s) & (labelled_times[e1,x,i]<t)):
                    eval_time = u-labelled_times[e1,x,i]+1.0
                    power_comp = pow(eval_time,1.0-decay_coefficients[e1,x])
                    result_memview[e1,x] += power_comp
    return result
        
def compute_partial_at_arrival_times(
                    double[:] intensity, 
                    double[:] intensity_inverse,
                    double[:] arrival_times,
                    double [:,:] impact_coefficients,
                    double [:,:] decay_coefficients,
                    double [:,:,:] labelled_times,
                    long [:,:] count,
                    int num_event_types,
                    int num_states,
                    int len_labelled_times,
                    int num_arrival_times,                 
                    double [:,:,:] ESSE,
                    double [:,:,:] ESSE_one
                   ):
    'The function assumes that intensity has been initialised equal to base rate, and ESSE and ESSE_one have been initialised equal to zero.'
    cdef double l_plus = 0
    cdef double grad_base_rate = 0
    cdef Py_ssize_t e1, x, i,t
    cdef double  eval_time = 0
    cdef double power_comp = 0
    cdef double  log_comp = 0
    cdef int count_zero_intensity = 0
#     for t in prange(num_arrival_times,nogil=True): #schedule='guided'):
#     The implementation with prange is "prange_compute_partial_at_arrival_times"
    with nogil:
        for t in range(num_arrival_times):    
            for e1 in range(num_event_types):
                for x in range(num_states):
                    for i in range(count[e1,x]):
                        if ((labelled_times[e1,x,i]<arrival_times[t])):
                            eval_time = arrival_times[t]-labelled_times[e1,x,i]+1.0
                            power_comp = pow(eval_time,-decay_coefficients[e1,x])
                            log_comp = log(eval_time)
                            ESSE[e1,x,t] += power_comp
                            ESSE_one[e1,x,t] += power_comp*log_comp
                        else:
                            break
                    intensity[t]+=ESSE[e1,x,t]*impact_coefficients[e1,x]
            if (intensity[t]>0):
                intensity_inverse[t]=1.0/intensity[t]
                l_plus +=log(intensity[t])
                grad_base_rate +=intensity_inverse[t]
            else:
                count_zero_intensity+=1
    if (count_zero_intensity >0):
        print('compute_partial_at_arrival_times: count_zero_intensity={}'.format(count_zero_intensity))    
    return l_plus,grad_base_rate      
        
def prange_compute_partial_at_arrival_times(
                    double[:] intensity, 
                    double[:] intensity_inverse,
                    double[:] arrival_times,
                    double [:,:] impact_coefficients,
                    double [:,:] decay_coefficients,
                    double [:,:,:] labelled_times,
                    long [:,:] count,
                    int num_event_types,
                    int num_states,
                    int len_labelled_times,
                    int num_arrival_times,                 
                    double [:,:,:] ESSE,
                    double [:,:,:] ESSE_one
                   ):
    'The function assumes that intensity has been initialised equal to base rate, and ESSE and ESSE_one have been initialised equal to zero.'
    cdef double l_plus = 0
    cdef double grad_base_rate = 0
    cdef Py_ssize_t e1, x, i,t
    cdef double  eval_time = 0
    cdef double power_comp = 0
    cdef double  log_comp = 0
    cdef int count_zero_intensity = 0
    for t in prange(num_arrival_times,nogil=True, schedule='guided'): 
        for e1 in range(num_event_types):
            for x in range(num_states):
                for i in range(count[e1,x]):
                    if ((labelled_times[e1,x,i]<arrival_times[t])):
                        eval_time = arrival_times[t]-labelled_times[e1,x,i]+1.0
                        power_comp = pow(eval_time,-decay_coefficients[e1,x])
                        log_comp = log(eval_time)
                        ESSE[e1,x,t] += power_comp
                        ESSE_one[e1,x,t] += power_comp*log_comp
                    else:
                        break
                intensity[t]+=ESSE[e1,x,t]*impact_coefficients[e1,x]
        if (intensity[t]>0):
            intensity_inverse[t]=1/intensity[t]
            l_plus +=log(intensity[t])
            grad_base_rate +=intensity_inverse[t]
        else:
            count_zero_intensity+=1
    if (count_zero_intensity >0):
        print('compute_partial_at_arrival_times: count_zero_intensity={}'.format(count_zero_intensity))    
    return l_plus,grad_base_rate   
    
def compute_l_plus_partial_and_gradient_partial(
    int index_event,
    int num_event_types,
    int num_states,
    double time_start,
    double base_rate,
    np.ndarray[DTYPEf_t, ndim=2] impact_coefficients,
    np.ndarray[DTYPEf_t, ndim=2] decay_coefficients,      
    np.ndarray[DTYPEf_t, ndim=3] labelled_times,
    np.ndarray[DTYPEi_t, ndim=2] count,
    np.ndarray[DTYPEf_t, ndim=1] arrival_times,
    int num_arrival_times,
    int len_labelled_times,
    use_prange=False,
):
    cdef Py_ssize_t e1, x, i
    cdef int M
    cdef DTYPEf_t [:,:] imp_coef_memview = impact_coefficients
    cdef DTYPEf_t [:,:] dec_coef_memview = decay_coefficients
    cdef DTYPEf_t [:,:,:] lt_memview = labelled_times
    cdef DTYPEi_t [:,:] count_memview = count
    cdef DTYPEf_t [:] arrival_times_memview = arrival_times
    'Notice that lambda_ is initialised equal to base rate'
    cdef np.ndarray[DTYPEf_t,ndim=1] lambda_ = base_rate*np.ones(num_arrival_times,dtype=DTYPEf)
    cdef DTYPEf_t [:] lambda_memview = lambda_
    cdef np.ndarray[DTYPEf_t,ndim=1] lambda_inverse = np.ones(num_arrival_times,dtype=DTYPEf)
    cdef DTYPEf_t [:] lambda_inverse_memview = lambda_inverse
    'ESSE_at_arrival_times and ESSE_one_at_arrival_times must be initilised qual to zero for usage in compute_partial_at_arrival_times'
    cdef np.ndarray[DTYPEf_t,ndim=3] ESSE_at_arrival_times = np.zeros(
        (num_event_types,num_states,num_arrival_times),dtype=DTYPEf)
    cdef DTYPEf_t [:,:,:] ESSE = ESSE_at_arrival_times
    cdef np.ndarray[DTYPEf_t,ndim=3] ESSE_one_at_arrival_times = np.zeros(
        (num_event_types,num_states,num_arrival_times),dtype=DTYPEf)
    cdef DTYPEf_t [:,:,:] ESSE_one = ESSE_one_at_arrival_times
    cdef double l_plus, grad_base_rate
    if use_prange:
        l_plus,grad_base_rate = prange_compute_partial_at_arrival_times(
                        lambda_memview, 
                        lambda_inverse_memview,
                        arrival_times_memview,
                        imp_coef_memview,
                        dec_coef_memview,
                        lt_memview,
                        count_memview,
                        num_event_types,
                        num_states,
                        len_labelled_times,
                        num_arrival_times,                 
                        ESSE,
                        ESSE_one
                       )
    else:
        l_plus,grad_base_rate = compute_partial_at_arrival_times(
                        lambda_memview, 
                        lambda_inverse_memview,
                        arrival_times_memview,
                        imp_coef_memview,
                        dec_coef_memview,
                        lt_memview,
                        count_memview,
                        num_event_types,
                        num_states,
                        len_labelled_times,
                        num_arrival_times,                 
                        ESSE,
                        ESSE_one
                       )
    cdef np.ndarray[DTYPEf_t,ndim=2] grad_imp_coef = np.zeros((num_event_types, num_states), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t,ndim=2] grad_dec_coef = np.zeros((num_event_types, num_states), dtype=DTYPEf)
    for e1 in range(num_event_types):
        for x in range(num_states):
            grad_imp_coef[e1,x] = np.dot(
                ESSE_at_arrival_times[e1,x,:],
                lambda_inverse)
            grad_dec_coef[e1,x] = -impact_coefficients[e1,x]*np.dot(
                    ESSE_one_at_arrival_times[e1,x,:],
                    lambda_inverse)
    return l_plus, grad_base_rate, grad_imp_coef, grad_dec_coef                  

def labelled_times_within_range(double s, double t,
                                double [:,:,:] labelled_times,
                                long [:,:] count,
                                int n_event_types,
                                int n_states
                               ):
    cdef np.ndarray[DTYPEi_t, ndim = 2] num = np.zeros((n_event_types,n_states),dtype=DTYPEi)
    cdef long [:,:] num_memview = num
    cdef int e1,x,i
    with nogil:
        for e1 in range(n_event_types):
            for x in range(n_states):
                for i in range(count[e1,x]):
                    if ((labelled_times[e1,x,i]>=s) & (labelled_times[e1,x,i]<t)):
                        num_memview[e1,x]+=1
    return num

def compute_sum_of_S2(
    double time_start, double time_end, int num_event_types, int num_states,
    double [:,:] decay_coefficients, double [:,:,:] labelled_times, long [:,:] count
):
    cdef np.ndarray[DTYPEf_t,ndim=2] result = np.zeros((num_event_types, num_states), dtype=DTYPEf)
    cdef double [:,:] result_memview = result
    cdef int e1, x, i 
    cdef double  eval_time = 1.0
    cdef double  power_comp = 1.0
    cdef double  log_comp = 0.0
    with nogil:
        for e1 in range(num_event_types):
            for x in range(num_states):
                for i in range(count[e1,x]):
                    if (labelled_times[e1,x,i]<time_end): #add ESSE_two(T)
                        eval_time = time_end-labelled_times[e1,x,i]+1.0
                        power_comp = pow(eval_time,1-decay_coefficients[e1,x])
                        log_comp = log(eval_time)
                        result_memview[e1,x] += power_comp*log_comp
                        if (labelled_times[e1,x,i]<time_start): #subtract ESSE_two (t_0)
                            eval_time = time_start-labelled_times[e1,x,i]+1.0
                            power_comp = pow(eval_time,1-decay_coefficients[e1,x])
                            log_comp = log(eval_time)
                            result_memview[e1,x] += -power_comp*log_comp
    return result

def prange_compute_sum_of_S2(
    double time_start, double time_end, int num_event_types, int num_states,
    double [:,:] decay_coefficients, double [:,:,:] labelled_times, long [:,:] count
):
    cdef np.ndarray[DTYPEf_t,ndim=2] result = np.zeros((num_event_types, num_states), dtype=DTYPEf)
    cdef double [:,:] result_memview = result
    cdef int e1, x, i 
    cdef double  eval_time = 1.0
    cdef double  power_comp = 1.0
    cdef double  log_comp = 0.0
    for e1 in prange(num_event_types, nogil=True,  ):
        for x in range(num_states):
            for i in range(count[e1,x]):
                if (labelled_times[e1,x,i]<time_end): #add ESSE_two(T)
                    eval_time = time_end-labelled_times[e1,x,i]+1.0
                    power_comp = pow(eval_time,1-decay_coefficients[e1,x])
                    log_comp = log(eval_time)
                    result_memview[e1,x] += power_comp*log_comp
                    if (labelled_times[e1,x,i]<time_start): #subtract ESSE_two (t_0)
                        eval_time = time_start-labelled_times[e1,x,i]+1.0
                        power_comp = pow(eval_time,1-decay_coefficients[e1,x])
                        log_comp = log(eval_time)
                        result_memview[e1,x] += -power_comp*log_comp
    return result

def compute_sum_of_S3(
    DTYPEf_t time_start, DTYPEf_t time_end, int num_event_types, int num_states,
    DTYPEf_t [:,:] decay_coefficients, DTYPEf_t [:,:,:] labelled_times, DTYPEi_t [:,:] count
):
    cdef np.ndarray[DTYPEf_t,ndim=2] result = np.zeros((num_event_types, num_states), dtype=DTYPEf)
    cdef double [:,:] result_memview = result
    cdef int e1, x, i
    cdef double eval_time = 0.0
    cdef double  power_comp = 0.0
    with nogil:
        for e1 in range(num_event_types):
            for x in range(num_states):
                for i in range(count[e1,x]):
                    if (labelled_times[e1,x,i]<time_start): #compute ESSE_three(t_0,t_0) - ESSE_three(t_0,T)
                        eval_time = time_start-labelled_times[e1,x,i]+1.0
                        power_comp = pow(eval_time, 1.0 -decay_coefficients[e1,x])
                        result_memview[e1,x]+= power_comp # add ESSE_three(t_0,t_0)
                        eval_time = time_end-labelled_times[e1,x,i]+1.0
                        power_comp = pow(eval_time, 1.0 -decay_coefficients[e1,x])
                        result_memview[e1,x]+= -power_comp # subtract ESSE_three(t_0,T)
                    if ((labelled_times[e1,x,i]>=time_start) & (labelled_times[e1,x,i]<time_end)): # compute num_times - ESSE_three(t_0,T,T)
                        eval_time = time_end-labelled_times[e1,x,i]+1.0
                        power_comp = pow(eval_time, 1.0 -decay_coefficients[e1,x])
                        result_memview[e1,x] += 1.0 -  power_comp
    return result                     

def prange_compute_sum_of_S3(
    DTYPEf_t time_start, DTYPEf_t time_end, int num_event_types, int num_states,
    DTYPEf_t [:,:] decay_coefficients, DTYPEf_t [:,:,:] labelled_times, DTYPEi_t [:,:] count
):
    cdef np.ndarray[DTYPEf_t,ndim=2] result = np.zeros((num_event_types, num_states), dtype=DTYPEf)
    cdef double [:,:] result_memview = result
    cdef int e1, x, i
    cdef double eval_time = 0.0
    cdef double  power_comp = 0.0
    for e1 in prange(num_event_types, nogil=True,  ):
        for x in range(num_states):
            for i in range(count[e1,x]):
                if (labelled_times[e1,x,i]<time_start): #compute ESSE_three(t_0,t_0) - ESSE_three(t_0,T)
                    eval_time = time_start-labelled_times[e1,x,i]+1.0
                    power_comp = pow(eval_time, 1.0 -decay_coefficients[e1,x])
                    result_memview[e1,x]+= power_comp # add ESSE_three(t_0,t_0)
                    eval_time = time_end-labelled_times[e1,x,i]+1.0
                    power_comp = pow(eval_time, 1.0 -decay_coefficients[e1,x])
                    result_memview[e1,x]+= -power_comp # subtract ESSE_three(t_0,T)
                if ((labelled_times[e1,x,i]>=time_start) & (labelled_times[e1,x,i]<time_end)): # compute num_times - ESSE_three(t_0,T,T)
                    eval_time = time_end-labelled_times[e1,x,i]+1.0
                    power_comp = pow(eval_time, 1.0 -decay_coefficients[e1,x])
                    result_memview[e1,x] += 1.0 -  power_comp
    return result                     

def compute_l_minus_partial_and_gradient_partial(
    int num_event_types, int num_states,
    double base_rate,
    np.ndarray[DTYPEf_t, ndim=2] decay_coefficients,
    np.ndarray[DTYPEf_t, ndim=2] impact_decay_ratios,
    np.ndarray[DTYPEf_t, ndim=3] labelled_times,
    np.ndarray[DTYPEi_t, ndim=2] count,                     
    np.float time_start,
    np.float time_end,
    use_prange = False
):
    cdef np.ndarray[DTYPEf_t,ndim=2] sum_of_S2= np.zeros(
        (num_event_types, num_states),
        dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t,ndim=2] sum_of_S3= np.zeros(
        (num_event_types, num_states),
        dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t,ndim=1] delta = impact_decay_ratios.flatten()
    cdef DTYPEf_t s=  -1.0  # the input s=-1.0 accounts for summing over all times smaller than t
    if use_prange:
        sum_of_S3 = prange_compute_sum_of_S3(
            time_start, time_end, num_event_types, num_states,
            decay_coefficients, labelled_times, count
        )
        sum_of_S2 = prange_compute_sum_of_S2(
            time_start, time_end, num_event_types, num_states,
            decay_coefficients, labelled_times, count
        )
    else:
        sum_of_S3 = compute_sum_of_S3(
            time_start, time_end, num_event_types, num_states,
            decay_coefficients, labelled_times, count
        )
        sum_of_S2 = compute_sum_of_S2(
            time_start, time_end, num_event_types, num_states,
            decay_coefficients, labelled_times, count
        )
    cdef double l_minus =  base_rate*(time_end-time_start)+np.dot(delta,sum_of_S3.flatten())
    cdef double grad_base_rate = time_end-time_start
    cdef np.ndarray[DTYPEf_t, ndim=2] grad_imp_coef = sum_of_S3/(decay_coefficients-1.0)
    cdef np.ndarray[DTYPEf_t, ndim=2] grad_dec_coef = impact_decay_ratios*(sum_of_S2 - grad_imp_coef)
    return l_minus, grad_base_rate, grad_imp_coef, grad_dec_coef
                           
"""
tree of function calls:
compute_event_loglikelihood_partial_and_gradient_partial --> [compute_l_plus_partial_and_gradient_partial, compute_l_minus_partial_and_gradient_partial];
compute_l_plus_partial_and_gradient_partial --> compute_partial_at_arrival_times;
compute_l_minus_partial_and_gradient_partial --> [compute_sum_of_S3, compute_sum_of_S2];
"""
def compute_event_loglikelihood_partial_and_gradient_partial(
    int event_type,
    int n_event_types,
    int n_states,
    double base_rate,
    np.ndarray[DTYPEf_t, ndim=2] imp_coef,
    np.ndarray[DTYPEf_t, ndim=2] dec_coef,
    np.ndarray[DTYPEf_t, ndim=2] ratio,
    np.ndarray[DTYPEf_t, ndim=3] labelled_times,
    np.ndarray[DTYPEi_t, ndim=2] count,
    np.ndarray[DTYPEf_t, ndim=1] arrival_times,
    int num_arrival_times,
    int len_labelled_times,
    double time_start,
    double time_end,
    use_prange = False
):
    cdef int e = event_type
    cdef DTYPEf_t l_plus, l_minus, l_plus_base,l_minus_base
    cdef np.ndarray[DTYPEf_t, ndim=2] l_plus_imp = np.zeros((n_event_types,n_states),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] l_plus_dec = np.zeros((n_event_types,n_states),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] l_minus_imp = np.zeros((n_event_types,n_states),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] l_minus_dec = np.zeros((n_event_types,n_states),dtype=DTYPEf)
    l_plus,l_plus_base,l_plus_imp,l_plus_dec = compute_l_plus_partial_and_gradient_partial(
        e, n_event_types, n_states, time_start,
        base_rate, imp_coef, dec_coef,
        labelled_times,count, arrival_times, num_arrival_times, len_labelled_times, use_prange
    )
    l_minus,l_minus_base,l_minus_imp,l_minus_dec = compute_l_minus_partial_and_gradient_partial(
        n_event_types, n_states, base_rate, dec_coef,ratio, labelled_times, count, time_start,time_end,
        use_prange,
    )
    cdef double log_likelihood = l_plus-l_minus
    cdef DTYPEf_t log_likelihood_base = l_plus_base - l_minus_base
    cdef np.ndarray[DTYPEf_t, ndim=2] log_likelihood_imp = l_plus_imp-l_minus_imp
    cdef np.ndarray[DTYPEf_t, ndim=2] log_likelihood_dec = l_plus_dec-l_minus_dec
    cdef np.ndarray[DTYPEf_t, ndim=1] gradient = parameters_to_array_partial(
        log_likelihood_base,
        log_likelihood_imp,
        log_likelihood_dec)
    return log_likelihood, gradient

"FUNCTIONS CALLED TO COMPUTE RESIDUALS"

def compute_time_integral_of_intensity_partial(
    int num_event_types, int num_states,
    DTYPEf_t base_rate,
    np.ndarray[DTYPEf_t, ndim=2] decay_coefficients,
    np.ndarray[DTYPEf_t, ndim=2] impact_decay_ratios,
    np.ndarray[DTYPEf_t, ndim=3] labelled_times,
    np.ndarray[DTYPEi_t, ndim=2] count,                     
    double time_start,
    double time_end,
    use_prange=False):
    cdef np.ndarray[DTYPEf_t,ndim=2] sum_of_S3= np.zeros((num_event_types, num_states),  dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t,ndim=1] delta = impact_decay_ratios.flatten()
    if use_prange:
        sum_of_S3 = prange_compute_sum_of_S3(
            time_start, time_end, num_event_types, num_states,
            decay_coefficients, labelled_times, count
        )
    else:
        sum_of_S3 = compute_sum_of_S3(
            time_start, time_end, num_event_types, num_states,
            decay_coefficients, labelled_times, count
        )
    return base_rate*(time_end-time_start)+np.dot(delta,sum_of_S3.flatten())

def extract_arrival_times_of_event(int event_index,
                                   int num_event_types,
                                   int num_states,
                                   np.ndarray[DTYPEf_t, ndim=3] labelled_times,
                                   np.ndarray[DTYPEi_t, ndim=2] count,
                                   int len_labelled_times,
                                   double time_start
                                  ):
    cdef int e1, x, i
    cdef np.ndarray[DTYPEf_t, ndim = 2] labelled_times_of_event = labelled_times[event_index,:,:]
    cdef double [:,:] times_memview = labelled_times_of_event
    cdef long [:,:] count_memview = count
    cdef np.ndarray[DTYPEf_t, ndim=1] tt = np.zeros(num_states*len_labelled_times,dtype=DTYPEf)
    cdef double [:] tt_memview = tt
    cdef int n =0
    with nogil:
        for x in range(num_states):
                for i in range(count_memview[event_index,x]):
                    if (times_memview[x,i]>=time_start):
                        tt_memview[n] = times_memview[x,i]
                        n+=1           
    cdef int num_arrival_times = n
    cdef np.ndarray[DTYPEf_t,ndim=1] arrival_times = np.sort(tt[:num_arrival_times],axis=None)      
    return arrival_times                                 

def compute_event_residual(int event_index,
                           int num_event_types,
                           int num_states,
                           int len_labelled_times,
                           double base_rate,
                           np.ndarray[DTYPEf_t, ndim=2] decay_coefficients,
                           np.ndarray[DTYPEf_t, ndim=2] impact_decay_ratios,
                           np.ndarray[DTYPEf_t, ndim=3] labelled_times,
                           np.ndarray[DTYPEi_t, ndim=2] count,
                           use_prange = False
                          ):
    cdef double t_0 = -1.0
    cdef np.ndarray[DTYPEf_t,ndim=1] arrival_times =\
    extract_arrival_times_of_event(event_index,
                                   num_event_types,
                                   num_states,
                                   labelled_times,
                                   count,
                                   len_labelled_times,
                                   t_0
                                  )
    cdef int num_arrival_times = len(arrival_times)
    cdef Py_ssize_t n=0
    cdef np.ndarray[DTYPEf_t, ndim=1] residual = np.zeros(num_arrival_times-1,dtype=DTYPEf)
    for n in range(num_arrival_times-1):
        residual[n]=compute_time_integral_of_intensity_partial(num_event_types,
                                                               num_states,
                                                               base_rate,
                                                               decay_coefficients,
                                                               impact_decay_ratios,
                                                               labelled_times,
                                                               count,
                                                               arrival_times[n],
                                                               arrival_times[n+1],
                                                               use_prange
                                                              )
    return residual
    
def compute_residuals(int n_event_types,
                      int n_states,
                      np.ndarray[DTYPEf_t, ndim=1] base_rate,
                      np.ndarray[DTYPEf_t, ndim=3] imp_coef,
                      np.ndarray[DTYPEf_t, ndim=3] dec_coef,
                      np.ndarray[DTYPEf_t, ndim=1] times,
                      np.ndarray[DTYPEi_t, ndim=1] events,
                      np.ndarray[DTYPEi_t, ndim=1] states,
                      use_prange=False,
                     ):
    labelled_times,count = distribute_times_per_event_state(
        n_event_types,
        n_states,
        times,
        events,
        states)
    cdef int len_labelled_times = labelled_times.shape[2]
    cdef np.ndarray[DTYPEf_t, ndim=3] ratios = imp_coef/(dec_coef - 1)
    residuals = []
    cdef int e
    print('Computation of residuals')
    for e in range(n_event_types):
        residuals.append( compute_event_residual(e,
                                                 n_event_types,
                                                 n_states,
                                                 len_labelled_times,
                                                 base_rate[e],
                                                 dec_coef[:,:,e],
                                                 ratios[:,:,e],
                                                 labelled_times,
                                                 count,
                                                 use_prange
                                                )
                        ) 
    print('Computation of residuals terminates.')    
    return residuals

def produce_state_trajectory(np.ndarray[DTYPEi_t, ndim=1] states,
                             np.ndarray[DTYPEf_t, ndim=1] times):
    cdef np.ndarray[DTYPEf_t, ndim=1] result_time = np.zeros_like(times,dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=1] result_state = np.zeros_like(states,dtype=DTYPEi)
    result_time[0] = times[0]
    result_state[0] = states[0]
    cdef Py_ssize_t n
    cdef Py_ssize_t i=1
    for n in range(1,len(times)):
        if states[n]!= states[n-1]:
            result_time[i] = times[n]
            result_state[i] = states[n]
            i+=1
    return result_time[:i], result_state[:i]         

def compute_time_integral_of_ex_intensity(
    int num_event_types, int num_states,
    double base_rate,
    np.ndarray[DTYPEf_t, ndim=2] decay_coefficients,
    np.ndarray[DTYPEf_t, ndim=2] impact_decay_ratios,
    np.ndarray[DTYPEf_t, ndim=3] labelled_times,
    np.ndarray[DTYPEi_t, ndim=2] count,
    np.ndarray[DTYPEf_t, ndim=1] trans_prob,              
    np.ndarray[DTYPEf_t, ndim=1] st_evol_times,
    np.ndarray[DTYPEi_t, ndim=1] st_evol_states,              
    double time_start, double time_end,
    use_prange = False
):
    cdef int idx_t0 = bisect.bisect_left(list(st_evol_times),time_start)
    cdef int idx_T = bisect.bisect_right(list(st_evol_times),time_end)
    partition=list(st_evol_times)[idx_t0:idx_T]
    st_trajectory=list(st_evol_states)[idx_t0:idx_T]
    partition.insert(0,time_start)
    partition.append(time_end)
    st_trajectory.insert(0,st_evol_states[max(idx_t0-1,0)])
    st_trajectory.append(st_evol_states[min(idx_T+1,len(st_evol_states)-1)])
    cdef np.ndarray[DTYPEf_t, ndim=1] time_partition = np.array(partition,dtype=DTYPEf)
    cdef Py_ssize_t n = 0
    cdef double result = 0.0
    for n in range(len(partition)-1):
        if time_partition[n+1]>time_partition[n]:
            result+=\
            (trans_prob[st_trajectory[n]]*
             compute_time_integral_of_intensity_partial(num_event_types,
                                                        num_states,
                                                        base_rate,
                                                        decay_coefficients,
                                                        impact_decay_ratios,
                                                        labelled_times,
                                                        count,
                                                        time_partition[n],
                                                        time_partition[n+1],
                                                        use_prange
                                                        )
            )
    return result

def compute_total_residual_ex(int e, int x,
                              int num_event_types, int num_states, 
                            double base_rate,
                            np.ndarray[DTYPEf_t, ndim=2] decay_coefficients,
                            np.ndarray[DTYPEf_t, ndim=2] impact_decay_ratios,
                            np.ndarray[DTYPEf_t, ndim=3] labelled_times,
                            np.ndarray[DTYPEi_t, ndim=2] count,
                            np.ndarray[DTYPEf_t, ndim=1] trans_prob,              
                            np.ndarray[DTYPEf_t, ndim=1] st_evol_times,
                            np.ndarray[DTYPEi_t, ndim=1] st_evol_states):
    cdef np.ndarray[DTYPEf_t, ndim=1] residual = np.zeros(max(1,count[e,x]-1),dtype=DTYPEf)
    cdef int n=0
    if count[e,x]==0:
        return residual
    else:
        for n in range(count[e,x]-1):
            residual[n] = compute_time_integral_of_ex_intensity(
                num_event_types, num_states,
                base_rate,
                decay_coefficients,
                impact_decay_ratios,
                labelled_times,
                count,
                trans_prob,
                st_evol_times,
                st_evol_states,
                labelled_times[e,x,n],
                labelled_times[e,x,n+1]
            )
        return residual

def compute_total_residuals(int n_event_types,
                      int n_states,
                      np.ndarray[DTYPEf_t, ndim=1] base_rate,
                      np.ndarray[DTYPEf_t, ndim=3] imp_coef,
                      np.ndarray[DTYPEf_t, ndim=3] dec_coef,
                      np.ndarray[DTYPEf_t, ndim=3] trans_prob,      
                      np.ndarray[DTYPEf_t, ndim=1] times,
                      np.ndarray[DTYPEi_t, ndim=1] events,
                      np.ndarray[DTYPEi_t, ndim=1] states
                     ):
    labelled_times,count = distribute_times_per_event_state(
        n_event_types,
        n_states,
        times,
        events,
        states)
    cdef np.ndarray[DTYPEf_t, ndim=3] impact_decay_ratios = imp_coef/(dec_coef - 1.0)
    state_traj_times,state_traj_states = produce_state_trajectory(states, times)
    cdef np.ndarray[DTYPEf_t, ndim=1] st_evol_times = state_traj_times
    cdef np.ndarray[DTYPEi_t, ndim=1] st_evol_states = state_traj_states
    residuals = []
    cdef Py_ssize_t e,x
    for e in range(n_event_types):
        for x in range(n_states):
            residuals.append(compute_total_residual_ex(e, x, n_event_types, n_states,
                                                       base_rate[e],
                                                       dec_coef[:,:,e],
                                                       impact_decay_ratios[:,:,e],
                                                       labelled_times,
                                                       count,
                                                       trans_prob[:,e,x],
                                                       st_evol_times,
                                                       st_evol_states)
                            )
    return residuals

"IMPACT PROFILE A' LA BACRY-MUZY" 

def compute_kernel_of_bm_profile_intensity(double t, int num_states,
                                           double [:,:] imp_coef,
                                           double [:,:] dec_coef,
                                           double [:,:,:] labelled_times,
                                           long [:,:] count):

    cdef np.ndarray[DTYPEf_t, ndim=1] S = np.zeros(num_states,dtype=DTYPEf)
    cdef double [:] S_memview = S
    cdef np.ndarray[DTYPEf_t, ndim=1] alpha = np.array(imp_coef[0,:],copy=True)
    cdef int x1,i
    cdef double eval_time, power_comp
    with nogil:
        for x1 in range(num_states):
            for i in range(count[0,x1]):
                if (labelled_times[0,x1,i]<t):
                    eval_time = t-labelled_times[0,x1,i]+1.0
                    power_comp = pow(eval_time,-dec_coef[0,x1])
                    S_memview[x1] += power_comp
    return np.dot(alpha,S)

def compute_intensity_of_bm_profile(
                                 DTYPEf_t t,
                                 int n_event_types,
                                 int n_states,   
                                 list deflationary_states,
                                 list inflationary_states,
                                 np.ndarray[DTYPEf_t, ndim=1] times,
                                 np.ndarray[DTYPEi_t, ndim=1] states,
                                 np.ndarray[DTYPEf_t, ndim=1] base_rates,
                                 np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
                                 np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
                                 np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
                                 np.ndarray[DTYPEf_t, ndim=3] labelled_times,
                                 np.ndarray[DTYPEi_t, ndim=2] count,
                                 int is_liquidator_active = 1  ):
    """
    It is assumed that index 0 refers to liquidator's interventions
    """
    cdef DTYPEf_t lambda_tilda_0_defl = 0.0
    cdef DTYPEi_t current_state=find_current_state(t,times,states)
    if is_liquidator_active:
        for x in deflationary_states:
            lambda_tilda_0_defl+=transition_probabilities[current_state,0,x]
        lambda_tilda_0_defl*=compute_intensity_partial(
            t, n_event_types, n_states, 
            base_rates[0], impact_coefficients[:,:,0], decay_coefficients[:,:,0], labelled_times,count)
    cdef np.ndarray[DTYPEf_t, ndim=1] time_kernel = np.zeros(n_event_types,dtype=DTYPEf)
    "Notice that the first entry time_kernel[0] will remain null; this is because the index e=0 corresponds to the liquidator"
    cdef np.ndarray[DTYPEf_t, ndim=1] state_transition = np.zeros(n_event_types,dtype=DTYPEf)
    for e in range(1,n_event_types):
        for x in deflationary_states:
            state_transition[e]+=transition_probabilities[current_state,e,x]
        for x in inflationary_states:
            state_transition[e]-=transition_probabilities[current_state,e,x]
        time_kernel[e] = compute_kernel_of_bm_profile_intensity(
            t,n_states,
            impact_coefficients[:,:,e],
            decay_coefficients[:,:,e],
            labelled_times,count)
    return lambda_tilda_0_defl+np.dot(time_kernel,state_transition)
        
def compute_history_of_bm_profile_intensity(
        int n_event_types,
        int n_states,
        list deflationary_states,
        list inflationary_states,         
        DTYPEf_t liquid_start_time,
        DTYPEf_t liquid_termination_time,
        np.ndarray[DTYPEf_t, ndim=1] times,
        np.ndarray[DTYPEi_t, ndim=1] events,
        np.ndarray[DTYPEi_t, ndim=1] states,
        np.ndarray[DTYPEf_t, ndim=1] base_rates,
        np.ndarray[DTYPEf_t, ndim=3] imp_coef,
        np.ndarray[DTYPEf_t, ndim=3] dec_coef,
        np.ndarray[DTYPEf_t, ndim=3] trans_prob,
        int density_of_eval_points=1000,
       ):
    cdef DTYPEf_t time_liquidation_starts = liquid_start_time
    cdef DTYPEf_t time_liquidation_ends = liquid_termination_time
    cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times =np.zeros((n_event_types,n_states,len(times)),dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] count=np.zeros((n_event_types,n_states),dtype=DTYPEi)
    labelled_times,count=distribute_times_per_event_state(
        n_event_types,
        n_states,
        times,
        events,
        states,
        len_labelled_times = len(times)
    )
    cdef np.ndarray[DTYPEf_t, ndim=1] tt = np.sort(
        np.concatenate([times,
                        np.linspace(times[0],times[len(times)-1],num=density_of_eval_points,dtype=DTYPEf)],
                       axis=0),
        axis=0)
    idx=np.concatenate([[1],(np.diff(tt)>1.0e-8)]).astype(np.bool)
    cdef np.ndarray[DTYPEf_t, ndim=2] eval_points =np.array(tt[idx].reshape(-1,1),dtype=DTYPEf,copy=True)
    cdef int idx_liquidation_starts = bisect.bisect_left(np.squeeze(eval_points),time_liquidation_starts)
    cdef int idx_liquidation_ends = min(len(eval_points)-1,
                                        bisect.bisect(np.squeeze(eval_points),time_liquidation_ends))
    cdef np.ndarray[DTYPEf_t, ndim=2] history =\
    np.zeros((len(eval_points),1),dtype=DTYPEf)
    args_0=[n_event_types,n_states,
          deflationary_states,inflationary_states,
          times,states,
          base_rates,imp_coef,dec_coef,trans_prob,
          labelled_times,count,0]
    args_1=[n_event_types,n_states,
          deflationary_states,inflationary_states,
          times,states,
          base_rates,imp_coef,dec_coef,trans_prob,
          labelled_times,count,1]
#     print('computation.compute_history_of_bm_profile_intensity: initialisation completed')
    if idx_liquidation_starts>0:
        history[:idx_liquidation_starts,:]=np.apply_along_axis(
            compute_intensity_of_bm_profile,1,
            eval_points[:idx_liquidation_starts],
            *args_0).reshape(-1,1)
#         print('computation.compute_history_of_bm_profile_intensity: before liquidation completed')
    if idx_liquidation_starts<idx_liquidation_ends:
        history[idx_liquidation_starts:idx_liquidation_ends,:]=np.apply_along_axis(
            compute_intensity_of_bm_profile,1,
            eval_points[idx_liquidation_starts:idx_liquidation_ends,:],
            *args_1).reshape(-1,1)
#         print('computation.compute_history_of_bm_profile_intensity: during liquidation completed')
    if idx_liquidation_ends<len(eval_points)-1:
        history[idx_liquidation_ends:,:]=np.apply_along_axis(
            compute_intensity_of_bm_profile,1,
            eval_points[idx_liquidation_ends:,:],
            *args_0).reshape(-1,1)
#         print('computation.compute_history_of_bm_profile_intensity: after liquidation completed')   
    cdef np.ndarray[DTYPEf_t, ndim=2] result =\
    np.concatenate([eval_points,history],axis=1)
    return result

def compute_bm_impact_profile(np.ndarray[DTYPEf_t, ndim=2] history_of_intensity):
    cdef np.ndarray[DTYPEf_t, ndim=2] eval_points = np.array(history_of_intensity[1:,0].reshape(-1,1),
                                                             dtype=DTYPEf, copy=True)
    cdef int final_left_point_idx=len(history_of_intensity)-1
    cdef np.ndarray[DTYPEf_t, ndim=1] fun = 0.5*(history_of_intensity[1:,1]
                                                 +history_of_intensity[:final_left_point_idx,1])
    cdef np.ndarray[DTYPEf_t, ndim=2] integral =\
    np.array(
        np.cumsum(
            fun*np.diff(history_of_intensity[:,0])
        )
    ).reshape(-1,1)
    cdef np.ndarray[DTYPEf_t, ndim=2] result =\
    np.concatenate(
        [eval_points,
         integral
        ],
        axis=1)
    return result
    
def assess_symmetry(int n_states, 
    np.ndarray[DTYPEf_t, ndim=1] base_rates,
    np.ndarray[DTYPEf_t, ndim=3] imp_dec_ratio,
    np.ndarray[DTYPEf_t, ndim=3] trans_prob,
    list deflationary_states,
    list inflationary_states,
   ):
    cdef int n_event_types = len(base_rates)
    cdef np.ndarray[DTYPEf_t, ndim=1] rho = base_rates + np.sum(imp_dec_ratio.reshape(-1,n_event_types),axis=0)
    cdef np.ndarray[DTYPEf_t, ndim=1] eta = np.zeros(n_event_types,dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] deflationary_pressure = np.zeros(n_states,dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] inflationary_pressure = np.zeros(n_states,dtype=DTYPEf)
    cdef int y,x
    for y in range(n_states):
        for x in inflationary_states:
            eta=copy.copy(trans_prob[y,:,x])
            inflationary_pressure[y] += np.dot(eta,rho)
        for x in deflationary_states:
            eta=copy.copy(trans_prob[y,:,x])
            deflationary_pressure[y] += np.dot(eta,rho)    
    cdef DTYPEf_t asymmetry = np.linalg.norm(deflationary_pressure - inflationary_pressure)   
    return inflationary_pressure, deflationary_pressure, asymmetry

def produce_phi_for_symmetry(int n_states, 
    np.ndarray[DTYPEf_t, ndim=1] base_rates,
    np.ndarray[DTYPEf_t, ndim=3] imp_dec_ratio,
    np.ndarray[DTYPEf_t, ndim=3] trans_prob,
    list deflationary_states,
    list inflationary_states,
   ):
    cdef int n_event_types = len(base_rates)
    cdef int y=0,e=0,x=0
    cdef np.ndarray[DTYPEf_t, ndim=1] rho = base_rates + np.sum(imp_dec_ratio.reshape(-1,n_event_types),axis=0)
    inflationary_pressure, deflationary_pressure, asymmetry =\
    assess_symmetry(
        n_states, base_rates, imp_dec_ratio,trans_prob,
        deflationary_states, inflationary_states
    )
    cdef np.ndarray[DTYPEf_t, ndim=1] iota = inflationary_pressure
    cdef np.ndarray[DTYPEf_t, ndim=1] delta = deflationary_pressure
    cdef np.ndarray[DTYPEf_t, ndim=1] iota_delta = (iota+delta)/2
    cdef list inf_def_states = list(set(inflationary_states) | set(deflationary_states))
    cdef np.ndarray[DTYPEf_t, ndim=2] U = np.zeros((n_states,n_states),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] P = np.zeros((n_states,n_event_types,2),dtype=DTYPEf)
    for y in range(n_states):
        for x in range(n_states):
            U[y,x]=np.dot(trans_prob[y,:,x],rho)
            if x in inf_def_states:
                P[y,:,1]+=trans_prob[y,:,x]
            else:
                P[y,:,0]+=trans_prob[y,:,x]
    cdef np.ndarray[DTYPEf_t, ndim=3] u = np.array(trans_prob,dtype=DTYPEf,copy=True)
    for y in range(n_states):
        for x in range(n_states):
            u[y,:,x] /= U[y,x]
    cdef np.ndarray[DTYPEf_t, ndim=3] q = np.zeros_like(trans_prob,dtype=DTYPEf)
    for y in range(n_states):
        for x in range(n_states):
            if x in inflationary_states:
                q[y,:,x] = iota_delta[y]*u[y,:,x]/(len(inflationary_states))
            elif x in deflationary_states:
                q[y,:,x] = iota_delta[y]*u[y,:,x]/(len(deflationary_states))
            else:
                q[y,:,x] = trans_prob[y,:,x]/P[y,:,0]
    cdef np.ndarray[DTYPEf_t, ndim=2] Q = np.zeros((n_states,n_event_types), dtype=DTYPEf)
    for y in range(n_states):
        for x in inf_def_states:
            Q[y,:]+= q[y,:,x]
    cdef np.ndarray[DTYPEf_t, ndim=1] Q_bar = np.sum(Q,axis=1)
    cdef np.ndarray[DTYPEf_t, ndim=3] new_phi = np.zeros_like(trans_prob,dtype=DTYPEf)
    for y in range(n_states):
        for x in range(n_states):
            if x in inf_def_states:
                new_phi[y,:,x] = q[y,:,x]/Q_bar[y]
            else:
                new_phi[y,:,x] = (1-Q[y,:]/Q_bar[y])*q[y,:,x]
            
    return new_phi                
    
"MISCELLANEOUS TOOLS"    

def find_current_state(DTYPEf_t t, 
                       np.ndarray[DTYPEf_t, ndim=1] times,
                       np.ndarray[DTYPEi_t, ndim=1] states):
    cdef int idx=bisect.bisect_right(times,t)
    idx=min(len(states)-1,idx)
    cdef DTYPEi_t current_state=copy.copy(states[idx])
    return current_state

def random_choice(np.ndarray[DTYPEf_t, ndim=1] weights):
    cdef DTYPEf_t total, cumulative_sum, random_uniform
    cdef int result, dim, n, done
    dim = weights.shape[0]
    total = 0.0
    for n in range(dim):
        total += weights[n]
    random_uniform =  np.random.uniform(0, total)
    result = 0
    done = 0
    cumulative_sum = weights[result]
    if random_uniform <= cumulative_sum:
        done = 1
    while done == 0:
        result += 1
        cumulative_sum += weights[result]
        if random_uniform <= cumulative_sum:
            done = 1
    return result

def compute_probability_of_volimb_constraint(
    int upto_level,
    int idx_bid_level_1,
    int idx_ask_level_1,
    double [:] constraint,
    double [:] dir_param,
    int N_samples = 9999
):
    """
    The volumes are assumed to be reported as in LOBSTER, with the alternation [ask_level_1, bid_level_1, ask_level_2, bid_level_2 ...]    
    """
    cdef int uplim = 1+2*upto_level#This defines the index of the flattened array of volumes to be used in the computation of volume imbalances.  
    cdef np.ndarray[DTYPEf_t, ndim=2] samples = scipy_dirichlet.rvs(dir_param, size=N_samples)
    cdef np.ndarray[DTYPEf_t, ndim=1] vol_imb =\
    (np.sum(samples[:,idx_bid_level_1:uplim:2],axis=1) -
     np.sum(samples[:,idx_ask_level_1:uplim:2],axis=1))
    idx = np.logical_and(vol_imb>= constraint[0], vol_imb<constraint[1])
    return np.sum(idx)/N_samples

def produce_probabilities_of_volimb_constraints(
    int upto_level,
    int n_states,
    int num_of_st2,
    np.ndarray[DTYPEf_t, ndim=1] volimb_limits,
    np.ndarray[DTYPEf_t, ndim=2] dir_param,
    int N_samples = 9999
):
    """
    It is assumed that dir_param.shape[0] = n_states, and dir_param.shape[1] = 2*self.n_levels. 
    The volumes are reported as in LOBSTER, with the alternation [ask_level_1, bid_level_1, ask_level_2, bid_level_2 ...]    
    """
    cdef int idx_bid_level_1 = 1
    cdef int idx_ask_level_1 = 0
    cdef int x = 0, st2 = 0
    cdef np.ndarray[DTYPEf_t, ndim=1] result = np.zeros(n_states, dtype=DTYPEf)
    for x in range(n_states):
        st2 = x%num_of_st2
        result[x] = compute_probability_of_volimb_constraint(
            upto_level, idx_bid_level_1, idx_ask_level_1,
            volimb_limits[st2:st2+2], dir_param[x,:], N_samples
        )
    return result

def compute_dirichlet_mass(np.ndarray[DTYPEf_t, ndim=1] dir_param):
    cdef DTYPEf_t mass = 1.0
    cdef int k = 0
    for k in range(len(dir_param)):
        mass*= scipy_gamma_fun(dir_param[k])
    mass/=scipy_gamma_fun(np.sum(dir_param))
    return mass
    
def produce_dirichlet_masses(np.ndarray[DTYPEf_t, ndim=2] dir_param):
    cdef int N=dir_param.shape[0]
    cdef int n=0
    cdef np.ndarray[DTYPEf_t, ndim=1] masses = np.ones(N,dtype=DTYPEf)
    for n in range(N):
        masses[n]=compute_dirichlet_mass(dir_param[n,:])
    return masses
    
def compute_maximum_unnormalised_pseudo_dirichlet_density(np.ndarray[DTYPEf_t, ndim=1] exponents, verbose=False):
    """
    It is assumed that all the :math: `N+1` entries :math: `e_i` of the numpy array `exponents` are strictly positive. If not the default value 1.0 is returned. This is aligned with the usage of the array RejectionSampling.bound in the class RejectionSampling.
    The objective function is 
    :math: `f(x) = \prod_{i...N} x_{i}^{e_i} (1-\sum_{i...N} x_i)^{e_{N+1}}`
    """
    if np.any(exponents <=0):
        if verbose:
            print('compute_maximum_unnormalised_pseudo_dirichlet_density:')
            print('  WARNING: exponents <=0 in at least one entry')
            print('  exponents = {}'.format(exponents))
            print('  I am returning the default value 1.0')
        return 1.0
    cdef int N = len(exponents)-1
    idx_diag = np.diag_indices(N)
    cdef np.ndarray[DTYPEf_t, ndim=2] A = np.ones((N,N),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim = 1] diag_entries = (exponents[:N] + exponents[N])/exponents[:N]
    A[idx_diag] = diag_entries
    cdef np.ndarray[DTYPEf_t, ndim=1] b = np.ones(N,dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] maximiser = np.zeros(N+1,dtype=DTYPEf)
    maximiser[:N] = np.linalg.solve(A,b)
    maximiser[N]= 1.0 - np.sum(maximiser[:N])
    cdef DTYPEf_t maximum = np.prod(np.power(maximiser,exponents),dtype=DTYPEf)
    return maximum

def compute_optimal_proposal_param( gamma,
        DTYPEf_t l=-0.2, DTYPEf_t u=0.2, DTYPEf_t c=1.0,
        int uplim=5, int maxiter=1000):
    def objfun(rho):
        return compute_maximum_unnormalised_pseudo_dirichlet_density(gamma-rho)\
        +c*expectation_constr(rho)
    def expectation_constr(rho):
        cdef DTYPEf_t v=np.sum(rho[1:uplim:2]-rho[0:uplim:2])/np.sum(rho)
        return (max(0.0,v-u)+max(0.0,l-v))**2
    bounds=[(1.0e-8,gamma[k]) for k in range(len(gamma))]
    res=scipy_minimize(
        objfun,0.95*gamma,
        method='TNC',jac=False,
        bounds=bounds,options={'maxiter': maxiter})
    cdef np.ndarray[DTYPEf_t, ndim=1] rho = np.array(res['x'],dtype=DTYPEf)
    return rho
def compute_acceptance_prob(long state, np.ndarray[DTYPEf_t, ndim=2] proposal_dir_param,
    np.ndarray[DTYPEf_t, ndim=2]  difference_of_dir_params,
    np.ndarray[DTYPEf_t, ndim=1] inverse_bound,
    long [:] is_target_equal_to_proposal,
    int num_of_st2, double [:] volimb_limits, int upto_lim = 5, long num_samples = 999999
):
#    Notice that upto_lim is the index of the flattened array containing alternating ask/bid-volumes.     This means upto_lim=1+2*volume_imbalance_upto_level, where "volume_imbalance_upto_level" is the leve    l
    cdef int st_2 = state%num_of_st2
    cdef DTYPEf_t lower_bound = volimb_limits[st_2]
    cdef DTYPEf_t upper_bound = volimb_limits[1+st_2]
    cdef double [:] gamma_tilde = proposal_dir_param[state,:]
    cdef np.ndarray[DTYPEf_t, ndim=2] delta_gamma = np.zeros((1,len(gamma_tilde)), dtype=DTYPEf)
    delta_gamma[0,:]=difference_of_dir_params[state,:]
    cdef DTYPEf_t K = inverse_bound[state]
    cdef np.ndarray[DTYPEf_t, ndim=1] sample = np.zeros((num_samples,len(gamma_tilde)), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1]  vol_imb = np.zeros((num_samples,), dtype=DTYPEf)
    cdef DTYPEf_t u = 0.0, prob=0.0
    cdef int count=0, n=0
    if is_target_equal_to_proposal[state]:
        sample = scipy_dirichlet.rvs(gamma_tilde,size=num_samples)
        vol_imb = np.sum(sample[:,1:upto_lim:2] - sample[:,0:upto_lim:2], axis=1)
        idx=np.logical_and(lower_bound <= vol_imb,vol_imb<=upper_bound)
        prob=float(np.sum(idx))/num_samples
    else:
        sample = scipy_dirichlet.rvs(gamma_tilde, size=num_samples)
        vol_imb = np.sum(sample[:,1:upto_lim:2] - sample[:,0:upto_lim:2], axis=1)
        idx=np.logical_and(lower_bound <= vol_imb,vol_imb<=upper_bound)
        val = K*np.prod(np.power(sample[idx,:],delta_gamma), axis=1)
        assert len(val)==np.sum(idx)
        for n in range(len(val)):
            u = rand()/fRANDMAX#np.random.uniform(low=0.0, high=1.0)
            if u < val[n]:
                count+=1
        prob=float(count)/num_samples
    return prob

    
def parameters_to_array(np.ndarray[DTYPEf_t,ndim=1]  base_rate,
                        np.ndarray[DTYPEf_t,ndim=3] imp_coef,
                        np.ndarray[DTYPEf_t,ndim=3] dec_coef):
    return np.concatenate([base_rate,imp_coef.flatten(),dec_coef.flatten()])

def array_to_parameters(int num_event_types, int num_states, np.ndarray[DTYPEf_t, ndim=1] arr):
    cdef int break_pnt_1 = num_event_types
    cdef int break_pnt_2 = break_pnt_1+ num_event_types*num_states*num_event_types
    cdef np.ndarray[DTYPEf_t, ndim=1] base_rates = arr[0:break_pnt_1]
    cdef np.ndarray[DTYPEf_t, ndim=3] imp_coef =\
    arr[break_pnt_1:break_pnt_2].reshape(num_event_types, num_states, num_event_types)
    cdef np.ndarray[DTYPEf_t, ndim=3] dec_coef =\
    arr[break_pnt_2:len(arr)].reshape(num_event_types, num_states, num_event_types)
    return base_rates, imp_coef, dec_coef

def parameters_to_array_partial(DTYPEf_t base_rate,
                                np.ndarray[DTYPEf_t,ndim=2] imp_coef,
                                np.ndarray[DTYPEf_t,ndim=2] dec_coef):
    return np.concatenate([np.atleast_1d(base_rate),imp_coef.flatten(),dec_coef.flatten()])

def array_to_parameters_partial(int num_event_types, int num_states, np.ndarray[DTYPEf_t,ndim=1] array):
    cdef int break_point_1=1
    cdef int break_point_2=break_point_1+num_event_types*num_states
    cdef DTYPEf_t base_rate = float(array[0])
    cdef np.ndarray[DTYPEf_t, ndim=2] imp_coef = array[break_point_1:break_point_2].reshape(
        num_event_types,num_states)
    cdef np.ndarray[DTYPEf_t, ndim=2] dec_coef = array[break_point_2:len(array)].reshape(
        num_event_types,num_states)
    return base_rate, imp_coef, dec_coef

class Partition:
    def __init__(self, int num_pnts = 100, two_scales=False, DTYPEf_t t_max=1.0, DTYPEf_t t_min=1.0e-04, DTYPEf_t tol=1.0e-15):
        self.two_scales=two_scales
        self.t_min=t_min
        self.t_max=t_max
        self.num_pnts=2*num_pnts # If two_scales==True then this is meant to accommodate two scale: linear up to t_min and logarithmic from t_min to t_max 
        self.tol=tol
        self.create_grid()
    def create_grid(self):
        cdef int Q_half = self.num_pnts//2
        cdef np.ndarray[DTYPEf_t, ndim=1] partition = np.zeros(1+2*Q_half,dtype=DTYPEf)
        if self.two_scales:
            partition[:Q_half] = np.linspace(self.tol,self.t_min, num=Q_half, endpoint=False)
            partition[Q_half:] = np.exp(np.linspace(log(self.t_min), log(self.t_max), num=Q_half+1))
        else:
            partition=np.linspace(self.tol,self.t_max, num=1+2*Q_half)
        self.partition=partition
        
def merge_sorted_arrays(np.ndarray[DTYPEf_t, ndim=1] a,np.ndarray[DTYPEf_t, ndim=1] b):
    cdef int m=len(a), n=len(b)
    # Get searchsorted indices
    cdef np.ndarray[DTYPEi_t, ndim=1] idx = np.searchsorted(a,b)

    # Offset each searchsorted indices with ranged array to get new positions
    # of b in output array
    cdef np.ndarray[DTYPEi_t, ndim=1] b_pos = np.arange(n) + idx

    mask = np.ones(m+n,dtype=bool)
    mask[b_pos] = False
    cdef np.ndarray[DTYPEf_t, ndim=1] result = np.empty(m+n,dtype=DTYPEf)
    result[b_pos] = b
    result[mask] = a
    return result, b_pos        

def select_interval(x, DTYPEf_t t0, DTYPEf_t t1):
    cdef int ndims=x.ndim
    if ndims==1:
        idx0=bisect.bisect_left(x,t0)
        idx1=bisect.bisect_right(x,t1)
        return np.array(x[idx0:idx1],copy=True)
    elif ndims==2:
        idx0=bisect.bisect_left(x[:,0],t0)
        idx1=bisect.bisect_right(x[:,0],t1)
        return np.array(x[idx0:idx1,:],copy=True)
    else:
        print("Error: x.ndim expected to be either 1 or 2, but x.ndim={}".format(ndims))
        raise ValueError("Incorrect shape")
    return 1

