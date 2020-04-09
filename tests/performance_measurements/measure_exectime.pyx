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
import sys
sys.path.append(path_lobster_pyscripts+'/')
sys.path.append(path_sdhawkes+'/modelling/')
sys.path.append(path_sdhawkes+'/resources/')

import numpy as np
cimport numpy as np
import pandas as pd
import pickle
import datetime
import time
import datetime
import timeit

import model as sd_hawkes_model
import lob_model
import computation

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t


class PerformanceMeasure:
    def __init__(self, model, type_of_input='simulated'):
        print("PerformanceMeasure is being initialised on model.name_of_model={}".format(model.name_of_model))
        self.model = model
        self.type_of_input = type_of_input
        cdef int d_E = model.number_of_event_types
        cdef int d_S = model.number_of_states
        self.num_event_types = d_E
        self.num_states = d_S
        if type_of_input == 'simulated':
            self.times=model.simulated_times
            self.events=model.simulated_events
            self.states=model.simulated_states
        elif type_of_input =='empirical':
            self.times=model.data.observed_times
            self.events=model.data.observed_events
            self.states=model.data.observed_states
        else:
            print("type_of_input={}".format(type_of_input))
            raise ValueError("type of input not recognised. It must be either 'simulated' or 'empirical'")
        cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times = np.zeros((d_E,d_S,len(self.times)), dtype=DTYPEf)
        cdef np.ndarray[DTYPEi_t, ndim=2] count = np.zeros((d_E,d_S), dtype=DTYPEi)
        labelled_times, count = computation.distribute_times_per_event_state(
            d_E,d_S, self.times, self.events, self.states)
        self.labelled_times = labelled_times
        self.count = count
    def target_intensities(self, use_fast = False, print_res = False):
        cdef DTYPEf_t t = np.amax(self.labelled_times)
        cdef np.ndarray[DTYPEf_t, ndim=1] base_rates = self.model.base_rates
        cdef np.ndarray[DTYPEf_t, ndim=3] imp_coef = self.model.impact_coefficients
        cdef np.ndarray[DTYPEf_t, ndim=2] reshaped_imp_coef = imp_coef.reshape((-1,self.num_event_types))
        cdef np.ndarray[DTYPEf_t, ndim=3] dec_coef = self.model.decay_coefficients
        if use_fast:
            intensities =  computation.fast_compute_intensities_given_lt(
                t, self.labelled_times, self.count,
                base_rates, imp_coef, dec_coef, reshaped_imp_coef,           
                self.num_event_types, self.num_states)
        else:
            intensities =  computation.compute_intensities_given_lt(
                t, self.labelled_times, self.count,
                base_rates, imp_coef, dec_coef, reshaped_imp_coef,           
                self.num_event_types, self.num_states)
        if print_res:
            print("intensities:{}".format(intensities))        
    def target_loglikelihood(self, int event_type=0, use_prange=False, print_res = False):
        cdef int e=event_type
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        idx_e = (self.events==e)
        cdef int len_labelled_times = self.labelled_times.shape[2]
        cdef int num_arrival_times = np.sum(idx_e)
        cdef np.ndarray[DTYPEf_t, ndim=1] arrival_times = np.array(self.times[idx_e],copy=True)
        cdef DTYPEf_t time_start = arrival_times[0]
        cdef DTYPEf_t time_end = arrival_times[num_arrival_times-1] 
        cdef DTYPEf_t base_rate = self.model.base_rates[e]
        cdef np.ndarray[DTYPEf_t, ndim=2] imp_coef = self.model.impact_coefficients[:,:,e]
        cdef np.ndarray[DTYPEf_t, ndim=2] dec_coef = self.model.decay_coefficients[:,:,e]
        cdef np.ndarray[DTYPEf_t, ndim=2] ratio = imp_coef / (dec_coef - 1.0)
        cdef np.ndarray[DTYPEf_t, ndim=1] intensity = base_rate*np.ones(num_arrival_times, dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] intensity_inverse = base_rate*np.ones(num_arrival_times, dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=3] ESSE = np.zeros((d_E,d_S,num_arrival_times), dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=3] ESSE_one = np.zeros((d_E,d_S,num_arrival_times), dtype=DTYPEf)
        f, grad =  computation.compute_event_loglikelihood_partial_and_gradient_partial(
            e, d_E, d_S, base_rate, imp_coef, dec_coef, ratio,
            self.labelled_times, self.count, arrival_times, 
            num_arrival_times,len_labelled_times, time_start, time_end, use_prange)
        if print_res:
            print("f={}".format(f))
            print("grad_f={}".format(grad))
        
        
    