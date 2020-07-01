#cython: boundscheck=False, wraparound=False, nonecheck=False

import os
cdef str path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<6):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    raise ValueError("path_pipest not found. Instead: {}".format(path_pipest))
cdef str path_models=path_pipest+'/models'    
cdef str path_sdhawkes=path_pipest+'/sdhawkes_powerlaw'
cdef str path_lobster=path_pipest+'/lobster_for_sdhawkes'
cdef str path_lobster_pyscripts=path_lobster+'/py_scripts'
import sys
sys.path.append(path_sdhawkes+'/')
sys.path.append(path_sdhawkes+'/resources/')
sys.path.append(path_sdhawkes+'/modelling/')
sys.path.append(path_lobster_pyscripts+'/')

import pandas as pd
import numpy as np
cimport numpy as np
from scipy import linalg

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t
DTYPEfd = np.longdouble
DTYPEil = np.int64
ctypedef np.longdouble_t DTYPEfd_t
ctypedef np.int64_t DTYPEil_t

import mle_estimation as mle_estim
import nonparam_estimation as nonparam_estim
import goodness_of_fit
import computation
import simulation
import dirichlet
import lob_model

class UncertQuant:
    def __init__(self,
                 int number_of_event_types, int number_of_states,
                 int n_levels, state_enc, volume_enc,
                 np.ndarray[DTYPEf_t, ndim=1] br,
                 np.ndarray[DTYPEf_t, ndim=3] imp_coef,
                 np.ndarray[DTYPEf_t, ndim=3] dec_coef,
                 np.ndarray[DTYPEf_t, ndim=3] trans_prob,
                 np.ndarray[DTYPEf_t, ndim=2] dirichlet_param,
                 copy=True
                ):
        self.number_of_event_types = number_of_event_types
        self.number_of_states = number_of_states
        self.n_levels = n_levels
        self.state_enc = state_enc
        self.volume_enc = volume_enc
        self.transition_probabilities = np.array(trans_prob, copy=copy)
        self.base_rates =  np.array(br, copy=copy)
        self.impact_coefficients =  np.array(imp_coef, copy=copy)
        self.decay_coefficients =  np.array(dec_coef, copy=copy)
        impact_decay_ratios = imp_coef/(dec_coef-1.0)
        self.impact_decay_ratios = impact_decay_ratios
        self.dirichlet_param=dirichlet_param
    def simulate(self, DTYPEf_t time_start, DTYPEf_t time_end, 
                 initial_condition_times=[], initial_condition_events=[],
                 initial_condition_states=[], initial_condition_volumes=[], 
                 int max_number_of_events=10**5,add_initial_cond=False,
                ):
         # Convert the initial condition to np.arrays if required
        if type(initial_condition_times)!=np.ndarray:
            initial_condition_times = np.asarray(initial_condition_times, dtype=np.float)
        cdef int num_initial_events=initial_condition_times.shape[0]
        if type(initial_condition_events)!=np.ndarray:
            initial_condition_events = np.asarray(initial_condition_events, dtype=np.int)
            initial_condition_events= initial_condition_events[:num_initial_events]
        if type(initial_condition_states)!=np.ndarray:
            initial_condition_states = np.asarray(initial_condition_states, dtype=np.int)
            initial_condition_states=initial_condition_states[:num_initial_events]
        if type(initial_condition_volumes)!=np.ndarray:
            initial_condition_volumes = np.atleast_2d(
                np.asarray(
                    initial_condition_volumes, dtype=np.float
                )
            ).reshape(-1,2*self.n_levels)    
            initial_condition_volumes = initial_condition_volumes[:num_initial_events,:]
        if num_initial_events<1:
            initial_condition_times=np.insert(initial_condition_times,0,0.0,axis=0)
            initial_condition_events=np.insert(
                initial_condition_events,
                0,
                np.random.randint(low=0,high=self.number_of_event_types),
                axis=0)
            initial_condition_states=np.insert(
                initial_condition_states,
                0,
                np.random.randint(low=0,high=self.number_of_states),
                axis=0)
            initial_condition_volumes=np.insert(
                initial_condition_volumes,
                0,
                np.random.dirichlet(np.ones(2*self.n_levels)),
                axis=0)            
        times, events, states, volumes =  simulation.launch_simulation(
                                              self.number_of_event_types,
                                              self.number_of_states,
                                              self.n_levels,
                                              self.base_rates,
                                              self.impact_coefficients,
                                              self.decay_coefficients,
                                              self.transition_probabilities,
                                              self.dirichlet_param,
                                              initial_condition_times,
                                              initial_condition_events,
                                              initial_condition_states,
                                              initial_condition_volumes,
                                              self.volume_enc.rejection_sampling,                
                                              time_start,
                                              time_end,
                                              max_number_of_events,
                                              add_initial_cond,
                                              num_preconditions = 1,         
                                              largest_negative_time = -100.0,
                                              initialise_intensity_on_history = 1, 
                                              report_full_volumes=False)
        lt,count=computation.distribute_times_per_event_state(
                self.number_of_event_types,
                self.number_of_states,
                times, events,states)
        self.labelled_times = lt
        self.count = count
        self.simulated_times = times
        self.simulated_events = events
        self.simulated_states = states
    def create_goodness_of_fit(self,parallel=True):
        cdef str type_of_input = 'simulated'
        self.goodness_of_fit=goodness_of_fit.good_fit(
            self.number_of_event_types,self.number_of_states,
            self.base_rates,self.impact_coefficients,self.decay_coefficients,
            self.transition_probabilities,
            self.simulated_times, self.simulated_events, self.simulated_states,
            type_of_input=type_of_input, parallel = parallel
        )
    def create_nonparam_estim(self, 
                              int num_quadpnts = 100, DTYPEf_t quad_tmax=1.0, DTYPEf_t quad_tmin=1.0e-04,
                              int num_gridpnts = 100, DTYPEf_t grid_tmax=1.0, DTYPEf_t grid_tmin=1.0e-04,
                              DTYPEf_t tol=1.0e-15, two_scales=False
                             ):
        times=self.simulated_times
        events=self.simulated_events
        states=self.simulated_states
        self.nonparam_estim=nonparam_estim.EstimProcedure(
            self.number_of_event_types,self.number_of_states,
            times,events,states,
            'simulated',
            num_quadpnts, quad_tmax, quad_tmin,
            num_gridpnts, grid_tmax, grid_tmin,
            tol, two_scales
        )
    def create_mle_estim(self, store_dirichlet_param=False):
        times=self.simulated_times
        events=self.simulated_events
        states=self.simulated_states
        volumes=None
        self.mle_estim=mle_estim.EstimProcedure(
            self.number_of_event_types, self.number_of_states,
            times, events, states,
            volumes = volumes,
            n_levels = self.n_levels, 
            type_of_input = 'simulated',
            store_trans_prob = True,    
            store_dirichlet_param = store_dirichlet_param
        )
    def calibrate_on_simulated_data(self,
                                    str type_of_preestim = 'ordinary_hawkes',
                                    DTYPEf_t max_imp_coef = 100.0,
                                    DTYPEf_t learning_rate = 0.0001,
                                    int maxiter = 50,
                                    int num_of_random_guesses=0,
                                    parallel=False,
                                    use_prange=True,
                                    int number_of_attempts = 2,
                                    int num_processes = 0,
                                    int batch_size = 5000,
                                    int num_run_per_minibatch = 1,
                                    DTYPEf_t tol = 1.0e-7,
                                   ):
        times=self.simulated_times
        events=self.simulated_events
        states=self.simulated_states
        time_start=float(times[0])
        time_end=float(times[len(times)-1])
        cdef list list_init_guesses = []
        list_init_guesses.append(
            computation.parameters_to_array(self.base_rates,
                                            self.impact_coefficients,
                                            self.decay_coefficients)
        )
        num_of_random_guesses=max(num_of_random_guesses,2)
        preestim_ordinary_hawkes = (type_of_preestim =='ordinary_hawkes')
        if type_of_preestim == 'nonparam':
            list_init_guesses = self.nonparam_estim.produce_list_init_guesses_for_mle_estimation(
                num_additional_random_guesses = max(1,num_of_random_guesses//2),
                max_imp_coef = max_imp_coef)
        self.create_mle_estim()
        self.mle_estim.set_estimation_of_hawkes_param(
            time_start, time_end,
            list_of_init_guesses = list_init_guesses,
            max_imp_coef = max_imp_coef,
            learning_rate = learning_rate,
            maxiter = maxiter,
            number_of_additional_guesses = max(1,num_of_random_guesses//2),
            parallel = parallel,
            use_prange = use_prange,
            number_of_attempts = number_of_attempts,
            num_processes = num_processes,
            batch_size = batch_size,
            num_run_per_minibatch = num_run_per_minibatch,
        )
        self.mle_estim.launch_estimation_of_hawkes_param(partial=False)
        self.mle_estim.store_hawkes_parameters()
        self.mle_estim.create_goodness_of_fit()
    def adjust_baserates(self, 
            np.ndarray[DTYPEf_t, ndim=1] target_avgrates,
            DTYPEf_t adj_coef=1.0e-3, int num_iter=5,
            DTYPEf_t t0=0.0, DTYPEf_t t1=2.0*60*60, int max_number_of_events=10**4):
        cdef int d_E = self.number_of_event_types
        assert len(target_avgrates)==d_E
        print("target_avgrates: {}". format(target_avgrates))
        print("Original base_rates: {}". format(self.base_rates))
        cdef int i=0
        while i<num_iter:
            i+=1
            self.simulate(t0, t1, max_number_of_events=max_number_of_events)
            avgrates=computation.avg_rates(d_E, self.simulated_times, self.simulated_events, partial=False)
            self.base_rates*=np.exp(adj_coef*(target_avgrates-avgrates))
        avgrates=computation.avg_rates(d_E, self.simulated_times, self.simulated_events, partial=False)
        print("Adjusted base_rates: {}".format(self.base_rates))
        print("target_avgrates: {}". format(target_avgrates))
        print("Adjusted average rates: {}".format(avgrates))








