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

import time
from cython.parallel import prange
import multiprocessing as mp
from sklearn.linear_model import Ridge

from scipy import linalg
import numpy as np
cimport numpy as np
import bisect
from libc.math cimport pow
from libc.math cimport exp
from libc.math cimport log

import model
import computation
from computation import Partition
import goodness_of_fit
from mle_estimation import estimate_transition_probabilities
import minimisation_algo as minim_alg
import dirichlet

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t




class Filter:
    def __init__(self, enforce_positivity=False,
                 DTYPEf_t cutoff=100.0, DTYPEf_t scale=1.0, num_addpnts=2000):
        self.set_parameters(
            enforce_positivity=enforce_positivity,
            cutoff=cutoff,
            scale=scale,
            num_addpnts=num_addpnts
        )
    def set_parameters(self,
                       enforce_positivity=False,
                       DTYPEf_t cutoff=100.0, DTYPEf_t scale=1.0, num_addpnts=2000):
        self.enforce_positivity=enforce_positivity
        self.cutoff=cutoff
        self.scale=scale
        self.num_addpnts=num_addpnts  
        
class EstimProcedure:
    def __init__(self, int num_event_types, int num_states,
                 np.ndarray[DTYPEf_t , ndim=1] times, 
                 np.ndarray[DTYPEi_t , ndim=1] events,
                 np.ndarray[DTYPEi_t , ndim=1] states,
                 str type_of_input = 'simulated',
                 int num_quadpnts = 100, DTYPEf_t quad_tmax=1.0, DTYPEf_t quad_tmin=1.0e-04,
                 int num_gridpnts = 100, DTYPEf_t grid_tmax=1.0, DTYPEf_t grid_tmin=1.0e-04,
                 DTYPEf_t tol=1.0e-10, two_scales=False):
        print("nonparam_estimation.EstimProcedure is being initialised")
        self.type_of_input = type_of_input
        self.tol = tol
        if not (len(times)==len(states) & len(times)==len(events)):
            raise ValueError("All shapes must agree, but input was:\n len(times)={} \n len(events)={} \n len(states)={}".format(len(times),len(events),len(states)))
        self.num_event_types = num_event_types
        self.num_states = num_states
        cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times = np.zeros((num_event_types,num_states,len(times)), dtype=DTYPEf)
        cdef np.ndarray[DTYPEi_t, ndim=2] count = np.zeros((num_event_types, num_states), dtype=DTYPEi)
        labelled_times, count = computation.distribute_times_per_event_state(
            num_event_types,num_states,
            times,events, states)
        self.labelled_times=labelled_times
        self.count=count
        self.times=times
        self.time_horizon=times[len(times)-1]
        self.events=events
        self.states=states
        self.quadrature=Partition(num_pnts=num_quadpnts, two_scales=two_scales, t_max=quad_tmax, t_min=quad_tmin)
        self.grid=Partition(num_pnts=num_gridpnts, two_scales=two_scales, t_max=grid_tmax, t_min=grid_tmin)
        self.hawkes_kernel=model.HawkesKernel(num_event_types, num_states, 
                 num_quadpnts, two_scales, quad_tmax, quad_tmin)
        self.store_distribution_of_marks()
        self.store_transition_probabilities()
        self.store_expected_intensities()
        print("EstimProcedure has been successfully initialised") 
    def store_runtime(self,DTYPEf_t run_time):
        self.estimation_runtime=run_time
    def prepare_estimation_of_hawkes_kernel(self, use_filter=False, parallel=True):
        print("I am preparing estimation of hawkes kernel")
        if use_filter:
            self.filter_nonsingular_expected_jumps(
                pre_estimation=True, parallel=parallel)
        else:
            self.store_nonsingular_expected_jumps(parallel=parallel)
        self.store_convolution_kernels(use_filter=use_filter)
        self.store_matrix_A()
        print("Estimation of hawkes kernel is now ready")
    def estimate_hawkes_kernel(self, store_L1_norm=False,
                               use_filter=False, enforce_positive_g_hat=False,
                               DTYPEf_t filter_cutoff=100.0, DTYPEf_t filter_scale=1.0, num_addpnts_filter=2000,
                               parallel=False, parallel_prep=True):
        self.use_filter=use_filter
        if use_filter:
            self.set_filter_param(enforce_positivity=enforce_positive_g_hat,
                           cutoff=filter_cutoff, scale=filter_scale, num_addpnts=num_addpnts_filter)
        self.prepare_estimation_of_hawkes_kernel(
            use_filter=use_filter, parallel=parallel_prep)
        cdef int N = max(1,self.num_event_types)
        cdef DTYPEf_t run_time=-time.time()
        cdef list results = []
        #The pool procedure runs out of memory when executed on my EliteBook with resonably large data. Use the plain serial map instead
        if parallel:
            print("I am performing estimation of hawkes kernel in parallel. num_processes: {}".format(N))
            with mp.Pool(N) as pool:
                solver=pool.map_async(
                   self.solve_linear_system_partial,
                   list(range(self.num_event_types))
                )
                pool.close()
                pool.join()
                results=solver.get()
        else:        
        #Here the serial implementation is favoured because of limited RAM memory
            print("I am performing estimation of hawkes kernel serially")
            results=list(
                map(self.solve_linear_system_partial,
                    list(range(self.num_event_types))
                   )
            )       
        run_time+=time.time()
        self.results_estimation_of_hawkes_kernel=results
        print("Estimation of hawkes kernel terminates. run_time={}".format(run_time))
        cdef int d_E = self.num_event_types
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=4] kappa = np.zeros((d_E, self.num_states, d_E, Q), dtype=DTYPEf)
        cdef int e=0
        for e in range(d_E):
            kappa+=np.array(results[e], copy=True)
        self.hawkes_kernel.store_values_at_quadpnts(kappa)
        if store_L1_norm:
            self.hawkes_kernel.store_L1_norm(from_param=False)
    def fit_powerlaw(self,compute_L1_norm=False,DTYPEf_t ridge_param=1.0, DTYPEf_t tol=1.0e-9):
        fit_powerlaw=FitPowerlaw(self.num_event_types, self.num_states,
              self.quadrature,
              self.hawkes_kernel.values_at_quadpnts,
              ridge_param=ridge_param, tol=tol)
        fit_powerlaw.fit()
        self.fit_powerlaw = fit_powerlaw
        self.hawkes_kernel.store_parameters(self.fit_powerlaw.imp_coef, self.fit_powerlaw.dec_coef)
        if compute_L1_norm:
            #self.hawkes_kernel.compute_values_at_quadpnts_from_parametric_kernel()
            self.hawkes_kernel.store_L1_norm()
    def create_goodness_of_fit(self, str type_of_input='simulated', parallel=True):
        "type_of_input can either be 'simulated' or 'empirical'"
        self.goodness_of_fit=goodness_of_fit.good_fit(
            self.num_event_types,self.num_states,
            self.base_rates,self.hawkes_kernel.alphas,self.hawkes_kernel.betas,self.transition_probabilities,
            self.times,self.events,self.states,type_of_input=type_of_input, parallel=parallel
        )        
    def produce_list_init_guesses_for_mle_estimation(self, int num_additional_random_guesses = 0,
                                                     DTYPEf_t max_imp_coef = 100.0, DTYPEf_t tol=1.0e-07,
                                                    ):
        cdef list list_init_guesses = []
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int break_point_1 = d_E
        cdef int break_point_2 = d_E + d_E*d_S*d_E
        cdef np.ndarray[DTYPEf_t, ndim=1] guess = np.zeros(d_E+2*d_E*d_S*d_E,dtype=DTYPEf)
        guess = computation.parameters_to_array(self.base_rates, self.hawkes_kernel.alphas, self.hawkes_kernel.betas)
        list_init_guesses.append(np.array(guess,copy=True))
        cdef np.ndarray[DTYPEf_t, ndim=2] cov =\
        max(tol,0.1*np.amin(np.abs(guess)))*np.eye(guess.shape[0],dtype=DTYPEf)
        for k in range(num_additional_random_guesses):
            guess=np.random.multivariate_normal(guess,cov)
            guess[0:break_point_1] = np.maximum(tol, guess[0:break_point_1])
            guess[break_point_1:break_point_2] = np.maximum(tol,
                                                            np.minimum(max_imp_coef, guess[break_point_1:break_point_2])
                                                           )
            guess[break_point_2:] = np.maximum(1.0+tol, guess[break_point_2:])
            list_init_guesses.append(np.array(guess, copy=True))
        return list_init_guesses    
                                 
    def store_base_rates(self):
        cdef np.ndarray[DTYPEf_t, ndim=1] base_rates = self.tol*np.ones(self.num_event_types,dtype=DTYPEf)
        if self.hawkes_kernel.max_spectral_radius>=1.0:
            print("Spectral radius of L1 norm of hawkes kernel is greater or equal 1.0: I am setting base rates to the default value of 0.0.")
        else:
            try:
                base_rates = np.dot(
                    (np.eye(self.num_event_types, dtype=DTYPEf)-self.hawkes_kernel.L1_norm),
                    self.expected_intensities)
            except:
                pass
        self.base_rates = base_rates    
        
    def store_distribution_of_marks(self):
        print("I am storing distribution of marks")
        cdef int len_events=len(self.events)
        cdef DTYPEi_t [:] events_memview = self.events
        cdef DTYPEi_t [:] states_memview = self.states
        cdef np.ndarray[DTYPEf_t, ndim=2] prob = np.zeros((self.num_event_types, self.num_states),dtype=DTYPEf)
        prob = store_distribution_of_marks(self.num_event_types, self.num_states,
                                           events_memview, states_memview, len_events)
        self.marks_distribution=prob
    def store_transition_probabilities(self,verbose=False):
        print('I am storing transition probabilities')
        cdef int d_S = self.num_states
        cdef int d_E = self.num_event_types
        cdef int v = int(verbose)
        self.transition_probabilities = estimate_transition_probabilities(
            d_E,d_S,self.events,self.states,verbose=v)    
    def store_expected_intensities(self):
        print("I am storing expected intensities")
        cdef DTYPEi_t [:] events_memview = self.events
        cdef np.ndarray[DTYPEf_t, ndim=1] Lambda = np.zeros(self.num_event_types, dtype=DTYPEf)
        Lambda=store_expected_intensities(self.num_event_types,events_memview, self.time_horizon)
        self.expected_intensities = Lambda
    def store_nonsingular_expected_jumps(self, parallel=True):
        cdef int N = max(1,self.num_event_types)
        cdef DTYPEf_t run_time=-time.time()
        if parallel:
            print("I am storing non-singular expected jumps in parallel. num_processes: {}".format(N))
            pool=mp.Pool(N)
            results=pool.map_async(
                self.estimate_nonsingular_expected_jumps_partial,
                list(range(self.num_event_types))
            ).get()
            pool.close()
            pool.join()
        else:
            print("I am storing non-singular expected jumps serially")
            results=list(
                map(self.estimate_nonsingular_expected_jumps_partial,
                    list(range(self.num_event_types))
                   )
            )
        run_time+=time.time()
        print("Execution of non-singular expected jumps terminates. run_time={}".format(run_time))
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef int G = self.grid.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=5] g_hat = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=5] g_hat_one = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] g_hat_at_quadpnts = np.zeros((d_E,d_S,d_E,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] g_hat_at_gridpnts = np.zeros((d_E,d_S,d_E,G),dtype=DTYPEf)
        cdef int e=0
        for e in range(self.num_event_types):
            g_hat+=results[e][0]
            g_hat_one+=results[e][1]
            g_hat_at_quadpnts+=results[e][2]
            g_hat_at_gridpnts+=results[e][3]
        self.g_hat = g_hat
        self.g_hat_one = g_hat_one
        self.g_hat_at_quadpnts = g_hat_at_quadpnts
        self.g_hat_at_gridpnts = g_hat_at_gridpnts
    def store_convolution_kernels(self,use_filter=False):
        print("I am storing convolution kernels")
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=6] K_hat = np.zeros((d_E,d_E,d_S,d_S,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=6] K_hat_one = np.zeros((d_E,d_E,d_S,d_S,Q,Q),dtype=DTYPEf)
        self.K_hat=K_hat
        self.K_hat_one=K_hat_one
        self.set_convolution_kernels(use_filter=use_filter)
        
    def store_matrix_A(self):
        print("I am storing matrix A")
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=2] matrix_A = np.zeros((d_E*d_S*Q,d_E*d_S*Q), dtype=DTYPEf)
        self.matrix_A = matrix_A
        self.set_matrix_A()
    
    def estimate_nonsingular_expected_jumps_partial(self, int e):
        cdef int process_id = os.getpid()
        print("I am estimating non-singular expected jumps for the component e={}. process_id: pid{}".format(e,process_id))
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef int G = self.grid.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=5] g_hat = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=5] g_hat_one = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] g_hat_at_quadpnts = np.zeros((d_E,d_S,d_E,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] g_hat_at_gridpnts = np.zeros((d_E,d_S,d_E,G),dtype=DTYPEf)
        cdef DTYPEf_t [:,:,:,:,:] g_hat_memview = g_hat
        cdef DTYPEf_t [:,:,:,:,:] g_hat_one_memview = g_hat_one
        cdef DTYPEf_t [:,:,:,:] g_hat_at_quadpnts_memview = g_hat_at_quadpnts
        cdef DTYPEf_t [:,:,:,:] g_hat_at_gridpnts_memview = g_hat_at_gridpnts
        cdef DTYPEf_t [:] Lambda = self.expected_intensities
        cdef DTYPEf_t [:,:,:] lt_memview = self.labelled_times
        cdef DTYPEi_t [:,:] count_memview = self.count
        cdef DTYPEf_t [:] quadpnts_memview = self.quadrature.partition
        cdef DTYPEf_t [:] gridpnts_memview = self.grid.partition
        cdef int e1=0, x1=0
#         for x1 in prange(d_S, nogil=True):
        for x1 in range(d_S):
            for e1 in range(d_E):
                estimate_g_hat_at_gridpnts(
                    e1, x1, e, d_S, G,
                    g_hat_at_gridpnts_memview, 
                    Lambda,
                    lt_memview, count_memview,
                    gridpnts_memview,
                )
#                 print("g_hat_at_gridpoints[{},{},{},:] has been estimated".format(e1,x1,e))
                set_nonsingular_expected_jumps_from_grid(
                    e1, x1, e, Q, G,
                    g_hat_memview, g_hat_one_memview,
                    g_hat_at_quadpnts_memview, g_hat_at_gridpnts_memview,
                    quadpnts_memview, gridpnts_memview,
                )
                
        return g_hat, g_hat_one, g_hat_at_quadpnts, g_hat_at_gridpnts           

    def set_convolution_kernels(self,use_filter=False):
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int num_quadpnts = self.quadrature.num_pnts
        if use_filter:
            g_hat=self.filtered_g_hat
            g_hat_one=self.filtered_g_hat_one
        else:    
            g_hat=self.g_hat
            g_hat_one=self.g_hat_one
        cdef DTYPEf_t [:,:,:,:,:] g_hat_memview = g_hat
        cdef DTYPEf_t [:,:,:,:,:] g_hat_one_memview = g_hat_one
        cdef DTYPEf_t [:] Lambda = self.expected_intensities
        cdef DTYPEf_t [:,:] mark_prob = self.marks_distribution
        cdef DTYPEf_t [:] quadpnts = self.quadrature.partition
        cdef DTYPEf_t [:,:,:,:,:,:] K_hat = self.K_hat
        cdef DTYPEf_t [:,:,:,:,:,:] K_hat_one = self.K_hat_one
        cdef int e1=0, eps=0, x1=0, y=0
        for x1 in prange(d_S, nogil=True):
            for y in range(d_S):
                for e1 in range(d_E):
                    for eps in range(d_E):
                        produce_convolution_kernel(
                            e1, eps, x1, y, num_quadpnts,
                            g_hat_memview, g_hat_one_memview,
                            Lambda, mark_prob,
                            K_hat, K_hat_one,
                            quadpnts
                        ) 
    def set_matrix_A(self):
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef DTYPEf_t [:,:,:,:,:,:] K_hat = self.K_hat
        cdef DTYPEf_t [:,:,:,:,:,:] K_hat_one = self.K_hat_one
        cdef DTYPEf_t [:,:] A = self.matrix_A
        cdef int e1=0, eps=0, x1=0, y=0
        for x1 in prange(d_S, nogil=True):
            for y in range(d_S):
                for e1 in range(d_E):
                    for eps in range(d_E):
                        produce_matrix_A(
                            e1, eps, x1, y, d_S, Q,
                            K_hat, K_hat_one, A
                        ) 
    def set_vector_b_e(self, int e):
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=1] b_e = np.zeros(
            d_E*d_S*Q, dtype=DTYPEf)
        cdef DTYPEf_t [:] b_e_memview = b_e
        if self.use_filter:
            g_hat_at_quadpnts = self.filtered_g_hat_at_quadpnts
        else:
            g_hat_at_quadpnts = self.g_hat_at_quadpnts
        cdef DTYPEf_t [:,:,:,:] g_hat_at_quadpnts_memview = g_hat_at_quadpnts
        cdef int e1=0, x1=0
        for x1 in prange(d_S, nogil=True):
            for e1 in range(d_E):
                produce_b_e(
                    e1, x1, e, d_S, Q,
                    g_hat_at_quadpnts_memview,
                    b_e_memview
                )
        return b_e        
    def solve_linear_system_partial(self, int e):
        cdef int process_id = os.getpid()
        print("I am solving for the component e={}. process_id: pid{}".format(e,process_id))
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=1] b_e = self.set_vector_b_e(e)
        cdef np.ndarray[DTYPEf_t, ndim=2] M = self.matrix_A + np.eye(d_E*d_S*Q, dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] xi_e = linalg.solve(M,b_e)
        cdef np.ndarray[DTYPEf_t, ndim=4] result = np.zeros(
            (d_E, d_S, d_E, Q), dtype=DTYPEf)
        cdef int e1=0, x1=0, m=0, i=0
        for e1 in range(d_E):
            for x1 in range(d_S):
                for m in range(Q):
                    i=(e1*d_S+x1)*Q+m
                    result[e1,x1,e,m]=xi_e[i]
        return result
    def set_filter_param(self, enforce_positivity=False, DTYPEf_t scale=1.0, DTYPEf_t cutoff=10.0, int num_addpnts=200):
        self.filter=Filter(enforce_positivity=enforce_positivity,
                           cutoff=cutoff, scale=scale, num_addpnts=num_addpnts)
    def filter_nonsingular_expected_jumps(self,
                                          pre_estimation=False,
                                          parallel=True,
                                         ):
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef int G = self.grid.num_pnts
        self.filter_pre_estimation=pre_estimation
        if pre_estimation:
            self.g_hat_at_gridpnts = np.zeros((d_E,d_S,d_E,G),dtype=DTYPEf)
        cdef DTYPEf_t run_time = -time.time()    
        cdef int N=max(1,self.num_event_types)
        if parallel:
            print("I am filtering non-singular expected jumps in parallel. num_process: {}".format(N))
            with mp.Pool(N) as pool:
                solver=pool.map_async(
                    self.filter_nonsingular_expected_jumps_partial,
                    list(range(self.num_event_types))
                )
                pool.close()
                pool.join()
                results=solver.get()
        else:        
            print("I am filtering non-singular expected jumps serially")
            results=list(
                   map(self.filter_nonsingular_expected_jumps_partial,
                   list(range(self.num_event_types))
                   )
                   )
        run_time+=time.time()
        print("Filtering terminates. run_time={}".format(run_time))
        cdef np.ndarray[DTYPEf_t, ndim=5] filtered_g_hat = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=5] filtered_g_hat_one = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] filtered_g_hat_at_quadpnts = np.zeros((d_E,d_S,d_E,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] filtered_g_hat_at_gridpnts = np.zeros((d_E,d_S,d_E,G),dtype=DTYPEf)
        cdef int e=0
        for e in range(self.num_event_types):
            filtered_g_hat+=results[e][0]
            filtered_g_hat_one+=results[e][1]
            filtered_g_hat_at_quadpnts+=results[e][2]
            filtered_g_hat_at_gridpnts+=results[e][3]
        self.filtered_g_hat=filtered_g_hat
        self.filtered_g_hat_one=filtered_g_hat_one
        self.filtered_g_hat_at_quadpnts=filtered_g_hat_at_quadpnts    
        self.filtered_g_hat_at_gridpnts=filtered_g_hat_at_gridpnts    
    def filter_nonsingular_expected_jumps_partial(self,int e):
        cdef int process_id = os.getpid()
        print("Filter non-singular expected jumps. component_e: {}; process_id: pid{}".format(e,process_id))
        pre_estimation=self.filter_pre_estimation
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef int G = self.grid.num_pnts
        cdef DTYPEf_t cutoff = max(self.filter.cutoff,-self.filter.cutoff)
        cdef int num_addpnts = self.filter.num_addpnts
        cdef np.ndarray[DTYPEf_t, ndim=1] left_partition = np.linspace(
            self.grid.partition[0], self.grid.partition[self.grid.num_pnts-1], 
            num=self.filter.num_addpnts, dtype=DTYPEf
        )
        cdef np.ndarray[DTYPEf_t, ndim=1] partition = np.zeros(
            1+self.grid.num_pnts+self.filter.num_addpnts, dtype=DTYPEf)
        cdef np.ndarray[DTYPEi_t, ndim=1] pos_gridpnts = np.zeros(1+self.grid.num_pnts,dtype=DTYPEi)
        partition, pos_gridpnts = computation.merge_sorted_arrays(left_partition, self.grid.partition)
        cdef DTYPEf_t [:] partition_memview = partition
        cdef DTYPEi_t [:] pos_gridpnts_memview = pos_gridpnts
        cdef DTYPEf_t [:] gridpnts_memview = self.grid.partition
        cdef DTYPEf_t [:] quadpnts_memview = self.quadrature.partition
        cdef np.ndarray[DTYPEf_t, ndim=4] interpol_g_hat_at_gridpnts = np.zeros(
            (d_E,d_S,d_E,G+self.filter.num_addpnts),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] conv_g_hat_at_gridpnts = np.zeros(
            (d_E,d_S,d_E,G+self.filter.num_addpnts),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] filtered_g_hat_at_gridpnts = np.zeros(
            (d_E,d_S,d_E,G),dtype=DTYPEf)
        cdef DTYPEf_t [:,:,:,:] filtered_g_hat_at_gridpnts_memview = filtered_g_hat_at_gridpnts
        cdef np.ndarray[DTYPEf_t, ndim=4] g_hat_at_gridpnts = np.maximum(-cutoff,np.minimum(cutoff,self.g_hat_at_gridpnts)) 
        cdef DTYPEf_t [:,:,:,:] g_hat_at_gridpnts_memview = g_hat_at_gridpnts
        cdef DTYPEf_t [:,:,:,:] interpol_g_hat_at_gridpnts_memview = interpol_g_hat_at_gridpnts
        cdef DTYPEf_t [:,:,:,:] conv_g_hat_at_gridpnts_memview = conv_g_hat_at_gridpnts
        cdef np.ndarray[DTYPEf_t, ndim=5] filtered_g_hat = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=5] filtered_g_hat_one = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] filtered_g_hat_at_quadpnts = np.zeros((d_E,d_S,d_E,Q),dtype=DTYPEf)
        cdef DTYPEf_t [:,:,:,:,:] filtered_g_hat_memview = filtered_g_hat
        cdef DTYPEf_t [:,:,:,:,:] filtered_g_hat_one_memview = filtered_g_hat_one
        cdef DTYPEf_t [:,:,:,:] filtered_g_hat_at_quadpnts_memview = filtered_g_hat_at_quadpnts
        cdef DTYPEf_t [:] Lambda = self.expected_intensities
        cdef DTYPEf_t [:,:,:] lt_memview = self.labelled_times
        cdef DTYPEi_t [:,:] count_memview = self.count
        cdef DTYPEf_t scale = self.filter.scale
        cdef DTYPEf_t weight = 1.0/(1.0 - exp(-scale*self.grid.partition[self.grid.num_pnts-1]))
        cdef int e1=0, x1=0
        for x1 in range(d_S):
            for e1 in range(d_E):
                if pre_estimation:
                    estimate_g_hat_at_gridpnts(
                        e1, x1, e, d_S, G,
                        g_hat_at_gridpnts_memview, 
                        Lambda,
                        lt_memview, count_memview,
                        gridpnts_memview,
                    )
                    g_hat_at_gridpnts = np.maximum(-cutoff,np.minimum(cutoff,g_hat_at_gridpnts))
                if self.filter.enforce_positivity:
                    g_hat_at_gridpnts[e1,x1,e,:]=np.maximum(0.0,g_hat_at_gridpnts[e1,x1,e,:])
                interpolate(
                    e1, x1,  e,
                    g_hat_at_gridpnts_memview, interpol_g_hat_at_gridpnts_memview,
                    partition_memview, G+num_addpnts,
                    pos_gridpnts_memview, gridpnts_memview)       
                convolute(
                    e1, x1,  e,
                    interpol_g_hat_at_gridpnts_memview, conv_g_hat_at_gridpnts_memview,
                    partition_memview, G+num_addpnts,
                    scale, weight)
                filtered_g_hat_at_gridpnts[e1,x1,e,:]=np.array(
                    conv_g_hat_at_gridpnts[e1,x1,e,pos_gridpnts[:G]],copy=True,dtype=DTYPEf)
                set_nonsingular_expected_jumps_from_grid(
                    e1, x1, e, Q, G,
                    filtered_g_hat_memview, 
                    filtered_g_hat_one_memview,
                    filtered_g_hat_at_quadpnts_memview,
                    filtered_g_hat_at_gridpnts_memview,
                    quadpnts_memview,
                    gridpnts_memview,
                )        
        return filtered_g_hat, filtered_g_hat_one, filtered_g_hat_at_quadpnts, filtered_g_hat_at_gridpnts        
                
        
        
cdef void estimate_g_hat_at_gridpnts(
    int e1, int x1, int e, int num_states, int num_gridpnts,
    DTYPEf_t [:,:,:,:] g_hat_at_gridpnts, 
    DTYPEf_t [:] Lambda,
    DTYPEf_t [:,:,:] labelled_times,
    DTYPEi_t [:,:] count,
    DTYPEf_t [:] gridpnts,
) nogil:    
    cdef DTYPEf_t prev_pnt=0.0, next_pnt=0.0, delta_pnt=0.0
    cdef DTYPEf_t tau=0.0
    cdef DTYPEf_t float_count=max(1.0, float(count[e1,x1]))
    cdef int n=0, x=0, k=0, k1=0
    for n in range(num_gridpnts):
        if n==0:
            prev_pnt=0.0
        else:
            prev_pnt=gridpnts[n-1]
        next_pnt=gridpnts[n+1]
        delta_pnt=next_pnt-prev_pnt
        for k1 in range(count[e1,x1]):
            for x in range(num_states):
                for k in range(count[e,x]):
                    tau=labelled_times[e,x,k]-labelled_times[e1,x1,k1]
                    if (tau>=prev_pnt) & (tau<=next_pnt):
                        g_hat_at_gridpnts[e1,x1,e,n]+=1.0
        g_hat_at_gridpnts[e1,x1,e,n]/=(float_count*delta_pnt)
        g_hat_at_gridpnts[e1,x1,e,n]+= -Lambda[e]
        
        
cdef void set_nonsingular_expected_jumps_from_grid(
    int e1, int x1, int e, int num_quadpnts, int num_gridpnts,
    DTYPEf_t [:,:,:,:,:] g_hat, 
    DTYPEf_t [:,:,:,:,:] g_hat_one,
    DTYPEf_t [:,:,:,:] g_hat_at_quadpnts,
    DTYPEf_t [:,:,:,:] g_hat_at_gridpnts,
    DTYPEf_t [:] quadpnts,
    DTYPEf_t [:] gridpnts,
) nogil:    
    cdef result_eval_integr res
    cdef DTYPEf_t t=0.0, t_0=0.0, t_1=0.0, k=0.0
    cdef int m=0, n=0
    for m in range(num_quadpnts):
        for n in range(num_quadpnts):
            if m<=n:
                t=quadpnts[n]
                t_0=quadpnts[n]-quadpnts[m]
                t_1=quadpnts[n+1]-quadpnts[m]
                k=quadpnts[n]-quadpnts[m]
                res=eval_integr_g_hat_at_gridpnts(
                    e1, x1, e, num_gridpnts,
                    g_hat_at_gridpnts, gridpnts,
                    t, t_0, t_1, k)
                g_hat[e1,x1,e,m,n]=res.integral_0/(quadpnts[n+1]-quadpnts[n])
                g_hat_one[e1,x1,e,m,n]=res.integral_1/(quadpnts[n+1]-quadpnts[n])
                if m==0:
                    g_hat_at_quadpnts[e1,x1,e,n]=res.value
            else:
                t=quadpnts[n]
                t_0=quadpnts[m]-quadpnts[n+1]
                t_1=quadpnts[m]-quadpnts[n]
                k=quadpnts[m]-quadpnts[n]
                res=eval_integr_g_hat_at_gridpnts(
                    e1, x1, e, num_gridpnts,
                    g_hat_at_gridpnts, gridpnts,
                    t, t_0, t_1, k)
                g_hat[e1,x1,e,m,n]=res.integral_0/(quadpnts[n+1]-quadpnts[n])
                g_hat_one[e1,x1,e,m,n]= - res.integral_1/(quadpnts[n+1]-quadpnts[n])                
    
        
cdef struct result_eval_integr:
    DTYPEf_t value
    DTYPEf_t integral_0
    DTYPEf_t integral_1
        
cdef result_eval_integr eval_integr_g_hat_at_gridpnts(
    int e1, int x1, int e, int num_gridpnts,
    DTYPEf_t [:,:,:,:] g_hat_at_gridpnts,
    DTYPEf_t [:] gridpnts,
    DTYPEf_t t, DTYPEf_t t_0, DTYPEf_t t_1, DTYPEf_t k
) nogil:                
    "It is assumed that t_0<t_1"
    cdef DTYPEf_t value=0.0, integral_0=0.0, integral_1=0.0
    cdef DTYPEf_t pnt=0.0, next_pnt=0.0, delta_pnt=0.0
    cdef DTYPEf_t upper=0.0, lower=0.0, pol_upper=0.0, pol_lower=0.0
    cdef int n=0
    for n in range(num_gridpnts):
        pnt=gridpnts[n]
        next_pnt=gridpnts[n+1]
        delta_pnt=next_pnt-pnt
        if pnt > max(t_1,t):
            break
        else:
            if (pnt<=t_1) & (next_pnt>= t_0):
                upper=min(t_1,next_pnt)
                lower=max(t_0,pnt)
                if pnt<t_0:
                    integral_0+=(upper-t_0)*g_hat_at_gridpnts[e1,x1,e,n]
                    integral_0+=(g_hat_at_gridpnts[e1,x1,e,n+1]-g_hat_at_gridpnts[e1,x1,e,n])\
                    *(pow(upper-pnt,2)-pow(t_0-pnt,2))/(2*delta_pnt)
                else:
                    integral_0+=(upper-pnt)*g_hat_at_gridpnts[e1,x1,e,n]
                    integral_0+=(g_hat_at_gridpnts[e1,x1,e,n+1]-g_hat_at_gridpnts[e1,x1,e,n])\
                    *(pow(upper-pnt,2))/(2*delta_pnt)
                integral_1+=0.5*(pow(upper-k,2)-pow(lower-k,2))*g_hat_at_gridpnts[e1,x1,e,n]
                pol_upper=pow(upper,3)/3 - (k+pnt)*pow(upper,2)/2 + k*pnt*upper
                pol_lower=pow(lower,3)/3 - (k+pnt)*pow(lower,2)/2 + k*pnt*lower
                integral_1+=(pol_upper-pol_lower)*(g_hat_at_gridpnts[e1,x1,e,n+1]-g_hat_at_gridpnts[e1,x1,e,n])/delta_pnt 
            if (pnt<=t) & (next_pnt>t):
                value=g_hat_at_gridpnts[e1,x1,e,n]\
                 +(g_hat_at_gridpnts[e1,x1,e,n+1]-g_hat_at_gridpnts[e1,x1,e,n])*(t-pnt)/delta_pnt
    cdef result_eval_integr res
    res.value = value
    res.integral_0 = integral_0
    res.integral_1 = integral_1
    return res

cdef DTYPEf_t evaluate_g_hat_from_grid(
    int e1, int x1, int e, int num_gridpnts,
    DTYPEf_t [:,:,:,:] g_hat_at_gridpnts,
    DTYPEf_t [:] gridpnts,
    DTYPEf_t t
) nogil:      
    cdef DTYPEf_t result=0.0
    cdef DTYPEf_t pnt=0.0, next_pnt=0.0, delta_pnt=0.0
    cdef int n=0
    for n in range(num_gridpnts):
        if (gridpnts[n]<=t) & (gridpnts[n+1]>t):
            pnt=gridpnts[n]
            next_pnt=gridpnts[n+1]
            delta_pnt=next_pnt-pnt
            result=g_hat_at_gridpnts[e1,x1,e,n]\
              +(g_hat_at_gridpnts[e1,x1,e,n+1]-g_hat_at_gridpnts[e1,x1,e,n])*(t-pnt)/delta_pnt
            return result
    return result    


   
cdef DTYPEf_t integrate_g_hat_at_gridpnts(
    int e1, int x1, int e, int num_gridpnts,
    DTYPEf_t [:,:,:,:] g_hat_at_gridpnts,
    DTYPEf_t [:] gridpnts,
    DTYPEf_t t_0, DTYPEf_t t_1
) nogil:        
    cdef DTYPEf_t result=0.0, upper=0.0
    cdef DTYPEf_t pnt=0.0, next_pnt=0.0, delta_pnt=0.0
    cdef int n=0
    for n in range(num_gridpnts):
        pnt=gridpnts[n]
        next_pnt=gridpnts[n+1]
        delta_pnt=next_pnt-pnt
        if pnt > t_1:
            break
        elif next_pnt>= t_0:
            upper=min(t_1,next_pnt)
            if pnt<t_0:
                result+=(upper-t_0)*g_hat_at_gridpnts[e1,x1,e,n]
                result+=(g_hat_at_gridpnts[e1,x1,e,n+1]-g_hat_at_gridpnts[e1,x1,e,n])\
                *(pow(upper-pnt,2)-pow(t_0-pnt,2))/(2*delta_pnt)
            else:
                result+=(upper-pnt)*g_hat_at_gridpnts[e1,x1,e,n]
                result+=(g_hat_at_gridpnts[e1,x1,e,n+1]-g_hat_at_gridpnts[e1,x1,e,n])\
                *(pow(upper-pnt,2))/(2*delta_pnt)
    return result 

cdef DTYPEf_t integrate_g_hat_at_gridpnts_one(
    int e1, int x1, int e, int num_gridpnts,
    DTYPEf_t [:,:,:,:] g_hat_at_gridpnts,
    DTYPEf_t [:] gridpnts,
    DTYPEf_t t_0, DTYPEf_t t_1, DTYPEf_t k
) nogil:        
    cdef DTYPEf_t result=0.0
    cdef DTYPEf_t pnt=0.0, next_pnt=0.0, delta_pnt=0.0
    cdef DTYPEf_t upper=0.0, lower=0.0, pol_upper=0.0, pol_lower=0.0
    cdef int n=0
    for n in range(num_gridpnts):
        pnt=gridpnts[n]
        next_pnt=gridpnts[n+1]
        delta_pnt=next_pnt-pnt
        if pnt > t_1:
            break
        elif next_pnt>= t_0:
            upper=min(t_1,next_pnt)
            lower=max(t_0,pnt)
            result+=0.5*(pow(upper-k,2)-pow(lower-k,2))*g_hat_at_gridpnts[e1,x1,e,n]
            pol_upper=pow(upper,3)/3 - (k+pnt)*pow(upper,2)/2 + k*pnt*upper
            pol_lower=pow(lower,3)/3 - (k+pnt)*pow(lower,2)/2 + k*pnt*lower
            result+=(pol_upper-pol_lower)*(g_hat_at_gridpnts[e1,x1,e,n+1]-g_hat_at_gridpnts[e1,x1,e,n])/delta_pnt
    return result       

cdef void produce_convolution_kernel(
    int e1, int eps, int x1, int y, int num_quadpnts,
    DTYPEf_t [:,:,:,:,:] g_hat, 
    DTYPEf_t [:,:,:,:,:] g_hat_one,
    DTYPEf_t [:] Lambda,
    DTYPEf_t [:,:] mark_prob,
    DTYPEf_t [:,:,:,:,:,:] K_hat,
    DTYPEf_t [:,:,:,:,:,:] K_hat_one,
    DTYPEf_t [:] quadpnts
) nogil:
    cdef int Q = num_quadpnts
    cdef DTYPEf_t delta_pnt_n =0.0
    cdef DTYPEf_t intensity_ratio = Lambda[eps]/Lambda[e1]
    cdef int m=0, n=0
    for m in range(Q):
        for n in range(Q):
            delta_pnt_n=quadpnts[n+1]-quadpnts[n]
            if m<=n:
                K_hat[e1,eps,x1,y,m,n] = intensity_ratio*mark_prob[eps,y]*delta_pnt_n*g_hat[eps,y,e1,m,n]
                K_hat_one[e1,eps,x1,y,m,n] = intensity_ratio*mark_prob[eps,y]*g_hat_one[eps,y,e1,m,n]
            else:
                K_hat[e1,eps,x1,y,m,n] = mark_prob[eps,y]*delta_pnt_n*g_hat[e1,x1,eps,m,n]
                K_hat_one[e1,eps,x1,y,m,n] = mark_prob[eps,y]*g_hat_one[e1,x1,eps,m,n]

cdef void produce_matrix_A(
    int e1, int eps, int x1, int y, int d_S, int Q,
    DTYPEf_t [:,:,:,:,:,:] K_hat,
    DTYPEf_t [:,:,:,:,:,:] K_hat_one,
    DTYPEf_t [:,:] A
) nogil:
    cdef int j=(eps*d_S+y)*Q
    cdef int i=0, m=0, n=0
    for m in range(Q):
        i=(e1*d_S+x1)*Q
        for n in range(Q):
            if n==0:
                A[i,j]=K_hat[e1,eps,x1,y,m,n]-K_hat_one[e1,eps,x1,y,m,n]
            else:
                A[i,j]=K_hat[e1,eps,x1,y,m,n]-K_hat_one[e1,eps,x1,y,m,n]+K_hat_one[e1,eps,x1,y,m,n-1]
            i+=1
        j+=1
              

cdef void produce_b_e(
    int e1, int x1, int e, int d_S, int Q,
    DTYPEf_t [:,:,:,:] g_hat_at_quadpnts,
    DTYPEf_t [:] b_e
) nogil:
    cdef int i=(e1*d_S+x1)*Q
    cdef int m=0
    for m in range(Q):
        b_e[i]=g_hat_at_quadpnts[e1,x1,e,m]
        i+=1
        
        

cdef DTYPEf_t estimate_Lambda_e(DTYPEi_t e,DTYPEi_t [:] events, DTYPEf_t time_horizon) nogil:
    cdef DTYPEi_t j=0
    cdef DTYPEi_t N_e=0
    for j in range(len(events)):
        if (events[j]==e):
            N_e+=1
    return float(N_e)/time_horizon  

def store_expected_intensities(int num_event_types, DTYPEi_t [:] events, DTYPEf_t time_horizon):
    cdef int e=0
    cdef np.ndarray[DTYPEf_t, ndim=1] Lambda = np.zeros(num_event_types, dtype=DTYPEf)
    cdef DTYPEf_t [:] Lambda_memview = Lambda
    for e in prange(num_event_types, nogil=True):
        Lambda_memview[e] = estimate_Lambda_e(e, events, time_horizon)
    return Lambda    

cdef DTYPEf_t estimate_mark_prob(int e, int y,
    DTYPEi_t [:] events, DTYPEi_t [:]  states, int len_events) nogil:
    cdef DTYPEi_t count_e=0,count_y=0
    cdef DTYPEi_t j=0
    for j in range(len_events):
        if (events[j]==e):
            count_e+=1
            if (states[j]==y):
                count_y+=1          
    cdef DTYPEf_t result = float(count_y)/float(count_e)
    return result

def store_distribution_of_marks(int num_event_types, int num_states,
                                DTYPEi_t [:] events, DTYPEi_t [:]  states, int len_events):
    cdef int e=0, y=0
    cdef np.ndarray[DTYPEf_t, ndim=2] prob = np.zeros((num_event_types,num_states),dtype=DTYPEf)
    cdef DTYPEf_t [:,:] prob_memview = prob
    for e in prange(num_event_types,nogil=True):
        for y in range(num_states):
            prob_memview[e,y]=estimate_mark_prob(e,y,events,states, len_events)        
    return prob        

    

cdef void convolute(
    int e1, int x1, int  e,
    DTYPEf_t [:,:,:,:] raw_data, DTYPEf_t [:,:,:,:] conv_data,
    DTYPEf_t [:] time, int len_time,
    DTYPEf_t scale, DTYPEf_t weight) nogil:
    """
    conv_data is supposed to be initialised to zero
    """
    cdef int k=0, j=0
    for k in range(len_time):
        for j in range(len_time):
            if j<=k:
                conv_data[e1,x1,e,k]+=raw_data[e1,x1,e,j]*exp(-scale*(time[k]-time[j]))*(time[j+1]-time[j])
            else:
                conv_data[e1,x1,e,k]+=raw_data[e1,x1,e,j]*exp(-scale*(time[j]-time[k]))*(time[j+1]-time[j])
        conv_data[e1,x1,e,k]*=weight*scale   
        

cdef void interpolate(
    int e1, int x1, int  e,
    DTYPEf_t [:,:,:,:] raw_data, DTYPEf_t [:,:,:,:] interp_data,
    DTYPEf_t [:] time, int len_time,
    DTYPEi_t [:] pos_gridpnts, DTYPEf_t [:] gridpnts ) nogil:
    cdef int k=0, j=0
    cdef DTYPEf_t m=0.0, q=0.0
    for k in range(len_time):
        if k==pos_gridpnts[j]:
            interp_data[e1,x1,e,k]=raw_data[e1,x1,e,j]
            m=(raw_data[e1,x1,e,j+1]-raw_data[e1,x1,e,j])/(gridpnts[j+1]-gridpnts[j])
            q=raw_data[e1,x1,e,j+1]-m*gridpnts[j+1]
            j+=1
        else:
            interp_data[e1,x1,e,k]=m*time[k]+q
        
    
    
class FitPowerlaw:
    def __init__(self,
                 int num_event_types, int num_states,
                 quadrature,
                 np.ndarray[DTYPEf_t, ndim=4] hawkes_kernel_at_quadpnts,
                 DTYPEf_t ridge_param=1.0, DTYPEf_t tol=1.0e-9):
        self.num_event_types = num_event_types
        self.num_states = num_states
        self.quadrature = quadrature
        self.hawkes_kernel_at_quadpnts = hawkes_kernel_at_quadpnts
        self.ridge_param = ridge_param
        self.tol=tol
        self.regression_model=Ridge(alpha=ridge_param,normalize=True)
    def fit(self):
        cdef int N = max(1,min(self.num_event_types,mp.cpu_count()))
        print("I am fitting the powerlaw kernel in parallel on {} cpus".format(N))
        cdef DTYPEf_t run_time = -time.time()
        pool=mp.Pool(N)
        results=pool.map_async(self.fit_partial,list(range(self.num_event_types))).get()
        pool.close()
        run_time+=time.time()
        print("Parallel fitting terminates. run_time={}".format(run_time))
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef np.ndarray[DTYPEf_t, ndim=3] imp_coef = np.zeros((d_E,d_S,d_E), dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=3] dec_coef = np.zeros((d_E,d_S,d_E), dtype=DTYPEf)
        cdef int e=0
        for e in range(d_E):
            imp_coef+=results[e][0]
            dec_coef+=results[e][1]
        self.imp_coef=imp_coef
        self.dec_coef=dec_coef
    def fit_partial(self,int e):
        cdef DTYPEf_t tol=self.tol
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=2] X = np.log(1.0+self.quadrature.partition[:Q]).reshape(Q,1)
        cdef np.ndarray[DTYPEf_t, ndim=1] y = np.zeros(Q,dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=3] imp_coef = np.zeros((d_E,d_S,d_E), dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=3] dec_coef = np.zeros((d_E,d_S,d_E), dtype=DTYPEf)
        cdef int e1=0, x1=0
        for e1 in range(d_E):
            for x1 in range(d_S):
                y=np.log(np.maximum(tol,self.hawkes_kernel_at_quadpnts[e1,x1,e,:]))
                fitted_model=self.regression_model.fit(X,y)
                imp_coef[e1,x1,e]=np.exp(fitted_model.intercept_)
                dec_coef[e1,x1,e]=np.maximum(1.0+tol,-fitted_model.coef_)
        return imp_coef, dec_coef        
        
    
