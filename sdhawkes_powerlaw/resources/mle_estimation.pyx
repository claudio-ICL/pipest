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

import numpy as np
cimport numpy as np
import bisect
import copy
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport pow
from libc.stdlib cimport rand, RAND_MAX

import model
import computation
import minimisation_algo as minim_algo
import goodness_of_fit
import dirichlet
from sklearn.linear_model import LinearRegression, Ridge

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t



class EstimProcedure:
    def __init__(self, int num_event_types, int num_states,
                 np.ndarray[DTYPEf_t , ndim=1] times, 
                 np.ndarray[DTYPEi_t , ndim=1] events,
                 np.ndarray[DTYPEi_t , ndim=1] states,
                 str type_of_input = 'simulated',
                 store_trans_prob = False, store_dirichlet_param = False,
                 volumes = None, int n_levels = 2,
                ):
        print("mle_estimation.EstimProcedure is being initialised")
        self.type_of_input = type_of_input
        if not (len(times)==len(states) & len(times)==len(events)):
            raise ValueError("All shapes must agree, but input was:\n len(times)={} \n len(events)={} \n len(states)={}".format(len(times),len(events),len(states)))
        self.num_event_types = num_event_types
        self.num_states = num_states
        self.n_levels = n_levels
        cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times = np.zeros((num_event_types,num_states,len(times)), dtype=DTYPEf)
        cdef np.ndarray[DTYPEi_t, ndim=2] count = np.zeros((num_event_types, num_states), dtype=DTYPEi)
        labelled_times, count = computation.distribute_times_per_event_state(
            num_event_types,num_states,
            times, events, states)
        self.labelled_times=labelled_times
        self.count=count
        self.times=np.array(times,copy=True,dtype=DTYPEf)
        self.time_horizon=copy.copy(times[len(times)-1])
        self.events=np.array(events,copy=True,dtype=DTYPEi)
        self.states=np.array(states,copy=True,dtype=DTYPEi)
        self.hawkes_kernel=model.HawkesKernel(num_event_types, num_states)
        if not (volumes is None):
            self.volumes = np.array(volumes, copy=True, dtype=DTYPEf)
            if store_dirichlet_param:
                self.store_dirichlet_parameters()
        if store_trans_prob:
            self.store_transition_probabilities()
        print("EstimProcedure has been successfully initialised")          
    def store_transition_probabilities(self,verbose=False):
        print('I am storing transition probabilities')
        cdef int v = int(verbose)
        cdef DTYPEf_t run_time = -time.time()
        self.transition_probabilities =  estimate_transition_probabilities(
            self.num_event_types,self.num_states,self.events,self.states,verbose=v)
        run_time += time.time()
        print("Transition probabilities have been estimated and store. run_time={}".format(run_time))
    def store_dirichlet_parameters(self,verbose=False):
        print("I am storing dirichlet parameters")
        cdef DTYPEf_t run_time = -time.time()
        self.dirichlet_param = estimate_dirichlet_parameters(
            self.num_states, self.n_levels,
            self.states, self.volumes,
            verbose=verbose)
        run_time += time.time()
        print("Dirichlet parameters have been estimated and store. run_time={}".format(run_time))
    def set_estimation_of_hawkes_param(self,
                                       DTYPEf_t time_start, DTYPEf_t time_end,
                                       list list_of_init_guesses=[],
                                       DTYPEf_t max_imp_coef = 100.0,
                                       DTYPEf_t learning_rate = 0.001,
                                       int maxiter=100,
                                       int number_of_additional_guesses=3,
                                       parallel=False, 
                                       pre_estim_ord_hawkes=False, pre_estim_parallel=False, 
                                       use_prange = False,
                                       int number_of_attempts = 2, DTYPEf_t tol = 1.0e-07,
                                       int num_processes = 0,
                                       int batch_size = 5000,
                                       int num_run_per_minibatch = 1,  
                                      ):
        print('I am setting the estimation of hawkes parameters, with time_start={}, time_end={}.'
              .format(time_start,time_end))
        print('The boundaries of arrival times are {}-{}'.format(self.times[0],self.times[len(self.times)-1]))
        self.time_start=time_start
        self.time_end=time_end
        self.given_list_of_guesses =\
        store_given_list_of_guesses(self.num_event_types,self.num_states,list_of_init_guesses)
        self.max_imp_coef = max_imp_coef
        self.learning_rate = learning_rate
        self.maxiter=maxiter
        self.number_of_additional_guesses=number_of_additional_guesses
        self.parallel, self.pre_estim_ord_hawkes, self.pre_estim_parallel = \
        parallel, pre_estim_ord_hawkes, pre_estim_parallel
        self.use_prange = use_prange
        self.number_of_attempts = number_of_attempts
        cdef list results_of_estimation = []
        self.results_of_estimation = results_of_estimation
        self.tol = tol
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.num_run_per_minibatch = num_run_per_minibatch  
        
    def prepare_list_init_guesses_partial(self,int e):
        cdef list list_init_guesses = copy.copy(self.given_list_of_guesses.get(e))
        if self.pre_estim_ord_hawkes:
            list_init_guesses.append(pre_estimate_ordinary_hawkes(
                e,
                self.num_event_types, 
                self.times,
                self.events,
                self.time_start,
                self.time_end,
                num_init_guesses = 1+self.number_of_additional_guesses,
                parallel = self.pre_estim_parallel,
                learning_rate = self.learning_rate,
                maxiter = self.maxiter,
                tol = self.tol,
                n_states = self.num_states, #used to reshape the results
                reshape_to_sd = True,
                return_as_flat_array = True,
                use_prange = self.use_prange,
                number_of_attempts = self.number_of_attempts,
                num_processes = self.num_processes,
                batch_size = self.batch_size,
                num_run_per_minibatch = self.num_run_per_minibatch,
            ))
        cdef np.ndarray[DTYPEf_t, ndim=1] guess = np.array(list_init_guesses[len(list_init_guesses)-1],dtype=DTYPEf,copy=True)   
        cdef np.ndarray[DTYPEf_t, ndim=1] new_guess = np.zeros_like(guess,dtype=DTYPEf) 
        cov =\
        np.maximum(0.05*np.amin(np.abs(guess)),self.tol)*np.eye(len(guess))
        cdef int j=0
        cdef int break_point = 1+self.num_event_types*self.num_states
        for j in range(max(1,self.number_of_additional_guesses//2)):
            new_guess = np.random.multivariate_normal(guess,cov)
            new_guess[0:break_point] = np.maximum(0.0,new_guess[0:break_point])
            new_guess[break_point:] = np.maximum(1.01,new_guess[break_point:])
            list_init_guesses.append(np.array(new_guess,copy=True))
        t0=float(self.times[0])
        idx_e = (self.events==e)
        num_e = np.sum(idx_e)
        max_base_rate = max(0.001,np.mean(np.arange(1,1+num_e)/np.maximum(0.0001,self.times[idx_e]-t0)))
        for j in range(max(1,self.number_of_additional_guesses//2)):
            new_guess[0:break_point] = np.random.uniform(low=0.0, high = 5.0, size=(break_point,))
            new_guess[break_point:] = np.random.uniform(low=1.5, high = 2.5, size=(len(new_guess)-break_point,))
            convex_coef = max(0.001,float(rand())/float(RAND_MAX))
            new_guess[0] = convex_coef*new_guess[0] + (1.0-convex_coef)*max_base_rate
            list_init_guesses.append(np.array(new_guess,copy=True))
        return list_init_guesses    
            
    def estimate_hawkes_param_partial(self, int e):
        list_init_guesses = self.prepare_list_init_guesses_partial(e)
        res = estimate_hawkes_param_partial(
            e, self.num_event_types, self.num_states,
            self.time_start, self.time_end,
            self.times, self.events, self.states,
            list_init_guesses = list_init_guesses,
            max_imp_coef = self.max_imp_coef,
            learning_rate=self.learning_rate,
            maxiter = self.maxiter,
            parallel = self.parallel,
            use_prange = self.use_prange,
            number_of_attempts = self.number_of_attempts,
            tol = self.tol,
            num_processes = self.num_processes,
            batch_size = self.batch_size,
            num_run_per_minibatch = self.num_run_per_minibatch,
        )
        print("mle_estimation.EstimProcedure.estimate_hawkes_param_partial. Estimation terminates for event_type {}".format(e))
        self.results_of_estimation.append(res)
        
        
    def launch_estimation_of_hawkes_param(self, partial=True, int e=0):
        if not partial:
            for e in range(self.num_event_types):
                self.estimate_hawkes_param_partial(e)
        else:
            assert e < self.num_event_types
            self.estimate_hawkes_param_partial(e)

            
    def store_results_of_estimation(self, list results_of_estimation):
        self.results_of_estimation = copy.copy(results_of_estimation)
    def store_hawkes_parameters(self):
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        assert len(self.results_of_estimation) == d_E #One partial result of estimation for every component of the hawkes process
        cdef np.ndarray[DTYPEf_t, ndim=1] base_rates = np.zeros(d_E, dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=3] imp_coef = np.zeros((d_E,d_S,d_E), dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=3] dec_coef = np.zeros((d_E,d_S,d_E), dtype=DTYPEf)
        cdef int e=0, e1=0
        for e1 in range(d_E):
            res=self.results_of_estimation[e1]
            e=res.get('component_e')
            base_rates[e]=res.get('base_rate')
            imp_coef[:,:,e]=res.get('imp_coef')
            dec_coef[:,:,e]=res.get('dec_coef')
        self.base_rates=base_rates
        self.hawkes_kernel.store_parameters(imp_coef,dec_coef)
        self.hawkes_kernel.compute_L1_norm_param()
    def create_goodness_of_fit(self, parallel=True):
        "type_of_input can either be 'simulated' or 'empirical'"
        self.goodness_of_fit=goodness_of_fit.good_fit(
            self.num_event_types, self.num_states,
            self.base_rates, self.hawkes_kernel.alphas, self.hawkes_kernel.betas, self.transition_probabilities,
            self.times, self.events,self.states, type_of_input=self.type_of_input, parallel=parallel
        )      
        


def estimate_transition_probabilities(int n_event_types, int n_states,
    long [:] events, long [:]  states, int verbose=1):
    cdef np.ndarray[DTYPEf_t, ndim=3] result = np.zeros((n_states, n_event_types, n_states),dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] count_of_states_events = np.zeros((n_states, n_event_types),dtype=DTYPEi)
    cdef double [:,:,:] result_memview = result
    cdef long [:,:] count_memview  = count_of_states_events
    cdef int N = len(events)-1
    cdef int n, x1, x2, e
    cdef int event,state_before,state_after
    for n in prange(1,N,nogil=True):
        event = events[n]
        state_before = states[n-1]
        state_after = states[n]
        count_memview[state_before, event] += 1
        result_memview[state_before, event, state_after] += 1.0
    cdef long size=0   
    for x1 in range(n_states):
        for e in range(n_event_types):
            size = count_of_states_events[x1, e]
            if size > 0:
                for x2 in range(n_states):
                    result[x1, e, x2] /= float(size)
            else:
                result[x1,e,:]=1/n_states
                if verbose:
                    message = 'Warning: transition_prob[{},{},:]'.format(x1,e)
                    message += ' cannot be estimated because'
                    message += ' events of type {} never occur in state {}'.format(e,x1)
                    print(message)
    return result

def estimate_liquidator_transition_prob(int n_states,
                                        np.ndarray[DTYPEi_t, ndim=1] events,
                                        np.ndarray[DTYPEi_t, ndim=1] states,
                                        int liquidator_index = 0):
    cdef np.ndarray[DTYPEf_t, ndim=2] result = np.zeros((n_states,n_states), dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=1] count = np.zeros(n_states, dtype=DTYPEi)
    cdef double [:,:] result_memview = result
    cdef long [:] count_memview = count
    cdef int N = len(events)-1
    cdef int n = 0
    cdef int state_before,state_after
    for n in prange(1,N,nogil=True):
        if events[n]==liquidator_index:
            state_before = states[n-1]
            state_after = states[n]
            count_memview[state_before] += 1
            result_memview[state_before,state_after] +=1
    cdef long size=0   
    for x1 in range(n_states):
        size = count[x1]
        if size > 0:
            for x2 in range(n_states):
                result[x1, x2] /= size
        else:
            message = 'Warning: Transition probabilities from state ' + str(x1)
            message += ' when events of type ' + str(liquidator_index) + ' occur cannot be estimated because'
            message += ' events of this type never occur this state'
            print(message)
    return result        
            
    


def estimate_dirichlet_parameters(int num_of_states, int n_levels,
                                  np.ndarray[DTYPEi_t, ndim=1] states,
                                  np.ndarray[DTYPEf_t, ndim=2] volumes,
                                  DTYPEf_t tolerance=1e-8,verbose=False, DTYPEf_t epsilon=1.0e-7):
    cdef DTYPEf_t tol = tolerance
    cdef np.ndarray[DTYPEf_t, ndim=2] estimator = np.zeros((num_of_states,2*n_levels),dtype=DTYPEf)
    cdef int s =0 
    for s in range(num_of_states):
        if verbose:
            print('I am estimating the dirichlet parameters for the state {}'.format(s))
        idx=np.array((states==s),dtype=np.bool)
        if np.sum(idx)<2:
            if verbose:
                print('estimate_dirichlet_parameters: Warning: state {} never observed.'
                      .format(s) + ' I am using a default value of 1.0 in this state')
            estimator[s,:]=np.ones(2*n_levels,dtype=DTYPEf)
        else:
            observations=np.array(volumes[idx,:],dtype=np.float,copy=True)
            success=False
            while (not success)&(tol<1):
                try:
                    if verbose:
                        print('estimate_dirichlet_parameters. attempt with tol={}'.format(tol))
                    estimator[s,:]=dirichlet.mle(observations,tol=tol)
                    success=True
                    idx_neg=np.array((estimator[s,:]<=0.0),dtype=np.bool)
                    if np.any(idx_neg):
                        print('estimate_dirichlet_parameters. WARNING: In state={}, non-positive dirichlet parameter results from mle estimation.'.format(s))
                        print('  I am correcting this manually.')
                        estimator[s,idx_neg]=epsilon
                    if verbose:
                        print('estimate_dirichlet_parameters. state={}. success with tol={}'.format(s,tol))
                except Exception:
                    tol=2*tol
            if not success:
                print('estimate_dirichlet_parameters: Warning:'
                      + ' Estimation did not converge for state {}. I am guessing the parameter'.format(s))
                first_moment_obs=np.mean(observations,axis=0)
                second_moment_obs=np.mean(np.square(observations),axis=0)
                guess=(first_moment_obs 
                       * (first_moment_obs[0]-second_moment_obs[0])
                       /(-first_moment_obs[0]**2 + second_moment_obs[0]))
                estimator[s,:]=guess
    return estimator    


def store_given_list_of_guesses(int d_E, int d_S, list given_list):
    cdef int e=0
    cdef list list_init_guesses_partial = []
    cdef np.ndarray[DTYPEf_t, ndim=1] nus = np.zeros(d_E, dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] alphas = np.zeros((d_E,d_S,d_E), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] betas = np.zeros((d_E,d_S,d_E), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] x = np.zeros(d_E+2*d_E*d_S*d_E,dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] y = np.zeros(1+2*d_E*d_S,dtype=DTYPEf)
    res={}
    if given_list == []:
        print("WARNING! In mle_estimation.store_given_list_of_guesses, empty list was passed. I am adding a random initial guess")
        x[0:d_E]=np.random.uniform(low=0.0, high=2.0, size=(d_E,))
        x[d_E:d_E+d_E*d_S*d_E]=np.random.uniform(low=0.0,high=1.0, size=(d_E*d_S*d_E,))
        x[d_E+d_E*d_S*d_E:len(x)]=np.random.uniform(low=1.1,high=5.0, size=(d_E*d_S*d_E,))
        given_list.append(np.array(x,copy=True))
    for e in range(d_E):
        list_init_guesses_partial = []
        for x in given_list:
            nus,alphas,betas=computation.array_to_parameters(d_E, d_S, x)
            y=computation.parameters_to_array_partial(nus[e],alphas[:,:,e],betas[:,:,e])
            list_init_guesses_partial.append(y)
        res.update({e:list_init_guesses_partial})
    return res    

cdef DTYPEi_t count_events_of_type(DTYPEi_t event_type, DTYPEi_t [:] events):
    cdef DTYPEi_t count = 0
    cdef int N = len(events)
    cdef int n = 0
    for n in prange(N,nogil=True):
        if events[n]==event_type:
            count+=1
    return count 

def preguess_ordinary_hawkes_param(int event_index,
                                   np.ndarray[DTYPEf_t, ndim=3] lt,
                                   np.ndarray[DTYPEi_t, ndim=2] count,
                                   DTYPEf_t max_imp_coef = 100.0,
                                   DTYPEf_t tol = 1.0e-6,
                                   print_res = False,
                                  ):
    assert lt.shape[1]==1
    assert count.shape[1] ==1
    cdef int d_E = lt.shape[0]
    cdef int e = event_index
    cdef np.ndarray[DTYPEf_t, ndim=1] Y = np.arange(1,1+count[e,0], dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] X = np.zeros((count[e,0],2), dtype=DTYPEf)
    X[:,0] = np.array(lt[e,0,0:count[e,0]],copy=True)
    X[:,1] = np.array(lt[e,0,0:count[e,0]],copy=True)-np.log(lt[e,0,0:count[e,0]]+1.0)
    reg=Ridge(alpha = 0.5, fit_intercept = False).fit(X,Y)
    cdef DTYPEf_t c_0 = reg.coef_[0]
    cdef DTYPEf_t c_1 = reg.coef_[1]
    if c_0 < -tol:
        print("mle_estimation.preguess_ordinary_hawkes_param: WARNING! c_0 < 0.0 after linear regression")
        print("  c_0={}, \n c_1={}".format(c_0,c_1))
#         raise ValueError("mle_estimation.preguess_ordinary_hawkes_param: Error! c_0 < 0.0 after linear regression")
    if c_1< -tol:
        print("mle_estimation.preguess_ordinary_hawkes_param: WARNING! c_1 < 0.0 after linear regression")
        print("  c_0={}, \n c_1={}".format(c_0,c_1))
#         raise ValueError("mle_estimation.preguess_ordinary_hawkes_param: Error! c_1 < 0.0 after linear regression")
    cdef DTYPEf_t base_rate = max(tol, c_0 )
    cdef np.ndarray[DTYPEf_t, ndim=2] alphas = tol*np.ones((d_E,1), dtype=DTYPEf)
    alphas[e,0] = min(max_imp_coef, max(tol, c_1) / base_rate)
    cdef np.ndarray[DTYPEf_t, ndim=2] betas = 2.0*np.ones((d_E,1), dtype=DTYPEf)
    if print_res:
        print("preguess_ordinary_hawkes_param: component e={}, base_rate={}, imp_coef[e,0,e]={}".
              format(e, base_rate,alphas[e,0]))
    return base_rate, alphas, betas
    

    
def pre_estimate_ordinary_hawkes(
    int event_index,
    int n_event_types, 
    np.ndarray[DTYPEf_t, ndim=1] times,
    np.ndarray[DTYPEi_t, ndim=1] events,
    DTYPEf_t time_start,
    DTYPEf_t time_end,
    int num_init_guesses = 3,
    parallel = False,
    DTYPEf_t max_imp_coef = 100.0,
    DTYPEf_t learning_rate = 0.0005,
    int maxiter = 50,
    DTYPEf_t tol = 1.0e-07,
    int n_states = 15, #used to reshape the results
    reshape_to_sd = False,
    return_as_flat_array = False,
    use_prange = False,
    int number_of_attempts = 3,
    int num_processes = 0,
    int batch_size = 5000,
    int num_run_per_minibatch = 1, 
):
    assert event_index<n_event_types
    assert len(times)==len(events)
    assert time_start <= time_end
    assert num_init_guesses >= 1
    assert number_of_attempts >= 1
    cdef np.ndarray[DTYPEi_t, ndim=1] states = np.zeros_like(events)
#     cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times = np.zeros((n_event_types,1,len(times)), dtype=DTYPEf) 
#     cdef np.ndarray[DTYPEi_t, ndim=2] count = np.zeros((n_event_types,1),dtype=DTYPEi)
#     labelled_times,count=computation.distribute_times_per_event_state(
#         n_event_types, 1,
#         times, events, states)
#     cdef np.ndarray [DTYPEf_t, ndim=3] lt_copy = np.array(labelled_times, copy=True)
#     cdef np.ndarray [DTYPEi_t, ndim=2] count_copy = np.array(count, copy=True)
#     preguess_base_rate, preguess_imp_coef, preguess_dec_coef =\
#     preguess_ordinary_hawkes_param(event_index,
#                                    lt_copy, count_copy,
#                                    max_imp_coef = max_imp_coef,
#                                    tol = tol,
#                                    print_res = True,
#                                   )
    cdef DTYPEf_t preguess_base_rate = float(rand())/float(RAND_MAX)
    cdef np.ndarray[DTYPEf_t, ndim=2] preguess_imp_coef = np.random.uniform(low=0.0, high=2.0, size=(n_event_types, 1))
    cdef np.ndarray[DTYPEf_t, ndim=2] preguess_dec_coef = np.random.uniform(low=1.5, high=2.5, size=(n_event_types, 1))
    cdef np.ndarray[DTYPEf_t, ndim=1] mean = preguess_imp_coef.flatten()
    cdef np.ndarray[DTYPEf_t, ndim=2] cov = np.maximum(0.01*np.amin(mean),tol)*np.eye(len(mean))
    cdef DTYPEf_t guess_base_rate = 1.0
    cdef np.ndarray[DTYPEf_t, ndim=2] guess_imp_coef = np.zeros((n_event_types,1),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] guess_dec_coef = np.ones((n_event_types,1),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] init_guess = np.ones(1+2*n_event_types*1,dtype=DTYPEf)
    cdef list list_init_guesses = []
    for n in range(num_init_guesses):
        guess_base_rate = preguess_base_rate+(rand()/float(RAND_MAX))*(1.0-preguess_base_rate)
        guess_imp_coef = np.maximum(tol,np.random.multivariate_normal(mean,cov).reshape(n_event_types, 1))
        guess_dec_coef = np.random.uniform(low=1.5,high=2.5,size=(n_event_types,1))
        init_guess = computation.parameters_to_array_partial(guess_base_rate, guess_imp_coef, guess_dec_coef)
        list_init_guesses.append(np.array(init_guess,copy=True,dtype=DTYPEf))   
    minim = minim_algo.MinimisationProcedure(
        times, events, states,
        time_start,time_end,
        n_event_types, 1,
        event_index,
        list_init_guesses = list_init_guesses,
        max_imp_coef = max_imp_coef,
        learning_rate = learning_rate,
        maxiter = maxiter,
        tol= tol,
        number_of_attempts = number_of_attempts,
        batch_size = batch_size,
        num_run_per_minibatch = num_run_per_minibatch,  
    )
    minim.prepare_batches()
    print('pre_estimate_ordinary_hawkes: event_index={} -- initialisation completed.'.format(event_index))
    cdef double run_time = -time.time()
    minim.launch_minimisation(use_prange=use_prange, parallel=parallel, num_processes = num_processes)    
    cdef np.ndarray[DTYPEf_t, ndim=1] x_min = np.array(minim.minimiser, copy=True, dtype=DTYPEf)
    base_rate,imp_coef,dec_coef=computation.array_to_parameters_partial(n_event_types, 1, x_min)
    run_time += time.time()
    if not reshape_to_sd:
        n_states = 1
    cdef DTYPEf_t nu = base_rate    
    cdef np.ndarray[DTYPEf_t, ndim=2] alphas = np.zeros((n_event_types, n_states),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] betas = np.zeros((n_event_types, n_states),dtype=DTYPEf)
    if reshape_to_sd:
        alphas = np.tile(imp_coef,n_states)
        betas = np.tile(dec_coef,n_states)
    else:
        alphas = np.array(imp_coef,copy=True,dtype=DTYPEf)
        betas = np.array(dec_coef,copy=True,dtype=DTYPEf)
    if return_as_flat_array:
        result = computation.parameters_to_array_partial(nu,alphas,betas)
    else:
        result = {'base_rate': nu, 'imp_coef': alphas, 'dec_coef': betas}
    print('pre_estimate_ordinary_hawkes: event type={}, run_time={}'.format(event_index,run_time))
    return result
    
    
def estimate_hawkes_param_partial(
    int event_index,
    int n_event_types, int n_states,
    DTYPEf_t time_start,
    DTYPEf_t time_end,
    np.ndarray[DTYPEf_t, ndim=1] times,
    np.ndarray[DTYPEi_t, ndim=1] events,
    np.ndarray[DTYPEi_t, ndim=1] states,
    list list_init_guesses = [],
    DTYPEf_t max_imp_coef = 100.0,
    DTYPEf_t learning_rate = 0.0005,
    int maxiter = 50,
    parallel=False,
    use_prange = False,
    int number_of_attempts = 3,
    DTYPEf_t tol = 1.0e-7,
    int num_processes = 0,
    int batch_size = 5000,
    int num_run_per_minibatch = 1,  
):
    assert event_index < n_event_types
    print("I am estimating hawkes parameters for the component e={}".format(event_index))
    minim = minim_algo.MinimisationProcedure(
        times, events, states,
        time_start, time_end,
        n_event_types, n_states,
        event_index,
        list_init_guesses = list_init_guesses,
        max_imp_coef = max_imp_coef,
        learning_rate = learning_rate,
        maxiter = maxiter,
        tol= tol,
        number_of_attempts = number_of_attempts,
        batch_size = batch_size,
        num_run_per_minibatch = num_run_per_minibatch,
    )
    minim.prepare_batches()
    print('mle_estimation.estimate_hawkes_param_partial: event_type {}: MinimisationProcedure has been initialised'.format(event_index))
    cdef double run_time = -time.time()
    minim.launch_minimisation(use_prange, parallel=parallel, num_processes=num_processes)
    run_time+=time.time()
    print('estimate_hawkes_power_partial, event_type={}, run_time = {}'.format(event_index,run_time))
    cdef np.ndarray[DTYPEf_t, ndim=1] x_min = np.array(minim.minimiser,copy=True,dtype=DTYPEf)
    base_rate,imp_coef,dec_coef=computation.array_to_parameters_partial(n_event_types, n_states, x_min)
    res={'component_e':event_index,
         'base_rate': base_rate,
         'imp_coef': imp_coef,
         'dec_coef': dec_coef,
         'MinimisationProcedure': minim,
         'run_time': run_time,
        }
    return res    
