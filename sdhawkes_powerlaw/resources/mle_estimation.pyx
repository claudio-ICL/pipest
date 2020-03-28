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
                 str type_of_input = 'simulated'
                ):
        print("mle_estimation.EstimProcedure is being initialised")
        self.type_of_input = type_of_input
        if not (len(times)==len(states) & len(times)==len(events)):
            raise ValueError("All shapes must agree, but input was:\n len(times)={} \n len(events)={} \n len(states)={}".format(len(times),len(events),len(states)))
        self.num_event_types = num_event_types
        self.num_states = num_states
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
        self.store_transition_probabilities()
        print("EstimProcedure has been successfully initialised")
        
    def store_runtime(self,DTYPEf_t run_time):
        self.estimation_runtime=run_time    
        
    def store_transition_probabilities(self,verbose=False):
        print('I am storing transition probabilities')
        cdef int v = int(verbose)
        self.transition_probabilities =  estimate_transition_probabilities(
            self.num_event_types,self.num_states,self.events,self.states,verbose=v)
    def set_estimation_of_hawkes_param(self,
                                       DTYPEf_t time_start, DTYPEf_t time_end,
                                       list list_of_init_guesses=[],
                                       DTYPEf_t learning_rate = 0.001,
                                       int maxiter=100,
                                       int number_of_additional_guesses=3,
                                       parallel=False, 
                                       pre_estim_ord_hawkes=False, pre_estim_parallel=False, 
                                       int number_of_attempts = 2, DTYPEf_t tol = 1.0e-07):
        print('I am setting the estimation of hawkes parameters, with time_start={}, time_end={}.'
              .format(time_start,time_end))
        print('The boundaries of arrival times are {}-{}'.format(self.times[0],self.times[len(self.times)-1]))
        self.time_start=time_start
        self.time_end = time_end
        self.given_list_of_guesses =\
        store_given_list_of_guesses(self.num_event_types,self.num_states,list_of_init_guesses)
        self.learning_rate = learning_rate
        self.maxiter=maxiter
        self.number_of_additional_guesses=number_of_additional_guesses
        self.parallel, self.pre_estim_ord_hawkes, self.pre_estim_parallel = \
        parallel, pre_estim_ord_hawkes, pre_estim_parallel
        self.number_of_attempts = number_of_attempts
        cdef list results_of_estimation = []
        self.results_of_estimation = results_of_estimation
        self.tol = tol
        
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
                number_of_attempts = self.number_of_attempts
            ))
        cdef np.ndarray[DTYPEf_t, ndim=1] guess = np.array(list_init_guesses[len(list_init_guesses)-1],dtype=DTYPEf,copy=True)   
        cdef np.ndarray[DTYPEf_t, ndim=1] new_guess = np.zeros_like(guess,dtype=DTYPEf) 
        cov =\
        np.maximum(0.05*np.amin(np.abs(guess)),self.tol)*np.eye(len(guess))
        cdef int j=0
        cdef int break_point = 1+self.num_event_types*self.num_states
        for j in range(self.number_of_additional_guesses):
            new_guess = np.random.multivariate_normal(guess,cov)
            new_guess[0:break_point] = np.maximum(0.0,new_guess[0:break_point])
            new_guess[break_point:] = np.maximum(1.01,new_guess[break_point:])
            list_init_guesses.append(new_guess)
        return list_init_guesses    
            
    def estimate_hawkes_param_partial(self, int e):
        list_init_guesses = self.prepare_list_init_guesses_partial(e)
        res = estimate_hawkes_param_partial(
            e, self.num_event_types, self.num_states,
            self.time_start, self.time_end,
            self.labelled_times, self.count,
            list_init_guesses = list_init_guesses,
            maxiter = self.maxiter,
            return_minim_proc = 0,
            parallel = self.parallel,
            number_of_attempts = self.number_of_attempts,
            tol = self.tol
        )
        self.results_of_estimation.append(res)
        
    def launch_estimation_of_hawkes_param(self, all_components=False, int e=0):
        if all_components:
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
    def create_goodness_of_fit(self):
        "type_of_input can either be 'simulated' or 'empirical'"
        self.goodness_of_fit=goodness_of_fit.good_fit(
            self.num_event_types, self.num_states,
            self.base_rates, self.hawkes_kernel.alphas, self.hawkes_kernel.betas, self.transition_probabilities,
            self.times, self.events,self.states, type_of_input=self.type_of_input
        )      
        


def estimate_transition_probabilities(int n_event_types, int n_states,
    long [:] events, long [:]  states, int verbose=1):
    cdef np.ndarray[DTYPEf_t, ndim=3] result = np.zeros((n_states, n_event_types, n_states),dtype=DTYPEf)
    cdef np.ndarray[DTYPEi_t, ndim=2] count_of_states_events = np.zeros((n_states, n_event_types),dtype=DTYPEi)
    cdef double [:,:,:] result_memview = result
    cdef long [:,:] count_memview  = count_of_states_events
    cdef int N = len(events)-1
    cdef int n
    cdef int event,state_before,state_after
    for n in prange(1,N,nogil=True):
        event = events[n]
        state_before = states[n-1]
        state_after = states[n]
        count_memview[state_before, event] += 1
        result_memview[state_before, event, state_after] += 1
    cdef long size=0   
    for x1 in range(n_states):
        for e in range(n_event_types):
            size = count_of_states_events[x1, e]
            if size > 0:
                for x2 in range(n_states):
                    result[x1, e, x2] /= size
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
                                  float tolerance=1e-8,verbose=False, DTYPEf_t epsilon=1.0e-7):
    cdef float tol = tolerance
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

#Write a new preguess function to guess base rate and imp_coeff of an ordinary hawkes process based on approximation of the formula in Theorem 2!!!

# def pre_guess_base_rate(
#         DTYPEi_t event_type,
#         np.ndarray[DTYPEf_t, ndim=1] times,
#         np.ndarray[DTYPEi_t, ndim=1] events,
#         np.float time_start,
#         np.float time_end
# ):
#     cdef int idx_start = bisect.bisect_left(times,time_start)
#     cdef int idx_end = bisect.bisect_right(times,time_end)
#     events_in_range = np.array(events[idx_start:idx_end],dtype=DTYPEi) 
#     cdef DTYPEi_t [:] events_memview = events_in_range
#     cdef long n_events = count_events_of_type(event_type,events_in_range)
#     cdef DTYPEf_t result = n_events/(time_end-time_start)
#     return result


# cdef double preguess_imp_coef_onedim(
#     double [:] times,
#     DTYPEf_t nu,
#     DTYPEf_t beta
# ) nogil:
#     cdef DTYPEf_t result = 0.0
#     cdef int n = 0
#     if ((beta>=1.9)&(beta<=2.1)):
#         for n in range(1,len(times)):
#             result += max(0.0,n/nu - times[n])/(times[n] - log(times[n]+1.0) )
#     else:
#         for n in range(1,len(times)):
#             result +=\
#             max(0.0,n/nu - times[n])\
#              /(times[n]/(beta -1.0) - 1.0 /((beta - 1.0)*(beta - 2.0))\
#                +pow(times[n]+ 1.0 , 2.0 - beta)/((beta - 1.0)*(beta - 2.0))
#               )

#     result/= (len(times)-1.0)
#     return result
 
    

# def pre_guess_impact_coefficients(int num_event_types, int num_states,
#                                   np.ndarray[DTYPEi_t, ndim = 1] events):
#     cdef np.ndarray[DTYPEf_t, ndim=3] imp_coef = np.zeros((num_event_types,num_states,num_event_types),dtype=DTYPEf)
#     cdef np.ndarray[DTYPEi_t, ndim=2] event_count = np.zeros((num_event_types,num_event_types),dtype=DTYPEi)
#     cdef int e, e1
#     idx= np.zeros((num_event_types,len(events)),dtype=np.bool)
#     for e1 in range(num_event_types):
#         idx[e1,:] = (events==e1)
#         for e in range(num_event_types):
#             event_count[e1,e] = count_events_of_type(e, events[np.roll(idx[e1,:],1)] )
#             imp_coef[e1,0,e] = event_count[e1,e]/count_events_of_type(e,events)
#     imp_coef = np.repeat(np.expand_dims(imp_coef[:,0,:],axis=1),repeats = num_states,axis=1)
#     return imp_coef


def preguess_ordinary_hawkes_param(int event_index,
                                   np.ndarray[DTYPEf_t, ndim=3] lt,
                                   np.ndarray[DTYPEi_t, ndim=2] count,
                                   DTYPEf_t max_imp_coef = 10.0,
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
    cdef np.ndarray[DTYPEf_t, ndim=2] alphas = np.zeros((d_E,1), dtype=DTYPEf)
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
    DTYPEf_t max_imp_coef = 10.0,
    DTYPEf_t learning_rate = 0.0005,
    int maxiter = 50,
    DTYPEf_t tol = 1.0e-07,
    int n_states = 15, #used to reshape the results
    reshape_to_sd = False,
    return_as_flat_array = False,
    int number_of_attempts = 3
):
    assert event_index<n_event_types
    assert len(times)==len(events)
    assert time_start <= time_end
    assert num_init_guesses >= 1
    assert number_of_attempts >= 1
    cdef np.ndarray[DTYPEi_t, ndim=1] states = np.zeros_like(events)
    cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times = np.zeros((n_event_types,1,len(times)), dtype=DTYPEf) 
    cdef np.ndarray[DTYPEi_t, ndim=2] count = np.zeros((n_event_types,1),dtype=DTYPEi)
    labelled_times,count=computation.distribute_times_per_event_state(
        n_event_types, 1,
        times, events, states)
    cdef np.ndarray [DTYPEf_t, ndim=3] lt_copy = np.array(labelled_times, copy=True)
    cdef np.ndarray [DTYPEi_t, ndim=2] count_copy = np.array(count, copy=True)
    preguess_base_rate, preguess_imp_coef, preguess_dec_coef =\
    preguess_ordinary_hawkes_param(event_index,
                                   lt_copy, count_copy,
                                   max_imp_coef = max_imp_coef,
                                   tol = tol,
                                   print_res = True,
                                  )
    cdef np.ndarray[DTYPEf_t, ndim=1] mean = preguess_imp_coef.flatten()
    cdef np.ndarray[DTYPEf_t, ndim=2] cov = np.maximum(0.01*np.amin(mean),tol)*np.eye(len(mean))
    cdef np.ndarray[DTYPEf_t, ndim=2] guess_imp_coef = np.zeros((n_event_types,1),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] guess_dec_coef = np.ones((n_event_types,1),dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] init_guess = np.ones(1+2*n_event_types*1,dtype=DTYPEf)
    cdef list list_init_guesses = []
    for n in range(num_init_guesses):
        guess_imp_coef = np.maximum(tol,np.random.multivariate_normal(mean,cov).reshape(n_event_types, 1))
        guess_dec_coef = np.random.uniform(low=1.5,high=2.5,size=(n_event_types,1))
        init_guess = computation.parameters_to_array_partial(preguess_base_rate/(n+1),guess_imp_coef,guess_dec_coef)
        list_init_guesses.append(init_guess)   
    minim = minim_algo.MinimisationProcedure(
        labelled_times,count,
        time_start,time_end,
        n_event_types, 1,
        event_index,
        list_init_guesses = list_init_guesses,
        max_imp_coef = max_imp_coef,
        learning_rate = learning_rate,
        maxiter = maxiter,
        tol= tol,
        number_of_attempts = number_of_attempts
    )
    print('pre_estimate_ordinary_hawkes: event_index={} -- initialisation completed.'.format(event_index))
    cdef double run_time = -time.time()
    minim.launch_minimisation(parallel=parallel, return_results=False)    
    cdef np.ndarray[DTYPEf_t, ndim=1] x_min = np.array(minim.minimiser, copy=True, dtype=DTYPEf)
    base_rate,imp_coef,dec_coef=computation.array_to_parameters_partial(n_event_types, 1, x_min)
    run_time += time.time()
    del minim
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
    np.ndarray[DTYPEf_t, ndim=3] labelled_times,
    np.ndarray[DTYPEi_t, ndim=2] count,
    list list_init_guesses = [],
    DTYPEf_t max_imp_coef = 100.0,
    int maxiter = 50,
    DTYPEf_t learning_rate = 0.0005,
    parallel=False,
    print_list=False,
    int number_of_attempts = 3,
    DTYPEf_t tol = 1.0e-7
):
    assert event_index < n_event_types
    print("I am estimating hawkes parameters for the component e={}".format(event_index))
    if print_list:
        print("list_init_guesses:")
        print(list_init_guesses)
    minim = minim_algo.MinimisationProcedure(
        labelled_times, count,
        time_start, time_end,
        n_event_types, n_states,
        event_index,
        list_init_guesses = list_init_guesses,
        max_imp_coef = max_imp_coef,
        learning_rate = learning_rate,
        maxiter = maxiter,
        tol= tol,
        number_of_attempts = number_of_attempts
    )
    print('mle_estimation.estimate_hawkes_param_partial: event_type {}: MinimisationProcedure has been initialised')
    cdef double run_time = -time.time()
    minim.launch_minimisation(parallel=parallel, return_results = False)
    run_time+=time.time()
    print('estimate_hawkes_power_partial, event_type={}, run_time = {}'.format(event_index,run_time))
    cdef np.ndarray[DTYPEf_t, ndim=1] x_min = np.array(minim.minimiser,copy=True,dtype=DTYPEf)
    base_rate,imp_coef,dec_coef=computation.array_to_parameters_partial(n_event_types, n_states, x_min)
    res={'component_e':event_index,
         'base_rate': base_rate,
         'imp_coeff': imp_coef,
         'dec_coef': dec_coef,
         'MinimisationProcedure': minim
        }
    return res    
    
# def produce_list_init_guesses(
#     int num_event_types, int num_states,
#     int num_additional_guesses = 4,
#     list given_list_of_guesses = [],
#     print_list=False
    
# ):
#     print("I am producing the list of initial guesses")
#     cdef list list_init_guesses = copy.copy(given_list_of_guesses)
#     cdef np.ndarray[DTYPEf_t, ndim=1] guess = np.ones(1+2*num_event_types*num_states, dtype=DTYPEf)
#     cdef j=0
#     cdef break_point = 1+num_event_types*num_states
#     for j in range(num_additional_guesses):
#         guess[0] = np.random.uniform(low=0.0, high=2.0)
#         guess[1:break_point] = np.random.uniform(low=0.0, high= 1.0, size=(break_point-1,))
#         guess[break_point:len(guess)] = np.random.uniform(low=1.1, high= 3.0, size=(break_point-1,))
#         list_init_guesses.append(np.array(guess,copy=True,dtype=DTYPEf))
#     if print_list:
#         print("list_init_guesses: \n")
#         print(list_init_guesses)    
#     return list_init_guesses


# def produce_list_init_guesses(
#     int event_index,
#     int n_event_types, int n_states,
#     np.ndarray[DTYPEf_t, ndim=1] times,
#     np.ndarray[DTYPEi_t, ndim=1] events,
#     np.ndarray[DTYPEi_t, ndim=1] states,
#     DTYPEf_t time_start,
#     DTYPEf_t time_end,
#     np.ndarray[DTYPEf_t, ndim=3] labelled_times,
#     np.ndarray[DTYPEi_t, ndim=2] count,
#     list list_init_guesses = [],
#     int num_additional_guesses = 3,
#     pre_estim_ord_hawkes = False,
#     pre_estim_parallel = False,
#     learning_rate = 0.0005,
#     maxiter = 100,
#     tol=1.0e-7,
#     print_list = False
# ):
#     print("I am producing list of initial guesses. pre_estim_ord_hawkes={}, pre_estim_parallel={}.".format(pre_estim_ord_hawkes,pre_estim_parallel))
#     cdef list list_of_guesses = copy.copy(list_init_guesses)
#     cdef list additional_guesses = []
#     cdef DTYPEf_t base_rate = 0.0
#     cdef np.ndarray[DTYPEf_t, ndim=2] imp_coef = np.zeros((n_event_types,n_states),dtype=DTYPEf)
#     cdef np.ndarray[DTYPEf_t, ndim=2] dec_coef = np.ones((n_event_types,n_states),dtype=DTYPEf)
#     cdef DTYPEf_t guess_base_rate = pre_guess_base_rate(event_index,times, events, time_start, time_end)
#     cdef np.ndarray[DTYPEf_t, ndim=2] guess_imp_coef = np.ones((n_event_types,n_states),dtype=DTYPEf)
#     cdef np.ndarray[DTYPEf_t, ndim=2] guess_dec_coef = 2*np.ones((n_event_types,n_states),dtype=DTYPEf)
#     cdef np.ndarray[DTYPEf_t, ndim=1] guess = 2*np.ones(1+2*n_event_types*n_states,dtype=DTYPEf)
#     for n in range(num_additional_guesses):
#         guess_imp_coef =\
#         np.random.uniform(low=0.001, high=n+1,size=(n_event_types,n_states))
#         guess_dec_coef =\
#         np.random.uniform(low=1.001, high=n+2,size=(n_event_types,n_states))
#         guess =\
#         computation.parameters_to_array_partial(guess_base_rate/(n+1),guess_imp_coef,guess_dec_coef)
#         additional_guesses.append(np.array(guess,dtype=DTYPEf,copy=True))
#     if pre_estim_ord_hawkes:    
#         ord_base_rate, ord_imp_coef, ord_dec_coef = pre_estimate_ordinary_hawkes(
#             event_index,
#             n_event_types,
#             times, events,
#             time_start,time_end,
#             num_init_guesses = num_additional_guesses,
#             maxiter=maxiter,
#             learning_rate=learning_rate,
#             parallel = pre_estim_parallel,
#         )
#         imp_coef = np.tile(ord_imp_coef,n_states)
#         dec_coef = np.tile(ord_dec_coef,n_states)
#         pre_estim_guess = computation.parameters_to_array_partial(ord_base_rate,imp_coef,dec_coef)
#         additional_guesses.append(np.array(pre_estim_guess,dtype=DTYPEf,copy=True))
#         cov =\
#         np.maximum(0.2*np.amin(np.abs(guess)),1.0e-6)*np.eye(len(guess))
#         break_point = 1+n_event_types*n_states
#         for n in range(num_additional_guesses):
#             guess=np.random.multivariate_normal(pre_estim_guess,cov)
#             guess[:break_point] = np.maximum(guess[:break_point],1.0e-7)
#             guess[break_point:] = np.maximum(guess[break_point:],1.01)
#             additional_guesses.append(np.array(guess,dtype=DTYPEf,copy=True))
#     cdef list new_list_of_guesses = list_of_guesses + additional_guesses        
#     if print_list:
#         print("mle_estimation.produce_list_init_guesses: new_list_of_guesses: \n")
#         print(new_list_of_guesses)
#     return new_list_of_guesses    
    
    
