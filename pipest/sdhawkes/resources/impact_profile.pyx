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

from cython.parallel import prange
cimport openmp
#openmp.omp_set_num_threads(min(16,os.cpu_count()))
#print("openmp.omp_get_max_threads(): {}".format(openmp.omp_get_max_threads()))
import time
import numpy as np
cimport numpy as np
import pandas as pd
import bisect
import copy
from libc.math cimport pow
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport isnan
from libc.math cimport ceil
# from libc.stdlib cimport rand, RAND_MAX, srand

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t
DTYPEfd = np.longdouble
DTYPEil = np.int64
ctypedef np.longdouble_t DTYPEfd_t
ctypedef np.int64_t DTYPEil_t

import computation
import minimisation_algo as minim_algo
import mle_estimation as mle_estim
import goodness_of_fit
    
import matplotlib.pyplot as plt    

class Multivariate_sdHawkesProcess:
    def __init__(self,
                 np.ndarray[DTYPEf_t, ndim=1] times,
                 np.ndarray[DTYPEi_t, ndim=1] events,
                 np.ndarray[DTYPEi_t, ndim=1] states,
                 np.ndarray[DTYPEf_t, ndim=1] base_rates,
                 np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
                 np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
                 np.ndarray[DTYPEf_t, ndim=3] trans_prob):
        assert len(times)==len(events)
        assert len(times)==len(states)
        assert base_rates.shape[0]==impact_coefficients.shape[0]
        assert impact_coefficients.shape[0]==decay_coefficients.shape[0]
        assert impact_coefficients.shape[0]==trans_prob.shape[1]
        assert impact_coefficients.shape[1]==trans_prob.shape[0]
        self.times=times
        self.events=events
        self.states=states
        self.base_rates=base_rates
        self.impact_coefficients=impact_coefficients
        self.decay_coefficients=decay_coefficients
        self.imp_dec_ratio=impact_coefficients/(decay_coefficients-1.0)
        self.transition_probabilities=trans_prob
        self.num_event_types=base_rates.shape[0]
        self.num_states=trans_prob.shape[0]
        cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times =\
        np.zeros((self.num_event_types,self.num_states,len(times)),dtype=DTYPEf)
        cdef np.ndarray[DTYPEi_t, ndim=2] count =\
        np.zeros((self.num_event_types,self.num_states),dtype=DTYPEi)
        labelled_times,count = computation.distribute_times_per_event_state(
            self.num_event_types, self.num_states,
            times, events,states, len_labelled_times = len(times))
        self.labelled_times=labelled_times
        self.count=count

class Univariate_Ordinary_HawkesProcess:
    def __init__(self,
                 np.ndarray[DTYPEf_t, ndim=1] times,
                 time_start=None, time_end=None,
                 int num_init_guesses = 3,
                 parallel = False,
                 DTYPEf_t max_imp_coef = 100.0,
                 DTYPEf_t learning_rate = 0.0005,
                 int maxiter = 50,
                 DTYPEf_t tol = 1.0e-07,
                 use_prange = False,
                 int number_of_attempts = 3,
                 int num_processes = 0,
                 int batch_size = 5000,
                 int num_run_per_minibatch = 1,
                ):
        self.sampled_times=times
        cdef DTYPEf_t t_first, t_last
        if time_start == None:
            t_first = copy.copy(times[0])
        else:
            t_first=copy.copy(time_start)
        if time_end == None:
            t_last = copy.copy(times[len(times)-1])
        else:
            t_last=copy.copy(time_end)
        self.time_start=t_first
        self.time_end=t_last
        cdef np.ndarray[DTYPEi_t, ndim=1] events = np.zeros_like(times, dtype=DTYPEi)
        base_rate, imp_coef, dec_coef =\
            mle_estim.estimate_ordinary_hawkes(
            0,
            1,
            times,
            events,
            time_start,
            time_end,
            num_init_guesses = num_init_guesses,
            parallel = parallel,
            max_imp_coef = max_imp_coef ,
            learning_rate = learning_rate ,
            maxiter =maxiter ,
            tol = tol,
            use_prange = use_prange,
            number_of_attempts = number_of_attempts,
            num_processes = num_processes,
            batch_size = batch_size,
            num_run_per_minibatch = num_run_per_minibatch)
        self.base_rate=base_rate
        self.imp_coef=imp_coef[0,0]
        self.dec_coef=dec_coef[0,0]
    def compute_expectation(self,DTYPEf_t eval_time):
        cdef DTYPEf_t result = self.base_rate*eval_time
        alpha=self.imp_coef
        beta=self.dec_coef
        nu=self.base_rate
        if np.isclose(beta,2.0):
            result += nu*(alpha*eval_time - alpha*log(eval_time+1))
        else:
            result += nu*(alpha*eval_time/(beta-1)
                          -alpha/((beta-1)*(beta-2)) + alpha*pow(eval_time+1,2-beta)/((beta-1)*(beta-2))
                         )
        print("Univariate_Ordinary_Hawkes.compute_exepctation: {}".format(result))
        return result
    def create_goodness_of_fit(self,):
        cdef np.ndarray[DTYPEf_t, ndim=3] phi = np.ones((1,1,1),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] nu = np.array(self.base_rate,copy=True,dtype=DTYPEf).reshape((1,))
        cdef np.ndarray[DTYPEf_t, ndim=3] alpha = np.array(self.imp_coef,copy=True,dtype=DTYPEf).reshape((1,1,1))
        cdef np.ndarray[DTYPEf_t, ndim=3] beta = np.array(self.dec_coef,copy=True,dtype=DTYPEf).reshape((1,1,1))
        self.goodness_of_fit=goodness_of_fit.good_fit(
            1,1,nu,alpha,beta, phi, self.sampled_times,
            np.zeros(len(self.sampled_times),dtype=DTYPEi), np.zeros(len(self.sampled_times),dtype=DTYPEi))
        

class Impact():
    def __init__(self,liquidator,
                 state_enc,
                 ):
        self.liquidator=liquidator
        self.state_enc=state_enc
        self.weakly_defl_states=state_enc.weakly_deflationary_states
        cdef np.ndarray[DTYPEi_t, ndim=1] ar_w_defl = np.array(
                state_enc.weakly_deflationary_states,dtype=DTYPEi)
        self.array_weakly_defl_states=ar_w_defl
        self.num_weakly_defl_states = len(self.weakly_defl_states)
        self.stored_onesided_impact_profiles=[]
    def store_sdhawkes(self,
        np.ndarray[DTYPEf_t, ndim=1] base_rates,
        np.ndarray[DTYPEf_t, ndim=3] impact_coefficients,
        np.ndarray[DTYPEf_t, ndim=3] decay_coefficients,
        np.ndarray[DTYPEf_t, ndim=3] trans_prob,
        np.ndarray[DTYPEf_t, ndim=1] times,
        np.ndarray[DTYPEi_t, ndim=1] events,
        np.ndarray[DTYPEi_t, ndim=1] states,
        ):
        assert len(times)==len(events)
        assert len(events)==len(states)
        self.num_event_types=int(len(base_rates))
        self.num_states=int(impact_coefficients.shape[1])
        self.sdhawkes = Multivariate_sdHawkesProcess(
            times,events,states,
            base_rates, impact_coefficients,
            decay_coefficients, trans_prob)
        idx=np.array(times>=0.0, dtype=np.bool)
        self.times=np.array(times[idx], copy=True)
        self.events=np.array(events[idx], copy=True)
        self.states=np.array(states[idx], copy=True)
        self.labelled_times, self.count = computation.distribute_times_per_event_state(
            self.num_event_types, self.num_states,
            self.times, self.events, self.states)

    def produce_reduced_weakly_defl_pp(self,
                 time_start=None, time_end=None,
                 int num_init_guesses = 3,
                 parallel = False,
                 DTYPEf_t max_imp_coef = 100.0,
                 DTYPEf_t learning_rate = 0.0005,
                 int maxiter = 50,
                 DTYPEf_t tol = 1.0e-07,
                 use_prange = False,
                 int number_of_attempts = 3,
                 int num_processes = 0,
                 int batch_size = 5000,
                 int num_run_per_minibatch = 2,
                ):
        if time_start==None:
            time_start=self.times[0]
        if time_end==None:
            time_end=self.times[len(self.times)-1]
        idx = np.logical_and(
            np.isin(self.states,self.weakly_defl_states),
            self.events>=1
        )        
        cdef np.ndarray[DTYPEf_t, ndim=1] reduced_times = np.array(self.times[idx],copy=True,dtype=DTYPEf)
        self.reduced_weakly_defl_pp = Univariate_Ordinary_HawkesProcess(
            reduced_times, time_start, time_end,
            num_init_guesses = num_init_guesses,
            parallel = parallel,
            max_imp_coef = max_imp_coef ,
            learning_rate = learning_rate ,
            maxiter =maxiter ,
            tol = tol,
            use_prange = use_prange,
            number_of_attempts = number_of_attempts,
            num_processes = num_processes,
            batch_size = batch_size,
            num_run_per_minibatch = num_run_per_minibatch)
        print('impact: reduced_weakly_defl_pp has been produced, with the following coefficients')
        message=' nu={}'.format(self.reduced_weakly_defl_pp.base_rate)
        message+=' alpha={}'.format(self.reduced_weakly_defl_pp.imp_coef)
        message+=' beta={}'.format(self.reduced_weakly_defl_pp.dec_coef)
        print(message)
    def evaluate_compensator_of_weakly_defl_pp(self, DTYPEf_t eval_time):
        return compute_compensator_of_weakly_defl_pp(
            eval_time,
            self.num_event_types,
            self.num_states,
            self.num_weakly_defl_states,
            self.array_weakly_defl_states,
            self.times, #times is supposed to start at 0.0, which is also the time when liquidation starts
            self.states,
            self.labelled_times,
            self.count,
            self.sdhawkes.base_rates,
            self.sdhawkes.imp_dec_ratio,
            self.sdhawkes.decay_coefficients,
            self.sdhawkes.transition_probabilities,
            self.liquidator.termination_time
        )
    
    def evaluate_impact_profile(self, DTYPEf_t eval_time, display_message=True):
        # It is assumed that liquidation starts at zero and times>=0
        cdef DTYPEf_t numerator =\
        compute_numerator(
            eval_time,
            self.liquidator.termination_time,
            self.times,
            self.states,
            self.labelled_times,
            self.count,
            self.liquidator.base_rate,
            self.sdhawkes.imp_dec_ratio[1,:,:], #sell market orders impacting lob events
            self.sdhawkes.decay_coefficients[1,:,:], #sell market orders impacting lob events
            self.sdhawkes.transition_probabilities,
            self.array_weakly_defl_states,
            self.sdhawkes.num_event_types,
            self.sdhawkes.num_states,
            self.num_weakly_defl_states,
            self.liquidator.num_orders
        )
        cdef DTYPEf_t sqrt_denominator =\
        (self.evaluate_compensator_of_weakly_defl_pp(eval_time)
         -self.sdhawkes.base_rates[0]*min(eval_time,self.liquidator.termination_time))
        sqrt_denominator -= self.reduced_weakly_defl_pp.compute_expectation(eval_time)
        cdef DTYPEf_t denominator = sqrt_denominator*sqrt_denominator
        if display_message:
            message='evaluate_impact_profile:'
            message+='  numerator={}'.format(numerator)
            message+='  denominator={}'.format(denominator)
            print(message)
        if np.isclose(numerator, 0.0, atol=1.0e-12) and np.isclose(denominator, 0.0, atol=1.0e-12):
            return 0.0
        return numerator/denominator
    def compute_imp_profile_history(self, DTYPEf_t t0,  DTYPEf_t t1, int num_extra_eval_points = 10):
        idx_liquidator = np.logical_and(self.events==0,
                np.logical_and(t0<=self.times, self.times<=t1))
        cdef np.ndarray[DTYPEf_t, ndim=1] eval_points = np.atleast_1d(np.array(self.times[idx_liquidator],copy=True))
        cdef np.ndarray[DTYPEf_t, ndim=1] extra_eval_points =\
                np.linspace(t0, t1, num=max(2,num_extra_eval_points), dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=2] eval_times =\
        np.sort(np.concatenate([eval_points,extra_eval_points],axis=0),axis=0).reshape(-1,1)
        cdef np.ndarray[DTYPEf_t, ndim=2] imp_profile = np.zeros_like(eval_times)
        message='\ncompute_imp_profile_history:\n'
        message+='t0={}, t1={}, '.format(t0,t1)
        message+='eval_times.shape=({},{}),'.format(eval_times.shape[0],eval_times.shape[1])
        print(message)
        imp_profile[:,0] = np.apply_along_axis(self.evaluate_impact_profile,1,eval_times)
        cdef np.ndarray[DTYPEf_t, ndim=2] result = np.concatenate([eval_times,imp_profile], axis=1)
        return result
    def store_impact_profile(self, DTYPEf_t t0, DTYPEf_t t1, int num_extra_eval_points = 10):
        impact_profile = self.compute_imp_profile_history(
                t0, t1, num_extra_eval_points)
        self.stored_onesided_impact_profiles.append(impact_profile)
        df=pd.DataFrame({'time': impact_profile[:,0], 'impact': impact_profile[:,1]})
        try:
            self.df_impact_profile=pd.concat([self.df_impact_profile, df])
        except (NameError, AttributeError):
            self.df_impact_profile = df
    def store_bm_impact(self, 
            int density_of_eval_points=1000):
        profile, intensity = self.compute_bm_impact(
                density_of_eval_points=density_of_eval_points)
        df_1=pd.DataFrame({'time': profile[:,0], 'impact': profile[:,1]})
        df_2=pd.DataFrame({'time': intensity[:,0], 'intensity': intensity[:,1]})
        df=df_1.merge(df_2,how='inner',on='time')
        self.bacrymuzy=df
        self.bm_impact_profile=profile
        self.bm_impact_intensity=intensity
    def compute_bm_impact(self,
        int density_of_eval_points=1000):
        cdef np.ndarray[DTYPEf_t, ndim=2] history_of_profile_intensity = \
        computation.compute_history_of_bm_profile_intensity(
            self.sdhawkes.num_event_types,
            self.sdhawkes.num_states,
            self.state_enc.deflationary_states,
            self.state_enc.inflationary_states,
            self.liquidator.start_time,
            self.liquidator.termination_time,
            self.sdhawkes.times,
            self.sdhawkes.events,
            self.sdhawkes.states,
            self.sdhawkes.base_rates,
            self.sdhawkes.impact_coefficients,
            self.sdhawkes.decay_coefficients,
            self.sdhawkes.transition_probabilities,
            density_of_eval_points=density_of_eval_points
        )
        cdef np.ndarray[DTYPEf_t, ndim=2] profile =\
                computation.compute_bm_impact_profile(history_of_profile_intensity)
        return profile, history_of_profile_intensity
    def plot_bm_impact(self,
            time_start=None, time_end=None,
            plot_intensity=False,
            plot_events=False,
            save_fig=False, path=path_pipest, name='bm_impact_profile',
            plot=True
            ):
        if time_start==None:
            time_start=self.liquidator.start_time
        if time_end==None:
            time_end=self.liquidator.termination_time
        profile=self.bm_impact_profile 
        idx_start=bisect.bisect_left(profile[:,0], time_start)
        idx_end=bisect.bisect(profile[:,0],time_end)
        impact_profile=profile[idx_start:idx_end,:]
        fig = plt.figure(figsize=(8,4))
        ax_profile=fig.add_subplot(111)
        if plot_events:
            idx=np.logical_and(self.sdhawkes.times>=time_start,
                    self.sdhawkes.times<=time_end)
            times=self.sdhawkes.times[idx]
            events=self.sdhawkes.events[idx]

        if plot_intensity:
            intensity=self.bm_impact_intensity
            idx_start_intensity=bisect.bisect_left(intensity[:,0], time_start)
            idx_end_intensity=bisect.bisect(intensity[:,0], time_end)
            impact_intensity = intensity[idx_start_intensity:idx_end_intensity,:]
            ax_intensity=ax_profile.twinx()
            ax_profile.plot(
                    impact_intensity[:,0], impact_intensity[:,1],
                    color=[0,0.4,0.4,0.4],label='profile intensity')
        ax_profile.plot(impact_profile[:,0], impact_profile[:,1],
                color='green',label='impact_profile',linewidth=2.0)
        ax_profile.set_xlabel('time')
        ax_profile.set_ylabel('impact')
        #ax_profile.set_yticks([])
        ax_profile.legend()
        fig.suptitle('Impact profile a` la Bacry-Muzy')
        if save_fig:    
            fname=path+'/'+name
            plt.savefig(fname)
        if plot:
            plt.show()   
        
        
        
"""
The following are the functions utilised by the classes above
"""

def compute_compensator_of_weakly_defl_pp(
    DTYPEf_t eval_time,
    int num_event_types,
    int num_states,
    int num_weakly_defl_states,
    long [:] weakly_defl_states,
    double [:] times,
    long [:] states,
    double [:,:,:] labelled_times,
    long [:,:] count,
    double [:] base_rates,
    double [:,:,:] imp_dec_ratio,
    double [:,:,:] beta,
    double [:,:,:] phi,
    DTYPEf_t liquid_termination_time,
): # It is assumed that liquidation starts at time 0.0 and all times are non-negative
    cdef int len_times=len(times)
    cdef int n=0, x=0, x1=0, e=1, e1=0, j=0
    cdef DTYPEf_t upper_time=0.0, T_n=copy.copy(times[0])#, summand=0.0, total=0.0
    cdef DTYPEf_t result = 0.0
    cdef int idx_state = 0
    cdef np.ndarray[DTYPEf_t, ndim=3] total = np.zeros((num_weakly_defl_states, num_event_types, len_times-1), dtype=DTYPEf)
    cdef DTYPEf_t [:,:,:] total_memview = total
    cdef np.ndarray[DTYPEf_t, ndim=5] summand = np.zeros((num_weakly_defl_states, num_event_types, len_times-1, num_event_types, num_states), dtype=DTYPEf)
    cdef DTYPEf_t [:,:,:, :,:] summand_memview = summand
    for idx_state in prange(num_weakly_defl_states, nogil=True):
        x=weakly_defl_states[idx_state]
        for e in range(1,num_event_types):
            for n in range(len_times-1):
                T_n = times[n]
                if T_n < eval_time:
                    upper_time=min(times[n+1],eval_time)
                    for e1 in range(num_event_types):
                        for x1 in range(num_states):
                            for j in range(count[e1,x1]):
                                if labelled_times[e1,x1,j] < T_n:
                                    summand_memview[idx_state,e,n,e1,x1] += pow(T_n - labelled_times[e1,x1,j] + 1.0, 1-beta[e1,x1,e])
                                    summand_memview[idx_state,e,n,e1,x1] += -pow(upper_time - labelled_times[e1,x1,j] + 1.0, 1-beta[e1,x1,e])
                                else:
                                    break
                            summand_memview[idx_state,e,n,e1,x1] *= imp_dec_ratio[e1,x1,e]
                            total_memview[idx_state,e,n] += summand_memview[idx_state,e,n,e1,x1]
                    total_memview[idx_state,e,n] +=  base_rates[e]*(upper_time-T_n)
                    total_memview[idx_state,e,n] *= phi[states[n],e,x]
                    result += total_memview[idx_state,e,n]
                else:
                    break
    result += base_rates[0]*min(eval_time,liquid_termination_time)
    return result



def find_current_state(DTYPEf_t t, 
                       double [:] times,
                       long [:] states,
                       int len_states):
    cdef int idx=bisect.bisect_right(times,t)
    idx=min(len_states-1,idx)
    cdef DTYPEi_t current_state=copy.copy(states[idx])
    return current_state

def compute_cdf_execution_horizon(DTYPEf_t eval_time,
                                  DTYPEf_t base_rate,
                                  DTYPEf_t size_child_order,
                                  DTYPEf_t initial_inventory):
    """
    It is assumed that the arrival of liquidator's orders follow a Poisson process with intensity equal to the base rate and that each child order has the constant fixed specified size.
    """
    
    cdef int n_orders = np.int(ceil(initial_inventory/size_child_order))
    cdef DTYPEf_t nu_t = base_rate*eval_time
    cdef int j=0
    cdef double P = exp(-nu_t)
    cdef long factorial=1
    with nogil:
        for j in range(1,min(14,n_orders+1)):
            factorial*=j
            P+= exp(-nu_t)*pow(nu_t,j)/factorial
    cdef DTYPEf_t cdf=1.0-P
    return cdf

def evaluate_anti_cdf_execution_horizon(DTYPEf_t eval_time,
                                  DTYPEf_t base_rate,
                                  int n_orders):
    return compute_anti_cdf_execution_horizon(eval_time, base_rate, n_orders)

cdef double compute_anti_cdf_execution_horizon(DTYPEf_t eval_time,
                                  DTYPEf_t base_rate,
                                  int n_orders
                                  ) nogil:
    """
    It is assumed that the arrival of liquidator's orders follow a Poisson process with intensity equal to the base rate and that each child order has a fixed constant size.
    """
    cdef DTYPEf_t nu_t = base_rate*eval_time
    cdef int j=0
    cdef double P = exp(-nu_t)
    cdef long factorial=1
    for j in range(1,min(14,n_orders+1)):
        factorial*=j
        P+= exp(-nu_t)*pow(nu_t,j)/factorial
    return P

cdef double compute_numerator(
    DTYPEf_t eval_time,
    DTYPEf_t termination_time,
    np.ndarray[DTYPEf_t, ndim=1] times,
    np.ndarray[DTYPEi_t, ndim=1] states,
    np.ndarray[DTYPEf_t, ndim=3] labelled_times,
    np.ndarray[DTYPEi_t, ndim=2] count,
    DTYPEf_t base_rate,
    np.ndarray[DTYPEf_t, ndim=2] imp_dec_ratio_one,
    np.ndarray[DTYPEf_t, ndim=2] beta_one,
    np.ndarray[DTYPEf_t, ndim=3] phi,
    long [:] weakly_defl_states,
    int num_event_types,
    int num_states,
    int num_weakly_defl_states,
    int num_orders,
):
    cdef DTYPEf_t result = 0.0, sqrt_result = 0.0
    sqrt_result = compute_first_summand_of_numerator(
        eval_time,termination_time, base_rate,num_orders)
    print("Impact.compute_numerator: first summand = {}".format(sqrt_result))
    sqrt_result += compute_second_summand_of_numerator(
        eval_time, times, states, labelled_times, count,
        base_rate, imp_dec_ratio_one, beta_one, phi,
        weakly_defl_states, num_event_types,
        num_states, num_weakly_defl_states, num_orders
    )
    print("Impact.compute_numerator: sqrt_result = {}".format(sqrt_result))
    result = sqrt_result*sqrt_result
    return result
    

cdef double compute_first_summand_of_numerator(
    DTYPEf_t eval_time, DTYPEf_t termination_time,
    DTYPEf_t base_rate,  int num_orders):
    cdef DTYPEf_t nu_zero_t = base_rate*eval_time
    cdef int N = max(0,min(15,num_orders-1))
    cdef int n =  0
    cdef long factorial = 1
    cdef DTYPEf_t result = (num_orders)*exp(-nu_zero_t)
    with nogil:
        for n in range(1,N+1):
            factorial*=n
            result+=(num_orders - n)*pow(nu_zero_t,n)*exp(-nu_zero_t)/factorial
    result+= base_rate*min(termination_time, eval_time) - num_orders
    return result

cdef double compute_second_summand_of_numerator(
    DTYPEf_t eval_time,
    double [:] times,
    long [:] states,
    double [:,:,:] labelled_times,
    long [:,:] count,
    DTYPEf_t base_rate,
    double [:,:] imp_dec_ratio_one,
    double [:,:] beta_one,
    double [:,:,:] phi,
    long [:] weakly_defl_states,
    int num_event_types,
    int num_states,
    int num_weakly_defl_states,
    int num_orders,
):
    cdef np.ndarray[DTYPEi_t, ndim=1] factorial = cache_factorials(num_orders)
    cdef long [:] factorial_memview = factorial
    cdef np.ndarray[DTYPEf_t, ndim=3] summands = np.zeros(
        (num_states,num_event_types,num_states),dtype=DTYPEf)
    cdef double[:,:,:] summands_memview = summands
    cdef DTYPEf_t result=0.0
    cdef DTYPEi_t x=0
    cdef int e=1, x1=0, idx_weakly_defl_state = 0, n=0
    cdef int len_times = len(times)
    
    
    for idx_weakly_defl_state in prange(num_weakly_defl_states,nogil=True):
        x=weakly_defl_states[idx_weakly_defl_state]
        for e in range(1,num_event_types):
            for  x1 in range(num_states):
                summands_memview[x,e,x1] = compute_fn(eval_time, times[0], times[1], states[0],
                    times,states,labelled_times,count,
                    base_rate,beta_one[x1,e],phi,
                    0,x1, e, x,
                    num_orders, factorial_memview
                )
                for n in range(1,len_times):
                    if times[n]<eval_time:
                        summands_memview[x,e,x1] += compute_fn(
                            eval_time, times[n], times[n+1], states[n],
                            times, states, labelled_times, count,
                            base_rate, beta_one[x1,e], phi,
                            n, x1, e, x,
                            num_orders,
                            factorial_memview
                        )
                        summands_memview[x,e,x1] += -compute_gn(
                            eval_time, times[n], times[n+1], times[n-1], states[n], states[n-1],
                            base_rate, beta_one[x1,e],phi,
                            x1, e, x, num_orders, factorial_memview
                        )
                    else:
                        break
                result += imp_dec_ratio_one[x1,e]*summands_memview[x,e,x1]
    return result            
                
                    

cdef double compute_fn(DTYPEf_t eval_time,
               DTYPEf_t T_n,
               DTYPEf_t T_np1,
               DTYPEi_t X_n,
               double [:] times,
               long [:] states,
               double [:,:,:] labelled_times,
               long [:,:] count,
               DTYPEf_t base_rate,
               DTYPEf_t beta_one_x1_e,
               double [:,:,:] phi,
               int n,
               int x1, int e, int x,
               int num_orders,
               long [:] factorial,        
               int discretisation_hyperparam = 10
              ) nogil:
    cdef DTYPEf_t fn = 0.0
    cdef DTYPEf_t upper_time = min(T_np1,eval_time)
    #assert upper_time >= T_n
    cdef DTYPEf_t integral = 0.0
    cdef DTYPEf_t P_T_geq_u = 0.0, nu_0_times_u = 0.0
    cdef DTYPEf_t du = 0.0, u=0.0
    cdef int M = discretisation_hyperparam
    cdef int j=0, m=0, k=0
    for j in range(n):
        integral=0.0
        if ( (times[j+1]-times[j]) > 1.0e-15 ):
            du=(times[j+1]-times[j])/M
            for m in range(M):
                u=times[j]+m*du
                nu_0_times_u = base_rate*u
                P_T_geq_u = exp(-nu_0_times_u)
                for k in range(1,min(14,num_orders+1)):
                    #assert factorial[4]>0
                    P_T_geq_u+=exp(-nu_0_times_u)*pow(nu_0_times_u,k)/factorial[k]        
                integral+= du*P_T_geq_u*(
                    - pow(upper_time-u+1.0, 1-beta_one_x1_e) + pow(T_n - u +1.0, 1-beta_one_x1_e)
                )
        fn-=phi[states[j],0,x1]*integral
    fn*=base_rate    
       
    for j in range(count[0,x1]):
        if labelled_times[0,x1,j] < T_n:
            fn-=pow(upper_time - labelled_times[0,x1,j]+1.0, 1-beta_one_x1_e)
            fn+=pow(T_n - labelled_times[0,x1,j]+1.0, 1-beta_one_x1_e)
        else:
            break
    fn*=phi[states[n],e,x]
    return fn

cdef double compute_gn(DTYPEf_t eval_time,
               DTYPEf_t T_n,
               DTYPEf_t T_np1,
               DTYPEf_t T_nm1,
               DTYPEi_t X_n,
               DTYPEi_t X_nm1,
               DTYPEf_t base_rate,
               DTYPEf_t beta_one_x1_e,
               double [:,:,:] phi,
               int x1,
               int e,
               int x,
               int num_orders,
               long [:] factorial,        
               int discretisation_hyperparam = 30
              ) nogil:
    cdef DTYPEf_t upper_time = min(T_np1,eval_time)
    cdef DTYPEf_t integral = 0.0
    cdef DTYPEf_t P_T_geq_u = 0.0, nu_0_times_u=0.0
    cdef DTYPEf_t du = 0.0, u=0.0
    cdef int M = discretisation_hyperparam
    cdef int m=0, k=0
    if (upper_time - T_n) > 1.0e-15 :
        du = (upper_time - T_n)/M
        for m in range(M):
            u = T_n + m*du
            nu_0_times_u = base_rate*u
            P_T_geq_u = exp(-nu_0_times_u)
            for k in range(1,min(14,num_orders+1)):
                #assert factorial[k] > 0
                P_T_geq_u+=exp(-nu_0_times_u)*pow(nu_0_times_u,k)/factorial[k]
            integral+=du*P_T_geq_u*(
                1 - pow(T_np1 - u +1, 1-beta_one_x1_e)
            )
    return base_rate*phi[X_n,0,x1]*phi[X_nm1,e,x]*integral

def cache_factorials(int n):
    cdef int k =0
    cdef int N=min(16,n)
    cdef np.ndarray[DTYPEi_t, ndim=1] factorial = np.ones(N, dtype=DTYPEi)
    for k in range(1,N):
        factorial[k] = factorial[k-1]*k
    return factorial
