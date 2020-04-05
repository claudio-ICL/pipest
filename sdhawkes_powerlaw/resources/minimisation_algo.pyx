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
import multiprocessing as mp
import numpy as np
cimport numpy as np
import bisect
import copy
from libc.math cimport isnan
from libc.math cimport pow
from libc.math cimport exp
from libc.math cimport log
from libc.stdlib cimport rand, RAND_MAX


DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t


import computation 

class MinimisationProcedure:
    def __init__(self,
                 np.ndarray[DTYPEf_t, ndim=1] times,
                 np.ndarray[DTYPEi_t, ndim=1] events,
                 np.ndarray[DTYPEi_t, ndim=1] states,
                 DTYPEf_t time_start,
                 DTYPEf_t time_end,
                 int num_event_types,
                 int num_states,
                 int event_type,
                 list list_init_guesses=[],
                 DTYPEf_t max_imp_coef = 100.0,
                 DTYPEf_t learning_rate = 0.001,
                 int maxiter = 100,
                 DTYPEf_t tol = 1.0e-6,
                 int  number_of_attempts = 3,
                 int batch_size = 5000,
                 int num_run_per_minibatch = 1,     
                ):
        print("MinimisationProcedure is being initialised: event_type={}, learning_rate={}, maxiter={}".format(event_type,learning_rate,maxiter))
        assert time_start<=time_end
        assert len(times)==len(events)
        assert len(times)==len(states)
        if list_init_guesses == []:
            print("No initial guess provided")
            raise ValueError("User needs to provide initial position for gradient descent")
        assert number_of_attempts >= 1
        self.num_event_types = num_event_types
        self.num_states = num_states
        self.event_type = event_type
        self.time_start = copy.copy(time_start)
        self.time_end = copy.copy(time_end)
        self.times=times
        self.events=events
        self.states=states
        self.list_init_guesses = copy.copy(list_init_guesses)
        self.max_imp_coef = copy.copy(max_imp_coef)
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.tol = tol   
        self.number_of_attempts = number_of_attempts
        self.batch_size =  batch_size 
        self.num_run_per_minibatch =  num_run_per_minibatch 
    def launch_minimisation(self, parallel=False, int num_processes = 0):
        print('I am launching minimisation. Number of initial guesses={}, parallel={}'.format(len(self.list_init_guesses),parallel))
        cdef list results = []
        self.parallel_minimisation = parallel
        if parallel:            
            results = parallel_minimisation(
                self.event_type, self.num_event_types, self.num_states,
                self.times, self.events, self.states,
                self.time_start, self.time_end,
                self.list_init_guesses,
                self.max_imp_coef,
                self.learning_rate, self.tol, self.maxiter, number_of_attempts = self.number_of_attempts,
                batch_size =  self.batch_size, 
                num_run_per_minibatch =  self.num_run_per_minibatch, 
                num_processes = num_processes
            )
        else:        
            print("I am performing gradient descent serially")    
            solver=map(lambda x: grad_descent_partial(
                               self.event_type, self.num_event_types, self.num_states,
                               self.times, self.events, self.states,
                               self.time_start, self.time_end,
                               x,
                               self.max_imp_coef,
                               self.learning_rate, self.tol, self.maxiter,
                               number_of_attempts = self.number_of_attempts,
                               batch_size =  self.batch_size,
                               num_run_per_minibatch =  self.num_run_per_minibatch),
                       self.list_init_guesses)
            results=list(solver)
        print("MinimisationProcedure: minimisation finished")
        self.results=results
        cdef list list_f_min=[]
        for n in range(len(results)):
            list_f_min.append(np.atleast_1d(results[n].get('f_min')))
        arr_f_min=np.squeeze(np.array(list_f_min))
        index_best=np.argmin(arr_f_min)
        best=results[index_best]
        self.best_result = best
        self.minimiser = best.get('x_min')
        self.minimum = best.get('f_min')
        self.grad_descent_history = best.get('f')

def parallel_minimisation(int event_type, int num_event_types, int num_states,
                          np.ndarray[DTYPEf_t, ndim=1] times,
                          np.ndarray[DTYPEi_t, ndim=1] events,
                          np.ndarray[DTYPEi_t, ndim=1] states,
                          DTYPEf_t time_start, DTYPEf_t time_end,
                          list list_init_guesses,
                          DTYPEf_t max_imp_coef = 100.0,
                          DTYPEf_t learning_rate = 0.001, DTYPEf_t tol=1.0e-07,
                          int maxiter=100, int number_of_attempts = 3,
                          int batch_size = 5000, int num_run_per_minibatch = 1,     
                          int num_processes = 0, 
                         ):
    cdef int tot_tasks = len(list_init_guesses)
    if num_processes <= 0:
        num_processes = max(1,min(mp.cpu_count(),tot_tasks))
    else:
        num_processes = max(1,min(num_processes, len(list_init_guesses)))
    cdef int max_num_tasks = max(1,tot_tasks//num_processes)    
    if tot_tasks%num_processes>=1:
        max_num_tasks+=1
    cdef list results = []
    cdef list async_res = []
    def store_res(res):
        try:
            results.append(res)
        except:
            print("storing result fails")
            pass
    def error_handler(_):
        print("parallel_minimisation: error in worker")
        pass
    print('I am performing parallel gradient descent. num_processes={} '.format(num_processes))
    with mp.Pool(processes = num_processes, maxtasksperchild = max_num_tasks) as pool:
        for x in list_init_guesses:
            async_res.append(
                pool.apply_async(
                    grad_descent_partial,
                    args=(
                           event_type, num_event_types, num_states,
                           times, events, states,
                           time_start, time_end,
                           x,
                           max_imp_coef,
                           learning_rate, tol, maxiter, number_of_attempts,
                           batch_size, num_run_per_minibatch     
                    ),
                    callback=store_res,
                    error_callback=error_handler,
                )
            )        
        pool.close()
        pool.join()
        pool.terminate()
    cdef int i=0
    for i in range(len(async_res)):
        async_res[i].successful()
    return results     
    
def grad_descent_partial(int event_type, int num_event_types, int num_states,
        np.ndarray[DTYPEf_t, ndim=1] times,
        np.ndarray[DTYPEi_t, ndim=1] events,
        np.ndarray[DTYPEi_t, ndim=1] states,                 
        DTYPEf_t time_start,
        DTYPEf_t time_end,
        np.ndarray[DTYPEf_t,ndim=1] initial_guess,
        DTYPEf_t max_imp_coef = 100.0,                 
        DTYPEf_t learning_rate=0.001,
        DTYPEf_t tol = 1.0e-7,
        int maxiter = 100,
        int number_of_attempts = 2,
        int batch_size = 5000,
        int num_run_per_minibatch = 1,                 
):  
    cdef list list_of_batches = []
    cdef list results_batches = []
    cdef int num_of_batches = max(1,len(times)//batch_size)
    if (len(times)//batch_size>=1) & (len(times)%batch_size > 0.75*batch_size):
        num_of_batches += 1
    cdef DTYPEf_t t_0 = 0.0, t_1=0.0    
    cdef int k=0, idx_0=0, idx_1=1
    cdef int len_lt=0, num_at=0
    for k in range(num_of_batches):
        idx_0 = k*batch_size
        idx_1 = min((k+1)*batch_size, len(times)-1)
        t_0 = copy.copy(times[idx_0])
        t_1 = copy.copy(times[idx_1])
        lt, count = computation.distribute_times_per_event_state(
                        num_event_types, num_states,
                        times[idx_0: idx_1],
                        events[idx_0: idx_1],
                        states[idx_0: idx_1])
        len_lt = len(lt)
        arrival_times = computation.extract_arrival_times_of_event(
            event_type, num_event_types, num_states,
            lt, count, len_lt, t_0)
        num_at = len(arrival_times)
        batch = {'lt': lt,
                 'count': count,
                 'at': arrival_times,
                 'num_at': num_at,
                 'len_lt': len_lt,
                 't_0': t_0, 't_1': t_1
                }
        list_of_batches.append(batch)
    cdef np.ndarray[DTYPEf_t,ndim=1] x = np.array(initial_guess, dtype=DTYPEf, copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=1] grad = np.zeros_like(x,dtype=DTYPEf)
    cdef DTYPEf_t f = 0.0 
    cdef int bid=0 # batch id
    for k in range(max(1,num_run_per_minibatch)):
        for batch in list_of_batches:
            print("bid: {}".format(bid))
            x, f, grad , res = descend_along_gradient(
                event_type, num_event_types, num_states,
                batch.get("lt"), batch.get("count"),batch.get("at"),
                batch.get("num_at"), batch.get("len_lt"),
                batch.get("t_0"), batch.get("t_1"),
                x, f, grad,
                max_imp_coef = max_imp_coef,
                learning_rate = learning_rate,
                tol = tol,
                maxiter = maxiter,
                number_of_attempts = number_of_attempts,
                minibatch_id = bid
            )
            results_batches.append(res)
            bid+=1
    result = {'x_min': x,
              'f_min': f,
              'grad_min': grad,
              'results_batches': results_batches
             }
    return result
            
        
def descend_along_gradient(int event_type, int num_event_types, int num_states,
        labelled_times,
        count,
        arrival_times,
        int num_arrival_times,
        int len_labelled_times,
        DTYPEf_t time_start,
        DTYPEf_t time_end,
        np.ndarray [DTYPEf_t,ndim=1] initial_guess,
        DTYPEf_t f_at_initial_guess,
        np.ndarray [DTYPEf_t, ndim=1] grad_at_initial_guess,                     
        DTYPEf_t max_imp_coef = 100.0,                 
        DTYPEf_t learning_rate=0.001,
        DTYPEf_t tol = 1.0e-7,
        int maxiter = 100,
        int number_of_attempts = 2,                   
        int minibatch_id = 0,                   
):
    def compute_f_and_grad(np.ndarray[DTYPEf_t, ndim=1] x):    
        base_rate, imp_coef, dec_coef = computation.array_to_parameters_partial(num_event_types, num_states, x)
#         print("compute_f_and_grad: Enered and param initialised")
        cdef np.ndarray[DTYPEf_t, ndim=2] ratio = imp_coef/(dec_coef-1.0)
        log_likelihood, grad_loglikelihood = computation.compute_event_loglikelihood_partial_and_gradient_partial(
            event_type,
            num_event_types,
            num_states,
            base_rate,
            imp_coef,
            dec_coef,
            ratio,
            labelled_times,
            count,
            arrival_times,
            num_arrival_times,
            len_labelled_times,
            time_start,
            time_end)
#         print("compute_f_and_grad: ready to return")
        return -log_likelihood,-grad_loglikelihood
    cdef int process_id = os.getpid()
    print("Launching descend_along_gradient:\ncomponent_e: {}; process_id: pid{}; minibatch_id: bid{}".format(event_type,process_id,minibatch_id))
    assert learning_rate>0.0
    assert learning_rate<1.0
    assert max_imp_coef>=1.0
    cdef int break_point = 1+num_event_types*num_states
    if minibatch_id == 0:
        f_at_initial_guess, grad_at_initial_guess = compute_f_and_grad(initial_guess)
    cdef int n = 0
    cdef int m = 0
    cdef np.ndarray[DTYPEf_t,ndim=1] x = np.array(initial_guess,copy=True,dtype=DTYPEf)
    cdef int d = len(x)
    cdef int maxiter_inner=maxiter
    cdef np.ndarray[DTYPEf_t, ndim = 2] eye_d = np.eye(d)
    cdef np.ndarray[DTYPEf_t, ndim = 2] cov = np.array(eye_d,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=1] grad = np.zeros_like(x,dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] f = np.zeros(maxiter,dtype=DTYPEf)
    cdef DTYPEf_t norm_grad  = 1.0
    cdef DTYPEf_t f_new = 0.0
    cdef np.ndarray[DTYPEf_t,ndim=1] x_new = np.array(initial_guess,copy=True,dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] grad_new = np.array(grad,copy=True,dtype=DTYPEf)
    cdef DTYPEf_t f_min =  0.0
    cdef np.ndarray[DTYPEf_t,ndim=1] x_min = np.array(x,copy=True,dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] grad_min = np.array(grad,copy=True,dtype=DTYPEf)
    cdef int idx_alt = max(1,maxiter-2)
    cdef int attempt_num=0
    minimisation_conclusive = False
    while (attempt_num<number_of_attempts) & (not minimisation_conclusive):
        print("descend_along_gradient pid{}: minibatch_id{}; attempt_num={}".format(process_id,minibatch_id,1+attempt_num))
        n, m = 0, 0
        f = f_at_initial_guess*np.ones(maxiter,dtype=DTYPEf)
        grad = np.array(grad_at_initial_guess, copy=True, dtype=DTYPEf)
        norm_grad = np.linalg.norm(grad,2)
        f_min =  copy.copy(f[0])
        x = np.array(initial_guess, copy=True,dtype=DTYPEf)
        x_min = np.array(initial_guess, copy=True,dtype=DTYPEf)
        grad_min = np.array(grad,copy=True,dtype=DTYPEf)
        while (norm_grad>=tol) & (n<=idx_alt):
            with nogil:
                n+=1
                m = 0
                f_new = f[n]
            while (f_min <= f_new) & (m<=maxiter_inner):
                x_new = x-learning_rate*grad
                x_new[0] = max(x_new[0], tol)
                x_new[1:break_point]=np.minimum(max_imp_coef,np.maximum(x_new[1:break_point],tol))
                x_new[break_point:len(x_new)]=np.maximum(x_new[break_point:len(x_new)],1.0001)
                f_new, grad_new = compute_f_and_grad(x_new)
                if m > 0:
                    learning_rate=max(2*tol,0.9*learning_rate)
                m+=1   
            with nogil:
                if (m>=maxiter_inner - 1):
                    learning_rate/= pow(0.9,max(m-2,1))
            """"
            if update along gradient direction did not produce 
            decrease of objective function, use random update near previous evaluation point
            """
            if f_new > 0.05*np.abs(f[n-1])+f[n-1]:
                x_new= x - tol*grad*rand()/float(RAND_MAX)
                x_new[0]=max(tol,x_new[0])
                x_new[1:break_point]=np.minimum(max_imp_coef,np.maximum(x_new[1:break_point],tol))
                x_new[break_point:len(x_new)]=np.maximum(x_new[break_point:len(x_new)],1.0001)
                f_new, grad_new = compute_f_and_grad(x_new)
                   
            f[n] = copy.copy(f_new)
            x = np.array(x_new,copy=True)
            grad = np.array(grad_new,copy=True)
            if (f_min > f[n]):
                f_min = copy.copy(f[n])
                x_min = np.array(x,copy=True)
                grad_min = np.array(grad,copy=True)
            norm_grad = np.linalg.norm(grad,2).astype(float)
            if (isnan(f[n])):
                print("descend_along_gradient pid{} bid{}: f[{}]=nan".format(process_id,n))
                raise ValueError('nan')   
        if n<maxiter-1:
            f[n:] = np.repeat(f[n],maxiter-n)
        if f_min>=f[0]:
            attempt_num+=1
            learning_rate = max(2*tol,learning_rate*rand()/float(RAND_MAX))
            if attempt_num<=number_of_attempts:
                print('descend_along_gradient pid{} bid{}: attempt_num {}/{} has failed'.format(process_id,minibatch_id,attempt_num,number_of_attempts))
        else:
            attempt_num+=1
            minimisation_conclusive = True
            print("descend_along_gradient pid{} bid{}: Minimisation conclusive after {} attempts".format(process_id, minibatch_id, attempt_num))   
    res={
        'pid': process_id,
        'bid': minibatch_id,
        'x_min':x_min,
        'f_min': f_min,
        'grad_min': grad_min,
        'f':f,
        'steps':n+1,
        'conclusive': minimisation_conclusive
    }
    return x_min, f_min, grad_min, res