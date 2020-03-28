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


DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t


import computation 

class MinimisationProcedure:
    def __init__(self,
                 np.ndarray[DTYPEf_t, ndim=3] labelled_times,
                 np.ndarray[DTYPEi_t, ndim=2] count,
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
                ):
        print("MinimisationProcedure is being initialised: event_type={}, learning_rate={}, maxiter={}".format(event_type,learning_rate,maxiter))
        assert time_start<=time_end
        assert labelled_times.shape[0] == num_event_types
        assert labelled_times.shape[1] == num_states
        assert count.shape[0] == num_event_types
        assert count.shape[1] == num_states
        if list_init_guesses == []:
            print("No initial guess provided")
            raise ValueError("User needs to provide initial position for gradient descent")
        assert number_of_attempts >= 1
        self.num_event_types = num_event_types
        self.num_states = num_states
        self.event_type = event_type
        self.labelled_times = np.array(labelled_times, copy=True,dtype=DTYPEf)
        self.count = np.array(count, copy=True,dtype=DTYPEi)
        cdef int len_labelled_times = int(labelled_times.shape[2])
        self.len_labelled_times = len_labelled_times
        self.time_start = copy.copy(time_start)
        self.time_end = copy.copy(time_end)
        cdef np.ndarray[DTYPEf_t, ndim=1] arrival_times =\
        computation.extract_arrival_times_of_event(
            event_type, num_event_types, num_states,
            labelled_times, count,len_labelled_times, time_start
        )
        cdef int num_arrival_times = len(arrival_times)
        self.arrival_times=arrival_times
        self.num_arrival_times=num_arrival_times
        self.list_init_guesses = copy.copy(list_init_guesses)
        self.learning_rate = learning_rate
        self.maxiter = maxiter
        self.tol = tol   
        self.number_of_attempts = number_of_attempts
        self.max_imp_coef = copy.copy(max_imp_coef)
    def launch_minimisation(self, parallel=False, return_results=True):
        print('I am launching minimisation. Number of initial guesses={}, parallel={}'.format(len(self.list_init_guesses),parallel))
        cdef list results = []
        self.parallel_minimisation = parallel
        if parallel:
            results = parallel_minimisation(
                self.event_type, self.num_event_types, self.num_states,
                self.labelled_times, self.count,
                self.arrival_times, self.num_arrival_times, self.len_labelled_times,
                self.time_start, self.time_end,
                self.list_init_guesses,
                self.max_imp_coef,
                self.learning_rate, self.tol, self.maxiter, number_of_attempts = self.number_of_attempts,
            )
        else:        
            print("I am performing gradient descent serially")    
            solver=map(lambda x: grad_descent_partial(
                               self.event_type, self.num_event_types, self.num_states,
                               self.labelled_times, self.count,
                               self.arrival_times, self.num_arrival_times, self.len_labelled_times,
                               self.time_start, self.time_end,
                               x,
                               self.max_imp_coef,
                               self.learning_rate, self.tol, self.maxiter),
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
        self.minimiser = best.get('x_min')
        self.minimum = best.get('f_min')
        self.grad_descent_history = best.get('f')
        if return_results:
            return results,best
              

    

def parallel_minimisation(int event_type, int num_event_types, int num_states,
                          np.ndarray[DTYPEf_t, ndim=3] labelled_times,
                          np.ndarray[DTYPEi_t, ndim=2] count,
                          np.ndarray[DTYPEf_t, ndim=1] arrival_times,
                          int num_arrival_times, int len_labelled_times,
                          DTYPEf_t time_start, DTYPEf_t time_end,
                          list list_init_guesses,
                          DTYPEf_t max_imp_coef = 100.0,
                          DTYPEf_t learning_rate = 0.001, DTYPEf_t tol=1.0e-07,
                          int maxiter=100, int number_of_attempts = 3,
                          int num_processes = 0, 
                         ):
    lt_copy = np.array(labelled_times, copy=True)
    count_copy = np.array(count, copy=True)
    arrtimes_copy = np.array(arrival_times, copy=True)
    cdef int tot_tasks = len(list_init_guesses)
    if num_processes <= 0:
        num_processes = max(1,min(mp.cpu_count(),tot_tasks))
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
                           lt_copy, count_copy,
                           arrtimes_copy, num_arrival_times, len_labelled_times,
                           time_start, time_end,
                           x,
                           max_imp_coef,
                           learning_rate, tol, maxiter, number_of_attempts
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
        labelled_times,
        count,
        arrival_times,
        int num_arrival_times,
        int len_labelled_times,
        DTYPEf_t time_start,
        DTYPEf_t time_end,
        np.ndarray[DTYPEf_t,ndim=1] initial_guess,
        DTYPEf_t max_imp_coef = 100.0,                 
        DTYPEf_t learning_rate=0.001,
        DTYPEf_t tol = 1.0e-7,
        int maxiter = 100,
        int number_of_attempts = 2                 
):  
    def compute_f_and_grad(np.ndarray[DTYPEf_t, ndim=1] x):    
#         cdef DTYPEf_t base_rate = 0.0
#         cdef np.ndarray[DTYPEf_t, ndim=2] imp_coef = np.zeros((num_event_types,num_states),dtype=DTYPEf)
#         cdef np.ndarray[DTYPEf_t, ndim=2] dec_coef = np.ones((num_event_types,num_states),dtype=DTYPEf)
        base_rate, imp_coef, dec_coef = computation.array_to_parameters_partial(num_event_types, num_states, x)
        cdef np.ndarray[DTYPEf_t, ndim=2] ratio = imp_coef/(dec_coef-1.0)
#         cdef DTYPEf_t log_likelihood  = 0.0
#         cdef np.ndarray[DTYPEf_t, ndim=1] grad_log_likelihood = np.zeros(len(x),dtype=DTYPEf)
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
    print("Launching grad_descent_partial. pid={}".format(process_id))
    assert learning_rate>0.0
    assert learning_rate<1.0
    assert max_imp_coef>=1.0
    cdef int break_point = 1+num_event_types*num_states
    cdef int n = 0
    cdef int m = 0
    cdef np.ndarray[DTYPEf_t,ndim=1] x = np.array(initial_guess,copy=True,dtype=DTYPEf)
    cdef int d = len(x)
    cdef int maxiter_inner=maxiter
    cdef np.ndarray[DTYPEf_t, ndim = 2] eye_d = np.eye(d)
    cdef np.ndarray[DTYPEf_t, ndim = 2] cov = np.array(eye_d,copy=True)
    cdef np.ndarray[DTYPEf_t, ndim=1] grad = np.zeros_like(x,dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] f = np.zeros(maxiter,dtype=DTYPEf)
    cdef DTYPEf_t [:] f_memview = f
    cdef DTYPEf_t norm_grad  = 1.0
    cdef DTYPEf_t f_new = copy.copy(f_memview[0])
    cdef np.ndarray[DTYPEf_t,ndim=1] x_new = np.array(initial_guess,copy=True,dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] grad_new = np.array(grad,copy=True,dtype=DTYPEf)
    cdef DTYPEf_t f_min =  copy.copy(f_memview[0])
    cdef np.ndarray[DTYPEf_t,ndim=1] x_min = np.array(x,copy=True,dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] grad_min = np.array(grad,copy=True,dtype=DTYPEf)
    cdef int idx_alt = max(1,maxiter-2)
    cdef int attempt_num=0
    minimisation_conclusive = False
    while (attempt_num<number_of_attempts) & (not minimisation_conclusive):
        print("grad_descent_partial pid{}: component_e={}, attempt_num={}".format(process_id,event_type,1+attempt_num))
        n, m = 0, 0
        f = np.zeros(maxiter,dtype=DTYPEf)
        f[0],grad = compute_f_and_grad(x)
        norm_grad = np.linalg.norm(grad,2)
        f[1:] = np.repeat(f[0],maxiter-1)
        f_min =  copy.copy(f_memview[0])
        x = np.array(initial_guess,copy=True,dtype=DTYPEf)
        x_min = np.array(x,copy=True,dtype=DTYPEf)
        grad_min = np.array(grad,copy=True,dtype=DTYPEf)
        while (norm_grad>=tol) & (n<=idx_alt):
    #         print("grad_descent_partial pid{}: count={}".format(process_id,n))
            with nogil:
                n+=1
                m = 0
                f_new = f_memview[n]
            while (f_min <= f_new) & (m<=maxiter_inner):
                x_new = x-learning_rate*grad
                x_new[0] = max(x_new[0], tol)
                x_new[1:break_point]=np.minimum(max_imp_coef,np.maximum(x_new[1:break_point],tol))
                x_new[break_point:]=np.maximum(x_new[break_point:len(x_new)],1.0001)
                f_new, grad_new = compute_f_and_grad(x_new)
                if m > 0:
                    learning_rate=np.maximum(2*tol,0.9*learning_rate)
                m+=1
            with nogil:
                if (m>=maxiter_inner - 1):
                    learning_rate/= pow(0.9,max(m-2,1))
            """"
            if update along gradient direction did not produce 
            decrease of objective function, use random update near previous evaluation point
            """
            if f_new > 0.05*np.abs(f_memview[n-1])+f_memview[n-1]:
                cov = np.maximum(np.amin(np.abs(x)),tol)*eye_d
                x_new = np.random.multivariate_normal(x,cov)
                x_new[:break_point]=np.minimum(10.0,np.maximum(x_new[:break_point],tol))
                x_new[break_point:]=np.maximum(x_new[break_point:],1.0001)
                f_new, grad_new = compute_f_and_grad(x_new)
                
                
            f[n] = copy.copy(f_new)
            x = np.array(x_new,copy=True)
            grad = np.array(grad_new,copy=True)
            if (f_min > f_memview[n]):
                f_min = f_memview[n]
                x_min = np.arrray(x,copy=True)
                grad_min = np.array(grad,copy=True)
            norm_grad = np.linalg.norm(grad,2).astype(float)
            if n<maxiter:
                print("grad_descent_partial pid{}: f[{}]={},  x_n[0]={}, norm(grad_n)={}".format(process_id,n,f[n],x[0],norm_grad))
            if (isnan(f_memview[n])):
                print("grad_descent_partial pid{}: f_memview[{}]=nan".format(process_id,n))
                raise ValueError('nan')   
        if n<maxiter-1:
            f[n:] = np.repeat(f[n],maxiter-n)
        if f_min>=f[0]:
            attempt_num+=1
            if attempt_num<=number_of_attempts:
                print('grad_descent_partial pid{}: attempt_num {}/{} has failed'.format(process_id,attempt_num,number_of_attempts))
        else:
            attempt_num+=1
            minimisation_conclusive = True
            print("grad_descent_partial pid{}: Minimisation conclusive after {} attempts".format(process_id, attempt_num))   
    res={
        'x_min':x_min,
        'f_min': f_min,
        'grad_min': grad_min,
        'f':f,
        'steps':n+1
    }
    try:
        return res
    except:
        print("gradient_descent_partial pid{} could not return".format(process_id))


