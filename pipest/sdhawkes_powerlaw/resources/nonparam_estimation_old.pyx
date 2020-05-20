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

from scipy import linalg
import numpy as np
cimport numpy as np
import bisect
from libc.math cimport pow
from libc.math cimport exp
from libc.math cimport log

import computation
import minimisation_algo as minim_alg
import dirichlet

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t



class Quadrature:
    def __init__(self, int num_pnts = 100, two_scales=False, DTYPEf_t t_max=1.0, DTYPEf_t t_min=1.0e-04, DTYPEf_t tol=1.0e-15):
        self.two_scales=two_scales
        self.t_min=t_min
        self.t_max=t_max
        self.num_pnts=2*num_pnts # If two_scales==True then this is meant to accommodate two scale: linear up to t_min and logarithmic from t_min to t_max 
        self.tol=tol
        self.create_grid()
    def create_grid(self):
        cdef int Q_half = self.num_pnts//2
        cdef np.ndarray[DTYPEf_t, ndim=1] grid = np.zeros(1+2*Q_half,dtype=DTYPEf)
        if self.two_scales:
            grid[:Q_half] = np.linspace(self.tol,self.t_min, num=Q_half, endpoint=False)
            grid[Q_half:] = np.exp(np.linspace(log(self.t_min), log(self.t_max), num=Q_half+1))
        else:
            grid=np.linspace(self.tol,self.t_max, num=1+2*Q_half)
        self.grid=grid

        
class EstimProcedure:
    def __init__(self, int num_event_types, int num_states,
                 np.ndarray[DTYPEf_t , ndim=1] times, 
                 np.ndarray[DTYPEi_t , ndim=1] events,
                 np.ndarray[DTYPEi_t , ndim=1] states,
                 int num_quadpnts = 100, DTYPEf_t quad_tmax=1.0, DTYPEf_t quad_tmin=1.0e-04, DTYPEf_t tol=1.0e-15):
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
        self.quadrature=Quadrature(num_pnts=num_quadpnts, t_max=quad_tmax, t_min=quad_tmin)
        self.store_distribution_of_marks()
        self.store_expected_intensities()
        print("EstimProcedure has been successfully initialised") 
    def prepare_estimation_of_hawkes_kernel(self,pool=False):
        print("I am preparing estimation of hawkes kernel")
        if pool:
            self.store_nonsingular_expected_jumps_pool()
        else:
            self.store_nonsingular_expected_jumps()
        self.store_convolution_kernels()
        self.store_matrix_A()
        print("Estimation of hawkes kernel is now ready")
    def estimate_hawkes_kernel(self,pool=False):
        self.prepare_estimation_of_hawkes_kernel(pool=pool)
        cdef int N = max(1,min(self.num_event_types,mp.cpu_count()))
        cdef DTYPEf_t run_time=-time.time()
        print("I am performing estimation of hawkes kernel in parallel on {} cpus".format(N))
        pool=mp.Pool(N)
        results=pool.map_async(
            self.solve_linear_system_partial,
            list(range(self.num_event_types))
        ).get()
        pool.close()
        run_time+=time.time()
        print("Parallel estimation of hawkes kernel terminates. run_time={}".format(run_time))
        cdef int d_E = self.num_event_types
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=4] kappa = np.zeros((d_E, self.num_states, d_E, Q), dtype=DTYPEf)
        cdef int e=0
        for e in range(d_E):
            kappa+=np.array(results[e][1], copy=True)
        self.hawkes_kernel=kappa        
        
    def store_distribution_of_marks(self):
        print("I am storing distribution of marks")
        cdef int len_events=len(self.events)
        cdef DTYPEi_t [:] events_memview = self.events
        cdef DTYPEi_t [:] states_memview = self.states
        cdef np.ndarray[DTYPEf_t, ndim=2] prob = np.zeros((self.num_event_types,self.num_states),dtype=DTYPEf)
        prob = store_distribution_of_marks(self.num_event_types, self.num_states,
                                           events_memview, states_memview, len_events)
        self.marks_distribution=prob
    def store_expected_intensities(self):
        print("I am storing expected intensities")
        cdef DTYPEi_t [:] events_memview = self.events
        cdef np.ndarray[DTYPEf_t, ndim=1] Lambda = np.zeros(self.num_event_types, dtype=DTYPEf)
        Lambda=store_expected_intensities(self.num_event_types,events_memview, self.time_horizon)
        self.expected_intensities = Lambda
    def store_nonsingular_expected_jumps(self):
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=5] g_hat = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=5] g_hat_one = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] g_hat_at_quadpnts = np.zeros((d_E,d_S,d_E,Q),dtype=DTYPEf)
        self.g_hat = g_hat
        self.g_hat_one = g_hat_one
        self.g_hat_at_quadpnts = g_hat_at_quadpnts
        self.estimate_nonsingular_expected_jumps()
    def store_nonsingular_expected_jumps_pool(self):
        cdef int N = max(1,min(self.num_event_types,mp.cpu_count()))
        print("I am storing non-singular expected jumps in parallel on {} cpus".format(N))
        cdef DTYPEf_t run_time=-time.time()
        pool=mp.Pool(N)
        results=pool.map_async(
            self.estimate_nonsingular_expected_jumps_partial,
            list(range(self.num_event_types))
        ).get()
        pool.close()
        run_time+=time.time()
        print("Parallel execution of non-singular expected jumps terminates. run_time={}".format(run_time))
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=5] g_hat = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=5] g_hat_one = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] g_hat_at_quadpnts = np.zeros((d_E,d_S,d_E,Q),dtype=DTYPEf)
        cdef int e=0
        for e in range(self.num_event_types):
            g_hat+=results[e][0]
            g_hat_one+=results[e][1]
            g_hat_at_quadpnts+=results[e][2]
        self.g_hat = g_hat
        self.g_hat_one = g_hat_one
        self.g_hat_at_quadpnts = g_hat_at_quadpnts    
    def store_convolution_kernels(self):
        print("I am storing convolution kernels")
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=6] K_hat = np.zeros((d_E,d_E,d_S,d_S,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=6] K_hat_one = np.zeros((d_E,d_E,d_S,d_S,Q,Q),dtype=DTYPEf)
        self.K_hat=K_hat
        self.K_hat_one=K_hat_one
        self.set_convolution_kernels()
    def store_matrix_A(self):
        print("I am storing matrix A")
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=2] matrix_A = np.zeros((d_E*d_S*Q,d_E*d_S*Q), dtype=DTYPEf)
        self.matrix_A = matrix_A
        self.set_matrix_A()

    def estimate_nonsingular_expected_jumps(self):
        print("I am estimating non-singular expected jumps")
        cdef DTYPEf_t run_time=-time.time()
        cdef DTYPEf_t [:,:,:,:,:] g_hat = self.g_hat
        cdef DTYPEf_t [:,:,:,:,:] g_hat_one = self.g_hat_one
        cdef DTYPEf_t [:,:,:,:] g_hat_at_quadpnts = self.g_hat_at_quadpnts
        cdef DTYPEf_t [:,:,:] labelled_times = self.labelled_times
        cdef DTYPEi_t [:,:] count = self.count
        cdef DTYPEf_t [:] quadpnts = self.quadrature.grid
        cdef int d_S = self.num_states, d_E = self.num_event_types, num_quadpnts = self.quadrature.num_pnts
        cdef int e1=0, x1=0, e=0
        for x1 in prange(d_S, nogil=True):
            for e1 in range(d_E):
                for e in range(d_E):
                    estimate_nonsingular_expected_jumps(
                        e1, x1, e, d_S, num_quadpnts,
                        g_hat, g_hat_one, g_hat_at_quadpnts,
                        labelled_times, count, quadpnts
                    )
        run_time+=time.time()            
        print("Estimation terminates. run_time={}".format(run_time))            
    
    def estimate_nonsingular_expected_jumps_partial(self, int e):
#         print("I am estimating non-singular expected jumps for the component e={}".format(e))
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int Q = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=5] g_hat = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=5] g_hat_one = np.zeros((d_E,d_S,d_E,Q,Q),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=4] g_hat_at_quadpnts = np.zeros((d_E,d_S,d_E,Q),dtype=DTYPEf)
        cdef DTYPEf_t [:,:,:,:,:] g_hat_memview = g_hat
        cdef DTYPEf_t [:,:,:,:,:] g_hat_one_memview = g_hat_one
        cdef DTYPEf_t [:,:,:,:] g_hat_at_quadpnts_memview = g_hat_at_quadpnts
        cdef DTYPEf_t [:,:,:] lt_memview = self.labelled_times
        cdef DTYPEi_t [:,:] count_memview = self.count
        cdef DTYPEf_t [:] quadpnts_memview = self.quadrature.grid
        cdef int e1=0, x1=0
        for x1 in prange(d_S, nogil=True):
            for e1 in range(d_E):
                estimate_nonsingular_expected_jumps(
                    e1, x1, e, d_S, Q,
                    g_hat_memview, g_hat_one_memview, g_hat_at_quadpnts_memview,
                    lt_memview, count_memview, quadpnts_memview
                )
        return g_hat, g_hat_one, g_hat_at_quadpnts        

    def set_convolution_kernels(self):
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int num_quadpnts = self.quadrature.num_pnts
        cdef DTYPEf_t [:,:,:,:,:] g_hat = self.g_hat
        cdef DTYPEf_t [:,:,:,:,:] g_hat_one = self.g_hat_one
        cdef DTYPEf_t [:] Lambda = self.expected_intensities
        cdef DTYPEf_t [:,:] mark_prob = self.marks_distribution
        cdef DTYPEf_t [:] quadpnts = self.quadrature.grid
        cdef DTYPEf_t [:,:,:,:,:,:] K_hat = self.K_hat
        cdef DTYPEf_t [:,:,:,:,:,:] K_hat_one = self.K_hat_one
        cdef int e1=0, eps=0, x1=0, y=0
        for x1 in prange(d_S, nogil=True):
            for y in range(d_S):
                for e1 in range(d_E):
                    for eps in range(d_E):
                        produce_convolution_kernel(
                            e1, eps, x1, y, num_quadpnts,
                            g_hat, g_hat_one,
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
        cdef DTYPEf_t [:,:,:,:] g_hat_at_quadpnts = self.g_hat_at_quadpnts
        cdef int e1=0, x1=0
        for x1 in prange(d_S, nogil=True):
            for e1 in range(d_E):
                produce_b_e(
                    e1, x1, e, d_S, Q,
                    g_hat_at_quadpnts,
                    b_e_memview
                )
        return b_e        
    def solve_linear_system_partial(self, int e):
#         print("I am solving for the component e={}".format(e))
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
                        
        
cdef int estimate_g_hat_at_gridpnts(
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
        
cdef int set_nonsingular_expected_jumps_from_grid(
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
    return 0
        
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
        
        

cdef int estimate_nonsingular_expected_jumps(
    int e1, int x1, int e, int num_states, int num_quadpnts,
    DTYPEf_t [:,:,:,:,:] g_hat, 
    DTYPEf_t [:,:,:,:,:] g_hat_one,
    DTYPEf_t [:,:,:,:] g_hat_at_quadpnts,
    DTYPEf_t [:,:,:] labelled_times,
    DTYPEi_t [:,:] count,
    DTYPEf_t [:] quadpnts,
) nogil:
    cdef int Q = num_quadpnts
    cdef DTYPEf_t quadpnt_m=0.0, next_pnt_m=0.0, prev_pnt_m=0.0, delta_pnt_m=0.0
    cdef DTYPEf_t quadpnt_n=0.0, next_pnt_n=0.0, prev_pnt_n=0.0, delta_pnt_n=0.0
    cdef DTYPEf_t tau=0.0, tau_0=0.0
    cdef DTYPEf_t float_count=max(1.0, float(count[e1,x1]))
    cdef int m=0, n=0, x=0, k=0, k1=0
    for m in range(Q):
        quadpnt_m = quadpnts[m]
        next_pnt_m = quadpnts[m+1]
        prev_pnt_m = quadpnts[max(0,m-1)]
        delta_pnt_m = next_pnt_m - prev_pnt_m # notice that this def is different between m and n
        for n in range(Q):
            quadpnt_n = quadpnts[n]
            next_pnt_n = quadpnts[n+1]
            prev_pnt_n = quadpnts[max(0,n-1)]
            delta_pnt_n = next_pnt_n - quadpnt_n # notice that this def is different between m and n 
            for k1 in range(count[e1,x1]):
                for x in range(num_states):
                    for k in range(count[e,x]):
                        if n==0:
                            if m>=1:
                                if (labelled_times[e,x,k]-labelled_times[e1,x1,k1]>= prev_pnt_m):
                                    tau_0 = (labelled_times[e,x,k]-labelled_times[e1,x1,k1] \
                                             -  prev_pnt_m)/(delta_pnt_m)
                                    if tau_0 <=1.0:
                                        g_hat_at_quadpnts[e1,x1,e,m]+=1.0
                            else:
                                if labelled_times[e,x,k]-labelled_times[e1,x1,k1]>= 0.0:
                                    tau_0 = (labelled_times[e,x,k]-labelled_times[e1,x1,k1])/(next_pnt_m)
                                    if tau_0 <=1.0:
                                        g_hat_at_quadpnts[e1,x1,e,m]+=1.0
                        if m<=n:
                            if labelled_times[e,x,k]-labelled_times[e1,x1,k1]>= quadpnt_n-quadpnt_m:
                                tau=(labelled_times[e,x,k]-labelled_times[e1,x1,k1] \
                                     - quadpnt_n+quadpnt_m)/(delta_pnt_n)
                                if tau<=1.0:
                                    g_hat[e1,x1,e,m,n]+=1.0
                                    g_hat_one[e1,x1,e,m,n]+= tau*(delta_pnt_n)
                        else:
                            if labelled_times[e,x,k]-labelled_times[e1,x1,k1]>= quadpnt_m-next_pnt_n:
                                tau=(labelled_times[e,x,k]-labelled_times[e1,x1,k1] \
                                     - quadpnt_m+next_pnt_n)/(delta_pnt_n)
                                if tau<=1.0:
                                    g_hat[e1,x1,e,m,n]+=1.0
                                    g_hat_one[e1,x1,e,m,n]+= quadpnt_m \
                                    -labelled_times[e,x,k]+labelled_times[e1,x1,k1]-quadpnt_n                         
            g_hat[e1,x1,e,m,n]/=(float_count*(delta_pnt_n))
            g_hat_one[e1,x1,e,m,n]/=(float_count*(delta_pnt_n))
            if n==0:
                if m>=1:
                    g_hat_at_quadpnts[e1,x1,e,m] /= (float_count*(delta_pnt_m))
                else:
                    g_hat_at_quadpnts[e1,x1,e,m] /= (float_count*(next_pnt_m))
    return 0

cdef int produce_convolution_kernel(
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
    return 0   


cdef int produce_matrix_A(
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
    return 0   

cdef int produce_b_e(
    int e1, int x1, int e, int d_S, int Q,
    DTYPEf_t [:,:,:,:] g_hat_at_quadpnts,
    DTYPEf_t [:] b_e
) nogil:
    cdef int i=(e1*d_S+x1)*Q
    cdef int m=0
    for m in range(Q):
        b_e[i]=g_hat_at_quadpnts[e1,x1,e,m]
        i+=1
        
        
# class Quadrature:
#     def __init__(self,int num_quadrature_points=80, DTYPEf_t t_max=1.0, DTYPEf_t t_min=1.0e-07,DTYPEf_t tol=1.0e-15):
#         self.t_max=t_max
#         self.t_min=t_min
#         self.num_quadrature_points=max(1,num_quadrature_points)
#         self.store_quadrature_points()
        
#     def store_quadrature_points(self, DTYPEf_t tol=1.0e-15):
#         cdef int Q = self.num_quadrature_points
#         cdef np.ndarray[DTYPEf_t, ndim=2] gauss_legendre = np.zeros((Q,2), dtype=DTYPEf)
#         gauss_legendre[:,0], gauss_legendre[:,1] =\
#         np.polynomial.legendre.leggauss(Q)
#         # Adjust for log-scale and interval [t_min, t_max]
#         cdef DTYPEf_t l_max=log(self.t_max)
#         cdef DTYPEf_t l_min=log(self.t_min)
#         gauss_legendre[:,0]= np.exp(
#             0.5*\
#             (
#                 (l_max-l_min)*gauss_legendre[:,0] + (l_max+l_min)
#             )
#         )
#         gauss_legendre[:,1]=0.5*(l_max-l_min)*gauss_legendre[:,0]*gauss_legendre[:,1]
#         self.gauss_legendre=gauss_legendre
#         cdef int len_sigmas=Q+Q*(Q-1)/2
#         cdef np.ndarray[DTYPEf_t, ndim=1] sigmas = np.zeros(len_sigmas, dtype=DTYPEf)
#         sigmas[:Q]=np.array(gauss_legendre[:,0],copy=True)
#         cdef int i=0, j=0, s=Q
#         for i in range(Q):
#             for j in range(i+1, Q):
#                 sigmas[s]=gauss_legendre[j,0]-gauss_legendre[i,0]
#                 s+=1
#         sigmas.sort(axis=0)
#         cdef np.ndarray[DTYPEi_t, ndim=1] idx_quadpnts=np.zeros((Q,), dtype=DTYPEi)
#         s=0
#         for i in range(Q):
#             while (
#                 (
#                     (sigmas[s]<gauss_legendre[i,0]-tol) | (sigmas[s]>gauss_legendre[i,0]+tol)
#                 )
#                 &
#                 (s<len_sigmas-1)
#             ) :
#                 s+=1
#             idx_quadpnts[i]=s
#         cdef np.ndarray[DTYPEi_t, ndim=2] idx_distance_quadpnts=np.zeros((Q,Q), dtype=DTYPEi)
#         cdef DTYPEi_t [:,:] idx_distance_memview = idx_distance_quadpnts
#         cdef DTYPEf_t [:] sigmas_memview = sigmas
#         cdef DTYPEf_t [:] quadpnts_memview = gauss_legendre[:,0]
#         cdef DTYPEf_t dist = 0.0
#         cdef int n=0, m=0, start_index=0
#         for n in prange(Q, nogil=True):
#             for m in range(Q):
#                 if m>=n:
#                     dist=quadpnts_memview[m]-quadpnts_memview[n]
#                 else:
#                     dist=quadpnts_memview[n]-quadpnts_memview[m]
#                 start_index=0
#                 if dist> sigmas_memview[len_sigmas//4]:
#                     if dist > sigmas_memview[len_sigmas//2]:
#                         if dist > sigmas_memview[3*len_sigmas//4]:
#                             start_index=-1+3*len_sigmas//4
#                         else:
#                             start_index=-1+len_sigmas//2
#                     else:
#                         start_index=-1+len_sigmas//4
#                 for s in range(start_index,len_sigmas):
#                     if (dist-tol <= sigmas_memview[s]) & (sigmas_memview[s] <= dist+tol):
#                         idx_distance_memview[m,n]=s
#                         break 
#         self.idx_quadpnts=idx_quadpnts
#         self.idx_distance_quadpnts=idx_distance_quadpnts
#         self.sigmas=sigmas


# class EstimProcedure:
#     def __init__(self,int num_event_types, int num_states,
#                  np.ndarray[DTYPEf_t , ndim=1] times, 
#                  np.ndarray[DTYPEi_t , ndim=1] events, np.ndarray[DTYPEi_t , ndim=1] states,
#                  DTYPEf_t upperbound_of_support_of_kernel=1.0e+02,
#                  DTYPEf_t lowerbound_of_support_of_kernel=1.0e-09,
#                  int num_quadrature_points = 80, DTYPEf_t bandwidth = 1.0, DTYPEf_t tol = 1.0e-15):
#         if not (len(times)==len(states) & len(times)==len(events)):
#             raise ValueError("All shapes must agree, but input was:\n len(times)={} \n len(events)={} \n len(states)={}".format(len(times),len(events),len(states)))
#         self.num_event_types = num_event_types
#         self.num_states = num_states
#         cdef np.ndarray[DTYPEf_t, ndim=3] labelled_times = np.zeros((num_event_types,num_states,len(times)), dtype=DTYPEf)
#         cdef np.ndarray[DTYPEi_t, ndim=2] count = np.zeros((num_event_types, num_states), dtype=DTYPEi)
#         labelled_times, count = computation.distribute_times_per_event_state(
#             num_event_types,num_states,
#             times,events, states)
#         self.labelled_times=labelled_times
#         self.count=count
#         self.times=times
#         self.time_horizon=times[len(times)-1]
#         self.events=events
#         self.states=states
#         self.upperbound_of_support_of_kernel=upperbound_of_support_of_kernel
#         self.lowerbound_of_support_of_kernel=lowerbound_of_support_of_kernel
#         self.bandwidth=bandwidth
#         self.quadrature=Quadrature(num_quadrature_points=num_quadrature_points,
#                                    t_max=upperbound_of_support_of_kernel, t_min=lowerbound_of_support_of_kernel,
#                                    tol=tol)
#         self.store_distribution_of_marks()
#         self.store_expected_intensities()
#         self.store_nonsingular_conditional_expected_jumps()
#         self.store_primitives_of_convolution_kernel()
#         self.store_matrix_A_of_linear_system()
#         print("EstimProcedure has been successfully initialised")
        
        
#     def store_distribution_of_marks(self):
#         print("I am storing distribution of marks")
#         cdef int len_events=len(self.events)
#         cdef DTYPEi_t [:] events_memview = self.events
#         cdef DTYPEi_t [:] states_memview = self.states
#         cdef np.ndarray[DTYPEf_t, ndim=2] prob = np.zeros((self.num_event_types,self.num_states),dtype=DTYPEf)
#         prob = store_distribution_of_marks(self.num_event_types, self.num_states,
#                                            events_memview, states_memview, len_events)
#         self.marks_distribution=prob
#     def store_expected_intensities(self):
#         print("I am storing expected intensities")
#         cdef DTYPEi_t [:] events_memview = self.events
#         cdef np.ndarray[DTYPEf_t, ndim=1] Lambda = np.zeros(self.num_event_types, dtype=DTYPEf)
#         Lambda=store_expected_intensities(self.num_event_types,events_memview, self.time_horizon)
#         self.expected_intensities = Lambda
#     def store_nonsingular_conditional_expected_jumps(self):
#         print("I am storing non-singular conditional expected jumps")
#         cdef int N = max(1,min(self.num_event_types,mp.cpu_count()))
#         print("Computing non-singular conditional expected jumps in parallel on {} cpus".format(N))
#         pool=mp.Pool(N)
#         results=pool.map_async(
#             self.produce_nonsingular_conditional_expected_jumps_partial,
#             list(range(self.num_event_types))
#         ).get()
#         pool.close()
#         print("Parallel computation terminates")
#         cdef int len_sigmas=len(self.quadrature.sigmas)
#         cdef int e=0, e1=0
#         cdef np.ndarray[DTYPEf_t, ndim=4] g_hat = np.zeros(
#             (self.num_event_types, self.num_states, self.num_event_types, len_sigmas),dtype=DTYPEf)
#         for e in range(self.num_event_types):
#             for e1 in range(self.num_event_types):
#                 if results[e1][0]==e:
#                     g_hat[:,:,e,:]=np.array(results[e1][1],copy=True)
#         self.nonsingular_conditional_expected_jumps=g_hat
#     def produce_nonsingular_conditional_expected_jumps_partial(self, int e):
#         print("I am computing non-signular conditional expected jumps for component e={}".format(e))
#         cdef int N = len(self.quadrature.sigmas)
#         cdef int d_E = self.num_event_types
#         cdef int d_S = self.num_states
#         cdef np.ndarray[DTYPEf_t, ndim=3]  result = np.zeros((d_E,d_S,N),dtype=DTYPEf)
#         cdef DTYPEf_t [:,:,:] result_memview = result
#         cdef DTYPEf_t Lambda_e = self.expected_intensities[e]
#         cdef DTYPEf_t [:,:,:] lt_memview = self.labelled_times
#         cdef DTYPEi_t [:,:] count_memview = self.count
#         cdef DTYPEf_t [:] sigmas_memview = self.quadrature.sigmas
#         cdef DTYPEf_t h = self.bandwidth
#         cdef int x1=0, e1=0, n=0
#         for n in prange(N, nogil=True):
#             for e1 in range(d_E):
#                 for x1 in range(d_S):
#                         result_memview[e1,x1,n] = estimate_nonsingular_conditional_expected_jumps(
#                             d_S, sigmas_memview[n],  h,
#                             lt_memview, count_memview, Lambda_e,
#                             e1, x1, e)
#         return e,result
#     def store_primitives_of_convolution_kernel(self):
#         print("I am storing primitives of convolution kernel")
#         cdef int d_E = self.num_event_types
#         cdef int d_S = self.num_states
#         cdef int Q = self.quadrature.num_quadrature_points
#         cdef np.ndarray[DTYPEf_t, ndim=1] eval_times = np.array(self.quadrature.gauss_legendre[:,0],copy=True)
#         cdef DTYPEf_t [:] eval_times_memview=eval_times
#         cdef np.ndarray[DTYPEf_t, ndim=5] primitives = np.zeros(
#             (d_E,d_E,d_S,d_S,Q),dtype=DTYPEf)
#         cdef DTYPEf_t [:,:,:,:,:] primitives_memview= primitives
#         cdef DTYPEf_t [:,:,:,:] g_hat = self.nonsingular_conditional_expected_jumps
#         cdef DTYPEf_t [:] Lambda_memview =self.expected_intensities
#         cdef DTYPEf_t [:,:] mark_prob_memview = self.marks_distribution
#         cdef DTYPEf_t [:] sigmas_memview = self.quadrature.sigmas
#         cdef int len_sigmas=len(self.quadrature.sigmas)
#         cdef int e1=0,eps=0,x1=0,y=0,n=0
#         for e1 in prange(d_E,nogil=True):
#             for eps in range(d_E):
#                 for x1 in range(d_S):
#                     for y in range(d_S):
#                          eval_primitive_of_convolution_kernel_at_times(
#                             eval_times_memview, Q, primitives_memview[e1,eps,x1,y,:],
#                             e1, eps, x1, y, sigmas_memview, len_sigmas,
#                             g_hat, Lambda_memview, mark_prob_memview)
#         self.primitives_of_convolution_kernel = primitives
#     def store_matrix_A_of_linear_system(self):
#         print("I am storing the matrix A for linear system")
#         self.matrix_A=self.create_matrix_A()
#     def store_hawkes_kernel(self):
#         self.hawkes_kernel=self.solve_for_hawkes_kernel()
#     def store_smoothened_hawkes_kernel(self, DTYPEf_t scale=1.0, int num=100):
#         print("I am storing the smoothened hawkes kernel")
#         result, time_grid = self.smoothen_hawkes_kernel(scale = scale, num_additional_pnts = num)
#         self.smoothened_hawkes_kernel=result
#         self.time_grid = time_grid
    
#     def solve_for_hawkes_kernel(self):
#         cdef int N = max(1,min(self.num_event_types,mp.cpu_count()))
#         print("I am computing the hawkes_kernels")
#         print("Parallel computation on {} cpus".format(N))
#         pool=mp.Pool(N)
#         results=pool.map_async(
#             self.solve_linear_system,
#             list(range(self.num_event_types))
#         ).get()
#         pool.close()
#         print("Parallel computation terminates")
#         cdef np.ndarray[DTYPEf_t, ndim=4] kappa = np.zeros(
#             (self.num_event_types,self.num_states,self.num_event_types,self.quadrature.num_quadrature_points),
#             dtype=DTYPEf)
#         cdef int e, e1
#         for e in range(self.num_event_types):
#             for e1 in range(self.num_event_types):
#                 if results[e1][0]==e:
#                     kappa[:,:,e,:]=np.array(results[e1][1],dtype=DTYPEf,copy=True)
#         return kappa            
#     def solve_linear_system(self,int e):
#         print("I am solving the linear system for kappa[:,:,{},:]".format(e))
#         cdef np.ndarray[DTYPEf_t, ndim=1] b = self.create_vector_b(e)
#         cdef np.ndarray[DTYPEf_t, ndim=2] M = self.matrix_A + np.eye(self.matrix_A.shape[0],dtype=DTYPEf)
#         cdef np.ndarray[DTYPEf_t, ndim=1] x = np.linalg.solve(M,b)
#         cdef DTYPEf_t [:] x_memview = x
#         cdef int Q=self.quadrature.num_quadrature_points
#         cdef int d_S = self.num_states
#         cdef int d_E = self.num_event_types
#         cdef np.ndarray[DTYPEf_t, ndim=3] kernel = np.zeros((d_E,d_S,Q),dtype=DTYPEf)
#         cdef DTYPEf_t [:,:,:] kernel_memview = kernel
#         cdef int e1, x1, m, i
#         for m in prange(Q, nogil=True):    
#             for e1 in range(d_E):
#                 for x1 in range(d_S):
#                     i=(e1*d_S+x1)*Q+m
#                     kernel_memview[e1,x1,m]=x_memview[i]
#         return e,kernel
#     def create_vector_b(self, int e):
#         cdef int Q=self.quadrature.num_quadrature_points
#         cdef int d_S = self.num_states
#         cdef int d_E = self.num_event_types
#         cdef np.ndarray[DTYPEf_t, ndim=1] b = np.zeros(d_S*d_E*Q,dtype=DTYPEf)
#         cdef DTYPEf_t [:] b_memview = b
#         cdef DTYPEf_t [:,:,:,:] g_hat = self.nonsingular_conditional_expected_jumps
#         cdef DTYPEf_t [:] sigmas = self.quadrature.sigmas
#         cdef DTYPEf_t [:] points = self.quadrature.gauss_legendre[:,0]
#         cdef DTYPEi_t [:] idx_quadpnts = self.quadrature.idx_quadpnts
#         cdef int e1, x1, m, i, not_found, idx
#         cdef int len_sigmas = len(self.quadrature.sigmas)
#         for m in prange(Q, nogil=True):
#             for e1 in range(d_E):
#                 for x1 in range(d_S):
#                     i=(e1*d_S+x1)*Q+m
#                     b_memview[i]=g_hat[e1,x1,e,idx_quadpnts[m]]
#         return b            
    
#     def create_matrix_A(self):
#         cdef int Q=self.quadrature.num_quadrature_points
#         cdef int d_S = self.num_states
#         cdef int d_E = self.num_event_types
#         cdef np.ndarray[DTYPEf_t, ndim=2] A = np.eye(d_S*d_E*Q,dtype=DTYPEf)
#         cdef DTYPEf_t [:,:] A_memview = A
#         cdef DTYPEf_t [:] points = self.quadrature.gauss_legendre[:,0]
#         cdef DTYPEf_t [:] weights = self.quadrature.gauss_legendre[:,1]
#         cdef DTYPEf_t [:] sigmas = self.quadrature.sigmas
#         cdef DTYPEi_t [:] idx_quadpnts = self.quadrature.idx_quadpnts
#         cdef int len_sigmas=len(self.quadrature.sigmas)
#         cdef DTYPEf_t [:,:,:,:] g_hat = self.nonsingular_conditional_expected_jumps
#         cdef DTYPEf_t [:] Lambda = self.expected_intensities
#         cdef DTYPEf_t [:,:] mark_prob = self.marks_distribution
#         cdef DTYPEf_t [:,:,:,:,:] primitives_memview = self.primitives_of_convolution_kernel
#         cdef DTYPEi_t [:,:] idx_distance_memview = self.quadrature.idx_distance_quadpnts
#         cdef int e1, eps, x1, y, m, n, i, j
#         for n in prange(Q, nogil=True):
#             for m in range(Q):
#                 for e1 in range(d_E):
#                     for eps in range(d_E):
#                         for x1 in range(d_S):
#                             for y in range(d_S):
#                                 i=(e1*d_S+x1)*Q+m
#                                 j=(eps*d_S+y)*Q+n
#                                 if m==n:
#                                     A_memview[i,j]=primitives_memview[e1,eps,x1,y,m]
#                                 else:
#                                     A_memview[i,j]=weights[n]*eval_convolution_kernel_given_ghat_and_distance(
#                                         m, n, e1, eps, x1, y,
#                                         sigmas, len_sigmas, g_hat, Lambda, mark_prob, idx_distance_memview)
# #                                     A_memview[i,j]=weights[n]*eval_convolution_kernel_given_ghat(
# #                                         points[m]-points[n], e1, eps, x1, y,
# #                                         sigmas, len_sigmas, g_hat, Lambda, mark_prob)
#         return A
    
#     def smoothen_hawkes_kernel(self,DTYPEf_t scale = 1.0, int num_additional_pnts = 100):
#         cdef int d_E = self.num_event_types
#         cdef int d_S = self.num_states
#         cdef int Q = self.quadrature.num_quadrature_points
#         cdef int len_grid = Q+num_additional_pnts
#         cdef np.ndarray[DTYPEf_t, ndim=1] time_grid = np.zeros(
#             len_grid, dtype=DTYPEf)
#         cdef np.ndarray[DTYPEf_t, ndim=1] left_grid=np.linspace(
#             self.quadrature.gauss_legendre[0,0], self.quadrature.gauss_legendre[Q-1, 0], num=num_additional_pnts, dtype=DTYPEf)
#         cdef np.ndarray[DTYPEi_t, ndim=1] idx_quadpnts = np.zeros(Q, dtype=DTYPEi)
#         time_grid, idx_quadpnts = merge_sorted_arrays(left_grid, self.quadrature.gauss_legendre[:, 0])
#         cdef DTYPEf_t [:] time_memview = time_grid
#         cdef DTYPEi_t [:] idx_quadpnts_memview = idx_quadpnts
#         cdef DTYPEf_t [:] quadpnts_memview = self.quadrature.gauss_legendre[:, 0]
#         cdef np.ndarray[DTYPEf_t, ndim=4] interpolated_hawkes_kernel = np.zeros(
#             (d_E, d_S, d_E, len_grid), dtype=DTYPEf)
#         cdef np.ndarray[DTYPEf_t, ndim=4] smoothened_hawkes_kernel = np.zeros(
#             (d_E, d_S, d_E, len_grid), dtype=DTYPEf)
        
#         cdef DTYPEf_t [:,:,:,:] kappa = self.hawkes_kernel
#         cdef DTYPEf_t [:,:,:,:] new_kappa = interpolated_hawkes_kernel
#         cdef DTYPEf_t [:,:,:,:] newnew_kappa = smoothened_hawkes_kernel
#         cdef int e1=0, x1=0, e=0
#         for e in prange(d_E, nogil=True):
#             for e1 in range(d_E):
#                 for x1 in range(d_S):
#                     interpolate_hawkes_kernels(
#                         e1, x1, e,
#                         kappa, new_kappa,
#                         time_memview, len_grid,
#                         idx_quadpnts_memview, quadpnts_memview)
#         self.interpolated_hawkes_kernel=interpolated_hawkes_kernel          
#         cdef DTYPEf_t weight =  1.0/(exp(-scale*time_grid[0])-exp(-scale*time_grid[len_grid-1]))            
#         for e in prange(d_E, nogil=True):
#             for e1 in range(d_E):
#                 for x1 in range(d_S):
#                     convolute_hawkes_kernels(
#                         e1, x1, e,
#                         new_kappa, newnew_kappa,
#                         time_memview, len_grid, scale, weight)
#         return  smoothened_hawkes_kernel, time_grid           
                    
        
          
        

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

# cdef DTYPEf_t estimate_nonsingular_conditional_expected_jumps(int num_states,
#     DTYPEf_t t, DTYPEf_t h, DTYPEf_t [:,:,:] labelled_times, DTYPEi_t [:,:] count, DTYPEf_t Lambda_e,
#     int e1, int x1, int e
# ) nogil:
#     cdef DTYPEf_t kernel=0.0, tau=0.0
#     cdef DTYPEi_t j=0, k=0
#     cdef DTYPEi_t x=0
#     cdef DTYPEf_t result= 0.0
#     h*=pow(float(count[e1,x1]),-0.33)
#     if count[e1,x1]>=1:
#         for j in range(count[e1,x1]):
#             for x in range(num_states):
#                 for k in range(count[e,x]):
#                     if (labelled_times[e,x,k] >= (labelled_times[e1,x1,j] + t)):
#                         tau=((labelled_times[e,x,k]-labelled_times[e1,x1,j]-t)/h)
#                         if (tau <= 1.0):
#                             kernel+=1.0 # or:  kernel+=3.0-6.0*pow(tau,2)
#                         else:
#                             break
#         result=kernel/(h*float(count[e1,x1]))  #- Lambda_e
#     else:
#         result = 0.0  # - Lambda_e                      
#     return max(0.0,result)
 

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

# cdef DTYPEf_t evaluate_convolution_kernel(int num_states,
#     DTYPEf_t t, DTYPEf_t [:,:] prob_marks, DTYPEf_t [:] expected_intensities,
#     DTYPEf_t h, DTYPEf_t [:,:,:] labelled_times, DTYPEi_t [:,:] count,
#     int e1, int x1, int eps, int y) nogil:
#     cdef DTYPEf_t result = 0.0
#     if (t>=0):
#         return prob_marks[eps,y]*estimate_nonsingular_conditional_expected_jumps(
#             num_states,
#             t, h, labelled_times, count, expected_intensities[eps],
#             e1, x1, eps)
#     else:
#         return ((expected_intensities[eps]/expected_intensities[e1])*
#                 prob_marks[eps,y]*estimate_nonsingular_conditional_expected_jumps(
#                 num_states, 
#                 -t, h, labelled_times, count, expected_intensities[e1],
#                 eps, y, e1)
#                )
    
cdef DTYPEf_t eval_convolution_kernel_given_ghat(
    DTYPEf_t t, 
    int e1, int eps, int x1, int y, DTYPEf_t [:] sigmas, int len_sigmas, 
    DTYPEf_t [:,:,:,:] g_hat, DTYPEf_t [:] Lambda, DTYPEf_t [:,:] mark_prob
) nogil:
    "Evaluates K_{x1,y}^{e1,eps} (t) assuming that |t| is in sigmas[:]"
    cdef DTYPEf_t sigma = 0.0
    if t < 0:
        sigma = -t
    else:
        sigma = t
    cdef int idx=0, not_found=1
    if sigma>sigmas[max(1,len_sigmas//4)]:
        idx=-1+len_sigmas//4
        if sigma>sigmas[max(1,len_sigmas//2)]:
            idx=-1+len_sigmas//2
            if sigma>sigmas[max(1,3*len_sigmas//4)]:
                idx=-1+3*len_sigmas//4
    while ((not_found) & (idx<len_sigmas-1)):
        if (sigmas[idx]<=sigma) & (sigma<sigmas[idx+1]):
            not_found=0
        else:
            idx+=1
    cdef DTYPEf_t result=0.0
    if t < 0:
        result = Lambda[eps]*mark_prob[eps,y]*g_hat[eps,y,e1,idx]/Lambda[e1]
    else:
        result = mark_prob[eps,y]*g_hat[e1,x1,eps,idx]
    return result  


    
cdef DTYPEf_t eval_convolution_kernel_given_ghat_and_distance(
    int m, int n, 
    int e1, int eps, int x1, int y, DTYPEf_t [:] sigmas, int len_sigmas, 
    DTYPEf_t [:,:,:,:] g_hat, DTYPEf_t [:] Lambda, DTYPEf_t [:,:] mark_prob,
    DTYPEi_t [:,:] idx_distance
) nogil:
    "Evaluates K_{x1,y}^{e1,eps} (t) assuming that |t|=quadpnts[m]-quadpnts[n]"
    cdef int idx = idx_distance[m,n]   
    cdef DTYPEf_t result=0.0
    if m < n:
        result = Lambda[eps]*mark_prob[eps,y]*g_hat[eps,y,e1,idx]/Lambda[e1]
    else:
        result = mark_prob[eps,y]*g_hat[e1,x1,eps,idx]
    return result  
    
    
    
cdef DTYPEf_t compute_primitive_of_convolution_kernel(
    DTYPEf_t t, int e1, int eps, int x1, int y, DTYPEf_t [:] sigmas, int len_sigmas,
    DTYPEf_t [:,:,:,:] g_hat, DTYPEf_t [:] Lambda, DTYPEf_t [:,:] mark_prob) nogil:
    cdef DTYPEf_t summand_1 = 0.0, summand_2 = 0.0
    cdef int n=0
    for n in range(len_sigmas-1):
        summand_2+=g_hat[eps,y,e1,n]*(sigmas[n+1]-sigmas[n])
        if sigmas[n]<t:
            summand_1 += g_hat[e1,x1,eps,n]*(sigmas[n+1]-sigmas[n])
    return mark_prob[eps,y]*summand_1 + Lambda[eps]*mark_prob[eps,y]*summand_2/Lambda[e1]
    
cdef int eval_primitive_of_convolution_kernel_at_times(
    DTYPEf_t [:] times, int len_times, DTYPEf_t [:] primitives,
    int e1, int eps, int x1, int y, DTYPEf_t [:] sigmas, int len_sigmas,
    DTYPEf_t [:,:,:,:] g_hat, DTYPEf_t [:] Lambda, DTYPEf_t [:,:] mark_prob) nogil:
    """It is assumed that primitives has been initialised to zero and it has the same lenght of times"""
    cdef DTYPEf_t prob = mark_prob[eps,y]
    cdef DTYPEf_t summand_2 = 0.0
    cdef int n=0
    for n in range(len_sigmas-1):
        summand_2+=g_hat[eps,y,e1,n]*(sigmas[n+1]-sigmas[n])
        for m in range(len_times):
            if sigmas[n]<times[m]:
                primitives[m] += g_hat[e1,x1,eps,n]*(sigmas[n+1]-sigmas[n])
    for m in range(len_times):
        primitives[m]*=prob
        primitives[m]+=Lambda[eps]*prob*summand_2/Lambda[e1]
    return 0    
    
    
cdef int interpolate_hawkes_kernels(
    int e1, int x1, int  e,
    DTYPEf_t [:,:,:,:] kappa, DTYPEf_t [:,:,:,:] new_kappa,
    DTYPEf_t [:] time, int len_grid,
    DTYPEi_t [:] idx_quadpnts, DTYPEf_t [:] quadpnts ) nogil:
    cdef int k=0, j=0
    cdef DTYPEf_t m=0.0, q=0.0
    for k in range(len_grid):
        if k==idx_quadpnts[j]:
            new_kappa[e1,x1,e,k]=kappa[e1,x1,e,j]
            m=(kappa[e1,x1,e,j+1]-kappa[e1,x1,e,j])/(quadpnts[j+1]-quadpnts[j])
            q=kappa[e1,x1,e,j+1]-m*quadpnts[j+1]
            j+=1
        else:
            new_kappa[e1,x1,e,k]=m*time[k]+q
    return 0          
                

cdef int convolute_hawkes_kernels(
    int e1, int x1, int  e,
    DTYPEf_t [:,:,:,:] kappa, DTYPEf_t [:,:,:,:] new_kappa,
    DTYPEf_t [:] time, int len_grid,
    DTYPEf_t scale, DTYPEf_t weight) nogil:
    """
    new_kappa is supposed to be initialised to zero
    """
    cdef int k=0, j=0
    for k in range(len_grid):
        for j in range(len_grid-1):
            if j<=k:
                new_kappa[e1,x1,e,k]+=kappa[e1,x1,e,j]*exp(-scale*(time[k]-time[j]))*(time[j+1]-time[j])
            else:
                new_kappa[e1,x1,e,k]+=kappa[e1,x1,e,j]*exp(-scale*(time[j]-time[k]))*(time[j+1]-time[j])
        new_kappa[e1,x1,e,k]*=weight*scale
    return 0      
    


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
    