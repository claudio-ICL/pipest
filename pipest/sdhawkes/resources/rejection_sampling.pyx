#cython: boundscheck=False, wraparound=False, nonecheck=False 
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
sys.path.append(Path(__file__).parent.parent)
#import header
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

import computation
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import dirichlet as scipy_dirichlet
from scipy.special import loggamma as LogGamma
from scipy.special import gamma as scipy_gamma

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t

class RejectionSampling:
    def __init__(self, 
            np.ndarray[DTYPEf_t, ndim=2] gamma,
            np.ndarray[DTYPEf_t, ndim=1] volimb_limits,
            int volimb_upto_level,
            int N_samples_for_prob_constraints=10**6):
        cdef int num_states = gamma.shape[0]
        self.num_states = num_states
        self.gamma=gamma
        self.volimb_limits=volimb_limits
        self.volimb_upto_level=volimb_upto_level
        cdef int num_of_st2 = len(volimb_limits)-1
        self.num_of_st2 = num_of_st2
        self.prob_constraint_physical = self.compute_prob_constraint(gamma)
    def store_gamma_and_rho(self,):
        self.physical_dir_param=self.gamma
        self.proposal_dir_param=self.rho
        self.difference_of_dir_params=self.gamma-self.rho
    def compute_prob_constraint(self, np.ndarray[DTYPEf_t, ndim=2] gamma,
            int N_samples_for_prob_constraints=10**6):
        return computation.produce_probabilities_of_volimb_constraints(
                self.volimb_upto_level, self.num_states, self.num_of_st2,
                self.volimb_limits, gamma, 
                N_samples=N_samples_for_prob_constraints)
    def store_proposal_parameters(self, int N_samples_for_prob_constraints=10**6):
        rho, is_equal, M_bound = self.produce_proposal_parameters()
        self.rho=rho
        self.is_target_equal_to_proposal=np.array(is_equal, dtype=DTYPEi)
        self.rejection_bound = np.array(M_bound, dtype=DTYPEf)
        self.prob_constraint_proposal = self.compute_prob_constraint(
                rho,N_samples_for_prob_constraints)
        self.store_gamma_and_rho()
    def produce_proposal_parameters(self,):
        print("Producing proposal density for rejection sampling.")
        cdef np.ndarray[DTYPEf_t, ndim=2] rho = np.array(self.gamma, copy=True)
        cdef np.ndarray[DTYPEi_t, ndim=1] is_equal = np.ones(self.num_states,dtype=DTYPEi)
        cdef np.ndarray[DTYPEf_t, ndim=1] M_bound = np.ones(self.num_states,dtype=DTYPEf)
        cdef DTYPEf_t sum_bid=0.0, sum_ask=0.0, imb=0.0
        cdef int st2=0, uplim=1+2*self.volimb_upto_level
        cdef DTYPEf_t l=0.0, u=0.0
        for j in range(self.num_states):
            st2=j%self.num_of_st2
            l=self.volimb_limits[st2]
            u=self.volimb_limits[st2+1]
            sum_bid=np.sum(self.gamma[j,1:uplim:2])
            sum_ask=np.sum(self.gamma[j,0:uplim:2])
            imb=(sum_bid-sum_ask)/(sum_bid+sum_ask)
            if (imb<l) or (imb>u):
                res=find_rho(self.gamma[j,:], l, u)
                if res['fun']<logB(self.gamma[j,:]):
                    is_equal[j]=0
                    rho[j,:]=np.array(res['x'],copy=True)
                    M_bound[j]=produce_M(self.gamma[j,:]-rho[j,:],l,u)
        return rho, is_equal, M_bound

def objfun(np.ndarray[DTYPEf_t, ndim=1] rho,
        np.ndarray[DTYPEf_t, ndim=1] gamma,
        DTYPEf_t l,DTYPEf_t u):
    alpha=gamma-rho
    def logM():
        return produce_logM(alpha,l,u)
    return logB(rho)+logM()
def logB(np.ndarray[DTYPEf_t, ndim=1] rho):
    return np.sum(LogGamma(rho))-LogGamma(np.sum(rho))
def find_rho(np.ndarray[DTYPEf_t, ndim=1] gamma,
        DTYPEf_t l, DTYPEf_t u,
        DTYPEi_t maxiter=10000, DTYPEf_t tol=1.0e-8):
    bounds=tuple([(tol,(1.0-tol)*gamma[k]) for k in range(len(gamma))])
    res=scipy_minimize(
        objfun,0.995*gamma,args=(gamma,l,u),
        method='TNC',jac=False,
        bounds=bounds,options={'maxiter': maxiter})
    return res
def compute_AZ(np.ndarray[DTYPEf_t, ndim=1] alpha, DTYPEf_t c):
    assert c!=0.0
    cdef DTYPEf_t bid=np.sum(alpha[1::2])
    cdef DTYPEf_t ask=np.sum(alpha[0::2])
    cdef DTYPEf_t diff=bid-ask
    cdef DTYPEf_t tot=bid+ask
    cdef DTYPEf_t A=(tot-diff/c)/(diff-tot/c)
    cdef DTYPEf_t Z=(1.0-A)*bid+(1.0+A)*ask
    return A, Z
def produce_candidate(np.ndarray[DTYPEf_t, ndim=1] alpha,DTYPEf_t c):
    A,Z=compute_AZ(alpha,c)
    cdef np.ndarray[DTYPEf_t, ndim=1] v=np.zeros((len(alpha),),dtype=DTYPEf)
    v[1::2]=(1.0-A)*alpha[1::2]/Z
    v[0::2]=(1.0+A)*alpha[0::2]/Z
    return v
def logdir(np.ndarray[DTYPEf_t, ndim=1] v,np.ndarray[DTYPEf_t, ndim=1] alpha): #for the production of M
    return np.sum(alpha*np.log(v))
def produce_M(np.ndarray[DTYPEf_t, ndim=1] alpha,DTYPEf_t l,DTYPEf_t u):
    cdef np.ndarray[DTYPEf_t, ndim=1] v_l=produce_candidate(alpha,l)
    assert np.all(v_l>=0.0)
    cdef np.ndarray[DTYPEf_t, ndim=1] v_u=produce_candidate(alpha,u)
    assert np.all(v_u>=0.0)
    return np.exp(max(logdir(v_l,alpha),logdir(v_u,alpha)))
def produce_logM(np.ndarray[DTYPEf_t, ndim=1] alpha,DTYPEf_t l,DTYPEf_t u,DTYPEf_t tol=1.0e-10):
    v_l=np.maximum(tol,produce_candidate(alpha,l))
    v_u=np.maximum(tol,produce_candidate(alpha,u))
    return (max(logdir(v_l,alpha),logdir(v_u,alpha)))
