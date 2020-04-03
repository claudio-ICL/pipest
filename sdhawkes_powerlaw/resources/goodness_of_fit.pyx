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
from scipy import stats

import numpy as np
cimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bisect
from libc.math cimport pow
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport isnan
DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t

import computation



class good_fit:
    def __init__(self,
              int n_event_types,
              int n_states,
              np.ndarray[DTYPEf_t, ndim=1] base_rate,
              np.ndarray[DTYPEf_t, ndim=3] imp_coef,
              np.ndarray[DTYPEf_t, ndim=3] dec_coef,
              np.ndarray[DTYPEf_t, ndim=3] trans_prob,   
              np.ndarray[DTYPEf_t, ndim=1] times,
              np.ndarray[DTYPEi_t, ndim=1] events,
              np.ndarray[DTYPEi_t, ndim=1] states,
              parallel=True,   
              compute_total_residuals = False,
              str type_of_input = 'simulated' #it can be either 'simulated' or 'empirical'
              ):
        self.n_event_types = n_event_types
        self.n_states =n_states
        self.base_rate = base_rate
        self.imp_coef = imp_coef
        self.dec_coef = dec_coef
        self.trans_prob = trans_prob
        self.ratios = imp_coef/(dec_coef - 1)
        self.times = times
        self.events= events
        self.states = states
        self.type_of_input=type_of_input #it can be either 'simulated' or 'empirical'
        self.state_traj_times,self.state_traj_states = computation.produce_state_trajectory(states, times)
        self.labelled_times,self.count = computation.distribute_times_per_event_state(
                                                        n_event_types,
                                                        n_states,
                                                        times,
                                                        events,
                                                        states)
        cdef int len_labelled_times = self.labelled_times.shape[2]
        self.len_labelled_times = len_labelled_times
        self.residuals = self.compute_residuals(parallel=parallel)
        self.timechanged_pp = self.produce_timechanged_point_process(self.residuals,self.n_event_types)
        if compute_total_residuals:
            self.total_residuals = self.compute_total_residuals_pc()
            self.omnibus_pp = self.produce_timechanged_point_process(self.total_residuals,self.n_event_types*self.n_states)


    def produce_timechanged_point_process(self,residuals,n_marks):
        event_times = []
        for e in range(n_marks):
            event_times.append(np.cumsum(residuals[e]))
        new_pp=pd.DataFrame({'mark':np.zeros_like(event_times[0],dtype=np.int),
                         'time':np.array(event_times[0],dtype=np.float)})
        for e in range(1,max(n_marks,1)):
            to_add=pd.DataFrame({'mark':e*np.ones_like(event_times[e],dtype=np.int),
                             'time':np.array(event_times[e],dtype=np.float)})
            new_pp=pd.concat([new_pp,to_add],ignore_index=True)
        new_pp.sort_values(by='time',inplace=True)
        new_pp.reset_index(drop=True,inplace=True)
        return new_pp
    

    def ks_test_on_residuals(self):
        print('Kolmogorov-Smirnov test to check that the residuals are iid with distribution Exp(1)')
        ks_ans=list(map(lambda x: stats.kstest(x,'expon'),self.residuals))
        kstest_residuals=[]
        for e in range(len(ks_ans)):
            ks_stat = ks_ans[e][0]
            p_val = ks_ans[e][1]
            kstest_residuals.append([e,ks_stat,p_val])
            print('event type={}, ks_answer: {}'.format(e,ks_ans[e]))
        df=pd.DataFrame(kstest_residuals)
        df.columns=['event','ks_stat','p_val']    
        self.kstest_residuals=df    
    def ks_test_on_total_residuals(self):
        print('Kolmogorov-Smirnov test to check that the total residuals are iid with distribution Exp(1)')
        ks_ans=list(map(lambda x: stats.kstest(x,'expon'),self.total_residuals))
        kstest_tot_res=[]
        for n in range(len(ks_ans)):
            e=n//self.n_states
            x=n%self.n_states
            kstest_tot_res.append([e,x,ks_ans[n][0],ks_ans[n][1]])
            print('event type={}, state={}, ks_answer: {}'.format(e,x,ks_ans[n]))
        df=pd.DataFrame(kstest_tot_res)
        df.columns=['event','state','ks_stat','p_val']    
        self.kstest_total_residuals=df
        
    def ad_test_on_residuals(self,distr='expon'):
        print('Anderson-Darling test to check distribution of residuals')
        print('Null hypothesis is "{}" '.format(distr))
        if distr=='expon':
            ad_ans=list(map(lambda x: stats.anderson(x,'expon'),self.residuals))
        elif distr=='norm':
            ad_ans=list(map(lambda x: stats.anderson(np.log(x),distr),self.residuals))
        else:
            ad_ans=list(map(lambda x: stats.anderson(x-np.mean(x),distr),self.residuals))
        adtest_residuals=[]
        print('Significance levels: {}'.format(ad_ans[0][2]))
        print('Critical values: {}'.format(ad_ans[0][1]))
        for e in range(len(ad_ans)):
            ad_stat = ad_ans[e][0]
            idx = min(len(ad_ans[e][1])-1,bisect.bisect_right(ad_ans[e][1],ad_stat))
            signif_lev_noReject = ad_ans[e][2][idx]
            critical_val = ad_ans[e][1]
            adtest_residuals.append([e,ad_stat,critical_val,signif_lev_noReject])
            print('event type={}, ad_stat: {}'.format(e,ad_stat))
        if (distr=='expon'):    
            df=pd.DataFrame(adtest_residuals)
            df.columns=['event','ad_stat','critical_val','signif_lev_noReject']    
            self.adtest_residuals=df
       
    
        
    def obj_compute_res(self,int e):
        cdef int process_id = os.getpid()
        print("goodness_of_fit.compute_residuals. component_e: {}; process_id: pid{}".format(e,process_id))
        return computation.compute_event_residual(
            e, self.n_event_types, self.n_states, self.len_labelled_times,
            self.base_rate[e], self.dec_coef[:,:,e], self.ratios[:,:,e],
            self.labelled_times, self.count)
    
    def compute_residuals(self, parallel=True):
        cdef double run_time= - time.time()
        cdef int num_processes = max(1, self.n_event_types)
        if parallel:
            print('I am computing residuals in parallel. num_process: {}'.format(num_processes))
            pool=mp.Pool(num_processes)
            residuals=pool.map(
                self.obj_compute_res,list(range(self.n_event_types))
            )                                
            pool.close()
            pool.join()
        else:
            print("I am computing residuals serially")
            residuals = list(
                map(self.obj_compute_res,list(range(self.n_event_types))
                )
            )
        run_time+=time.time()
        print('Computation of residuals terminates. run_time={}'.format(run_time))
        return residuals
    def obj_pc_compute_tot_res(self,int n):
        cdef int e = n//self.n_states
        cdef int x = n%self.n_states
        return computation.compute_total_residual_ex(e,x,
                                          self.base_rate[e],
                                          self.dec_coef[:,:,e],
                                          self.ratios[:,:,e],
                                          self.labelled_times,
                                          self.count,
                                          self.trans_prob[:,e,x],
                                          self.state_traj_times,
                                          self.state_traj_states)
    def compute_total_residuals_pc(self):
        cdef double run_time= - time.time()
        cdef int n_parallel = max(1,min(self.n_event_types*self.n_states,mp.cpu_count()))
        print('Compute total residuals in parallel on {} cpus'.format(n_parallel))
        cdef list eval_list = list(range(self.n_event_types*self.n_states))
#         print('good_fit.compute_total_residuals_pc: eval_list = \n {}',format(eval_list))
        pool=mp.Pool(n_parallel)        
        residuals=pool.map(
            self.obj_pc_compute_tot_res, eval_list
        )               
        pool.close()
        pool.join()
        run_time+=time.time()
        print('Parallel computation of total residuals terminates. run_time={}'.format(run_time))
        return residuals
    
    def qq_plot_total_residuals(self,save_fig=False,path=path_pipest,name='qq_plot_total_residuals'):
        cdef int n = len(self.total_residuals)
        cdef int n_cols=max(1,min(4,self.n_states))
        cdef int n_rows=1+n//n_cols
        qq_plots=[]
        fig=plt.figure(figsize=(12,4*n_rows))
        for i in range(n):
            e = i//self.n_states
            x = i%self.n_states
            ax=fig.add_subplot(n_rows,n_cols,i+1)
            qq_plots.append(stats.probplot(self.total_residuals[e*self.n_states+x],dist=stats.expon,plot=ax))
            ax.set_title('Event type: {}, state: {}'.format(e,x))
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.suptitle('QQ plot for total residuals')
        if save_fig:
            fname=path+'/'+name
            fig.savefig(fname)
        plt.show()
        
    def qq_plot_residuals(self, int index_of_first_event_type=0, int tot_event_types = 4, int title_per_event_type=1,
                          fig_suptitle='QQ plot residuals',
                          save_fig=False,path=path_pipest,name='qq_plot_residuals',):
        cdef int n=len(self.residuals)
        cdef int n_rows=1+n//4
        cdef int n_cols=min(4,tot_event_types)
        qq_plots=[]
        fig=plt.figure(figsize=(1+3*n_cols,4*n_rows))
        for i in range(n):
            ax=fig.add_subplot(n_rows,n_cols,i+1)
            qq_plots.append(stats.probplot(self.residuals[i],dist=stats.expon,plot=ax))
            if title_per_event_type:
                ax.set_title('Event type: {}'.format(i+index_of_first_event_type))
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.suptitle(fig_suptitle)
        if save_fig:
            fname=path+'/'+name
            plt.savefig(fname)
        plt.show()
    
    def qq_plot_omnibus(self):
        fig=plt.figure()
        x=self.n_states*self.n_event_types*np.diff(self.omnibus_pp['time'].values)
        ax=fig.add_subplot(111)
        qq_plot=stats.probplot(x,dist=stats.expon,plot=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title('QQ plot for omnibus point process')
        plt.show()
                            

        

        
        
