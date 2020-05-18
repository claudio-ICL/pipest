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
import time
import datetime
import pickle

from cython.parallel import prange

import pandas as pd
import numpy as np
cimport numpy as np
from scipy import linalg
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


import mle_estimation as mle_estim
import nonparam_estimation as nonparam_estim
import goodness_of_fit
import uncertainty_quant
import computation
import simulation
import impact_profile
import dirichlet
import lob_model


class HawkesKernel:
    def __init__(self, int num_event_types=1,int num_states=1, 
                 int num_quadpnts = 100, two_scales=False, DTYPEf_t quad_tmax=10.0, DTYPEf_t quad_tmin = 1.0e-2):
        self.num_event_types=num_event_types
        self.num_states=num_states
        self.quadrature=computation.Partition(num_pnts=num_quadpnts, two_scales=two_scales, t_max=quad_tmax, t_min=quad_tmin)
    def store_parameters(self, np.ndarray[DTYPEf_t, ndim=3] alphas, np.ndarray[DTYPEf_t, ndim=3] betas):
        d_E, d_S, d_EE = alphas.shape[0],alphas.shape[1],alphas.shape[2]
        d_E_1, d_S_1, d_EE_1 = betas.shape[0],betas.shape[1],betas.shape[2]
        assert (d_E == self.num_event_types) & (d_E_1 == self.num_event_types)
        assert (d_EE == self.num_event_types) & (d_EE_1 == self.num_event_types)
        assert (d_S == self.num_states) & (d_S_1 == self.num_states)
        self.alphas = alphas
        self.betas = betas
    def compute_values_at_quadpnts_from_parametric_kernel(self,):
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef int num_quadpnts = self.quadrature.num_pnts
        cdef np.ndarray[DTYPEf_t, ndim=4] values = np.zeros((d_E,d_S,d_E,num_quadpnts), dtype=DTYPEf)
        cdef int e1=0,x1=0,e=0
        for e in range(d_E):
            for e1 in range(d_E):
                for x1 in range(d_S):
                    values[e1,x1,e,:] =\
                    self.alphas[e1,x1,e]*np.power(self.quadrature.partition[:num_quadpnts]+1.0,-self.betas[e1,x1,e])
        self.store_values_at_quadpnts(values, from_param=True)            
        
    def store_values_at_quadpnts(self, np.ndarray[DTYPEf_t, ndim=4] values, from_param = False):
        if from_param:
            self.values_at_quadpnts_param = values
        else:
            self.values_at_quadpnts = values
    def compute_L1_norm(self):
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef np.ndarray[DTYPEf_t, ndim=3] L1_norm = np.zeros((d_E, d_S, d_E), dtype=DTYPEf)
        L1_norm = self.alphas/(self.betas - 1.0)
        return L1_norm
    def store_L1_norm(self):
        self.L1_norm=self.compute_L1_norm()
    def store_spectral_radius(self,):
        cdef int d_E = self.num_event_types
        cdef int d_S = self.num_states
        cdef np.ndarray[DTYPEf_t, ndim=2] eigenvalues = np.zeros((d_S,d_E), dtype=DTYPEf)
        for x in range(d_S):
            eigenvalues[x,:]=linalg.eigvals(self.L1_norm[:,x,:])
        cdef np.ndarray[DTYPEf_t, ndim=1] radii = np.amax(eigenvalues, axis=1)
        cdef DTYPEf_t max_radius = np.amax(radii)
        self.spectral_radii = radii
        self.max_spectral_radius = max_radius
        
        

class Calibration:
    def __init__(self, data, str name_of_model='sdhawkes_model', 
                 partial=False, int event_type=0,
                 str type_of_preestim = 'ordinary_hawkes' #or 'nonparam'
                ):
        self.data=data
        self.name_of_model=name_of_model
        os_info = os.uname()
        self.type_of_preestim=type_of_preestim
        print('Calibration is being performed on the following machine:\n {}'.format(os_info))
        self.os_info=os_info    
        self.partial=partial
        cdef list mle_info = []
        self.mle_info = mle_info
        if partial:
            print("Partial calibration for event_type={}".format(event_type))
            self.event_type=event_type
    def store_runtime_for_state_processes(self,DTYPEf_t run_time):
        self.runtime_for_state_processes = run_time
    def store_mle_info(self, list results_of_estimation):
        info_e={}
        for res in results_of_estimation:
            info_e=copy.copy(res)
            del info_e["base_rate"]
            del info_e["imp_coef"]
            del info_e["dec_coef"]
            self.mle_info.append(info_e)    

class Liquidator:
    def __init__(self,
                 DTYPEf_t initial_inventory,
                 DTYPEf_t base_rate,
                 np.ndarray[DTYPEf_t, ndim=2] imp_coef,
                 np.ndarray[DTYPEf_t, ndim=2] dec_coef,
                 str type_of_liquid, 
                 DTYPEf_t control,
                 str control_type,
                 time_start = None,
                 termination_time = None,
                ):
        self.initial_inventory = initial_inventory
        self.base_rate = base_rate
        self.imp_coef = imp_coef
        self.dec_coef = dec_coef
        self.type_of_liquid = type_of_liquid
        self.control = control
        self.control_type=control_type
        if not time_start == None:
            self.store_time_start(time_start)
        if not termination_time == None:
            self.store_termination_time(termination_time)
        if type_of_liquid =='constant_intensity':
            if control_type=='fraction_of_inventory':
                self.num_orders=int(ceil(initial_inventory/control))
    def store_time_start(self, DTYPEf_t time_start):
        self.time_start=copy.copy(time_start)
    def store_termination_time(self, DTYPEf_t termination_time):
        self.termination_time = copy.copy(termination_time)
    def store_inventory_trajectory(self,times,inventory):
        cdef np.ndarray[DTYPEf_t, ndim=2] t=times.reshape(-1,1)
        cdef np.ndarray[DTYPEf_t, ndim=2] inv=inventory.reshape(-1,1)
        cdef np.ndarray[DTYPEf_t, ndim=2] trajectory=np.concatenate([t,inv],axis=1)
        self.inventory_trajectory=trajectory
    def store_bm_impact_profile(self,
                                np.ndarray[DTYPEf_t, ndim=2] bm_impact_profile,
                                np.ndarray[DTYPEf_t, ndim=2] history_of_intensity):
        df_1=pd.DataFrame({'time': bm_impact_profile[:,0], 'impact': bm_impact_profile[:,1]})
        df_2=pd.DataFrame({'time': history_of_intensity[:,0], 'intensity': history_of_intensity[:,1]})
        df=df_1.merge(df_2,how='inner',on='time')
        self.bm_impact_profile=df
    def store_impact_profile(self,np.ndarray[DTYPEf_t, ndim=2] impact_profile):
        df=pd.DataFrame({'time': impact_profile[:,0], 'impact': impact_profile[:,1]})
        self.impact_profile=df    


class SDHawkes:
    """
    This class implements state-dependent Hawkes processes with power-law kernels, a subclass of hybrid marked point
    processes.
    The main features it provides include simulation and statistical inference (estimation).

    :type number_of_event_types: int
    :param number_of_event_types: number of different event types.
    :type number_of_states: int
    :param number_of_states: number of possible states.
    :type events_labels: list of strings
    :param events_labels: names of the different event types.
    :type states_labels: list of strings
    :param states_labels: names of the possible states.
    """
    def __init__(self, int number_of_event_types=1, int number_of_states = 0,
                 events_labels=False, states_labels=False,
                 int number_of_lob_levels=1,
                 int volume_imbalance_upto_level =1,
                 list list_of_n_states=[3,5], int st1_deflationary=0, int st1_inflationary=2, int st1_stationary=1,
                 str name_of_model = 'SDHawkel_model',
                ):
        """
        Initialises an instance.

        :type number_of_event_types: int
        :param number_of_event_types: number of different event types.
        :type number_of_states: int
        :param number_of_states: number of possible states.
        :type events_labels: list of strings
        :param events_labels: names of the different event types.
        :type states_labels: list of strings
        :param states_labels: names of the possible states.
        """
        #print("Initialising an instance of SDHawkes")
        cdef str path_pipest = os.path.abspath('./')
        cdef int n=0
        while (not os.path.basename(path_pipest)=='pipest') and (n<6):
            path_pipest=os.path.dirname(path_pipest)
            n+=1 
        if not os.path.basename(path_pipest)=='pipest':
            raise ValueError("path_pipest not found. Instead: {}".format(path_pipest))
        cdef str path_models=path_pipest+'/models'    
        cdef str path_sdhawkes=path_pipest+'/sdhawkes_powerlaw'
        self.path_sdhawkes=path_sdhawkes
        self.path_pipest=path_pipest
        self.path_models=path_models
        self.name_of_model = name_of_model
        self.datetime_of_initialisation=datetime.datetime.now()
        if not events_labels:
            self.events_labels = list(range(number_of_event_types))
        else:
            self.events_labels = events_labels
        if not states_labels:
            self.states_labels = list(range(number_of_states))
        else:
            self.states_labels = states_labels
        self.state_enc = lob_model.state_encoding(
            list_of_n_states=list_of_n_states,
            st1_deflationary=st1_deflationary, st1_inflationary=st1_inflationary, st1_stationary=st1_stationary
        )
        if number_of_states == 0:
            self.number_of_states = self.state_enc.tot_n_states
        else:
            self.number_of_states = number_of_states
            if number_of_states != self.state_enc.tot_n_states:
                message='SDHawkes.__init__: WARNING:'
                message+= 'Given number_of_states inconststent with list_of_n_states'
                message+= ' \n number_of_states={},'.format(number_of_states)
                message+= '  list_of_n_states={}'.format(list_of_n_states)
                print(message)
        self.number_of_event_types = number_of_event_types
        self.n_levels = number_of_lob_levels
        self.volume_enc = lob_model.volume_encoding(
            self.state_enc.volimb_limits,self.n_levels,volume_imbalance_upto_level)
        cdef np.ndarray[DTYPEf_t, ndim=3] transition_probabilities = \
        np.zeros((number_of_states, number_of_event_types, number_of_states),dtype=DTYPEf)
        self.transition_probabilities = transition_probabilities
        cdef np.ndarray[DTYPEf_t, ndim=1] base_rates = np.zeros(number_of_event_types,dtype=DTYPEf)
        self.base_rates = base_rates
        cdef np.ndarray[DTYPEf_t, ndim=3] impact_coefficients =\
        np.zeros((number_of_event_types, number_of_states, number_of_event_types),dtype=DTYPEf)
        self.impact_coefficients = impact_coefficients
        cdef np.ndarray[DTYPEf_t, ndim=3] decay_coefficients = np.ones(
            (number_of_event_types, number_of_states, number_of_event_types),dtype=DTYPEf)
        self.decay_coefficients = decay_coefficients
        cdef np.ndarray[DTYPEf_t, ndim=3] impact_decay_ratios = \
        np.zeros((number_of_event_types, number_of_states, number_of_event_types),dtype=DTYPEf)
        self.impact_decay_ratios = impact_decay_ratios
        cdef np.ndarray[DTYPEf_t, ndim=2] dirichlet_param = np.ones((number_of_states,2*number_of_lob_levels),dtype=DTYPEf)
        self.dirichlet_param = dirichlet_param
        self.hawkes_kernel=HawkesKernel(self.number_of_event_types, self.number_of_states)
    def set_name_of_model(self,str name):
        self.name_of_model=name
    def dump(self, str name='', str path=''):
        if name=='':
            name=self.name_of_model
        if path=='':
            path=self.path_models
        print('\nI am dumping the model with name "{}" in the directory: {}/ \n'
              .format(name,path))
        with open(path+'/'+name,'wb') as outfile:
            pickle.dump(self,outfile)       
    def introduce_liquidator(self,
                             DTYPEf_t initial_inventory = 0.0,
                             time_start=None,
                             type_of_liquid='with_the_market',
                             liquidator_base_rate=None,
                             self_excitation=False,
                             liquidator_control=0.5,
                             liquidator_control_type='fraction_of_inventory'):
        cdef str control_type = copy.copy(liquidator_control_type)
        cdef DTYPEf_t control=copy.copy(liquidator_control)
        cdef int idx_type = 0 
        if type_of_liquid=='with_the_market':
            idx_type=0
        elif type_of_liquid=='against_the_market':
            idx_type=1
        elif type_of_liquid!='constant_intensity':
            raise ValueError ('{}: type_of_liquid not recognised'.format(type_of_liquid))
        if liquidator_base_rate==None:
            base_rates=np.insert(self.base_rates,0,values=self.base_rates[idx_type],axis=0)
        else:
            base_rates=np.insert(self.base_rates,0,values=liquidator_base_rate,axis=0)
        """    
        First, impact_coeff of other participants as response to the liquidator' interventions is set as the impact_coeff of the same participants as response to sell_I
        """
        impact_coeff=np.insert(self.impact_coefficients,
                               obj=0,
                               values=self.impact_coefficients[0,:,:],
                               axis=0)
        """
        Second, impact_coeff of liquidator as response to the other participants is set as the impact_coeff of either sell_I or buy_I depending on type_of_liquid
        """ 
        if type_of_liquid == 'constant_intensity':
            impact_coeff=np.insert(
                impact_coeff,obj=0,
                values=np.zeros((1+self.number_of_event_types,self.number_of_states),dtype=DTYPEf),
                axis=2)
        else:    
            impact_coeff=np.insert(impact_coeff,obj=0,values=impact_coeff[:,:,idx_type],axis=2)
            if not self_excitation:
                impact_coeff[0,:,0] = np.zeros(self.number_of_states,dtype=DTYPEf)
        """
        Third, decay_coeff of other participants as response to the liquidator' interventions is set as the decay_coeff of the same participants as response to sell_I
        """
        decay_coeff=np.insert(self.decay_coefficients,
                              obj=0,
                              values=self.decay_coefficients[0,:,:],
                              axis=0)
        """
        Fourth, decay_coeff of liquidator as response to the other participants is set as the decay_coeff of either sell_I or buy_I depending on type_of_liquid
        """
        decay_coeff=np.insert(decay_coeff,obj=0,values=decay_coeff[:,:,idx_type],axis=2)
        """
        impact_decay_ratios updated accordingly
        """
        impact_decay_ratios=impact_coeff/(decay_coeff-1)
        """
        Finally, transition probabilities are updated, but this does not matter since   it will never be used, but it is done to modify the dimension
        """
        trans_prob=np.insert(self.transition_probabilities,
                             obj=0,
                             values=self.transition_probabilities[:,0,:],
                             axis=1)
        self.base_rates=base_rates
        self.impact_coefficients=impact_coeff
        self.decay_coefficients=decay_coeff
        self.impact_decay_ratios = impact_decay_ratios
        self.transition_probabilities=trans_prob
        self.number_of_event_types +=1
        if self_excitation:
            with_without_self_excitation='with'
        else:
            with_without_self_excitation='without'
        print(('SDHawkes.introduce_liquidator:\n  type_of_liquid:{},' 
               +' {} self excitation.'
               +'\n new number of event types= {}')
              .format(type_of_liquid,with_without_self_excitation,self.number_of_event_types))
        self.liquidator = Liquidator(initial_inventory,
                                     self.base_rates[0],
                                     self.impact_coefficients[:,:,0],
                                     self.decay_coefficients[:,:,0],
                                     type_of_liquid,
                                     control,
                                     control_type,
                                     time_start = time_start
                                    )

    def configure_liquidator_param(self,
                                   initial_inventory=None,
                                   time_start=None,
                                   liquidator_base_rate=None,
                                   type_of_liquid='with_the_market',
                                   self_excitation=False,
                                   liquidator_control=0.5,
                                   liquidator_control_type='fraction_of_inventory'
                                  ):
        
        "Notice that it is assumed that the liquidator has already been introduced, with event type = 0"
        if initial_inventory==None:
            initial_inventory=self.liquidator.initial_inventory
        if time_start == None:
            try:
                time_start=self.liquidator.time_start
            except:
                pass
        cdef str control_type = copy.copy(liquidator_control_type)
        cdef DTYPEf_t control=copy.copy(liquidator_control)
        cdef int idx_type = 0
        if type_of_liquid=='with_the_market':
            idx_type=1
        elif type_of_liquid=='against_the_market':
            idx_type=2
        elif type_of_liquid!='constant_intensity':
            raise ValueError ('{}: type_of_liquid not recognised'.format(type_of_liquid))
        rates=self.base_rates
        imp_coef=self.impact_coefficients
        decay_coef=self.decay_coefficients
        i_d_ratio=self.impact_decay_ratios
        if liquidator_base_rate==None:
            rates[0]=rates[idx_type]
        else:
            rates[0]=liquidator_base_rate
        if type_of_liquid=='constant_intensity':
            imp_coef[:,:,0]=np.zeros((self.number_of_event_types,self.number_of_states),dtype=DTYPEf)
        else:
            imp_coef[:,:,0]=imp_coef[:,:,idx_type]
            if not self_excitation:
                imp_coef[0,:,0]=np.zeros(self.number_of_states,dtype=DTYPEf)
        decay_coef[:,:,0]=decay_coef[:,:,idx_type]
        i_d_ratio=imp_coef/(decay_coef-1)
        self.base_rates=rates
        self.impact_coefficients=imp_coef
        self.decay_coefficients=decay_coef
        self.impact_decay_ratios=i_d_ratio
        self.liquidator = Liquidator(initial_inventory,
                                     self.base_rates[0], self.impact_coefficients[:,:,0], self.decay_coefficients[:,:,0],
                                     type_of_liquid, control, control_type,
                                     time_start = time_start)
        
    def set_transition_probabilities(self, transition_probabilities):
        r"""
        Fixes the transition probabilities :math:`\phi` of the state-dependent Hawkes process.
        The are used to :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.simulate` and
        :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.compute_total_residuals`.

        :type transition_probabilities: 3D numpy array
        :param transition_probabilities: shape should be :math:`(d_x, d_e,d_x)` where :math:`d_e` and :math:`d_x`
                                         are the number of event types and states, respectively.
                                         The entry :math:`i, j, k` is the probability of going from state :math:`i`
                                         to state :math:`k` when an event of type :math:`j` occurs.
        :return:
        """
        'Raise ValueError if the given parameters do not have the right shape'
        if np.shape(transition_probabilities) != (self.number_of_states, self.number_of_event_types,
                                                  self.number_of_states):
            raise ValueError('given transition probabilities have incorrect shape')
        assert np.all(transition_probabilities>=0.0)
        assert np.all(transition_probabilities<=1.0)    
        cdef np.ndarray[DTYPEf_t, ndim=3] phi = np.array(transition_probabilities,copy=True,dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=2] masses = np.ones((self.number_of_states, self.number_of_event_types), dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=3] normalisation = np.ones((self.number_of_states, self.number_of_event_types, self.number_of_states), dtype=DTYPEf)
        masses = np.sum(phi, axis=2)
        tol=1.0e-10
        normalisation=(1.0+tol)*np.repeat(np.expand_dims(masses,axis=2),self.number_of_states,axis=2)
        idx_zero_mass = (normalisation<=0.0)
        phi[idx_zero_mass]=tol/float(self.number_of_states)
        normalisation=np.maximum(tol,normalisation)
        phi/=normalisation
        if not np.all(np.sum(phi,axis=2)<=1.0):
            print(np.sum(phi,axis=2))
        assert np.all(np.sum(phi,axis=2)<=1.0)
        if np.any(np.isnan(phi)):
            print(phi)
        assert not np.any(np.isnan(phi))
        self.transition_probabilities = phi
        self.inflationary_pressure, self.deflationary_pressure, asymmetry = computation.assess_symmetry(
            self.number_of_states,
            self.base_rates,
            self.impact_decay_ratios,
            self.transition_probabilities,
            self.state_enc.deflationary_states,
            self.state_enc.inflationary_states)
        print('SDHawkes: asymmetry in transition_probabilities = {}'.format(asymmetry))
        print('Transition probabilities have been set')
    
    def enforce_symmetry_in_transition_probabilities(self, int is_liquidator_present = 0):
        cdef int i = is_liquidator_present
        cdef np.ndarray[DTYPEf_t, ndim=3] ratios = copy.copy(self.impact_decay_ratios[i:,:,i:])
        cdef np.ndarray[DTYPEf_t, ndim=1] nu = copy.copy(self.base_rates[i:])
        cdef np.ndarray[DTYPEf_t, ndim=3] phi = copy.copy(self.transition_probabilities[:,i:,:])
        new_phi = computation. produce_phi_for_symmetry(
            self.number_of_states,
            nu,ratios,phi,
            self.state_enc.deflationary_states,
            self.state_enc.inflationary_states)
        self.transition_probabilities[:,i:,:] = copy.copy(new_phi)
        self.inflationary_pressure, self.deflationary_pressure, asymmetry = computation.assess_symmetry(
            self.number_of_states,
            nu,
            ratios,
            self.transition_probabilities[:,i:,:],
            self.state_enc.deflationary_states,
            self.state_enc.inflationary_states)
        print('SDHawkes: new asymmetry in transition_probabilities = {}'.format(asymmetry))
        

    def set_hawkes_parameters(self, np.ndarray[DTYPEf_t, ndim=1] base_rates,
                              np.ndarray[DTYPEf_t, ndim=3 ]impact_coefficients,
                              np.ndarray[DTYPEf_t, ndim=3] decay_coefficients):
        r"""
        Fixes the parameters :math:`(\nu, \alpha, \beta)` that define the intensities (arrival rates) of events.
        The are used in
        :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.simulate`,
        :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.compute_events_residuals`
        and :py:meth:`~mpoints.hybrid_hawkes_exp.HybridHawkesExp.compute_total_residuals`.

        :type base_rates: 1D numpy array
        :param base_rates: one base rate :math:`\nu_e` per event type :math:`e`.
        :type impact_coefficients: 3D numpy array
        :param impact_coefficients: the alphas :math:`\alpha_{e'xe}`.
        :type decay_coefficients: 3D numpy array
        :param decay_coefficients: the betas :math:`\beta_{e'xe}`.
        :return:
        """
        'Raise ValueError if the given parameters do not have the right shape'
        if np.shape(base_rates) != (self.number_of_event_types,):
            raise ValueError('given base rates have incorrect shape')
        if np.shape(impact_coefficients) != (self.number_of_event_types, self.number_of_states,
                                              self.number_of_event_types):
            raise ValueError('given impact coefficients have incorrect shape')
        if np.shape(decay_coefficients) != (self.number_of_event_types, self.number_of_states,
                                             self.number_of_event_types):
            raise ValueError('given decay coefficients have incorrect shape')
        cdef np.ndarray[DTYPEf_t, ndim=1] nu = np.array(base_rates,copy=True,dtype=DTYPEf)    
        self.base_rates = nu
        cdef np.ndarray[DTYPEf_t, ndim=3] alphas = np.array(impact_coefficients,copy=True,dtype=DTYPEf)
        self.impact_coefficients = alphas
        cdef np.ndarray[DTYPEf_t, ndim =3] betas = np.array(decay_coefficients,copy=True,dtype=DTYPEf)
        self.decay_coefficients = betas
        cdef np.ndarray[DTYPEf_t, ndim=3] ratios = np.divide(self.impact_coefficients, self.decay_coefficients-1)
        self.impact_decay_ratios = ratios
        self.hawkes_kernel.store_parameters(alphas,betas)
        self.hawkes_kernel.compute_L1_norm()
        print('Hawkes parameters have been set')
        
        
    def set_dirichlet_parameters(self, dir_param, N_samples_for_prob_constraints = 9999):
        r"""
        Fixes the parameters :math:`\gamma` that defines the dirichlet distributions of volume proportions.

        :type dir_param: 3D numpy array
        :param dir_param: for every event :math:`e` and every state :math:`s`, 
        the :math:`2n`-dimensional vector dir_param[e,s,:] is the dirichlet parameters of 
        volume distribution when event is :math:`e` and  state is :math:`s`.
        The integer :math:`n` denotes number of levels in the order book.
        :return:
        """
        'Raise ValueError if the given parameters do not have the right shape'
        if dir_param.shape != (self.number_of_states,2*self.n_levels):
            raise ValueError('given parameter has incorrect shape, given shape={}, expected shape=({},{})'.format(dir_param.shape,self.number_of_states,2*self.n_levels))
        idx_neg=np.array((dir_param<=0.0), dtype=np.bool) 
        cdef int s=0, l=0
        cdef str side=''
        if np.any(idx_neg):
            print('SDHawkes.set_dirichlet_parameters. ERROR: The following components are non-positive:')
            for s in range(self.number_of_states):
                for l in range(2*self.n_levels):
                    if idx_neg[s,l]:
                        if l%2==0:
                            side='ask'
                        else:
                            side='bid'
                        print(' state={}, side={}, level={}: dir_param={}'.format(s,side,1+l//2,dir_param[s,l]))
            raise ValueError('SDHawkes.set_dirichlet_parameters. Given parameters has non-positive components')
        cdef np.ndarray[DTYPEf_t, ndim = 2] dirichlet_param = np.array(dir_param,copy=True,dtype=DTYPEf)   
        self.dirichlet_param = dirichlet_param
        self.volume_enc.store_dirichlet_param(dirichlet_param, num_of_st2 = self.state_enc.array_of_n_states[1])
        self.volume_enc.store_param_for_rejection_sampling()
        print('Dirichlet parameters have been set')
    
    def initialise_from_partial(self,list partial_models, 
                                dump_after_merging=True, str name_of_model='', str path = '',
                                type_of_input = 'simulated', parallel = False):
        """
        It is assumed that all part models have been estimated on the same sample.
        """
        cdef int e=0, event_type=0
        cdef list list_of_mle_results = []
        for e in range(len(partial_models)):
            model=partial_models[e]
            assert model.n_levels == self.n_levels
            assert np.all(model.state_enc.list_of_n_states == self.state_enc.list_of_n_states)
            assert model.number_of_event_types == self.number_of_event_types
            assert model.number_of_states == self.number_of_states
            event_type=copy.copy(model.mle_estim.results_of_estimation[0].get("component_e"))
            print('I am reading from model number {}, referring to event_type={}'.format(e, event_type))
            print("model name is: {}".format(model.name_of_model))
            print("len(mle_estim.results_of_estimation) = {}".format(len(model.mle_estim.results_of_estimation)))
            for res in model.mle_estim.results_of_estimation:
                list_of_mle_results.append(res)
        print("{} mle results have been loaded".format(len(list_of_mle_results)))
        if len(list_of_mle_results)!= self.number_of_event_types:
            warning_message="WARNING: It was expected that len(list_of_mle_results) == self.number_of_event_types"
            warning_message+="\nBut:\n"
            warning_message+="len(list_of_mle_results)={};  ".format(len(list_of_mle_results))
            warning_message+="self.number_of_event_types={}.\n".format(self.number_of_event_types)
            print(warning_message)
            print("list_of_mle_results=\n{}".format(list_of_mle_results))
        self.create_mle_estim(type_of_input=type_of_input,store_trans_prob=True, store_dirichlet_param=True)
        self.mle_estim.store_results_of_estimation(list_of_mle_results)
        self.mle_estim.store_hawkes_parameters()
        self.mle_estim.create_goodness_of_fit(parallel = parallel)
        if dump_after_merging:
            if name_of_model == '':
                name_of_model = self.name_of_model
            if path == '':
                path = path_models
            self.dump(name=name_of_model, path=path)
            
    
    def initialise_from_partial_calibration(self,list partial_models,
                                            set_parameters = False,
                                            adjust_base_rates = False, DTYPEf_t leading_coef=0.66,
                                            dump_after_merging=True, str name_of_model='',
                                           ):
        """
        It is assumed that all part models have been calibrated on the same dataset.
        """
        cdef int e=0, event_type=0
        cdef list list_of_mle_results = []
        for e in range(len(partial_models)):
            model=partial_models[e]
            assert model.data.n_levels == self.n_levels
            assert np.all(model.data.state_enc.list_of_n_states == self.state_enc.list_of_n_states)
            assert model.data.number_of_event_types == self.number_of_event_types
            assert model.data.number_of_states == self.number_of_states
            event_type=copy.copy(model.calibration.event_type)
            assert event_type == model.mle_estim.results_of_estimation[0].get('component_e')
            print('I am reading from model number {}, referring to event_type={}'.format(e,event_type))
            print("model name is: {}".format(model.name_of_model))
            print("len(mle_estim.results_of_estimation) = {}".format(len(model.mle_estim.results_of_estimation)))
            for res in model.mle_estim.results_of_estimation:
                list_of_mle_results.append(res)
        model=partial_models[0]        
        self.get_input_data(model.data, copy=True) #copy from first model in the list partial models
        type_of_preestim=model.calibration.type_of_preestim
        print("{} mle results have been loaded".format(len(list_of_mle_results)))
        if len(list_of_mle_results)!= self.number_of_event_types:
            warning_message="WARNING: It was expected that len(list_of_mle_results) == self.number_of_event_types"
            warning_message+="\nBut:\n"
            warning_message+="len(list_of_mle_results)={};  ".format(len(list_of_mle_results))
            warning_message+="self.number_of_event_types={}.\n".format(self.number_of_event_types)
            print(warning_message)
            print("list_of_mle_results=\n{}".format(list_of_mle_results))
        self.create_mle_estim(type_of_input='empirical',store_trans_prob=True, store_dirichlet_param=True)
        self.mle_estim.store_results_of_estimation(list_of_mle_results)
        self.mle_estim.store_hawkes_parameters()
        self.mle_estim.create_goodness_of_fit()
        if set_parameters:
            if adjust_base_rates:
                d_E = self.number_of_event_types
                A=leading_coef*np.eye(d_E, dtype=np.float)
                antidiag = ~np.eye(d_E//2, dtype=np.bool)
                A[:d_E//2,:d_E//2][antidiag] = (1.0-leading_coef)/max(1,d_E//2-1)
                antidiag = ~np.eye(d_E - d_E//2, dtype=np.bool)
                A[d_E//2:,d_E//2:][antidiag] = (1.0-leading_coef)/max(1,d_E-d_E//2 -1)
                base_rates=np.matmul(A,self.mle_estim.base_rates)
            else:
                base_rates=self.mle_estim.base_rates
            self.set_hawkes_parameters(base_rates,
                                       self.mle_estim.hawkes_kernel.alphas,
                                       self.mle_estim.hawkes_kernel.betas)
            self.set_transition_probabilities(self.mle_estim.transition_probabilities)
            self.set_dirichlet_parameters(self.mle_estim.dirichlet_param, N_samples_for_prob_constraints = 15000)
            self.create_goodness_of_fit(type_of_input='empirical')
        
        self.calibrate_on_input_data(partial=False,
                                     type_of_preestim = type_of_preestim,
                                     skip_mle_estim=True,
                                     dump_after_calibration=dump_after_merging,
                                     verbose=True,
                                     )        
    def get_input_data(self, data, copy=False):
        assert data.n_levels == self.n_levels
        assert np.all(data.state_enc.list_of_n_states == self.state_enc.list_of_n_states)
        assert data.number_of_event_types == self.number_of_event_types
        assert data.number_of_states == self.number_of_states
        self.data = data
        #if copy:
        #    self.data = copy.copy(data)
        #else:
        #    self.data = data    
                
    def calibrate_on_input_data(self, partial=True, int e=0,
                                str name_of_model='',
                                str type_of_preestim='ordinary_hawkes', #'ordinary_hawkes' or 'nonparam'
                                DTYPEf_t max_imp_coef = 100.0,
                                DTYPEf_t learning_rate = 0.0001,
                                int maxiter = 50,
                                int num_of_random_guesses=0,
                                parallel=True,
                                use_prange=False,
                                int number_of_attempts = 2,
                                int num_processes = 0,
                                int batch_size = 5000,
                                int num_run_per_minibatch = 1,  
                                skip_mle_estim=False,
                                store_trans_prob=True, store_dirichlet_param=False,
                                dump_after_calibration=False,
                                verbose=False,
                                DTYPEf_t tol = 1.0e-7,
                               ):
        times=self.data.observed_times
        events=self.data.observed_events
        states=self.data.observed_states
        volumes=self.data.observed_volumes
        "By default, time_start and time_end for the calibration are set equal to the boundaries of self.data.observed_times"
        time_start=float(times[0])
        time_end=float(times[len(times)-1])
        if name_of_model=='':
            if not partial:
                name_of_model =\
                '{}_sdhawkes_{}_{}-{}'.format(
                    self.data.symbol, self.data.date, self.data.initial_time, self.data.final_time)
            else:
                name_of_model =\
                '{}_sdhawkes_{}_{}-{}_partial{}'.format(
                    self.data.symbol, self.data.date, self.data.initial_time, self.data.final_time, e)
        self.name_of_model=name_of_model
        self.calibration=Calibration(self.data,
                                     name_of_model=name_of_model,
                                     partial=partial, event_type=e,
                                     type_of_preestim = type_of_preestim,
                                    )
        cdef list list_init_guesses = []
        if not skip_mle_estim:
            n_cpus = os.cpu_count()
            if num_of_random_guesses<=0:
                num_of_random_guesses = 2*n_cpus
            if type_of_preestim == 'ordinary_hawkes':
                pre_estim_ord_hawkes=True
            else:
                pre_estim_ord_hawkes=False
            if type_of_preestim == 'nonparam':
                list_init_guesses = self.nonparam_estim.produce_list_init_guesses_for_mle_estimation(
                    num_additional_random_guesses = max(1,num_of_random_guesses//2),
                    max_imp_coef = max_imp_coef
                )    
            self.create_mle_estim(type_of_input='empirical',
                                  store_trans_prob=store_trans_prob,
                                  store_dirichlet_param=store_dirichlet_param)
            self.mle_estim.set_estimation_of_hawkes_param(
                time_start, time_end,
                list_of_init_guesses = list_init_guesses,
                max_imp_coef = max_imp_coef,
                learning_rate = learning_rate,
                maxiter=maxiter,
                number_of_additional_guesses = max(1,num_of_random_guesses//2),
                parallel=parallel,
                pre_estim_ord_hawkes=pre_estim_ord_hawkes,
                pre_estim_parallel=parallel,
                use_prange = use_prange,
                number_of_attempts = number_of_attempts,
                num_processes = num_processes,
                batch_size = batch_size,
                num_run_per_minibatch = num_run_per_minibatch,
            )
            self.mle_estim.launch_estimation_of_hawkes_param(partial=partial, e=e)
            self.calibration.store_mle_info(self.mle_estim.results_of_estimation)
        if dump_after_calibration:
            name=name_of_model
            try:
                path=self.path_models+'/{}_{}'.format(self.data.symbol, self.data.date)
                self.dump(name=name,path=path)      
            except:
                path=self.path_models+'/{}'.format(self.data.symbol)
                self.dump(name=name,path=path)    
    
    "Functions to estimate model's parameters"
    
    def estimate_dirichlet_parameters(self,np.ndarray[DTYPEf_t, ndim=2] volumes,
                                      np.ndarray[DTYPEi_t, ndim=1] states,
                                      tolerance=1e-7,verbose=False):
        print('SDHawkes: I am estimating dirichlet parameters')
        return mle_estim.estimate_dirichlet_parameters(self.number_of_states,
                    self.n_levels,states,volumes,tolerance,verbose)

    def estimate_transition_probabilities(self,
                                          np.ndarray[DTYPEi_t, ndim=1] events,
                                          np.ndarray[DTYPEi_t, ndim=1] states,
                                          verbose=True):
        r"""
        Estimates the transition probabilities :math:`\phi` of the state process from the data.
        This method returns the maximum likelihood estimate.
        One can prove that it coincides with the empirical transition probabilities.

        :type events: 1D array of int
        :param events: the sequence of event types, `events[n]` is the event type of the `n` th event.
        :type states: 1D array of int
        :param states: the sequence of states, `states[n]` is the new state of the system following the `n` th event.
        :rtype: 3D array
        :return: the estimated transition probabilities :math:`\phi`.
        """
        print('SDHawkes: I am estimating transition probabilities')
        n_states = self.number_of_states
        n_event_types = self.number_of_event_types
        cdef int v = int(verbose)
        return mle_estim.estimate_transition_probabilities(
            self.number_of_event_types,self.number_of_states,events,states,verbose=v)
    
           
    "Specification testing and simulation"

    def simulate(self, time_start, time_end, initial_condition_times=[], initial_condition_events=[],
                 initial_condition_states=[],  
                 initial_condition_volumes=[], max_number_of_events=10**4,add_initial_cond=False,
                 store_results=False,report_full_volumes=False,
                ):
        """
        Simulates a sample path of the state-dependent Hawkes process.
        The methods wraps a C implementation that was obtained via Cython.

        :type time_start: float
        :param time_start: time at which the simulation starts.
        :type time_end: float
        :param time_end: time at which the simulation ends.
        :type initial_condition_times: array
        :param initial_condition_times: times of events before and including `time_start`.
        :type initial_condition_events: array of int
        :param initial_condition_events: types of the events that occurred at `initial_condition_times`.
        :type initial_condition_states: array of int
        :param initial_condition_states: values of the state process just after the `initial_condition_times`.
        :param initial_state: if there are no event times before `time_start`, this is used as the initial state.
        :type max_number_of_events: int
        :param max_number_of_events: the simulation stops when this number of events is reached
                                     (including the initial condition).
        :rtype: array, array of int, array of int
        :return: the times at which the events occur, their types and the values of the state process right after
                 each event. 
        """
        os_info = os.uname()
        print('Simulation is being performed on the following machine:\n {}'.format(os_info))
        time_start=np.array(time_start,dtype=np.float)
        time_end=np.array(time_end,dtype=np.float)
        max_number_of_events=np.array(max_number_of_events,dtype=int)
         # Convert the initial condition to np.arrays if required
        if type(initial_condition_times)!=np.ndarray:
            initial_condition_times = np.asarray(initial_condition_times, dtype=np.float)
        num_initial_events=initial_condition_times.shape[0]
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
        print('SDHawkes.simulate: initial conditions have been acknowledged')   
        times, events, states,volumes =  simulation.launch_simulation(self.number_of_event_types,
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
                                              report_full_volumes=report_full_volumes)
        if store_results:
            lt,count=computation.distribute_times_per_event_state(
                self.number_of_event_types,
                self.number_of_states,
                times, events,states)
            self.labelled_times = np.array(lt, copy=True)
            self.count = np.array(count,copy=True)
            self.simulated_times=np.array(times, copy=True)
            self.simulated_events=np.array(events, copy=True)
            self.simulated_states=np.array(states, copy=True)
            self.simulated_volume=np.array(volumes, copy=True)
        return times, events, states, volumes    
     
   
   
    def simulate_liquidation(self, DTYPEf_t time_end,
                             initial_condition_times=[],
                             initial_condition_events=[],
                             initial_condition_states=[],
                             initial_condition_volumes=[],
                             max_number_of_events=10**6,
                             add_initial_cond=False,
                             num_preconditions=1,
                             largest_negative_time=-1.0,
                             verbose=False,
                             report_history_of_intensities =False,
                             store_results=True,
                             report_full_volumes=False,
                            ):
        """
        Simulates a sample path of the state-dependent Hawkes process.
        The methods wraps a C implementation that was obtained via Cython.

        :type time_start: float
        :param time_start: time at which the simulation starts.
        :type time_end: float
        :param time_end: time at which the simulation ends.
        :type initial_inventory: float
        :param initial_inventory: initial inventory to liquidate
        :type initial_condition_times: array
        :param initial_condition_times: times of events before and including `time_start`.
        :type initial_condition_events: array of int
        :param initial_condition_events: types of the events that occurred at `initial_condition_times`.
        :type initial_condition_states: array of int
        :param initial_condition_states: values of the state process just after the `initial_condition_times`.
        :type initial_partial_sums: 3D numpy array
        :param initial_partial_sums: the initial condition can also be given implicitly via the partial sums
                                     :math:`S_{e',x,e}(-\infty, \mbox{time_start}]`.
        :type initial_state: int
        :param initial_state: if there are no event times before `time_start`, this is used as the initial state.
        :type max_number_of_events: int
        :param max_number_of_events: the simulation stops when this number of events is reached
                                     (including the initial condition).
        :rtype: array, array of int, array of int
        :return: the times at which the events occur, their types and the values of the state process right after
                 each event. Note that these include the initial condition as well.
        """
        os_info = os.uname()
        print('Simulation is being performed on the following machine:\n {}'.format(os_info))
        time_start=self.liquidator.time_start
        max_number_of_events=np.array(max_number_of_events,dtype=int)
         # Convert the initial condition to np.arrays if required
        if type(initial_condition_times)!=np.ndarray:
            initial_condition_times = np.asarray(initial_condition_times, dtype=np.float)
        num_initial_events=initial_condition_times.shape[0]
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
        print('SDHawkes: simulate_liquidation. initial conditions have been acknowledged')
        cdef int report_intensities = np.array(report_history_of_intensities, dtype=np.intc)
        times, events, states, volumes,\
        inventory, liquid_termination_time, history_of_intensities = \
        simulation.launch_liquidation(self.state_enc, self.volume_enc,
                                       self.number_of_event_types,
                                       self.number_of_states,
                                       self.n_levels,
                                       self.liquidator.initial_inventory,
                                       self.state_enc.array_of_n_states,
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
                                       num_preconditions,
                                       largest_negative_time,
                                       self.liquidator.control,
                                       self.liquidator.control_type,
                                       verbose=verbose,
                                       report_history_of_intensities = report_intensities,
                                       report_full_volumes=report_full_volumes,
                                    )
        "Update the transition prombabilities corresponding to liquidator's interventions"
        self.transition_probabilities[:,0,:] = mle_estim.estimate_liquidator_transition_prob(
            self.number_of_states, events, states, liquidator_index = 0)
        self.liquidator.store_termination_time(liquid_termination_time)
        "Store inventory trajectory"
        self.liquidator.store_inventory_trajectory(times,inventory)
        if store_results:
            lt,count=computation.distribute_times_per_event_state(
                self.number_of_event_types,
                self.number_of_states,
                times, events,states)
            self.labelled_times = np.array(lt, copy=True)
            self.count = np.array(count,copy=True)
            self.simulated_times=np.array(times, copy=True)
            self.simulated_events=np.array(events, copy=True)
            self.simulated_states=np.array(states, copy=True)
            if report_intensities:
                self.history_of_intensities = np.array(history_of_intensities, copy=True)
        return times, events, states, volumes,inventory, history_of_intensities
    
    
   
    'Miscellaneous tools'
    def make_start_liquid_origin_of_times(self,delete_negative_times=False):
        time_start=copy.copy(self.liquidator.time_start)
        self.liquidator.termination_time -= time_start
        cdef np.ndarray[DTYPEf_t, ndim=1] times = self.simulated_times - time_start*np.ones_like(self.simulated_times)
        if delete_negative_times:
            idx = times>=0
            self.simulated_times=np.array(times[idx],copy=True)
            self.simulated_events=np.array(self.simulated_events[idx],copy=True)
            self.simulated_states=np.array(self.simulated_states[idx],copy=True)
            print('SDHawkes: start_liquidation has been set as origin of times, and negative times have been deleted')
            message='  self.liquidator.termination_time={}'.format(self.liquidator.termination_time)
            message+='\n  self.simulated_times.shape={}'.format(self.simulated_times.shape)
            message+='\n  self.simulated_events.shape={}'.format(self.simulated_events.shape)
            message+='\n  self.simulated_states.shape={}'.format(self.simulated_states.shape)
            print(message)
        else:
            print('SDHawkes: start_liquidation has been set as origin of times')
        try:
            self.liquidator.inventory_trajectory[:,0]-=time_start
        except:
            pass
        try:
            lt, count = computation.distribute_times_per_event_state(
                self.number_of_event_types, self.number_of_states,
                self.simulated_times,self.simulated_events,self.simulated_states)
            self.labelled_times = lt
            self.count = count
            print('  self.labelled_times.shape={}'.format(self.labelled_times.shape))
        except:
            pass
        try:
            df=self.liquidator.bm_impact_profile
            df['time']-=time_start
            idx_row=df.loc[df['time']<0].index
            df.drop(axis=0,index=idx_row,inplace=True)
        except:
            pass
        self.liquidator.time_start = 0.0 
    def store_calibration(self,calibration):
        self.calibration=calibration    
    def store_goodness_of_fit(self,goodness_of_fit):
        self.goodness_of_fit = goodness_of_fit
    def create_goodness_of_fit(self, str type_of_input='simulated', parallel=True):
        "type_of_input can either be 'simulated' or 'empirical'"
        if type_of_input == 'simulated':
            times=self.simulated_times
            events=self.simulated_events
            states=self.simulated_states
        elif type_of_input =='empirical':
            times=self.data.observed_times
            events=self.data.observed_events
            states=self.data.observed_states
        else:
            print("type_of_input={}".format(type_of_input))
            raise ValueError("type of input not recognised. It must be either 'simulated' or 'empirical'")
        self.goodness_of_fit=goodness_of_fit.good_fit(
            self.number_of_event_types,self.number_of_states,
            self.base_rates,self.impact_coefficients,self.decay_coefficients,self.transition_probabilities,
            times,events,states,type_of_input=type_of_input, parallel = parallel
        )
    def create_uq(self,):
        self.uncertainty_quantification=uncertainty_quant.UncertQuant(
            self.number_of_event_types, self.number_of_states,
            self.n_levels, self.state_enc, self.volume_enc,
            self.base_rates,
            self.impact_coefficients,
            self.decay_coefficients,
            self.transition_probabilities,
            self.dirichlet_param,
        )
    def create_nonparam_estim(self, str type_of_input = 'simulated',
                              int num_quadpnts = 100, DTYPEf_t quad_tmax=1.0, DTYPEf_t quad_tmin=1.0e-04,
                              int num_gridpnts = 100, DTYPEf_t grid_tmax=1.0, DTYPEf_t grid_tmin=1.0e-04,
                              DTYPEf_t tol=1.0e-15, two_scales=False
                             ):
        if type_of_input == 'simulated':
            times=self.simulated_times
            events=self.simulated_events
            states=self.simulated_states
        elif type_of_input =='empirical':
            times=self.data.observed_times
            events=self.data.observed_events
            states=self.data.observed_states
        else:
            print("type_of_input: {}".format(type_of_input))
            raise ValueError("type of input not recognised. It must be either 'simulated' or 'empirical'")
        self.nonparam_estim=nonparam_estim.EstimProcedure(
            self.number_of_event_types,self.number_of_states,
            times,events,states,
            type_of_input,
            num_quadpnts, quad_tmax, quad_tmin,
            num_gridpnts, grid_tmax, grid_tmin,
            tol, two_scales
        )
    def store_nonparam_estim_class(self,nonparam_estim):
        self.nonparam_estim = copy.copy(nonparam_estim)
    def store_mle_estim(self,mle_estim):
        self.mle_estim = mle_estim
    def create_mle_estim(self, str type_of_input = 'simulated', store_trans_prob=True, store_dirichlet_param=False):
        if type_of_input == 'simulated':
            times=self.simulated_times
            events=self.simulated_events
            states=self.simulated_states
            try:
                volumes = self.simulated_volumes
            except:
                volumes = None
        elif type_of_input =='empirical':
            times=self.data.observed_times
            events=self.data.observed_events
            states=self.data.observed_states
            volumes = self.data.observed_volumes
            assert self.n_levels == self.data.n_levels
        else:
            print("ERROR: type_of_input = '{}' not recognised".format(type_of_input))
            raise ValueError("type of input not recognised. It must be either 'simulated' or 'empirical'")
        self.mle_estim=mle_estim.EstimProcedure(
            self.number_of_event_types, self.number_of_states,
            times, events, states,
            volumes = volumes,
            n_levels = self.n_levels, 
            type_of_input = type_of_input,
            store_trans_prob = store_trans_prob,
            store_dirichlet_param = store_dirichlet_param
        )    
        

    def compute_history_of_intensities(self,times,events,states,
                                       inventory=None,
                                       start_time_zero_event=-1.0,
                                       end_time_zero_event=-2.0,
                                       density_of_eval_points=100):
        cdef np.ndarray[DTYPEf_t, ndim=1] arrival_times = np.array(times,dtype=DTYPEf,copy=True)
        cdef np.ndarray[DTYPEi_t, ndim=1] history_of_events = np.array(events,dtype=DTYPEi, copy=True)
        cdef np.ndarray[DTYPEi_t, ndim=1] history_of_states = np.array(states,dtype=DTYPEi, copy=True)
        if not inventory==None:
            end_time_zero_event=np.squeeze(times[np.argmin(inventory)])
            if  start_time_zero_event ==  -1.0:
                idx=(np.diff(np.concatenate([inventory,[0.0]],axis=0,dtype=DTYPEf))<0.0)
                start_time_zero_event=np.squeeze(times[idx][0])
        start_time_zero_event=np.array(start_time_zero_event,dtype=DTYPEf)
        end_time_zero_event=np.array(end_time_zero_event,dtype=DTYPEf)
        return computation.compute_history_of_intensities(self.number_of_event_types,
                                                       self.number_of_states,
                                                       arrival_times,
                                                       history_of_events,
                                                       history_of_states,
                                                       self.base_rates,
                                                       self.impact_coefficients,
                                                       self.decay_coefficients,
                                                       start_time_zero_event=start_time_zero_event,
                                                       end_time_zero_event=end_time_zero_event,
                                                       density_of_eval_points=density_of_eval_points
                                                      )
    
    def compute_history_of_tilda_intensities(self,times,events,states,
                                       inventory=None,
                                       start_time_zero_event=-1.0,
                                       end_time_zero_event=-2.0,
                                       density_of_eval_points=100):
        cdef np.ndarray[DTYPEf_t, ndim=1] arrival_times = np.array(times,dtype=DTYPEf,copy=True)
        cdef np.ndarray[DTYPEi_t, ndim=1] history_of_events = np.array(events,dtype=DTYPEi, copy=True)
        cdef np.ndarray[DTYPEi_t, ndim=1] history_of_states = np.array(states,dtype=DTYPEi, copy=True)
        if not inventory==None:
            end_time_zero_event=np.squeeze(times[np.argmin(inventory)])
            if  start_time_zero_event ==  -1.0:
                idx=(np.diff(np.concatenate([inventory,[0.0]],axis=0,dtype=DTYPEf))<0.0)
                start_time_zero_event=np.squeeze(times[idx][0])
        start_time_zero_event=np.array(start_time_zero_event,dtype=DTYPEf)
        end_time_zero_event=np.array(end_time_zero_event,dtype=DTYPEf)
        return computation.compute_history_of_tilda_intensities(self.number_of_event_types,
                                                       self.number_of_states,
                                                       arrival_times,
                                                       history_of_events,
                                                       history_of_states,
                                                       self.base_rates,
                                                       self.impact_coefficients,
                                                       self.decay_coefficients,
                                                       self.transition_probabilities,         
                                                       start_time_zero_event=start_time_zero_event,
                                                       end_time_zero_event=end_time_zero_event,
                                                       density_of_eval_points=density_of_eval_points
                                                      )
    
    def compute_bm_impact_profile(self,times,events,states,
                                  inventory,
                                  start_liquidation_time = -1.0,
                                  int density_of_eval_points=1000,store=True):
        cdef np.ndarray[DTYPEf_t, ndim=1] arrival_times = np.array(times,dtype=DTYPEf,copy=True)
        cdef np.ndarray[DTYPEi_t, ndim=1] history_of_events = np.array(events,dtype=DTYPEi, copy=True)
        cdef np.ndarray[DTYPEi_t, ndim=1] history_of_states = np.array(states,dtype=DTYPEi, copy=True)
        cdef np.ndarray[DTYPEf_t, ndim=1] history_of_inventory = np.array(inventory,dtype=DTYPEf, copy=True)
        cdef DTYPEf_t start_liquid_time = np.array(start_liquidation_time, dtype=DTYPEf, copy=True)
        cdef np.ndarray[DTYPEf_t, ndim=2] history_of_profile_intensity = \
        computation.compute_history_of_bm_profile_intensity(
            self.number_of_event_types,
            self.number_of_states,
            self.state_enc.deflationary_states,
            self.state_enc.inflationary_states,
            arrival_times,
            history_of_events,
            history_of_states,
            history_of_inventory,
            self.base_rates,
            self.impact_coefficients,
            self.decay_coefficients,
            self.transition_probabilities,
            start_liquid_time,
            density_of_eval_points=density_of_eval_points
        )
        cdef np.ndarray[DTYPEf_t, ndim=2] bm_impact_profile = \
        computation.compute_bm_impact_profile(history_of_profile_intensity)
        if store:
            self.liquidator.store_bm_impact_profile(bm_impact_profile,history_of_profile_intensity)
        return history_of_profile_intensity, bm_impact_profile
    
    def produce_2Dstates(self,states):
        df=self.state_enc.translate_labels(states)
        cdef np.ndarray[DTYPEi_t, ndim=2] states_2D = \
        np.array(
            np.concatenate(
                [np.expand_dims(df['st_1'].values,axis=1),
                 np.expand_dims(df['st_2'].values,axis=1)]
                ,axis=1),
            dtype=DTYPEi
        )
        states_2D[:,0] -=1
        states_2D[:,1] -=2
        return states_2D
    
    def produce_impact_profile(self,
                               int num_init_guesses = 8,
                               int maxiter = 100,
                               time_start=None, time_end=None):
        self.make_start_liquid_origin_of_times(delete_negative_times=True)
        impact=impact_profile.impact(
            self.liquidator,
            self.state_enc.weakly_deflationary_states,
            self.simulated_times,
            self.simulated_events,
            self.simulated_states,)
        impact.produce_weakly_defl_pp(
            self.base_rates, self.impact_coefficients, self.decay_coefficients,
            self.transition_probabilities)
        impact.produce_reduced_weakly_defl_pp(
            num_init_guesses = num_init_guesses,
            maxiter = maxiter, time_start=time_start, time_end=time_end)
        self.impact=impact
        print('\nSDHawkes: the object "impact" has been initialised\n\n')
        

    def generate_base_rates_labels(self):
        r"""
        Produces labels for the base rates :math:`\nu`.
        This uses the events labels of the model.

        :rtype: list of string
        :return: `list[e]` returns a label for :math:`\nu_e`.
        """
        labels = []
        for e in range(self.number_of_event_types):
            l = r'$\nu_{' + self.events_labels[e] + '}$'
            labels.append(l)
        return labels
    
    
cdef void compute_L1_norm_of_hawkes_kernel(
    int e1, int e, int num_states, int num_quadpnts,
    DTYPEf_t [:,:,:,:] kappa,
    DTYPEf_t [:,:] hawkes_norm,
    DTYPEf_t [:] quadpnts
) nogil:
    """It utilises trapezoidal rule for numerical integration"""
    cdef DTYPEf_t integrand = 0.0
    cdef int n=0, x1=0
    for n in range(num_quadpnts-1):
        integrand=0.0
        for x1 in range(num_states):
            integrand+=max(0.0,kappa[e1,x1,e,n])+max(0.0,kappa[e1,x1,e,n+1]) # kappa is expected to be non-negative, but we cut it off at 0.0 in case of negative values
        hawkes_norm[e1,e]+=0.5*integrand*(quadpnts[n+1]-quadpnts[n])    

#     def generate_impact_coefficients_labels(self):
#         r"""
#         Produces labels for the impact coefficients :math:`\alpha`.
#         This uses the events and states labels of the model.

#         :rtype: list of list of list of string
#         :return: `list[e'][x][e]` returns a label for :math:`\alpha_{e'xe}`.
#         """
#         labels = []
#         for e1 in range(self.number_of_event_types):
#             l1 = []
#             for x in range(self.number_of_states):
#                 l2 = []
#                 for e2 in range(self.number_of_event_types):
#                     s = r'$\alpha_{' + self.events_labels[e1]
#                     s += r' \rightarrow ' + self.events_labels[e2] + '}('
#                     s += self.states_labels[x] + ')$'
#                     l2.append(s)
#                 l1.append(l2)
#             labels.append(l1)
#         return labels

#     def generate_decay_coefficients_labels(self):
#         r"""
#         Produces labels for the decay coefficients :math:`\beta`.
#         This uses the events and states labels of the model.

#         :rtype: list of list of list of string
#         :return: `list[e'][x][e]` returns a label for :math:`\beta_{e'xe}`.
#         """
#         labels = []
#         for e1 in range(self.number_of_event_types):
#             l1 = []
#             for x in range(self.number_of_states):
#                 l2 = []
#                 for e2 in range(self.number_of_event_types):
#                     s = r'$\beta_{' + self.events_labels[e1]
#                     s += r' \rightarrow ' + self.events_labels[e2] + '}('
#                     s += self.states_labels[x] + ')$'
#                     l2.append(s)
#                 l1.append(l2)
#             labels.append(l1)
#         return labels

#     def generate_product_labels(self):
#         r"""
#         Produces labels for all the couples of events types and possible states.
#         This uses the events and states labels of the model.

#         :rtype: list of strings
#         :return: the label for the couple `(e, x)` is given by by `list[n]` where
#                  :math:`n = x + e \times \mbox{number_of_event_types}`.
#         """
#         r = []
#         for e in self.events_labels:
#             for x in self.states_labels:
#                 r.append(e + ', ' + x)
#         return r
