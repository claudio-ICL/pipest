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
cdef str path_sdhawkes=path_pipest+'/sdhawkes'
cdef str path_lobster=path_pipest+'/lobster'
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
from impact_profile import Impact
import dirichlet
import lob_model
import plot_tools
import matplotlib.pyplot as plt
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
                 start_time = None,
                 termination_time = None,
                ):
        self.initial_inventory = initial_inventory
        self.base_rate = base_rate
        self.imp_coef = imp_coef
        self.dec_coef = dec_coef
        self.type_of_liquid = type_of_liquid
        self.control = control
        self.control_type=control_type
        if not start_time == None:
            self.store_start_time(start_time)
        if not termination_time == None:
            self.store_termination_time(termination_time)
        if type_of_liquid =='constant_intensity':
            if control_type=='fraction_of_inventory':
                self.num_orders=int(ceil(initial_inventory/control))
    def print_info(self,):
        print("liquidator.initial_inventory: {}".format(self.initial_inventory))
        print("liquidator.type_of_liquid: {}".format(self.type_of_liquid))
        print("liquidator.control_type: {}".format(self.control_type))
        print("liquidator.control: {}".format(self.control))
        print("liquidator.base_rate: {}".format(self.base_rate))
        print("liquidator.start_time: {}".format(self.start_time))
        try:
            print("liquidator.termination_time: {}".format(self.termination_time))
        except:
            pass
    def store_start_time(self, DTYPEf_t time):
        self.start_time=time
    def store_termination_time(self, DTYPEf_t termination_time):
        self.termination_time = copy.copy(termination_time)
    def store_inventory_trajectory(self,times,inventory):
        cdef np.ndarray[DTYPEf_t, ndim=2] t=times.reshape(-1,1)
        cdef np.ndarray[DTYPEf_t, ndim=2] inv=inventory.reshape(-1,1)
        cdef np.ndarray[DTYPEf_t, ndim=2] trajectory=np.concatenate([t,inv],axis=1)
        self.inventory_trajectory=trajectory
    def get_impact(self,impact):
        self.impact=impact


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
    def get_configuration(self, model, create_goodness_of_fit=False):
        assert self.number_of_event_types==model.number_of_event_types
        assert self.number_of_states==model.number_of_states
        self.set_name_of_model(model.name_of_model)
        try:
            self.get_input_data(model.data, copy=True)
            type_of_preestim=model.calibration.type_of_preestim
            print('Copying data from given model')
            self.create_mle_estim(type_of_input='empirical',store_trans_prob=True, store_dirichlet_param=True)
            self.mle_estim.store_results_of_estimation(model.mle_estim.results_of_estimation)
            self.mle_estim.store_hawkes_parameters()
            if create_goodness_of_fit:
                self.mle_estim.create_goodness_of_fit()
        except:
            print("No data or mle_estim found in given model")
            pass
        self.set_hawkes_parameters(model.base_rates,
                model.impact_coefficients, model.decay_coefficients)
        self.set_transition_probabilities(model.transition_probabilities)
        self.set_dirichlet_parameters(model.dirichlet_param)
        if create_goodness_of_fit:
            try:
                self.create_goodness_of_fit(type_of_input='empirical')
            except:
                print("I couldn't create goodness of fit on empirical data")
        try:
            self.calibrate_on_input_data(partial=False, 
                    type_of_preestim = model.calibration.type_of_preestim,
                    skip_mle_estim=True,
                    dump_after_calibration=False, verbose=False)
        except:
            print("I couldn't create Calibration")

    def setup_liquidator(self,
            DTYPEf_t initial_inventory = 0.0,
            time_start=None,
            type_of_liquid='with_the_market',
            liquidator_base_rate=None,
            self_excitation=False,
            DTYPEf_t  liquidator_control=0.5,
            liquidator_control_type='fraction_of_inventory',
            DTYPEf_t liq_excit = 1.0,
            DTYPEf_t liq_dec = 10.0):
        try:
            self.liquidator.print_info()
            print("\nI am overwriting the above")
            self.configure_liquidator_param(
                                   initial_inventory=initial_inventory,
                                   time_start=time_start,
                                   liquidator_base_rate=liquidator_base_rate,
                                   type_of_liquid=type_of_liquid,
                                   self_excitation=self_excitation,
                                   liquidator_control=liquidator_control,
                                   liquidator_control_type=liquidator_control_type,
                                   liq_excit = liq_excit,
                                   liq_dec = liq_dec)
        except:
            self.introduce_liquidator(
                                   initial_inventory=initial_inventory,
                                   time_start=time_start,
                                   liquidator_base_rate=liquidator_base_rate,
                                   type_of_liquid=type_of_liquid,
                                   self_excitation=self_excitation,
                                   liquidator_control=liquidator_control,
                                   liquidator_control_type=liquidator_control_type,
                                   liq_excit = liq_excit,
                                   liq_dec = liq_dec)
    def introduce_liquidator(self,
                             DTYPEf_t initial_inventory = 0.0,
                             time_start=None,
                             type_of_liquid='with_the_market',
                             liquidator_base_rate=None,
                             self_excitation=False,
                             DTYPEf_t  liquidator_control=0.5,
                             liquidator_control_type='fraction_of_inventory',
                             DTYPEf_t liq_excit = 1.0,
                             DTYPEf_t liq_dec = 10.0):
        try:
            self.liquidator.print_info()
            print("Liquidator exists. Use method 'setup_liquidator' instead")
            raise ValueError("Liquidator exists")
        except:
            pass
        cdef str control_type = copy.copy(liquidator_control_type)
        cdef DTYPEf_t control=copy.copy(liquidator_control)
        cdef int idx_type = 0 
        if type_of_liquid=='with_the_market':
            idx_type=0
        elif type_of_liquid=='against_the_market':
            idx_type=1
        elif type_of_liquid=='with_price_move':
            alpha=np.zeros((1+self.number_of_event_types, self.number_of_states), dtype=DTYPEf)
            for e1 in range(1,self.number_of_event_types):
                for x in self.state_enc.deflationary_states:
                    alpha[e1,x]=liq_excit
            beta=max(1.1,liq_dec)*np.ones((1+self.number_of_event_types, self.number_of_states), dtype=DTYPEf)
        elif type_of_liquid=='against_price_move':
            alpha=np.zeros((1+self.number_of_event_types, self.number_of_states), dtype=DTYPEf)
            for e1 in range(1,self.number_of_event_types):
                for x in self.state_enc.inflationary_states:
                    alpha[e1,x]=liq_excit
            beta=max(1.1,liq_dec)*np.ones((1+self.number_of_event_types, self.number_of_states), dtype=DTYPEf)
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
        Second, decay_coeff of other participants as response to the liquidator' interventions is set as the decay_coeff of the same participants as response to sell_I
        """
        decay_coeff=np.insert(self.decay_coefficients,
                              obj=0,
                              values=self.decay_coefficients[0,:,:],
                              axis=0)
        """
        Third, impact_coeff decay_coeff of liquidator as response to the other participants is set as the impact_coeff of either sell_I or buy_I depending on type_of_liquid
        """ 
        if type_of_liquid=='with_the_market' or type_of_liquid=='against_the_market':
            impact_coeff=np.insert(impact_coeff,obj=0,values=impact_coeff[:,:,idx_type],axis=2)
            decay_coeff=np.insert(decay_coeff,obj=0,values=decay_coeff[:,:,idx_type],axis=2)
            if not self_excitation:
                impact_coeff[0,:,0] = np.zeros(self.number_of_states,dtype=DTYPEf)
        elif type_of_liquid == 'constant_intensity':
            impact_coeff=np.insert(
                impact_coeff,obj=0,
                values=np.zeros((1+self.number_of_event_types,self.number_of_states),dtype=DTYPEf),
                axis=2)
            decay_coeff=np.insert(decay_coeff,obj=0,values=1.1*np.ones((1+self.number_of_event_types, self.number_of_states)),axis=2)
        else:    
            impact_coeff=np.insert(impact_coeff,obj=0,values=alpha,axis=2)
            decay_coeff=np.insert(decay_coeff,obj=0,values=beta,axis=2)
        """
        Finally, transition probabilities are updated, but this does not matter since   it will never be used, but it is done to modify the dimension
        """
        trans_prob=np.insert(self.transition_probabilities,
                             obj=0,
                             values=self.transition_probabilities[:,0,:],
                             axis=1)
        self.number_of_event_types +=1
        self.hawkes_kernel=HawkesKernel(self.number_of_event_types, self.number_of_states)
        self.set_hawkes_parameters(base_rates, impact_coeff, decay_coeff)
        self.set_transition_probabilities(trans_prob)
        if self_excitation:
            with_without_self_excitation='with'
        else:
            with_without_self_excitation='without'
        self.liquidator = Liquidator(initial_inventory,
                                     self.base_rates[0],
                                     self.impact_coefficients[:,:,0],
                                     self.decay_coefficients[:,:,0],
                                     type_of_liquid,
                                     control,
                                     control_type,
                                     start_time = time_start
                                    )
        self.liquidator.print_info()
    def configure_liquidator_param(self,
                                   initial_inventory=None,
                                   time_start=None,
                                   liquidator_base_rate=None,
                                   type_of_liquid='with_the_market',
                                   self_excitation=False,
                                   liquidator_control=0.5,
                                   liquidator_control_type='fraction_of_inventory',
                                   DTYPEf_t liq_excit = 1.0,
                                   DTYPEf_t liq_dec = 10.0):
        "Notice that it is assumed that the liquidator has already been introduced, with event type = 0"
        if initial_inventory==None:
            initial_inventory=self.initial_inventory
        if time_start == None:
            try:
                time_start=self.liquidator.start_time
            except:
                pass
        cdef str control_type = copy.copy(liquidator_control_type)
        cdef DTYPEf_t control=copy.copy(liquidator_control)
        cdef int idx_type = 0
        if type_of_liquid=='with_the_market':
            idx_type=1
        elif type_of_liquid=='against_the_market':
            idx_type=2
        elif type_of_liquid=='with_price_move':
            alpha=np.zeros((self.number_of_event_types, self.number_of_states), dtype=DTYPEf)
            for e1 in range(1,self.number_of_event_types):
                for x in self.state_enc.deflationary_states:
                    alpha[e1,x]=liq_excit
            beta=max(1.1,liq_dec)*np.ones((self.number_of_event_types, self.number_of_states), dtype=DTYPEf)
        elif type_of_liquid=='against_price_move':
            alpha=np.zeros((self.number_of_event_types, self.number_of_states), dtype=DTYPEf)
            for e1 in range(1,self.number_of_event_types):
                for x in self.state_enc.inflationary_states:
                    alpha[e1,x]=liq_excit
            beta=max(1.1,liq_dec)*np.ones((self.number_of_event_types, self.number_of_states), dtype=DTYPEf)
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
        elif type_of_liquid=='with_the_market' or type_of_liquid=='against_the_market':
            imp_coef[:,:,0]=np.array(imp_coef[:,:,idx_type], copy=True)
            decay_coef[:,:,0]=np.array(decay_coef[:,:,idx_type], copy=True)
            if not self_excitation:
                imp_coef[0,:,0]=np.zeros(self.number_of_states,dtype=DTYPEf)
        else:
            imp_coef[:,:,0]=alpha
            decay_coef[:,:,0]=beta
        self.set_hawkes_parameters(rates, imp_coef, decay_coef)
        self.liquidator = Liquidator(initial_inventory,
                                     self.base_rates[0], self.impact_coefficients[:,:,0], self.decay_coefficients[:,:,0],
                                     type_of_liquid, control, control_type,
                                     time_start = time_start)
        
    def set_transition_probabilities(self, transition_probabilities):
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
        tol=1.0e-12
        if np.any(masses<=tol):
            print("{} state-event pairs are not associated with a distribution of state updates".format(np.sum(masses<=tol)))
        normalisation=(1.0+tol)*np.repeat(np.expand_dims(masses,axis=2),self.number_of_states,axis=2)
        idx_zero_mass = (normalisation<=tol)
        for x in range(self.number_of_states):
            for e in range(self.number_of_event_types):
                if idx_zero_mass[x,e,0]:
                    st2=x%self.state_enc.num_of_st2 #st2 corresponding to same volume
                    x_next=self.state_enc.num_of_st2+st2 #one_label of state with same volume and no price change
                    phi[x,e,x_next]=1.0# by default if no information is given, the state remains such that volumes are the same and there is no price change
                    normalisation[x,e,:]=1.0
        phi/=normalisation
        if not np.all(np.sum(phi,axis=2)<=1.0):
            print("Masses exceed 1.0")
            print(np.sum(phi,axis=2))
        assert np.all(np.sum(phi,axis=2)<=1.0)
        if np.any(np.isnan(phi)):
            print(phi)
        assert not np.any(np.isnan(phi))
        self.transition_probabilities = np.array(phi, copy=True)
        self.inflationary_pressure, self.deflationary_pressure, asymmetry =\
            computation.assess_symmetry(
                self.number_of_event_types,
                self.number_of_states,
                self.base_rates,
                self.transition_probabilities,
                self.state_enc.deflationary_states,
                self.state_enc.inflationary_states)
        print('Transition probabilities have been set. Price Asymmetry = {}'.format(asymmetry))
    def enforce_price_symmetry(self, int is_liquidator_present = 0):
        print("I am enforcing price symmetry")
        cdef int i = is_liquidator_present
        cdef np.ndarray[DTYPEf_t, ndim=1] nu = np.array(self.base_rates[i:], copy=True)
        cdef np.ndarray[DTYPEf_t, ndim=3] alpha = np.array(self.impact_coefficients[i:,:,i:], copy=True)
        cdef np.ndarray[DTYPEf_t, ndim=3] beta = np.array(self.decay_coefficients[i:,:,i:], copy=True)
        cdef np.ndarray[DTYPEf_t, ndim=3] phi = np.array(self.transition_probabilities[:,i:,:], copy=True)
        new_nu, new_alpha, new_beta = computation.produce_hawkes_param_for_symmetry(
            nu, alpha, beta)
        self.base_rates[i:]=np.array(new_nu, copy=True)
        self.impact_coefficients[i:,:,i:]=np.array(new_alpha, copy=True)
        self.decay_coefficients[i:,:,i:]=np.array(new_beta, copy=True)
        self.set_hawkes_parameters(self.base_rates, self.impact_coefficients, self.decay_coefficients)
        new_phi = computation.produce_phi_for_symmetry(
            self.number_of_event_types - i,
            self.number_of_states,
            self.state_enc.num_of_st2,
            phi,
            self.state_enc.deflationary_states,
            self.state_enc.inflationary_states,
            self.state_enc.stationary_states)
        self.transition_probabilities[:,i:,:] = np.array(new_phi, copy=True)
        self.set_transition_probabilities(self.transition_probabilities)
    def reduce_price_volatility(self,DTYPEf_t reduction_coef=0.66):
        cdef DTYPEf_t coef=min(1.0,max(0.0,reduction_coef))
        cdef np.ndarray[DTYPEf_t, ndim=3] phi = np.array(self.transition_probabilities, copy=True)
        cdef int num_of_st1= self.state_enc.num_of_st1
        cdef int num_of_st2= self.state_enc.num_of_st2
        for e in range(self.number_of_event_types):
            for x1 in range(self.number_of_states):
                for st2 in range(num_of_st2):
                    x=0*num_of_st2+st2
                    y=1*num_of_st2+st2
                    z=2*num_of_st2+st2
                    less=min(phi[x1,e,z],min(phi[x1,e,x],phi[x1,e,y]))
                    more=max(phi[x1,e,z],max(phi[x1,e,x],phi[x1,e,y]))
                    phi[x1,e,x]=coef*less+(1.0-coef)*phi[x1,e,x]
                    phi[x1,e,z]=coef*less+(1.0-coef)*phi[x1,e,z]
                    if phi[x1,e,y]<more:
                        phi[x1,e,y]=coef*more+(1.0-coef)*phi[x1,e,y]
        self.set_transition_probabilities(phi)




    def set_base_rates(self, np.ndarray[DTYPEf_t, ndim=1] base_rates):
        'Raise ValueError if the given parameters do not have the right shape'
        if np.shape(base_rates) != (self.number_of_event_types,):
            raise ValueError('given base rates have incorrect shape')
        cdef np.ndarray[DTYPEf_t, ndim=1] nu = np.array(base_rates,copy=True,dtype=DTYPEf)    
        self.base_rates = nu
    def set_hawkes_parameters(self, np.ndarray[DTYPEf_t, ndim=1] base_rates,
                              np.ndarray[DTYPEf_t, ndim=3 ]impact_coefficients,
                              np.ndarray[DTYPEf_t, ndim=3] decay_coefficients):
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
    def set_dirichlet_parameters(self, dir_param,
            int N_samples_for_prob_constraints = 10**5):
        'Raise ValueError if the given parameters do not have the right shape'
        if dir_param.shape != (self.number_of_states,2*self.n_levels):
            print('given parameter has incorrect shape, given shape={}, expected shape=({},{})'.format(dir_param.shape,self.number_of_states,2*self.n_levels))
            raise ValueError("Incorrect shape")
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
        self.volume_enc.store_dirichlet_param(dirichlet_param) 
        self.volume_enc.create_rejection_sampling(N_samples_for_prob_constraints)
        self.volume_enc.rejection_sampling.store_proposal_parameters(N_samples_for_prob_constraints)
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
    
    "Methods to estimate model's parameters"
    
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
                 store_results=False,report_full_volumes=False,return_results=False,
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
            self.store_2Dstates(type_of_input='simulated')
        if return_results:
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
        os_info = os.uname()
        print('Simulation is being performed on the following machine:\n {}'.format(os_info))
        time_start=self.liquidator.start_time
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
        inventory, liquid_start_time, liquid_termination_time, history_of_intensities = \
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
            self.number_of_states, events, states,
            self.state_enc.weakly_deflationary_states, liquidator_index=0)
        self.liquidator.store_start_time(liquid_start_time)
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
            self.store_2Dstates(type_of_input='simulated')
            if report_intensities:
                self.history_of_intensities = np.array(history_of_intensities, copy=True)
        else:
            return times, events, states, volumes,inventory, history_of_intensities
    
    'Miscellaneous tools'
    def store_price_trajectory(self, type_of_input='simulated', DTYPEf_t initial_price=0.0, DTYPEf_t ticksize=100.0):
        traj=self.draw_price_trajectory(type_of_input=type_of_input,
                initial_price=initial_price, ticksize=ticksize)
        if type_of_input=='simulated':
            self.simulated_price=traj
        elif type_of_input=='empirical':
            self.reconstructed_empirical_price=traj
    def draw_price_trajectory(self, type_of_input='simulated', DTYPEf_t initial_price=0.0, DTYPEf_t ticksize=100.0):
        if type_of_input=='simulated':
            times=self.simulated_times
            states=self.simulated_states
        elif type_of_input=='empirical':
            times=self.data.observed_times
            states=self.data.observed_states
        else:
            print("type_of_input not recognised")
            raise ValueError("type_of_input not recognised")
        df=self.state_enc.translate_labels(states)
        cdef np.ndarray[DTYPEi_t, ndim=1] price_sign = np.array(df['st_1'].values, dtype=DTYPEi) - self.state_enc.num_of_st1//2
        cdef np.ndarray[DTYPEf_t, ndim=1] price = initial_price+0.5*ticksize*np.cumsum(price_sign) 
        cdef np.ndarray[DTYPEf_t, ndim=2] price_trajectory = np.concatenate([
            times.reshape(-1,1), price.reshape(-1,1)
            ],axis=1)
        return price_trajectory
    def plot_price_trajectories(self, DTYPEf_t t0 =0.0, DTYPEf_t t1 = 100.0, 
            save_fig=False, path='./', name='prices', plot=True, return_ax=False):
        def prepare_traj(np.ndarray[DTYPEf_t, ndim=2] x):
            cdef np.ndarray[DTYPEf_t, ndim=2] price = np.array(x, copy=True)
            price[:,0]-=price[0,0]
            price[:,0]+=earliest_time
            return computation.select_interval(price,t0,t1)
        fig=plt.figure(figsize=(10,8))
        ax=fig.add_subplot(111)
        try:
            earliest_time=self.simulated_price[0,0]
        except:
            earliest_time=0.0
        try:
            p=prepare_traj(self.simulated_price)
            ax.plot(p[:,0],p[:,1], label='simulation')
        except:
            print("I could not plot simulated price")
            pass
        try:
            p=prepare_traj(self.data.mid_price.values)
            ax.plot(p[:,0],p[:,1], label='data')
        except:
            print("I could not plot data")
            pass
        try:
            p=prepare_traj(self.reconstructed_empirical_price)
            ax.plot(p[:,0],p[:,1], label='reconstruction', linestyle='--')
        except:
            print("I could not plot reconstructed_empirical_price")
            pass
        ax.set_ylabel('price')
        ax.set_xlabel('time')
        ax.legend()
        fig.suptitle('Price trajectories')
        if save_fig:    
            fname=path+name
            plt.savefig(fname)
        if plot:
            plt.show()   
        if return_ax:
            return ax


    def make_start_liquid_origin_of_times(self,delete_negative_times=False):
        cdef DTYPEf_t time_start = self.liquidator.start_time
        self.liquidator.termination_time -= time_start
        cdef np.ndarray[DTYPEf_t, ndim=1] times = self.simulated_times - time_start*np.ones_like(self.simulated_times)
        if delete_negative_times:
            idx = times>=0.0
            self.simulated_times=np.array(times[idx],copy=True)
            self.simulated_events=np.array(self.simulated_events[idx],copy=True)
            self.simulated_states=np.array(self.simulated_states[idx],copy=True)
            print('SDHawkes: liquidator.start_time has been set as origin of times, and negative times have been deleted')
        else:
            self.simulated_times= np.array(times, copy=True, dtype=DTYPEf)
            print('SDHawkes: liquidator.start_time has been set as origin of times')
        try:
            self.liquidator.inventory_trajectory[:,0]-=time_start
            if delete_negative_times:
                idx=self.liquidator.inventory_trajectory[:,0]>=0.0
                self.liquidator.inventory_trajectory=\
                        np.array(self.liquidator.inventory_trajectory[idx,:], copy=True)
        except:
            pass
        try:
            lt, count = computation.distribute_times_per_event_state(
                self.number_of_event_types, self.number_of_states,
                self.simulated_times,self.simulated_events,self.simulated_states)
            self.labelled_times = lt
            self.count = count
        except:
            pass
        try:
            df=self.liquidator.bm_impact_profile
            df['time']-=time_start
            if delete_negative_times:
                idx_row=df.loc[df['time']<0].index
                df.drop(axis=0,index=idx_row,inplace=True)
        except:
            pass
        try:
            self.simulated_intensities[:,0]-=time_start
        except:
            pass
        self.liquidator.start_time = 0.0 
    def store_calibration(self,calibration):
        self.calibration=calibration    
    def create_archive(self,):
        self.archive={'n_items': 0}
    def stack_to_archive(self,item, str name_of_item= '', idx=None):
        if idx==None:
            if name_of_item == '':
                name_of_item = 'item_'+str(int(self.archive['n_items']+1))
            self.archive[name_of_item]={'n_subitems': 1, name_of_item: item}
            self.archive['n_items']+=1
        else:
            if name_of_item == '':
                name_of_item = 'subitem_{}.{}'.format(idx, self.archive[idx]['n_subitems']+1)
            self.archive[idx][name_of_item]=item
            self.archive[idx]['n_subitems']+=1
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
    def create_uq(self,copy_parameters=False,):
        self.uncertainty_quantification=uncertainty_quant.UncertQuant(
            self.number_of_event_types, self.number_of_states,
            self.n_levels, self.state_enc, self.volume_enc,
            self.base_rates,
            self.impact_coefficients,
            self.decay_coefficients,
            self.transition_probabilities,
            self.dirichlet_param,
            copy=copy_parameters,
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
    def create_mle_estim(self, str type_of_input = 'simulated', store_trans_prob=True, store_dirichlet_param=False, times=None, events=None, states = None, volumes=None):
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
        elif type_of_input == 'passed':
            times = np.array(times, copy=True, dtype=DTYPEf)
            events = np.array(events, copy=True, dtype=DTYPEi)
            states = np.array(states, copy=True, dtype=DTYPEi)
            try:
                volumes = np.array(volumes, copy=True, dtype=DTYPEf)
            except:
                volumes = np.zeros((len(times),2*self.n_levels), dtype=DTYPEf)
            assert len(times)==len(events)
            assert len(events)==len(states)
        else:
            print("ERROR: type_of_input = '{}' not recognised".format(type_of_input))
            raise ValueError("type of input not recognised. It must be either 'simulated', or 'empirical', or 'passed'")
        self.mle_estim=mle_estim.EstimProcedure(
            self.number_of_event_types, self.number_of_states,
            times, events, states,
            volumes = volumes,
            n_levels = self.n_levels, 
            type_of_input = type_of_input,
            store_trans_prob = store_trans_prob,
            store_dirichlet_param = store_dirichlet_param
        )    
        
    def store_history_of_intensities(self,type_of_input='simulated',density_of_eval_points=200):
        try:
            liquid_start=self.liquidator.start_time
            liquid_end=self.liquidator.termination_time
            liquidator_present=True
        except:
            liquid_start=None 
            liquid_end=None 
            liquidator_present=False
        if type_of_input=='simulated':
            times=self.simulated_times
            events=self.simulated_events
            states=self.simulated_states
            self.simulated_intensities=self.compute_history_of_intensities(
                    times,events,states,
                    liquidator_present, liquid_start, liquid_end,
                    density_of_eval_points=density_of_eval_points)

    def compute_history_of_intensities(self,
            times,events,states,
            liquidator_present=False,
            liquid_start=None, liquid_end=None,
            density_of_eval_points=200):
        cdef np.ndarray[DTYPEf_t, ndim=1] arrival_times = np.array(times,dtype=DTYPEf,copy=True)
        cdef np.ndarray[DTYPEi_t, ndim=1] history_of_events = np.array(events,dtype=DTYPEi, copy=True)
        cdef np.ndarray[DTYPEi_t, ndim=1] history_of_states = np.array(states,dtype=DTYPEi, copy=True)
        if liquidator_present:
            if (liquid_start!=None)&(liquid_end!=None):
                start_time_zero_event=float(liquid_start)
                end_time_zero_event=float(liquid_end)
            else:
                liquidator_present=False
        else:
            start_time_zero_event=float(times[0])
            end_time_zero_event=float(times[len(times)-1])
        return computation.compute_history_of_intensities(self.number_of_event_types,
                                                       self.number_of_states,
                                                       arrival_times,
                                                       history_of_events,
                                                       history_of_states,
                                                       self.base_rates,
                                                       self.impact_coefficients,
                                                       self.decay_coefficients,
                                                       liquidator_present=liquidator_present,
                                                       start_time_zero_event=start_time_zero_event,
                                                       end_time_zero_event=end_time_zero_event,
                                                       density_of_eval_points=density_of_eval_points
                                                      )
    
    def compute_history_of_tilda_intensities(self,times,events,states,
            liquidator_present=False,
            liquid_start=None, liquid_end=None,
            density_of_eval_points=100):
        cdef np.ndarray[DTYPEf_t, ndim=1] arrival_times = np.array(times,dtype=DTYPEf,copy=True)
        cdef np.ndarray[DTYPEi_t, ndim=1] history_of_events = np.array(events,dtype=DTYPEi, copy=True)
        cdef np.ndarray[DTYPEi_t, ndim=1] history_of_states = np.array(states,dtype=DTYPEi, copy=True)
        if liquidator_present:
            if (liquid_start!=None)&(liquid_end!=None):
                start_time_zero_event=float(liquid_start)
                end_time_zero_event=float(liquid_end)
            else:
                liquidator_present=False
        else:
            start_time_zero_event=float(times[0])
            end_time_zero_event=float(times[len(times)-1])
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
    
    def store_2Dstates(self,str type_of_input='simulated'):
        if type_of_input=='simulated':
            self.simulated_2Dstates = self.state_enc.produce_2Dstates(self.simulated_states)
        elif type_of_input=='empirical':
            self.data.observed_2Dstates = self.data.state_enc.produce_2Dstates(
                    self.data.observed_states)
    def create_impact_profile(self,
            delete_negative_times=False,
            int num_init_guesses = 6,
            int maxiter = 100,
            time_start=None, time_end=None, 
            produce_weakly_defl_pp=False,
            mle_estim=False):
        self.make_start_liquid_origin_of_times(delete_negative_times=delete_negative_times)
        impact=Impact(
            self.liquidator,
            self.state_enc)
        impact.store_sdhawkes(
            self.base_rates,
            self.impact_coefficients,
            self.decay_coefficients,
            self.transition_probabilities,
            self.simulated_times,
            self.simulated_events,
            self.simulated_states)
        if produce_weakly_defl_pp:
#            impact.produce_weakly_defl_pp(
#                self.base_rates, self.impact_coefficients, self.decay_coefficients,
#                self.transition_probabilities)
            if mle_estim:
                impact.produce_reduced_weakly_defl_pp(
                    num_init_guesses = num_init_guesses,
                    maxiter = maxiter, time_start=time_start, time_end=time_end)
        self.liquidator.get_impact(impact)
        print('\nSDHawkes:  "liquidator.impact" has been initialised\n\n')
        

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
    def plot_events_and_states(self,t_0=None,t_1=None,str type_of_input='simulated',int first_event_index=1,save_fig=False,name='events_and_states_traject'):
        if type_of_input=='simulated':
            if t_0==None:
                t_0=float(self.simulated_times[0])
            if t_1==None:
                t_1=float(self.simulated_times[len(self.simulated_times)-1])
            idx=np.logical_and(self.simulated_times>=t_0, self.simulated_times<=t_1)
            times=self.simulated_times[idx]
            events=first_event_index+self.simulated_events[idx]
            states2D=self.simulated_2Dstates[idx,:]
            intensities=computation.select_interval(self.simulated_intensities, t_0, t_1)
            plot_tools.plot_events_and_states(events,times,intensities,states2D,plot=True,save_fig=save_fig,name=name)
    def plot_intensities(self,t_0=None,t_1=None,str type_of_input='simulated',int first_event_index=1,save_fig=False,name='events_and_states_traject'):
        if type_of_input=='simulated':
            if t_0==None:
                t_0=float(self.simulated_times[0])
            if t_1==None:
                t_1=float(self.simulated_times[len(self.simulated_times)-1])
            intensities=computation.select_interval(self.simulated_intensities, t_0, t_1)
            plot_tools.plot_intensities(intensities,first_event_index=first_event_index,plot=True,save_fig=save_fig,name=name)
    def plot_bm_impact_profile(
            self, 
            time_start=None, time_end=None,
            plot_bm_intensity=False,
            save_fig=False, path=None,name='bm_impact_profile', plot=True
        ):
        if time_start==None:
            time_start=self.liquidator.start_time
        if time_end==None:
            time_end=self.liquidator.termination_time
        plot_tools.plot_bm_impact_profile(
                self.liquidator.impact.sdhawkes.times,
                self.liquidator.impact.sdhawkes.events,
                self.simulated_price,
                self.liquidator.inventory_trajectory,
                self.simulated_intensities,
                self.liquidator.impact.bm_impact_profile,
                self.liquidator.impact.bm_impact_intensity,
                time_start, time_end,
                plot_bm_intensity=plot_bm_intensity,
                save_fig=save_fig,
                path=path, name=name, plot=plot)



    
    
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
