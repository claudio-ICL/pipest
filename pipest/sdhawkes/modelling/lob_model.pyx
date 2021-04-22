#cython: boundscheck=False, wraparound=False, nonecheck=False 

import os
cdef str path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<6):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    raise ValueError("path_pipest not found. Instead: {}".format(path_pipest))
cdef str path_sdhawkes=path_pipest+'/sdhawkes'     
import sys
sys.path.append(path_sdhawkes+'/')
sys.path.append(path_sdhawkes+'/resources/')
sys.path.append(path_sdhawkes+'/modelling/')

import copy
import bisect 
import numpy as np
cimport numpy as np
import pandas as pd

import computation
from rejection_sampling import RejectionSampling

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t


"""
Dictionary for trades and orders' direction.
We follow LOBSTER convention, i.e. we adopt the perspective of the orderbook
and we label events that happend on the bid side with the integer 1,
and events that happened on the ask side with the integer -1. 
This means for example that a limit order to buy a certain number of shares at a certain price is referred to as having direction 1,
and a limit order to sell a certain number of shares at a certain price is referred to as having direction -1. 
Consistently, executions of sell limit orders have direction -1, 
but notice that such executions are buyer initiated trades. 
Analogously, executions of buy limit orders have direction 1, 
but they are seller initiated trades. 
As a consequence, market orders in the LOBSTER message file are identified as being the counterparts to  executions of limit orders,
and a market order is a buy market order when it causes the execution of a sell limit order (hence direction -1),
and a market order is a sell market order when it causes the execution of a buy limit order (hence direction 1). 

Dictionary for event types.
We keep consistency with Bacry and Muzy (2014), 
and we think of event types as having alternatively deflationary consequences on prices 
and inflationary consequences. 
Additionally, we distinguish between executions of orders (i.e. actual trades) 
and posting/cancelling of orders (i.e. events that modify the configuration of the orderbook yet without involving a transaction). 
If our event types are e=1,2,3,4, then: 
e=1 represents trades on the bid side (i.e. execution of buy limit order, i.e. trades initiated by sell market orders). Their direction as encoded in LOBSTER is 1, and their LOBSTER event category is 4 and 5. These trades are likely to deflate prices. 
e=2 represents trades on the ask side (i.e. execution of sell limit order, i.e. trades initiated by buy market orders). Their direction as encoded in LOBSTER is -1, and their LOBSTER event category is 4 and 5. These trades are likely to inflate prices. 
e=3 represents ask limit orders posted inside the spread (hence cause ask price to drop) 
and bid limit orders cancelled from the fiist bid side that depletes the entire liquidity available at the first bid level (hence casuing the bid price to drop).
e=4 represents bid limit orders posted inside the spread (hence cause bid price to increase) 
and ask limit orders cancelled from the first  ask level that depletes the entire liquidity available at the first ask level (hence casuing the ask price to increase).
Remark. Notice that because of Python conventions, when the liquidator is not present in the model, our four event types e=1,2,3,4 are actually labelled e=0,1,2,3. Instead, when the liquidator is present, her interventions will be labelled with event of type e=0, so that the above dictionary e=1,2,3,4 will apply without shifting to 0,1,2,3.

"""

class event_encoding:
    def __init__(self, int n_levels=1, int is_liquidator_present = 0):
        self.n_event_types = 4
        self.n_levels=n_levels   
        print('event_encoding: self.n_event_types={}, self.n_levels={}'.format(self.n_event_types,self.n_levels))
        self.create_df_of_event_encoding()
        
    def store_messagefile(self, messagefile):
        self.messagefile=messagefile
    
    def create_df_of_event_encoding(self):
        n_levels=self.n_levels
        cdef int n_rows = 2*2 + (self.n_event_types-2)*3*n_levels
        df=pd.DataFrame({'hawkes_mark':np.zeros(n_rows,dtype=np.int),
                         'event_type':np.zeros(n_rows,dtype=np.int),
                         'direction':np.zeros(n_rows,dtype=np.int),
                         'level':np.ones(n_rows,dtype=np.int)})        
        df.iloc[:2,0]=0
        df.iloc[:2,2]=1
        df.iloc[2:4,0]=1
        df.iloc[2:4,2]=-1
        df.iloc[[0,2],1]=4
        df.iloc[[1,3],1]=5
        df.iloc[4:4+3*n_levels,0]=2
        df.iloc[4:4+3*n_levels,1]=np.repeat(np.arange(1,4),self.n_levels)
        df.iloc[4:4+3*n_levels,2]=np.repeat(np.array([-1,1,1],dtype=int),n_levels)
        df.iloc[4:4+3*n_levels,3]=np.tile(np.arange(1,1+n_levels),3)
        df.iloc[4+3*n_levels:4+6*n_levels,0]=3
        df.iloc[4+3*n_levels:4+6*n_levels,1]=np.repeat(np.arange(1,4),n_levels)
        df.iloc[4+3*n_levels:4+6*n_levels,2]=np.repeat(np.array([1,-1,-1],dtype=int),n_levels)
        df.iloc[4+3*n_levels:4+6*n_levels,3]=np.tile(np.arange(1,1+n_levels),3)
        self.df_event_enc=df
    
    def produce_marked_messagefile(self):
        mf=self.messagefile
        cdef np.ndarray[DTYPEi_t,ndim=1] original_level = np.array(mf['level'].values,dtype=DTYPEi,copy=True)
        idx=np.array((mf['level'].values<=1),dtype=np.bool)
        mf.loc[idx,'level']=1
        marked_mf=mf.merge(self.df_event_enc, how='left', on=['event_type','direction','level'])
        marked_mf['level']=original_level
        idx=np.array((marked_mf['level']<=self.n_levels),dtype=np.bool)
        marked_mf=marked_mf.loc[idx,:].copy()
        marked_mf=marked_mf.astype({'hawkes_mark':'int'},copy=False)
        self.messagefile = marked_mf
        return marked_mf
        
        
        
        
        

class state_encoding:
    def __init__(self,list list_of_n_states=[],  
                 int st1_deflationary=0, int st1_inflationary=2, int st1_stationary=1):
        cdef np.ndarray[DTYPEi_t, ndim=1] array_of_n_states=np.array(list_of_n_states,dtype=DTYPEi)
        self.n_state_components=len(list_of_n_states)    
        self.list_of_n_states=list_of_n_states
        self.array_of_n_states=array_of_n_states
        cdef int tot_n_states = np.prod(self.array_of_n_states)
        self.tot_n_states= tot_n_states
        cdef int num_of_st1 = self.list_of_n_states[0]
        cdef int num_of_st2 = self.list_of_n_states[1]
        self.num_of_st1 = num_of_st1
        self.num_of_st2 = num_of_st2
        self.create_df_of_states_encoding()
        self.st1_deflationary=st1_deflationary
        cdef list deflationary_states = self.select_states_of_specified_st1(st1_deflationary)
        self.deflationary_states = deflationary_states
        self.st1_inflationary=st1_inflationary
        cdef list inflationary_states = self.select_states_of_specified_st1(st1_inflationary)
        self.inflationary_states = inflationary_states
        self.st1_stationary=st1_stationary
        cdef list stationary_states = self.select_states_of_specified_st1(st1_stationary)
        self.stationary_states = stationary_states
        cdef list weakly_deflationary_states = deflationary_states+stationary_states
        weakly_deflationary_states.sort()
        self.weakly_deflationary_states = weakly_deflationary_states
        cdef np.ndarray[DTYPEi_t, ndim=1] array_weakly_defl_states = np.array(weakly_deflationary_states,dtype=DTYPEi)
        self.array_weakly_defl_states = array_weakly_defl_states
        cdef np.ndarray[DTYPEf_t, ndim=1] volimb_limits = 2.0*np.arange(num_of_st2+1, dtype=DTYPEf)/num_of_st2 - 1.0
        self.volimb_limits = volimb_limits
        
        
    def create_df_of_states_encoding(self, return_df=False):
        list_n_state=self.list_of_n_states
        matrix = np.expand_dims(np.arange(list_n_state[0],dtype=np.int),axis=1)
        if len(list_n_state)>1:
            for n in list_n_state[1:]:
                prev_len=len(matrix)
                matrix=np.repeat(matrix,n,axis=0)
                rightmost=np.expand_dims(np.tile(np.arange(n),prev_len),axis=1)
                matrix=np.concatenate([matrix,rightmost],axis=1)
        cdef np.ndarray[DTYPEi_t, ndim=2] arr_state_enc = np.concatenate(
            [np.expand_dims(np.arange(len(matrix),dtype=DTYPEi),axis=1), matrix], axis=1)
        
        col=[]
        for n in range(len(list_n_state)):
            n+=1
            col.append('st_{}'.format(n))
            
        self.state_names=col
           
        df=pd.DataFrame(matrix,index=range(len(matrix)))
        df.columns=col
        df['one_dim_label']=np.arange(len(matrix),dtype=np.int)
        self.str_labels=(df['one_dim_label'].values).astype(str).tolist()
        subset = df[self.state_names]
        multidim_label = [tuple(x) for x in subset.values]
        df['multidim_label']=multidim_label
        
        if return_df:
            return df
        else:
            self.df_state_enc = df
            self.arr_state_enc = arr_state_enc
    
    def select_states_of_specified_st1(self,int st_1):
        df=self.df_state_enc
        idx=np.array(df['st_1']==st_1,dtype=np.bool)
        result=np.array(df.loc[idx,'one_dim_label'].values,dtype=np.int).tolist()
        return result
    
    def extract_volimb_limits(self,long state):
        cdef int st_2 = state%self.num_of_st2
        return np.array(self.volimb_limits[st_2:st_2+1],dtype=DTYPEf,copy=True)
    
    def convert_onedim_state_code(self,int state):
        return np.array(self.arr_state_enc[state,1:],dtype=DTYPEi)
        
    
    def convert_multidim_state_code(self, np.ndarray[DTYPEi_t, ndim=1] state):
        return convert_multidim_state_code(self.tot_n_states, self.arr_state_enc, state)
    
    def convert_state_code(self,x,multidim_input=False,verbose=False):
        dim_x=len(np.shape(x))
        x=np.array(x)
        list_x=x.tolist()
        if np.shape(x) == ():
            idx=(self.df_state_enc['one_dim_label']==x)
            converted_x=self.df_state_enc.loc[idx,self.state_names].values
        elif np.logical_and((dim_x ==1),np.logical_not(multidim_input)):
            idx=(self.df_state_enc['one_dim_label'].isin(list_x))
            converted_x=self.df_state_enc.loc[idx,self.state_names].values
        elif np.logical_and((dim_x ==1),(multidim_input)):
#             print('multidim_input={}'.format(multidim_input))
            if verbose:
                print('conversion of a single {}-dimensional array encoding one state'.format(dim_x))
                print('input x={}'.format(x))
            def comparison(a,b):
                return np.all(np.isclose(a,b))
            if x.shape==(self.n_state_components,):
                idx=np.apply_along_axis(comparison,1,
                                        self.df_state_enc.loc[:,self.state_names].values,
                                        x)                         
                converted_x=np.squeeze(self.df_state_enc.loc[idx,['one_dim_label']].values)
            else:
                raise ValueError('Conversion of one tuple label into one single label selected, but input has shape {}'
                                 .format(x.shape))
        elif np.logical_and(dim_x>=2,multidim_input):
            raise ValueError('too many input dimensions. Did you intend to use the function translate_labels?')
        converted_x=np.squeeze(converted_x)
        return converted_x
    
    def translate_labels(self,labels,multidim_input=False,tuple_label=False):
        if not multidim_input:
            length=len(labels)
            labels=pd.DataFrame(labels,index=range(length))
            labels.columns=['one_dim_label']
            translation=labels.merge(self.df_state_enc,how='left',on='one_dim_label')
        elif (tuple_label):
            labels.columns=['multidim_label']
            translation=labels.merge(self.df_state_enc,how='left',on='multidim_label')
        elif not tuple_label:
            translation=self.translate_multidim_labels(labels)
        return translation
    
    def translate_multidim_labels(self,np.ndarray[DTYPEi_t, ndim=2] labels):
        df=pd.DataFrame(labels,index=np.arange(len(labels)),dtype=DTYPEi)
        df.columns=self.state_names
        translation=df.merge(self.df_state_enc,how='left',on=self.state_names)
#         translation=translation.astype({'one_dim_label':'int'},copy=False)
        return translation
    
    
    def produce_volumebook_from_np_results(self,states,volumes):
        labels=[]
        n_levels=volumes.shape[1]//2
        for k in range(1,n_levels+1):
            labels.append('askVolume {}'.format(k))
            labels.append('bidVolume {}'.format(k))
        volumebook=pd.DataFrame(volumes)
        volumebook.columns=labels
        volumebook['one_dim_state']=states
        states_df=self.df_state_enc
        states_df['one_dim_state']=states_df['one_dim_label']
        volumebook=volumebook.merge(states_df,on='one_dim_state',how='left')
        return volumebook
    def generate_random_transition_prob(self,n_events=1, dirichlet_param=0):
        if dirichlet_param==0:
            dirichlet_param=np.random.uniform(low=0.5,high=10,size=(self.tot_n_states,))
        Q=np.random.dirichlet(dirichlet_param,size=(self.tot_n_states,n_events))
        return Q
    def compute_midprice_and_spread(self,LOB):
        best_ask_price=LOB['ask_price_1']
        best_bid_price=LOB['bid_price_1']
        mid_price=(best_ask_price+best_bid_price)//2
        spread=best_ask_price-best_bid_price
        return mid_price,spread
    def compute_midprice_change(self,np.ndarray[DTYPEf_t, ndim=1] mid_price):
        cdef np.ndarray[DTYPEf_t, ndim=1] change = np.diff(mid_price,prepend=mid_price[0])
        assert len(change)==len(mid_price)
        cdef np.ndarray[DTYPEi_t, ndim=1] result = np.sign(change,dtype=DTYPEi,casting='unsafe')
        return result
    def categorise_midprice_change(self,mid_price):
        cdef np.ndarray[DTYPEf_t, ndim=1] price = np.array(mid_price,dtype=DTYPEf)
        cdef np.ndarray[DTYPEi_t, ndim=1] change = self.compute_midprice_change(price)
        cdef np.ndarray[DTYPEi_t, ndim=1] result = np.ones_like(change) + change
        return result
    def categorise_spread(self,spread,tickSize=100,n_categories=2):
        spread=np.array(spread//tickSize,dtype=np.int)
        for k in range(n_categories):
            if k==0:
                idx=(spread<=1)
                spread[idx]=0
            elif k==(n_categories-1):
                idx=spread>=n_categories
                spread[idx]=k
            else:
                idx=(spread==k+1)
                spread[idx]=k
        return spread.astype(np.int)
    def reconstruct_price_trajectory(self,one_dim_states,initial_price=0,tickSize=100,time=None,is_mid_price=True):
        multi_dim_states=self.translate_labels(one_dim_states)
        price_movement=np.array(multi_dim_states['st_1'].values,dtype=np.int)-1
        if is_mid_price:
            price_trajectory=initial_price+0.5*tickSize*np.cumsum(price_movement)/2
        else:
            price_trajectory=initial_price+0.5*tickSize*np.cumsum(price_movement)
        if not time==None:
            price=pd.DataFrame({'time': time, 'recon_price': price_trajectory})
            return price
        else:
            return price_trajectory
    def produce_2Dstates(self,np.ndarray[DTYPEi_t, ndim=1] states):
        df=self.translate_labels(states)
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

        
        
class volume_encoding:
    def __init__(self, np.ndarray[DTYPEf_t, ndim=1] volimb_limits,
                 int n_levels=1,int volume_imbalance_upto_level=1, DTYPEf_t step=0.1,
                 ):
        if volume_imbalance_upto_level>n_levels:
            print('As per input: \n n_levels={}, volume_imbalance_upto_level={}'
                  .format(n_levels,volume_imbalance_upto_level))
            raise ValueError('volume_imbalance_upto_level must be smaller or equal than n_levels')
            
            
        self.n_levels=n_levels
        self.volume_imbalance_upto_level=volume_imbalance_upto_level
#         self.create_df_of_volume_encoding(step)
        try:
            self.store_volimb_limits(volimb_limits)
        except:
            pass
        
        
    def compute_volume_imbalance(self,np.ndarray[DTYPEf_t, ndim=2] liquidity_matrix, int upto_level=0):
        """
        liquidity matrix is supposed to have an instance every row. the columns are alternatively ask 
        and bid volumes from level 1 to level n
        """
        cdef np.ndarray[DTYPEf_t, ndim=2] matrix = liquidity_matrix
        if upto_level==0:
            upto_level=self.volume_imbalance_upto_level
        cdef int uplim=1+2*upto_level    
        cdef np.ndarray[DTYPEf_t, ndim=1] vol_ask = np.sum(matrix[:,0:uplim:2],axis=1)
        cdef np.ndarray[DTYPEf_t, ndim=1] vol_bid = np.sum(matrix[:,1:uplim:2],axis=1)
        cdef np.ndarray[DTYPEf_t, ndim=1] vol_imb = np.divide((vol_bid-vol_ask),np.maximum(1.0e-5,vol_bid+vol_ask))
        return vol_imb
    
    def compute_volume_imbalance_scalar(self,np.ndarray[DTYPEf_t, ndim=1] volumes, int upto_level=0):
        if upto_level==0:
            upto_level=self.volume_imbalance_upto_level     
        return compute_volume_imbalance_scalar(volumes,upto_level)    
       
    def classify_vol_imb_scalar(self, DTYPEf_t vol_imb):
        """
        volume imbalance is expected as a scalar with value between -1 and 1
        categories are sorted from the most negative volume imbalance to the most positive
        """
        return classify_vol_imb_scalar(vol_imb, self.volimb_limits)
    
    def classify_vol_imb_vector(self, np.ndarray[DTYPEf_t, ndim=1] vol_imb):
        """
        volume imbalance is expected as a one dimensional vector with values between -1 and 1
        categories are sorted from the most negative volume imbalance to the most positive according to self.volimb_limits
        """
        cdef Py_ssize_t k = 0
        cdef int len_vector = len(vol_imb)
        cdef np.ndarray[DTYPEi_t, ndim=1] classified_vi = np.zeros(len_vector, dtype=DTYPEi)
        for k in range(len_vector):
            classified_vi[k] =  -1+bisect.bisect_left(self.volimb_limits, vol_imb[k])
        assert np.all(classified_vi>=0)    
        return classified_vi    
    def store_dirichlet_param(self, np.ndarray[DTYPEf_t, ndim=2] dirichlet_param):                              
        self.dirichlet_param = dirichlet_param
    def store_volimb_limits(self,np.ndarray[DTYPEf_t, ndim=1] volimb_limits):
        self.volimb_limits = volimb_limits
    def create_rejection_sampling(self,int N_samples=10**6):
        self.rejection_sampling=RejectionSampling(self.dirichlet_param,
                self.volimb_limits, self.volume_imbalance_upto_level, N_samples_for_prob_constraints=N_samples)
        
        
cdef long convert_multidim_state_code(
    int num_of_states, DTYPEi_t [:,:] arr_state_enc, DTYPEi_t [:] state) nogil:
    cdef int i=0
    for i in range(num_of_states):
        if ( (arr_state_enc[i,1] == state[0]) & (arr_state_enc[i,2]==state[1]) ):
            return arr_state_enc[i,0]
    return 0         
        
        
cdef double compute_volume_imbalance_scalar(DTYPEf_t [:] volumes, int upto_level=2) nogil:
    cdef int n=0
    cdef DTYPEf_t vol_ask=0.0, vol_bid=0.0
    while n<upto_level:
        vol_ask+=volumes[2*n]
        vol_bid+=volumes[1+2*n]
        n+=1
    return ((vol_bid-vol_ask)/max(1.0e-10,vol_bid+vol_ask))

cdef int classify_vol_imb_scalar(DTYPEf_t vol_imb, DTYPEf_t [:] volimb_limits):
    """
    volume imbalance is expected as a scalar with value between -1 and 1
    categories are sorted from the most negative volume imbalance to the most positive
    """
    return int(max(0,-1+bisect.bisect_left(volimb_limits, vol_imb)))
        
def correct_null_transition_prob(transition_probabilities):
    n_events=transition_probabilities.shape[1]
    n_states=transition_probabilities.shape[0]
    if not n_states==transition_probabilities.shape[2]:
        print('shape of transition probabilities = {}'.transition_probabilities.shape)
        raise ValueError('shape of inserted transition_probabilities is incorrect')
    Q=np.array(transition_probabilities,copy=True)
    Q=np.reshape(Q,(-1,n_states))
    idx=Q==0
    idx_0=np.apply_along_axis(np.all,1,idx)
    idx_1=(np.squeeze(np.argwhere(idx_0))//n_events)
    Q[idx_0,idx_1]=1
    Q=Q.reshape((n_states,n_events,n_states))
    return Q        
        
        
           

#Old class (do not use)
class RejectSampling:    
    def __init__(self, np.ndarray[DTYPEf_t, ndim=2] gamma,
                 np.ndarray[DTYPEf_t, ndim=2] gamma_tilde,
                 np.ndarray[DTYPEf_t, ndim=1] prob_constraint,
                 np.ndarray[DTYPEf_t, ndim=1] bound,
                 np.ndarray[DTYPEf_t, ndim=1] volimb_limits, int volimb_upto_level, 
                 DTYPEf_t tol = 1.0e-8):
        self.target_dir_param = gamma
        self.proposal_dir_param = gamma_tilde
        self.difference_of_dir_params = gamma - gamma_tilde
        self.prob_constraint = prob_constraint
        self.bound = bound
        self.inverse_bound = np.divide(1.0, bound)
        cdef np.ndarray[DTYPEi_t, ndim=1] is_equal = np.zeros(gamma.shape[0], dtype=DTYPEi)
        cdef i = 0
        for i in range(len(is_equal)):
            is_equal[i] = np.allclose(gamma[i,:], gamma_tilde[i,:], tol, tol)
        self.is_target_equal_to_proposal = is_equal
        self.volimb_limits = volimb_limits
        self.volimb_upto_level = volimb_upto_level
        cdef int num_of_st2 = len(volimb_limits)-1
        self.num_of_st2 = num_of_st2
