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

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t


"""
Dictionary for trades and orders' direction.
We follow LOBSTER convention, i.e. we adopt the perspective of the orderbook and we label events that happend on the bid side with the integer 1, and event that happened on the ask side with the integer -1. This means for example that a limit order to buy a certain number of shares at a certain price is referred to as having direction 1, and a limit order to sell a certain number of shares at a certain price is referred to as having direction -1. Consistently, executions of sell limit orders have direction -1, but notice that such executions are a buyer initiated trades. Analogously, executions of buy limit orders have direction 1, but they are seller initiated trades. As a consequence, market orders in the LOBSTER message file are identified as being the counterparts to  executions of limit orders, and a market order is a buy market order when it causes the execution of a sell limit order (hence direction -1), and a market order is a sell market order when it causes the execution of a buy limit order (hence direction 1). 

Dictionary for event types.
We keep consistency with Bacry and Muzy (2014), and we think of event types as having alternatively deflationary consequences on prices and inflationary consequences. Additionally, we distinguish between executions of orders (i.e. actual trades) and posting/cancelling of orders (i.e. events that modify the configuration of the orderbook yet without involving a transaction). If our event types are e=1,2,3,4, then: 
e=1 represents trades on the bid side (i.e. execution of buy limit order, i.e. trades initiated by sell market orders). Their direction as encoded in LOBSTER is 1, and their LOBSTER event category is 4 and 5. These trades are likely to deflate prices. 
e=2 represents trades on the ask side (i.e. execution of sell limit order, i.e. trades initiated by buy market orders). Their direction as encoded in LOBSTER is -1, and their LOBSTER event category is 4 and 5. These trades are likely to inflate prices. 
e=3 represents limit orders posted on the ask side (hence LOBSTER direction -1, LOBSTER event category 1) and limit orders cancelled from the bid side (hence LOBSTER direction 1, LOBSTER event category 2 and 3). These activities are likely to deflate prices.
e=4 represents limit orders posted on the bid side (hence LOBSTER direction 1, LOBSTER event category 1) and limit orders cancelled from the ask side (hence LOBSTER direction -1, LOBSTER event category 2 and 3). These activities are likely to inflate prices.
By orderbook rules, executions happen at the best available price, hence events of type e=1 and e=2 will always occur on level 1. Instead, limit orders can be submitted to (or cancelled from) deeper levels of the orderbook. We restrict our attention to a small number of levels of the order book, stored by the integer n_levels.  
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
        mf=self.messagefile.copy()
        cdef np.ndarray[DTYPEi_t,ndim=1] original_level = np.array(mf['level'].values,dtype=DTYPEi,copy=True)
        idx=np.array((mf['level'].values<=1),dtype=np.bool)
        mf.loc[idx,'level']=1
        marked_mf=mf.merge(self.df_event_enc, how='left', on=['event_type','direction','level'])
        marked_mf['level']=original_level
        idx=np.array((marked_mf['level']<=self.n_levels),dtype=np.bool)
        marked_mf=marked_mf.loc[idx,:].copy()
        marked_mf=marked_mf.astype({'hawkes_mark':'int'},copy=False)
        self.marked_messagefile = marked_mf
        
        
        
        
        
#     def create_sell_category(self,messagefile):
#         is_category=self.define_event_category(messagefile,direction='sell')
#         is_sell=self.distribute_per_levels(messagefile,is_category,n_level_distribution=self.n_level_distribution)
#         return is_sell
#     def create_buy_category(self,messagefile):
#         is_category=self.define_event_category(messagefile,direction='buy')
#         is_buy=self.distribute_per_levels(messagefile,is_category,n_level_distribution=self.n_level_distribution)
#         return is_buy
#     def define_event_category(self,messagefile, int direction=1):
#         if (direction=='sell') | (direction == -1):
#             direction=np.array(-1,dtype=np.int)
#             opposite_direction=np.array(1,dtype=np.int)
#         else:
#             direction=np.array(1,dtype=np.int)
#             opposite_direction=np.array(-1,dtype=np.int)
#         is_market_order=np.logical_and(messagefile['direction']==opposite_direction,
#                                        messagefile['eventType'].isin([4,5]))
#         is_limit_order=np.logical_and(messagefile['direction']==direction,
#                                       messagefile['eventType']==1)
#         is_cancellation=np.logical_and(messagefile['direction']==opposite_direction,
#                                        messagefile['eventType'].isin([2,3]))
#         is_category=np.logical_or(
#             np.logical_or(is_market_order,
#                           is_limit_order),
#             is_cancellation)
#         return is_category
#     def distribute_per_levels(self,messagefile,is_category,n_level_distribution=1):
#         if 'level' not in list(messagefile.columns):
#             print('event_encoding: Error: messagefile has no indication of level')
#         else:
#             distribution=np.zeros((len(messagefile),n_level_distribution),dtype=np.bool)
#             for k in range(n_level_distribution):
# #                 print(' distribute_per_levels: self.n_level_distriburion={}, k={}'.format(self.n_level_distribution,k))
#                 if (k==0):
#                     if n_level_distribution==1:
#                         distribution[:,0]=is_category
#                     else:
# #                     print(' distribute_per_levels: enetering first level')
#                         distribution[:,0]=np.logical_and(is_category,
#                                                      messagefile['level']<=1)
#                 elif (k==(n_level_distribution-1)):
# #                     print(' distribute_per_levels: enetering last level')
#                     distribution[:,k]=np.logical_and(is_category,
#                                                      messagefile['level']>=n_level_distribution)
#                 else:
#                     distribution[:,k]=np.logical_and(is_category,
#                                                      messagefile['level']==k)
#             reconstruct=np.array(distribution[:,0],copy=True)
#             for k in range(distribution.shape[1]):
#                 reconstruct=np.logical_or(reconstruct,distribution[:,k])
#             check=np.all(reconstruct==is_category)
#             if not check:
#                 print('\n event_encoding: distribute_per_levels: Error:')
#                 print(' reconstruction failed')
#             return distribution
        
        

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
        self.stationary_sates = stationary_states
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
        mid_price=(best_ask_price+best_bid_price)/2
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
            price_trajectory=initial_price+tickSize*np.cumsum(price_movement)/2
        else:
            price_trajectory=initial_price+tickSize*np.cumsum(price_movement)
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
        
        
#     def create_df_of_volume_encoding(self,DTYPEf_t step):
#         values=np.round(np.arange(0,1,step,dtype=np.single),decimals=3)
#         self.volume_values=values
#         decimals=np.ceil(0.1*len(values)).astype(np.int)
#         self.decimals=decimals
#         self.unit_of_vol=np.array(10**decimals,dtype=np.int)
#         self.int_volume_values=np.array(self.unit_of_vol*self.volume_values,dtype=np.int)
# #         print('volume values=\n{}'.format(values))
# #         print('integer volume values=\n{}'.format(self.int_volume_values))
#         discrete_proportions=np.round(
#             values,
#             decimals=decimals)
# #         print('volume values=\n{}'.format(discrete_proportions))
#         n_discrete_proportions=len(discrete_proportions)
#         int_vol=np.array(discrete_proportions*(self.unit_of_vol),dtype=np.int)
#         matrix=np.expand_dims(int_vol,axis=1)
#         k=1
#         side=1
#         col=['ask_1']
#         while k<= self.n_levels:
#             side=np.mod(side,2)
#             while side <=1:
#                 prev_len=len(matrix)
#                 matrix=np.repeat(matrix,n_discrete_proportions,axis=0)
#                 rightmost=np.expand_dims(np.tile(int_vol,prev_len),axis=1)
#                 matrix=np.concatenate([matrix,rightmost],axis=1)
#                 if side == 0:
#                     col.append('ask_{}'.format(k))
#                 elif side ==1:
#                     col.append('bid_{}'.format(k))
#                 side +=1
#             k+=1
# #         matrix=np.round(matrix,decimals=decimals)
#         idx=(np.sum(matrix,axis=1)==self.unit_of_vol)
#         matrix=np.array(matrix[idx,:])
#         vol_imb=self.compute_volume_imbalance(matrix.astype(DTYPEf))
#         self.liquidity_matrix=matrix
#         self.volumes_names=col
#         df=pd.DataFrame(matrix,index=range(len(matrix)))
#         df.columns=col
#         df['one_dim_label']=np.arange(len(matrix),dtype=np.int)
#         self.str_volume_labels=(df['one_dim_label'].values).astype(str).tolist()
#         subset = df[self.volumes_names]
#         multidim_label = [tuple(x) for x in subset.values]
#         df['multidim_label']=multidim_label
#         self.df_vol=df.copy()
#         df.loc[:,self.volumes_names]=np.round(matrix/self.unit_of_vol,decimals=self.decimals)
        
#         df['vol_imb_{}'.format(self.volume_imbalance_upto_level)]=vol_imb
#         self.df_full=df.copy()
#         check = np.all(np.isclose(np.sum(df[self.volumes_names].values,axis=1),1))
#         if not check:
#             raise ValueError('sum of proportions of volumes does NOT sum up to 1')
        
#         if self.volume_imbalance_upto_level ==1:
#             check=np.all(np.isclose(
#                 df['vol_imb_1'],
#                 (df['bid_1']-df['ask_1'])/np.maximum((df['bid_1']+df['ask_1']),1.0e-7)
#             ))
#             if check:
#                 print('check of vol_imb passed')
#             else:
#                 print('vol_imb is wrong')

#     def convert_state_code(self,x,integer_input=False,multidim_input=False):
#         if np.logical_and(np.logical_not(integer_input),multidim_input):
#             x=np.array(self.unit_of_vol*np.round(x,decimals=self.decimals),dtype=np.int)
#         x=np.squeeze(x)
#         dim_x=len(np.shape(x))
# #         print('dim_x={}'.format(dim_x))
#         list_x=x.tolist()
#         if np.shape(x) == ():
#             idx=(self.df_vol['one_dim_label']==x)
#             converted_x=self.df_vol.loc[idx,self.volumes_names].values
#         elif np.logical_and((dim_x ==1),(np.logical_not(multidim_input))):
# #             print('multidim_input={}'.format(multidim_input))
#             print('conversion of {} one-dimensional labels'.format(x.shape[0]))
#             print('input x={}'.format(x))
#             idx=(self.df_vol['one_dim_label'].isin(list_x))
#             converted_x=self.df_vol.loc[idx,self.volumes_names].values
#         elif np.logical_and((dim_x ==1),(multidim_input)):
# #             print('multidim_input={}'.format(multidim_input))
#             print('conversion of a single {}-dimensional array of volume proportions'.format(dim_x))
#             print('input x={}'.format(x))
#             if not (np.sum(x)==self.unit_of_vol):
#                 raise ValueError('proportions do not sum up to the correct unit of volumes: \n np.sum(x)={}'.format(np.sum(x)))
#             def comparison(a,b):
#                 return np.all(np.isclose(a,b))
#             if x.shape==(2*self.n_levels,):
#                 idx=np.apply_along_axis(comparison,1,
#                                         self.liquidity_matrix,
#                                         x)                         
#                 converted_x=np.squeeze(self.df_vol.loc[idx,['one_dim_label']].values)
#             else:
#                 raise ValueError('Conversion of one tuple label into one single label selected, but input has shape {}'
#                                  .format(x.shape))
#         elif np.logical_and(dim_x>=2,multidim_input):
#             raise ValueError('too many input dimensions. Did you intend to use the function translate_labels?')
#         converted_x=np.squeeze(converted_x)    
#         return converted_x
    
#     def translate_labels(self,labels,multidim_input=False,tuple_label=False,integer_input=False):
#         cdef int length = len(labels)
#         if np.logical_and(np.logical_not(integer_input),multidim_input):
#             labels=np.round(labels,decimals=self.decimals)
#             labels=np.array(labels*(10**self.decimals),dtype=np.int)
#         labels=pd.DataFrame(labels,index=range(length))    
        
#         if not multidim_input:
#             labels.columns=['one_dim_label']
#             translation=labels.merge(self.df_vol,how='left',on='one_dim_label')
#         elif (tuple_label):
#             labels.columns=['multidim_label']
#             translation=labels.merge(self.df_vol,how='left',on='multidim_label')
#         elif not tuple_label:
#             labels.columns=self.volumes_names
#             translation=labels.merge(self.df_vol,how='left',on=self.volumes_names)
#         translation['one_dim_label']=np.array(translation['one_dim_label'].values,dtype=np.int)
#         return translation
    
#     def discretise_volumes(self,volumes,adjustment_position=0):
#         """
#         Given a matrix of instances of volume proportions (one instance every row), the function returns
#         the closest  matrix of volumes in the unit of the class  
#         """
        
#         instances=volumes.shape[0]
#         n_levels=volumes.shape[1]//2
#         if not adjustment_position<=volumes.shape[1]:
#             raise ValueError('adjustment_position > volumes.shape[1]')
#         if (adjustment_position==0):
#             pos_to_adjust=np.random.randint(volumes.shape[1],size=(instances,))
#         else:
#             pos_to_adjust=np.array((adjustment_position-1)*np.ones((instances,)),dtype=np.int)
            
#         alternative=np.mod(pos_to_adjust+1,volumes.shape[1])
        

#         def replace_with_closest(A):
#             original_shape=tuple(np.array(A.shape,copy=True))
#             A=A.flatten()
#             v=np.sort(self.int_volume_values)
#             for i in range(A.shape[0]):
#                 previous_value=np.array(A[i],copy=True)
#                 idx=np.amin([np.abs(v-previous_value).argmin()])
#                 A[i]=v[idx]
# #                 print('entry {} of A'.format(i))
# #                 print('previous_value={}'.format(previous_value))
# #                 print('new value={}'.format(v[idx]))
#             A=np.reshape(A,original_shape)
#             return A
#         disc_vol=np.round(self.unit_of_vol*volumes,decimals=3)
#         disc_vol=np.minimum(disc_vol,np.amax(self.int_volume_values))
            
# #         print('discretised_volumes: given volumes = \n {}'.format(disc_vol))
#         disc_vol=np.array(np.floor(replace_with_closest(disc_vol)),dtype=np.int)
#         sum_=np.sum(disc_vol,axis=1)-disc_vol[np.arange(disc_vol.shape[0]),pos_to_adjust]
#         idx=(sum_>self.unit_of_vol)
#         if np.any(idx):
#             print('Warning:sum_ > unit of volume \n unit={},np.amax(sum_)={}'.format(self.unit_of_vol,np.amax(sum_)))
#             print('I am correcting this manually')
#             disc_vol[idx,alternative[idx]]-=1
            
#         disc_vol=np.maximum(disc_vol,0)
#         disc_vol[np.arange(disc_vol.shape[0]),pos_to_adjust]=np.maximum(0,self.unit_of_vol-sum_)
# #         print('discretised_volumes: modified volumes = \n {}'.format(disc_vol))
#         if not np.all(np.sum(disc_vol,axis=1)==self.unit_of_vol):
#             print('Error')
#             print('disc_vol=\n {}'.format(disc_vol))
#             print('sum along lines =\n  {}'.format(np.sum(disc_vol,axis=1)))
#             raise ValueError('proportions do not sum up to unit of vol')
#         return disc_vol
    
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
    
#     def classify_vol_imb(self,np.ndarray[DTYPEf_t, ndim=1] vol_imb, n_categories=5):
#         """
#         volume imbalance is expected as a one dimensional vector with values between -1 and 1
#         categories are sorted from the most negative volume imbalance to the most positive
#         """
#         cdef np.ndarray[DTYPEf_t, ndim=1] vi=np.array(vol_imb,copy=True)
#         vi=np.minimum(np.maximum(vi,-1),1)
#         vi=np.minimum(np.maximum(0,0.5*(vi+1.0)),1)
#         vi=np.trunc(n_categories*vi)
#         cdef np.ndarray[DTYPEi_t, ndim=1] classified_vi = np.array(np.minimum(vi,n_categories-1),dtype=DTYPEi)
#         classified_vi=np.trunc(np.mod(vi,n_categories)).astype(DTYPEi)
#         return classified_vi
    
    def store_dirichlet_param(self,
                              np.ndarray[DTYPEf_t, ndim=2] dirichlet_param,
                              int num_of_st2 = 5, int N_samples_for_prob_constraints=9999):
        self.dirichlet_param = dirichlet_param
        upto_level = self.volume_imbalance_upto_level
        cdef int n_states = dirichlet_param.shape[0]
        cdef np.ndarray[DTYPEf_t, ndim=1] masses = np.zeros(n_states,dtype=DTYPEf)
        masses = computation.produce_dirichlet_masses(dirichlet_param)
        self.dirichlet_masses = masses
        cdef np.ndarray[DTYPEf_t, ndim=1] probs =\
        computation.produce_probabilities_of_volimb_constraints(
            upto_level,n_states,num_of_st2,
            self.volimb_limits, dirichlet_param, N_samples = N_samples_for_prob_constraints
        )
        self.prob_volimb_constraint = probs
        
    def store_volimb_limits(self,np.ndarray[DTYPEf_t, ndim=1] volimb_limits):
        self.volimb_limits = volimb_limits
   
    
    def store_param_for_rejection_sampling(self,DTYPEf_t epsilon = 0.1):
        upto_level = self.volume_imbalance_upto_level
        cdef int uplim = 1+2*upto_level
        dir_param=self.dirichlet_param
        volimb_limits = self.volimb_limits
        cdef int num_of_st2 = len(volimb_limits)-1
        cdef np.ndarray[DTYPEf_t, ndim=2] multiplier = np.ones_like(dir_param,dtype=DTYPEf)
        cdef DTYPEf_t sum_bid=0.0, sum_ask=0.0, l=0.0, h=0.0, imb = 0.0, tot_param=0.0
        cdef DTYPEf_t coef_ask = 1.0, coef_bid = 1.0, beta_bar = 1.0
        cdef int j=0, st_2=0
        for j in range(dir_param.shape[0]):
            st_2= j%num_of_st2
            l = volimb_limits[st_2]
            h = volimb_limits[st_2+1]
            sum_bid = np.sum(dir_param[j,1:uplim:2])
            sum_ask = np.sum(dir_param[j,0:uplim:2])
            imb = sum_bid - sum_ask
            tot_param = sum_bid+sum_ask
            if (imb < l*tot_param) or (imb > h* tot_param):
                coef_ask = (1-l)/((1+h-l)*(sum_ask))
                coef_bid = (1+h)/((1+h-l)*(sum_bid))
                beta_bar = max(coef_ask,coef_bid) + min(epsilon, np.amin(dir_param))
                multiplier[j,1:uplim:2]= coef_bid/beta_bar
                multiplier[j,0:uplim:2]= coef_ask/beta_bar
        cdef np.ndarray[DTYPEf_t, ndim=2] dir_param_rs = multiplier*dir_param
        cdef np.ndarray[DTYPEf_t, ndim=1] bound = np.ones(dir_param.shape[0])
        bound = np.apply_along_axis(
            computation.compute_maximum_unnormalised_pseudo_dirichlet_density,1,
            dir_param - dir_param_rs,
        )
        self.rejection_sampling = RejectSampling(
            dir_param, dir_param_rs, self.prob_volimb_constraint, bound, volimb_limits, upto_level)
        
        

        
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
        
        
            
