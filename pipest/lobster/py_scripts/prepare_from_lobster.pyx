from functools import lru_cache
from libc.math cimport ceil, floor
import lob_model
import pickle
import numpy as np
import pandas as pd
import sys
import os
cdef str path_pipest = os.path.abspath('./')
n = 0
while (not os.path.basename(path_pipest) == 'pipest') and (n < 6):
    path_pipest = os.path.dirname(path_pipest)
    n += 1
if not os.path.basename(path_pipest) == 'pipest':
    raise ValueError("path_pipest not found. Instead: {}".format(path_pipest))
cdef str path_sdhawkes = path_pipest + '/sdhawkes'
cdef str path_lobster = path_pipest + '/lobster'
cdef str path_lobster_data = path_lobster + '/data'
cdef str path_lobster_pyscripts = path_lobster + '/py_scripts'

sys.path.append(path_sdhawkes + '/')
sys.path.append(path_sdhawkes + '/resources/')
sys.path.append(path_sdhawkes + '/modelling/')
cimport numpy as np

DTYPEf = np.float
DTYPEi = np.int
ctypedef np.float_t DTYPEf_t
ctypedef np.int_t DTYPEi_t




@lru_cache()
def read_from_LOBSTER(str symbol='INTC', str date='2019-01-31',
                      int n_levels=10,
                      dump_after_reading=False,
                      int ticksize=100
                      ):
    cdef list column_name = produce_nameOfColumns(n_levels=n_levels)
    LOB = pd.read_csv(path_lobster_data + '/{}/{}_{}_34200000_57600000_orderbook_10.csv'.format(symbol, symbol, date),
                      header=None, index_col=None, names=column_name)
    LOB = LOB.astype('int', copy=False)
    print('Given shape of orderbook: {}x{}'.format(LOB.shape[0], LOB.shape[1]))
    head_message = [
        'time',
        'event_type',
        'orderID',
        'size',
        'price',
        'direction']
    messagefile = pd.read_csv(path_lobster_data + '/{}/{}_{}_34200000_57600000_message_10.csv'
                              .format(symbol, symbol, date), header=None, index_col=False)
    if messagefile.shape[1] == 7:
        print('I am dropping column 6 of messagefile. Please disregard any Warning about this column')
        messagefile.drop(columns=[6], inplace=True)
    messagefile.columns = head_message
    print('given shape of messagefile: {}x{}'.format(
        messagefile.shape[0], messagefile.shape[1]))
    if not (len(LOB) == len(messagefile)):
        raise ValueError(
            'lengths of order book and of message file do not agree')
    cdef np.ndarray[DTYPEf_t, ndim = 1] times = np.array(messagefile['time'].values, dtype=DTYPEf, copy=True)
    cdef str col
    for col in list(messagefile.columns):
        if col != 'time':
            messagefile = messagefile.astype({col: 'int'}, copy=False)
    messagefile = messagefile.astype({'time': 'float'}, copy=False)
    LOB['time'] = np.zeros(len(LOB), dtype=np.float)
    LOB = LOB.astype({'time': 'float'}, copy=False)
    LOB['time'] = np.array(times, dtype=np.float, copy=True)
    cols = LOB.columns.tolist()
    new_cols = cols[-1:] + cols[:-1]
    LOB = LOB[new_cols]
    initial_time = LOB.loc[0, 'time']
    final_time = LOB.loc[len(LOB) - 1, 'time']
    if not np.all(messagefile.index == LOB.index):
        raise ValueError('messagefile.index != LOB.index')
    cdef np.ndarray[DTYPEi_t, ndim = 1] original_idx = np.array(LOB.index, dtype=DTYPEi, copy=True)
    LOB['original_idx'] = original_idx
    messagefile['original_idx'] = original_idx
    print('read_from_LOBSTER: time window:')
    print('  initial_time={}'.format(initial_time))
    print('  final_time={}'.format(final_time))
    if dump_after_reading:
        print("I am dumping after reading")
        with open(path_lobster + '/data/{}/{}_orderbook_{}'
                  .format(symbol, symbol, date), 'wb') as outfile:
            pickle.dump(LOB, outfile)
        with open(path_lobster + '/data/{}/{}_messagefile_{}'
                  .format(symbol, symbol, date), 'wb') as outfile:
            pickle.dump(messagefile, outfile)
    return LOB, messagefile


class ManipulateMessageFile:
    def __init__(self, LOB=None, mf=None,
                 str symbol='INTC',
                 str date='2019-01-31',
                 int ticksize=100,
                 int n_levels=2,
                 **kwargs
                 ):
        self.symbol = symbol
        self.date = date
        self.n_levels = n_levels
        self.ticksize = ticksize
        self.event_enc = lob_model.event_encoding(n_levels=n_levels)
        self.set_from_lobster(LOB=LOB, mf=mf, **kwargs)
        self.event_enc.store_messagefile(self.messagefile)
        self.messagefile = self.event_enc.produce_marked_messagefile()
        validate_price_change(self.messagefile)
        validate_order_posting(self.messagefile)
        validate_execution(self.messagefile)

    def set_from_lobster(self, LOB=None, mf=None, **kwargs):     
        if LOB is None or mf is None:
            LOB, mf = read_from_LOBSTER(symbol=self.symbol, date=self.date, ticksize=self.ticksize, **kwargs)
        mf = mf.copy()
        LOB = LOB.copy()
        to_drop = mf['event_type'].isin([5,6,7])
        idx_to_drop = mf.loc[to_drop, :].index
        mf.drop(index=idx_to_drop, inplace=True)
        mf = add_lob_status_to_mf(LOB, mf)
        mf = declare_level(mf)
        mf = aggregate_executions(mf)
        mf = declare_midprice_change_in_mf(mf, define_midprice=True)
        LOB, mf = keep_lines_of_interest(LOB, mf)
        LOB, mf = drop_order_cancelling_errors(LOB, mf)
        LOB.reset_index(drop=True, inplace=True)
        mf.reset_index(drop=True, inplace=True)
        self.LOB = LOB
        self.messagefile=mf
        



class ManipulateOrderBook:
    def __init__(self, LOB=None,
                 str symbol='INTC',
                 str date='2019-01-31',
                 int ticksize=100,
                 int n_levels=2,
                 list list_of_n_states=[3, 5],
                 int st1_deflationary=0, int st1_inflationary=2, int st1_stationary=1,
                 int volume_imbalance_upto_level=1,
                 **kwargs
                 ):
        if 'original_idx' not in list(LOB.columns):
            raise ValueError(" 'original_idx' not in LOB.columns")
        self.symbol = symbol
        self.date = date
        self.ticksize = ticksize
        self.n_levels = n_levels
        self.state_enc = lob_model.state_encoding(
            list_of_n_states, st1_deflationary, st1_inflationary, st1_stationary)
        self.volume_enc = lob_model.volume_encoding(
            self.state_enc.volimb_limits, n_levels=self.n_levels, volume_imbalance_upto_level=volume_imbalance_upto_level)
        if LOB is None:
            LOB, _ = read_from_LOBSTER(symbol=self.symbol, date=self.date, ticksize=self.ticksize, **kwargs)
        self.LOB = LOB
    def set_states(self, midprice_changes=None):    
        self.store_midprice()
        self.produce_liquidity_matrix()
        self.encode_sdhawkes_states(midprice_changes=midprice_changes)
        self.create_df_of_volume_proportions()

    def encode_sdhawkes_states(self, midprice_changes = None):
        state_enc = self.state_enc
        volume_enc = self.volume_enc
        vol_imb = volume_enc.compute_volume_imbalance(self.liquidity_matrix)
        cdef np.ndarray[DTYPEi_t, ndim= 2] st_2_arr = volume_enc.classify_vol_imb_vector(vol_imb).reshape(-1, 1)
        cdef np.ndarray[DTYPEf_t, ndim= 1] mid_price = np.array(self.mid_price['mid_price'].values, dtype=DTYPEf)
        if midprice_changes is None:
            state_1 = state_enc.categorise_midprice_change(mid_price).reshape((-1, 1))
        else:
            state_1 = np.array(1 + midprice_changes, copy=True, dtype=DTYPEi).reshape((-1, 1))
        assert len(state_1)==len(st_2_arr)    
        cdef np.ndarray[DTYPEi_t, ndim= 2] st_1_arr = state_1
        cdef np.ndarray[DTYPEi_t, ndim = 2] states_2d = np.concatenate([st_1_arr, st_2_arr], axis=1)
        states_df = state_enc.translate_labels(states_2d, multidim_input=True)
        states_df.index = np.array(self.LOB.index, copy=True)
        states_df.insert(0, 'original_idx', self.LOB['original_idx'].values)
        self.states_df = states_df
        self.LOB.insert(-1+self.LOB.shape[1], 
            'one_dim_state', 
            np.array(
                states_df['one_dim_label'].values, dtype=DTYPEi, copy=True)
        )

    def produce_liquidity_matrix(self,):
        cdef list labels = []
        for k in range(self.n_levels):
            labels.append('ask_volume_{}'.format(k + 1))
            labels.append('bid_volume_{}'.format(k + 1))
        cdef np.ndarray[DTYPEf_t, ndim= 2] liquidity_matrix = np.array(self.LOB.loc[:, labels], dtype=DTYPEf, copy=True)
        self.liquidity_matrix = liquidity_matrix

    def store_midprice(self):
        LOB = self.LOB
        mid_price = np.array(
            (LOB['ask_price_1'] + LOB['bid_price_1']) // 2,
            dtype=DTYPEi)
        mid_price = pd.DataFrame(
                {'time': LOB['time'].values, 'mid_price': mid_price, 'original_idx': LOB['original_idx'].values}, index=LOB.index)
        self.mid_price = mid_price

    def create_df_of_volume_proportions(self):
        n_levels = self.n_levels
        orderbook = self.LOB

        cdef list labels = ['one_dim_state', 'original_idx', 'time']
        for k in range(n_levels):
            labels.append('ask_volume_{}'.format(k + 1))
            labels.append('bid_volume_{}'.format(k + 1))
        volumebook = orderbook.loc[:, labels].copy()
        labels.remove('one_dim_state')
        labels.remove('original_idx')
        labels.remove('time')
        volumebook['tot_vol'] = np.sum(
            volumebook.loc[:, labels].values, axis=1)
        cdef str label
        for label in labels:
            volumebook[label] = np.divide(
                volumebook[label], volumebook['tot_vol'])
        df_state_enc = self.state_enc.df_state_enc.copy()
        try:
            df_state_enc['one_dim_state'] = df_state_enc['one_dim_label']
        except BaseException:
            pass
        volumebook = volumebook.merge(
            df_state_enc, on='one_dim_state', how='left')
        cdef np.ndarray[DTYPEf_t, ndim = 2] volumes = np.array(volumebook.loc[:, labels].values, dtype=DTYPEf)
        self.volumes = volumes
        self.volumebook = volumebook


class DataToStore:
    def __init__(self, man_ob, man_mf, DTYPEf_t time_origin=0.0):
        if not man_ob.symbol == man_mf.symbol:
            raise ValueError('man_ob.symbol and man_mf.symbol do not match')
        else:
            self.symbol = man_ob.symbol
        if not man_ob.date == man_mf.date:
            raise ValueError('man_ob.date and man_mf.date do not match')
        else:
            self.date = man_ob.date
        if not man_ob.ticksize == man_mf.ticksize:
            raise ValueError(
                'man_ob.ticksize and man_mf.ticksize do not match')
        else:
            self.ticksize = man_ob.ticksize
        if not man_ob.n_levels == man_mf.n_levels:
            raise ValueError(
                'man_ob.n_levels and man_mf.n_levels do not match')
        else:
            self.n_levels = man_ob.n_levels
        cdef int number_of_states = man_ob.state_enc.tot_n_states
        cdef int number_of_event_types = man_mf.event_enc.n_event_types
        self.number_of_states = number_of_states
        self.number_of_event_types = number_of_event_types
        lob = man_ob.LOB.copy()
        mf = man_mf.messagefile.copy()
        states_df = man_ob.states_df.copy()
        lob = lob.merge(mf[['original_idx']],
                        on='original_idx',
                        how='inner',
                        validate='1:1')
        mf = mf.merge(lob[['original_idx']], on='original_idx',
                      how='inner', validate='1:1')
        states_df = states_df.merge(lob[['original_idx']], on='original_idx',
                      how='inner', validate='1:1')
        lob.sort_values(by=['original_idx'], inplace=True)
        mf.sort_values(by=['original_idx'], inplace=True)
        states_df.sort_values(by=['original_idx'], inplace=True)
        self.state_enc = man_ob.state_enc
        self.volume_enc = man_ob.volume_enc
        self.event_enc = man_mf.event_enc
        self.states_df = states_df
        self.messagefile = mf
        self.LOB = lob
        self.volumebook = man_ob.volumebook.copy()
        cdef np.ndarray[DTYPEf_t, ndim = 2] volumes = np.array(man_ob.volumes, dtype=DTYPEf, copy=True)
        self.observed_volumes = volumes
        self.store_midprice()
        self.store_states()
        self.store_events()
        self.time_origin = time_origin
        self.store_times(time_origin=time_origin)
        self.store_time_window()

    def store_midprice(self):
        LOB = self.LOB
        mid_price = np.array(
            (LOB['ask_price_1'] + LOB['bid_price_1']) / 2,
            dtype=DTYPEf)
        mid_price = pd.DataFrame(
            {'time': LOB['time'].values, 'mid_price': mid_price}, index=LOB.index)
        self.mid_price = mid_price

    def store_states(self):
        assert np.all(
            self.states_df['one_dim_label'].values == self.LOB['one_dim_state'].values)
        cdef np.ndarray[DTYPEi_t, ndim = 1] states = np.array(self.states_df['one_dim_label'].values, dtype=DTYPEi)
        assert np.all(states == self.volumebook['one_dim_state'].values)
        print('I am storing observed_states')
        self.observed_states = states

    def store_events(self):
        cdef np.ndarray[DTYPEi_t, ndim = 1] events = np.array(self.messagefile['hawkes_mark'].values, dtype=DTYPEi)
        assert len(events) == len(self.observed_states)
        print('I am storing observed_events')
        self.observed_events = events

    def store_times(self, DTYPEf_t time_origin=0.0):
        check_time_nondecreasing(self.messagefile)
        print('I am storing observed_times, with {} as time origin: Notice that this results in a shift in the values of self.observed_times'.format(time_origin))
        cdef np.ndarray[DTYPEf_t, ndim= 1] times = -time_origin + np.array(self.messagefile['time'].values, dtype=DTYPEf)
        assert len(times) == len(self.observed_events)
        self.observed_times = times

    def store_time_window(self):
        mf = self.messagefile
        cdef float first_time = float(mf.loc[0, 'time'])
        cdef float last_time = float(mf.loc[len(mf) - 1, 'time'])
        cdef int initial_time = int(floor(first_time))
        cdef int final_time = int(ceil(last_time))
        cdef list time_window = [initial_time, final_time]
        self.initial_time = initial_time
        self.final_time = final_time
        self.time_window = time_window


#################################################
## Utils
#############################################


def keep_lines_of_interest(LOB, mf):
    no_change_ask_price = mf['ask_price_1'] == mf['ask_price_1_ante']
    no_change_bid_price = mf['bid_price_1'] == mf['bid_price_1_ante']
    to_drop = (mf['event_type']!=4) & no_change_ask_price & no_change_bid_price
    idx_to_drop = mf.loc[to_drop, :].index
    mf.drop(index=idx_to_drop, inplace=True)
    LOB = restrict_lob_to_mf(LOB, mf)
    LOB.reset_index(drop=True, inplace=True)
    mf.reset_index(drop=True, inplace=True)
    return LOB, mf

def restrict_lob_to_mf(LOB, mf):
    LOB = LOB.merge(
            mf.loc[:, ['original_idx']],
            how='right',
            on='original_idx',
            suffixes = ('', '_mf'),
            validate='1:1'
            )
    return LOB

def drop_from_mf_and_restrict(LOB, mf, to_drop):
    idx_to_drop = mf.loc[to_drop, :].index
    mf.drop(index= idx_to_drop, inplace=True)
    LOB = restrict_lob_to_mf(LOB, mf)
    return LOB, mf

def drop_order_cancelling_errors(LOB, mf):
    for side in ['ask', 'bid']:
        to_drop = validate_order_cancelling(mf, side)
        LOB, mf = drop_from_mf_and_restrict(LOB, mf, to_drop)
    return LOB, mf

    
def aggregate_executions(mf,
    agg_dict = {
            'original_idx': 'first',
            'price': 'first',
            'size': 'sum',
            'direction': 'first',
            'level': 'min',
            'event_type': 'first',
            'ask_price_1': 'last',
            'ask_price_1_ante': 'first',
            'bid_price_1': 'last',
            'bid_price_1_ante': 'first',
            'mid_price': 'last',
            'mid_price_ante': 'first',
            },
    groupby_cols = ('time', ),
):
    for col in groupby_cols:
        assert col in mf.columns
    for col in agg_dict.keys():
        assert col in mf.columns
    return mf.groupby(list(groupby_cols)).agg(agg_dict).reset_index()

def declare_midprice_change_in_mf(mf, define_midprice=False):
    if not define_midprice:
        assert 'mid_price' in mf.columns
        assert 'mid_price_ante' in mf.columns
    else:    
        mf['mid_price'] =  (mf['bid_price_1'] + mf['ask_price_1'])//2
        mf['mid_price_ante'] =  (mf['bid_price_1_ante'] + mf['ask_price_1_ante'])//2
    mf.insert(-1+mf.shape[1], 'sign_delta_mid', np.sign(mf['mid_price']-mf['mid_price_ante']))
    return mf

def add_lob_status_to_mf(LOB, mf):
    LOB.insert(0, 'original_idx_plus1', LOB['original_idx'] + 1)
    LOB.set_index('original_idx', inplace=True)
    cols = ['ask_price_1', 'bid_price_1']
    res = mf.merge(
        LOB.loc[:, cols], 
        left_on='original_idx',
        right_index=True, 
        how='inner',
        suffixes=('', '_lob'),
        validate='1:1'
    )
    LOB.reset_index(inplace=True)
    LOB.set_index('original_idx_plus1', inplace=True)
    res = res.merge(
        LOB.loc[:, cols],
        left_on='original_idx',
        right_index=True, 
        how='inner',
        suffixes=('', '_ante'),
        validate='1:1'
    )
    LOB.reset_index(inplace=True)
    res.insert(-1+res.shape[1], 'mid_price', (res['bid_price_1'] + res['ask_price_1'])//2)
    res.insert(-1+res.shape[1], 'mid_price_ante', (res['bid_price_1_ante'] + res['ask_price_1_ante'])//2)
    return res

def add_mid_price(df):
    if ('ask_price_1' not in df.columns) or ('bid_price_1' not in df.columns):
        return df
    df.insert(0, 'mid_price', (df['ask_price_1'] + df['bid_price_1'])//2)
    return df



def validate_price_change(mf):
    no_change_ask_price = mf['ask_price_1'] == mf['ask_price_1_ante']
    no_change_bid_price = mf['bid_price_1'] == mf['bid_price_1_ante']
    no_change_mid_price = no_change_ask_price & no_change_bid_price
    errors = [
        np.logical_and(no_change_mid_price, mf['event_type']!=4),
        np.logical_and(mf['hawkes_mark']==3, mf['sign_delta_mid']!=1),
        np.logical_and(mf['hawkes_mark']==2, mf['sign_delta_mid']!=-1),
        np.logical_and(mf['hawkes_mark']==1, mf['sign_delta_mid']==-1),
        np.logical_and(mf['hawkes_mark']==0, mf['sign_delta_mid']==1),
        ]    
    for (n, error) in zip(range(1, 1+len(errors)), errors):
        if np.any(error):
            print('Validation of price change #{} failed. Num of errors: {}'.format(n, sum(error)))
        else:
            print('Validation of price #{} change passed'.format(n))

def validate_order_posting(mf, side=None):
    if side is None:
        validate_order_posting(mf, side='ask')
        validate_order_posting(mf, side='bid')
        return 0
    d = -1 if side == 'ask' else 1
    idx_price_improvement = find_price_improvement(mf, side)
    idx_order_posting = (mf['event_type']==1) & (mf['direction']==d)
    idx_error = idx_order_posting & ~idx_price_improvement
    if np.any(idx_error):
        print('Validation order posting failed. Side = {}; Num of errors = {}'.format(side, sum(idx_error)))
    else:    
        print('Validation order posting passed. Side = {}'.format(side))
    return 0    

def validate_execution(mf, side=None):        
    if side is None:
        validate_execution(mf, side='ask')
        validate_execution(mf, side='bid')
        return 0
    d = -1 if side == 'ask' else 1
    idx_price_improvement = find_price_improvement(mf, side)
    idx_execution = (mf['event_type']==4) & (mf['direction']==d)
    idx_error = idx_execution & idx_price_improvement
    if np.any(idx_error):
        print('Validation execution failed. Side = {}; Num of errors = {}'.format(side, sum(idx_error)))
    else:    
        print('Validation execution passed. Side = {}'.format(side))
    return 0    

def validate_order_cancelling(mf, side=None):        
    if side is None:
        validate_order_cancelling(mf, side='ask')
        validate_order_cancelling(mf, side='bid')
        return 0
    d = -1 if side == 'ask' else 1
    idx_price_improvement = find_price_improvement(mf, side)
    idx_order_cancelling = (mf['event_type'].isin([2,3])) & (mf['direction']==d)
    idx_error = idx_order_cancelling & idx_price_improvement
    if np.any(idx_error):
        print('Validation order cancelling failed. Side = {}; Num of errors = {}'.format(side, sum(idx_error)))
        return idx_error
    else:    
        print('Validation order cancelling passed. Side = {}'.format(side))
    return 0    

def find_price_improvement(mf, side='ask'):
    d = -1 if side == 'ask' else 1
    col = 'ask_price_1' if side=='ask' else 'bid_price_1'
    assert col in mf.columns 
    assert col+'_ante' in mf.columns
    idx_price_improvement = d*mf[col] > d*mf[col+'_ante']
    return idx_price_improvement



def produce_nameOfColumns(int n_levels=10, first_col_time=False):
    cdef list column_name = []
    if (first_col_time):
        column_name = [
            'time',
            'ask_price_1',
            'ask_volume_1',
            'bid_price_1',
            'bid_volume_1'
        ]
    else:
        column_name = ['ask_price_1',
                       'ask_volume_1',
                       'bid_price_1',
                       'bid_volume_1'
                       ]

    for i in range(2, n_levels + 1):
        column_name.append('ask_price_{}'.format(i))
        column_name.append('ask_volume_{}'.format(i))
        column_name.append('bid_price_{}'.format(i))
        column_name.append('bid_volume_{}'.format(i))
    return column_name


def produce_volume_column_names(self, n_levels=1):
    labels = []
    for k in range(1, self.n_levels + 1):
        labels.append('ask_volume_{}'.format(k))
        labels.append('bid_volume_{}'.format(k))
    return labels

def load_from_pickleFiles(str symbol='INTC', str date='2019-01-31'):
    with open(path_lobster + '/data/{}/{}_orderbook_{}'
              .format(symbol, symbol, date), 'rb') as source:
        LOB = pickle.load(source)
    with open(path_lobster + '/data/{}/{}_messagefile_{}'
              .format(symbol, symbol, date), 'rb') as source:
        messagefile = pickle.load(source)
    return LOB, messagefile

# analysis of a subset of the input matrix


def select_subset(LOB, messagefile,
                  random_timeSubWindow=False, double initial_time=36000, double final_time=46800):
    if (random_timeSubWindow):
        i = np.random.randint(2, 10)
        j = np.random.randint(0, 5)
        m = np.random.randint(
            i * 100,
            np.minimum(
                (1 + j) * i * i * 100,
                0.99 * len(LOB)))
        n = np.random.randint(
            m + j * m, np.minimum(m + i * (1 + j) * m, len(LOB)))
        initial_time = LOB.index[m]
        final_time = LOB.index[n]

    idx = np.array(
        np.logical_and(
            (LOB.time >= initial_time),
            (LOB.time <= final_time)),
        dtype=np.bool)
    LOB = pd.DataFrame(LOB[idx], copy=True)
    messagefile = pd.DataFrame(
        messagefile[idx], copy=True)

    return LOB, messagefile




def declare_level(messagefile, int ticksize=100):
    if 'level' in list(messagefile.columns):
        print("prepare_from_lobster.declare_level: WARNING: messagefile already has column 'level'. I am replacing it")
    if 'ask_price_1_ante' not in list(messagefile.columns):
        raise ValueError(
            "prepare_from_lobster.declare_level: ask_price_1_ante not in messagefile.columns.")
    if 'bid_price_1_ante' not in list(messagefile.columns):
        raise ValueError(
            "prepare_from_lobster.declare_level: bid_price_1_ante not in messagefile.columns.")
    mf = messagefile
    idx_ask = (mf['direction'] == -1)
    idx_bid = (mf['direction'] == 1)
    mf['level'] = np.zeros(len(mf), dtype=np.int)
    mf.loc[idx_ask, 'level'] =\
        1 + np.array(
        (mf.loc[idx_ask, 'price'] -
         mf.loc[idx_ask, 'ask_price_1_ante']) // ticksize,
        dtype=np.int)
    mf.loc[idx_bid, 'level'] =\
        1 + np.array(
        (-mf.loc[idx_bid, 'price'] +
         mf.loc[idx_bid, 'bid_price_1_ante']) // ticksize,
        dtype=np.int)
    idx_execution = np.array(mf['event_type'].isin([4, 5]), dtype=np.bool)
    check = np.array((mf.loc[idx_execution, 'level'] <= 1), dtype=np.bool)
    print('declare_level: check = {}'.format(np.all(check)))
    if not np.all(check):
        print('  number of errors ={}\n'.format(np.sum(check)))
    return mf


def add_delta_t(df):
    time_step = df['time'].diff()
    time_step[time_step.isna()] = df.loc[time_step.isna(), 'time']
    df['delta_t'] = time_step
    df = df.astype({'delta_t': 'float'}, copy=True)
    return df


def locate_same_time_stamp(messagefile):
    if 'delta_t' not in list(messagefile.columns):
        messagefile = add_delta_t(messagefile)
    idx = np.array(
        np.isclose(
            messagefile['delta_t'].values,
            0.0,
            atol=1.0e-13),
        dtype=np.bool)
    if not (len(idx) == len(messagefile)):
        raise ValueError('Error: index not of the same length as dataframe')
    idx = np.logical_or(idx, np.roll(idx, -1))
    return idx


def add_same_time_stamp(messagefile):
    messagefile['same_time_stamp'] = locate_same_time_stamp(messagefile)
    return messagefile


def refresh_same_time_stamp(messagefile, sort_by_original_idx=False):
    messagefile.drop(
        columns=[
            'delta_t',
            'same_time_stamp'],
        errors='ignore',
        inplace=True)
    if sort_by_original_idx:
        if 'original_idx' in list(messagefile.columns):
            messagefile = messagefile.sort_values(by=['original_idx'])
    messagefile = add_delta_t(messagefile)
    messagefile = add_same_time_stamp(messagefile)
    return messagefile


def check_time_nondecreasing(df):
    if 'original_idx' not in list(df.columns):
        print('prepare_from_lobster.check_time_nondecreasing: "original_idx" not found in columns ')
        return
    elif 'time' not in list(df.columns):
        print('prepare_from_lobster.check_time_nondecreasing: "time" not found in columns ')
        return
    else:
        df = df.sort_values(by=['original_idx'])
        error = (np.diff(df['time'].values) < 0)
        if np.any(error):
            print('prepare_from_lobster.check_time_nondecreasing: check has failed')
            print('  number of errors={}'.format(np.sum(error)))
        return np.any(error)


def separate_pair_events(messagefile, int first_eventType=-1, int second_eventType=-1):
    messagefile = refresh_same_time_stamp(messagefile)
    idx = locate_same_time_stamp(messagefile)
    idx_2 = np.array(messagefile['event_type'].shift(
        1) == first_eventType, dtype=np.bool)
    idx_1 = np.array(messagefile['event_type'] ==
                     second_eventType, dtype=np.bool)
    idx = np.logical_and(idx, idx_1)
    idx = np.logical_and(idx, idx_2)
    print('prepare_from_lobster.separate_{}{}_events: number of occurrencies = {} '
          .format(first_eventType, second_eventType, np.sum(idx)))
    messagefile.drop(
        columns=[
            'same_time_stamp',
            'delta_t'],
        inplace=True,
        errors='ignore')
    cdef np.ndarray[DTYPEf_t, ndim = 1] time = np.array(messagefile['time'].values, dtype=DTYPEf, copy=True)
    messagefile.loc[idx, 'time'] = 0.5 * \
        time[idx] + 0.5 * np.roll(time, -1)[idx]
    messagefile = refresh_same_time_stamp(messagefile)
    smallest_time_step = messagefile.loc[messagefile['delta_t'] > 0, 'delta_t'].min(
    )
    avg_time_step = messagefile.loc[messagefile['delta_t']
                                    > 0, 'delta_t'].mean()
    print(
        '   After separation, the average time step is {}, the smallest time step is {}.'.format(
            avg_time_step,
            smallest_time_step))
    check = check_time_nondecreasing(messagefile)
    if check:
        print('  Time has been modified and non-decreasing instances have been inserted!')
        print('  I am correcting this manually')
        idx_error = np.array(messagefile['delta_t'] < 0, dtype=np.bool)
        messagefile.loc[idx_error,
                        'time'] = messagefile.loc[idx_error,
                                                  'time'].shift(1)
        messagefile = refresh_same_time_stamp(messagefile)
    return messagefile


def separate_directions(messagefile,):
    messagefile = refresh_same_time_stamp(messagefile)
    idx = locate_same_time_stamp(messagefile)
    idx_ask = np.array(messagefile['direction'].shift(1) == -1, dtype=np.bool)
    idx_ask = np.logical_or(idx_ask, np.array(
        messagefile['direction'].shift(-1) == -1, dtype=np.bool))
    idx_bid = np.array(messagefile['direction'] == 1, dtype=np.bool)
    idx = np.logical_and(idx, idx_ask)
    idx = np.logical_and(idx, idx_bid)
    print('prepare_from_lobster.separate_directions: number of occurrencies = {} '
          .format(np.sum(idx)))
    messagefile.drop(
        columns=[
            'same_time_stamp',
            'delta_t'],
        inplace=True,
        errors='ignore')
    cdef np.ndarray[DTYPEf_t, ndim = 1] time = np.array(messagefile['time'].values, dtype=DTYPEf, copy=True)
    messagefile.loc[idx, 'time'] = 0.5 * \
        time[idx] + 0.5 * np.roll(time, -1)[idx]
    messagefile = refresh_same_time_stamp(messagefile)
    smallest_time_step = messagefile.loc[messagefile['delta_t'] > 0, 'delta_t'].min(
    )
    avg_time_step = messagefile.loc[messagefile['delta_t']
                                    > 0, 'delta_t'].mean()
    print(
        '   After separation, the average time step is {}, the smallest time step is {}.'.format(
            avg_time_step,
            smallest_time_step))
    check = check_time_nondecreasing(messagefile)
    if check:
        print('  Time has been modified and non-decreasing instances have been inserted!')
        print('  I am correcting this manually')
        idx_error = np.array(messagefile['delta_t'] < 0, dtype=np.bool)
        messagefile.loc[idx_error,
                        'time'] = messagefile.loc[idx_error,
                                                  'time'].shift(1)
        messagefile = refresh_same_time_stamp(messagefile)
    return messagefile


def equiparate_events_with_same_time_stamp(messagefile, int event_type_left, int event_type_right):
    messagefile = refresh_same_time_stamp(messagefile)
    idx_right = np.array(
        messagefile['event_type'] == event_type_right,
        dtype=np.bool)
    idx_left_before = np.array(
        messagefile['event_type'].shift(1) == event_type_left,
        dtype=np.bool)
    idx_left_before = np.logical_and(
        idx_left_before,
        np.array(messagefile['direction'].shift(1) ==
                 messagefile['direction'], dtype=np.bool)
    )
    idx_left_after = np.array(
        messagefile['event_type'].shift(-1) == event_type_left, dtype=np.bool)
    idx_left_after = np.logical_and(
        idx_left_after,
        np.array(messagefile['direction'].shift(-1) ==
                 messagefile['direction'], dtype=np.bool)
    )
    idx_left = np.logical_or(idx_left_before, idx_left_after)
    idx = np.array(messagefile['same_time_stamp'].values, dtype=np.bool)
    idx = np.logical_and(idx, idx_right)
    idx = np.logical_and(idx, idx_left)
    print(
        'prepare_from_lobster.equiparate_events_with_same_time_stamp:\n  event_left={}, event_right={}, occurrencies={}'.format(
            event_type_left,
            event_type_right,
            np.sum(idx)))
    messagefile.loc[idx, 'event_type'] = event_type_left
    return messagefile


def drop_events_with_same_time_stamp(messagefile, int event_type_left, int event_type_right):
    messagefile = messagefile.copy()
    messagefile = refresh_same_time_stamp(messagefile)
    idx = np.array(messagefile['same_time_stamp'].values, dtype=np.bool)
    idx_right = np.array(
        messagefile['event_type'] == event_type_right,
        dtype=np.bool)
    idx_left_before = np.array(
        (messagefile['event_type'].shift(1).isin([event_type_left]))
        & (messagefile['same_time_stamp'].shift(1) == True),
        dtype=np.bool)
    idx_left_after = np.array(
        (messagefile['event_type'].shift(-1).isin([event_type_left]))
        & (messagefile['same_time_stamp'].shift(-1) == True),
        dtype=np.bool)
    idx_left = np.logical_or(idx_left_before, idx_left_after)
    idx = np.logical_and(idx, idx_right)
    idx = np.logical_and(idx, idx_left)
    print(
        'prepare_from_lobster.drop_events_with_same_time_stamp:\n  event_left={}, event_right={}, occurrencies={}'.format(
            event_type_left,
            event_type_right,
            np.sum(idx)))
    messagefile.loc[idx, 'time'] = np.nan
    messagefile.dropna(axis=0, inplace=True)
    messagefile.drop(columns=['same_time_stamp'], inplace=True)
    messagefile = refresh_same_time_stamp(messagefile)
    return messagefile


def drop_same_time_stamp(df, list eventTypes_to_drop=[], str keep='the_last', tol=1.0e-8):
    def locate_time_repetition(df, tol=1.0e-8):
        if 'delta_t' not in list(df.columns):
            df = add_delta_t(df)
        idx_forward = pd.Series(
            np.isclose(
                df['delta_t'],
                0,
                atol=tol),
            dtype=np.bool)
        idx_backward = pd.Series(
            np.isclose(
                df['time'].diff(
                    periods=-1),
                0,
                atol=tol),
            dtype=np.bool)
        if keep == 'the_first':
            idx = idx_forward
        elif keep == 'the_last':
            idx = idx_backward
        else:
            idx = pd.Series(
                np.logical_or(
                    idx_forward,
                    idx_backward),
                dtype=np.bool)
        if not (len(idx) == len(df)):
            print('Error: index not of the same length as dataframe')
        idx = idx
        idx = np.asarray(idx.values, dtype=np.bool)
        return idx

    def add_encountered_time_stamp(df):
        df['encountered_time_stamp'] = locate_time_repetition(df, tol=tol)
        return df

    def refresh_encountered_time_stamp(df):
        df.drop(columns=['delta_t', 'same_time_stamp', 'encountered_time_stamp'],
                errors='ignore', inplace=True)
        df = add_delta_t(df)
        df = add_encountered_time_stamp(df)
        return df
    df = refresh_encountered_time_stamp(df)
    cdef int e = 0, s = 0
    for e in eventTypes_to_drop:
        idx = np.array(df['encountered_time_stamp'], dtype=np.bool)
        idx = np.logical_and(idx, df['event_type'] == e)
        print('I am dropping events with non-unique time stamp '
              + 'and event type={}, while keeping {} of the events in every batch'.format(e, keep))
        print('Numbers of events dropped: {}'.format(np.sum(idx)))
        s += np.sum(idx)
        df.loc[idx, 'time'] = np.nan
        df.dropna(inplace=True)
        df = refresh_encountered_time_stamp(df)
    print('Total number of events dropped: {}'.format(s))
    df.drop(columns=['same_time_stamp', 'encountered_time_stamp'],
            errors='ignore', inplace=True)
    return df


def dump_model_message_file(model, path=None, symbol=None, date=None):
    try:
        symbol = str(model.symbol)
        date = str(model.date)
    except BaseException:
        symbol = str(symbol)
        date = str(date)
    if path is None:
        path = path_lobster_data + '/' + symbol + '_' + date + '_messagefile_model'
    else:
        path = path + '/' + symbol + '_' + date + '_messagefile_model'
    with open(path, 'wb') as outfile:
        pickle.dump(model, outfile)


def dump_model_order_book(model, path=None, symbol=None, date=None):
    try:
        symbol = str(model.symbol)
        date = str(model.date)
    except BaseException:
        symbol = str(symbol)
        date = str(date)
    if path is None:
        path = path_lobster_data + '/' + symbol + '_' + date + '_orderbook_model'
    else:
        path = path + '/' + symbol + '_' + date + '_orderbook_model'
    with open(path, 'wb') as outfile:
        pickle.dump(model, outfile)
