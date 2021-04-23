#!/usr/bin/env python
# coding: utf-8

import os
path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<6):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    raise ValueError("path_pipest not found. Instead: {}".format(path_pipest))
path_sdhawkes=path_pipest+'/sdhawkes'
path_lobster=path_pipest+'/lobster'
path_lobster_data=path_lobster+'/data'
path_lobster_pyscripts=path_lobster+'/py_scripts'


import numpy as np
import pandas as pd
import pickle
import datetime
import time

import sys
sys.path.append(path_lobster_pyscripts+'/')
sys.path.append(path_sdhawkes+'/modelling/')
sys.path.append(path_sdhawkes+'/resources/')

import model as sd_hawkes_model
import lob_model
import prepare_from_lobster as from_lobster

def main(
    symbol='INTC',
    date='2019-01-22',
    initial_time=10.0*60*60,
    final_time=15.0*60*60,
    first_read_fromLOBSTER=True,
    dump_after_reading=False,
):
    time_window=str('{}-{}'.format(int(initial_time),int(final_time)))

    print('I am reading from lobster')
    print('symbol={}, date={}, time_window={}'.format(symbol,date,time_window))

    saveout=sys.stdout
    print('Output is being redirected to '+path_lobster_data+'/{}/{}_{}_{}_readout'.format(symbol,symbol,date,time_window))
#    fout=open(path_lobster_data+'/{}/{}_{}_{}_readout'.format(symbol,symbol,date,time_window),'w')
#    sys.stdout=fout


    print('symbol={}, date={}, time_window={}\n'.format(symbol,date,time_window))
    if (first_read_fromLOBSTER):
        LOB,messagefile=from_lobster.read_from_LOBSTER(symbol,date,
                                          dump_after_reading=dump_after_reading,
                                          )
    else:
        LOB,messagefile=from_lobster.load_from_pickleFiles(symbol,date)

    LOB,messagefile=from_lobster.select_subset(LOB,messagefile,
                                  initial_time=initial_time,
                                  final_time=final_time)

    print('\n\nDATA CLEANING\n')

    man_mf=from_lobster.ManipulateMessageFile(
         LOB=LOB, 
         mf=messagefile,
         symbol=symbol,
         date=date,
         )     

    man_ob=from_lobster.ManipulateOrderBook(
        LOB=man_mf.LOB,symbol=symbol,date=date,
        ticksize=man_mf.ticksize,n_levels=man_mf.n_levels,
        volume_imbalance_upto_level=2, 
        list_of_n_states = [3,3],
        )
    man_ob.set_states(midprice_changes = np.array(man_mf.messagefile['sign_delta_mid'].values, dtype=np.int))

    data=from_lobster.DataToStore(man_ob,man_mf,time_origin=initial_time)

    sym = data.symbol
    d = data.date
    t_0 = data.initial_time
    t_1 = data.final_time

    assert sym==symbol
    assert date==d

    print('\nData is being stored in {}'.format(path_lobster_data))
    with open(path_lobster_data+'/{}/{}_{}_{}-{}'.format(sym,sym,d,t_0,t_1), 'wb') as outfile:
        pickle.dump(data,outfile)
        
    print('read_from_lobster.py: END OF FILE')

#    sys.stdout=saveout
#    fout.close()

    print('read_from_lobster.py: END OF FILE')

    return data 

def test():
    data = main()
    model=sd_hawkes_model.SDHawkes(
        number_of_event_types=data.number_of_event_types,  number_of_states = data.number_of_states,
        number_of_lob_levels=data.n_levels, volume_imbalance_upto_level = data.volume_enc.volume_imbalance_upto_level,
        list_of_n_states=data.state_enc.list_of_n_states, st1_deflationary=data.state_enc.st1_deflationary,
        st1_inflationary=data.state_enc.st1_inflationary, st1_stationary=data.state_enc.st1_stationary
    )
    model.get_input_data(data)
    model.calibrate_on_input_data(partial=False, parallel=False, skip_mle_estim=True,
            store_trans_prob=True, 
            store_dirichlet_param=True,
            dump_after_calibration=False, 
            verbose=True)
    yield model
    trans_prob = model.estimate_transition_probabilities(
            data.observed_events,
            data.observed_states,
            verbose=True)
    yield trans_prob
    dir_param = model.estimate_dirichlet_parameters(data.observed_volumes,data.observed_states, verbose=True)
    yield dir_param




if __name__=='__main__':
    main()

