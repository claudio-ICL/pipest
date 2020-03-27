#!/usr/bin/env python
# coding: utf-8

import os
path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<4):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    raise ValueError("path_pipest not found. Instead: {}".format(path_pipest))
path_models=path_pipest+'/models'    
path_sdhawkes=path_pipest+'/sdhawkes_powerlaw'
path_lobster=path_pipest+'/lobster_for_sdhawkes'
path_lobster_data=path_lobster+'/data'
path_lobster_pyscripts=path_lobster+'/py_scripts'


import numpy as np
import pandas as pd
import pickle
import datetime
import time
import datetime

import sys
sys.path.append(path_lobster_pyscripts+'/')
sys.path.append(path_sdhawkes+'/modelling/')
sys.path.append(path_sdhawkes+'/resources/')

import model as sd_hawkes_model
import lob_model
import prepare_from_lobster as from_lobster


time_to_sleep = 60

# array_index=int(os.environ['PBS_ARRAY_INDEX'])


symbol='INTC'
initial_time=float(11*60*60)
final_time=float(11.06*60*60)
time_window=str('{}-{}'.format(int(initial_time),int(final_time)))
first_read_fromLOBSTER=True
dump_after_reading=False
add_level_to_messagefile=True

date='2019-01-02'

def main():

    now=datetime.datetime.now()

    print('\ndate of run: {}_{}_{} at {}:{}\n'.format(now.year,now.month,now.day, now.hour, now.minute))

    print('I am reading from lobster')
    print('symbol={}, date={}, time_window={}'.format(symbol,date,time_window))

    saveout=sys.stdout
    path_readout=path_models+'/{}/{}_{}_{}_readout'.format(symbol,symbol,date,time_window)
#     if array_index == 4:
    print('Output is being redirected to '+path_readout)
    fout=open(path_readout,'w')
    sys.stdout=fout
    print('\ndate of run: {}_{}_{} at {}:{}\n'
          .format(now.year,now.month,now.day, now.hour, now.minute))
    print('symbol={}, date={}'.format(symbol,date))

    if (first_read_fromLOBSTER):
        LOB,messagefile=from_lobster.read_from_LOBSTER(symbol,date,
                                          dump_after_reading=dump_after_reading,
                                          add_level_to_messagefile=add_level_to_messagefile)
    else:
        LOB,messagefile=from_lobster.load_from_pickleFiles(symbol,date)

    LOB,messagefile=from_lobster.select_subset(LOB,messagefile,
                                  initial_time=initial_time,
                                  final_time=final_time)


    print('\n\nDATA CLEANING\n')


    aggregate_time_stamp=True
    eventTypes_to_aggregate=[1,2,3,4,5]
    eventTypes_to_drop_with_nonunique_time=[]
    eventTypes_to_drop_after_aggregation=[3]
    only_4_events=False
    separate_directions=True
    separate_13_events=False
    separate_31_events=False
    separate_41_events=True
    separate_34_events=True
    separate_43_events=True
    equiparate_45_events_with_same_time_stamp=True
    drop_all_type3_events_with_nonunique_time=False
    drop_5_events_with_same_time_stamp_as_4=True
    drop_5_events_after_aggregation=True
    tolerance_when_dropping=1.0e-7
    add_hawkes_marks=True
    clear_same_time_stamp=True
    num_iter=4




    man_mf=from_lobster.ManipulateMessageFile(
         LOB,messagefile,
         symbol=symbol,
         date=date,
         aggregate_time_stamp=aggregate_time_stamp,
         eventTypes_to_aggregate=eventTypes_to_aggregate,  
         only_4_events=only_4_events,
         separate_directions=separate_directions,
         separate_13_events=separate_13_events,
         separate_31_events=separate_31_events,
         separate_41_events=separate_41_events,
         separate_34_events=separate_34_events,
         separate_43_events=separate_43_events,
         equiparate_45_events_with_same_time_stamp=
         equiparate_45_events_with_same_time_stamp,
         eventTypes_to_drop_with_nonunique_time=eventTypes_to_drop_with_nonunique_time,
         eventTypes_to_drop_after_aggregation=eventTypes_to_drop_after_aggregation,
         drop_all_type3_events_with_nonunique_time=drop_all_type3_events_with_nonunique_time,  
         drop_5_events_with_same_time_stamp_as_4=drop_5_events_with_same_time_stamp_as_4,
         drop_5_events_after_aggregation=drop_5_events_after_aggregation,
         tolerance_when_dropping=tolerance_when_dropping,
         clear_same_time_stamp=clear_same_time_stamp,
         add_hawkes_marks=add_hawkes_marks,
         num_iter=num_iter
    )



    man_ob=from_lobster.ManipulateOrderBook(
        man_mf.LOB_sdhawkes,symbol=symbol,date=date,
        ticksize=man_mf.ticksize,n_levels=man_mf.n_levels,volume_imbalance_upto_level=2)


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


    print('\n\nINSTANTIATE SD_HAWKES MODEL AND CALIBRATE\n')    

    model=sd_hawkes_model.sd_hawkes(
        number_of_event_types=data.number_of_event_types,  number_of_states = data.number_of_states,
        number_of_lob_levels=data.n_levels,volume_imbalance_upto_level = data.volume_enc.volume_imbalance_upto_level,
        list_of_n_states=data.state_enc.list_of_n_states, st1_deflationary=data.state_enc.st1_deflationary,
        st1_inflationary=data.state_enc.st1_inflationary, st1_stationary=data.state_enc.st1_stationary
    )

    model.get_input_data(data)

    list_of_partial_names=[]
    partial_models=[]


    for event_type in range(model.number_of_event_types):
        p_partial=path_models+'/{}/{}_{}_{}_partial{}_readout'.format(symbol,symbol,date,time_window,event_type)
        try:
            fout.close()
        except:
            pass
#         if array_index == 4:
        fout=open(path_readout,'a')    
        sys.stdout=fout 
        print('\nOutput is being redirected to '+p_partial)
        fout.close()
#         if (event_type == array_index):
#             print('\nOutput is being redirected to '+p_partial)
        fout=open(p_partial,'w')
        sys.stdout=fout
        print('\ndate of run: {}_{}_{} at {}:{}\n'
              .format(now.year,now.month,now.day, now.hour, now.minute))
        print('symbol={}, date={}'.format(symbol,date))
        model.calibrate_on_input_data_partial(
            event_type, maximum_number_of_iterations=4, number_of_random_guesses=2, store_after_calibration=True)
        n=datetime.datetime.now()
        print('\nCalibration of event_type {} terminates on {}_{}_{} at {}:{}\n'
              .format(event_type, n.year, n.month, n.day, n.hour, n.minute))
        fout.close()
        sys.stdout=saveout
        print('\nCalibration of event_type {} terminates on {}_{}_{} at {}:{}\n'
              .format(event_type, n.year, n.month, n.day, n.hour, n.minute))
#         if array_index == 4:
        n_model='{}_sdhawkes_{}_{}-{}_partial{}'.format(
                data.symbol,data.date,data.initial_time, data.final_time,event_type)
        list_of_partial_names.append(n_model)


#     if array_index==4:
    try:
        fout.close()
    except:
        pass
    fout=open(path_readout,'a')    
    sys.stdout=fout    

    print('\nlist_of_partial_names:\n{}'.format(list_of_partial_names))

    all_branches_terminated = False
    branches_terminated=np.zeros(model.number_of_event_types,dtype=np.bool)
    while not all_branches_terminated:
        event_type=0
        for mn in list_of_partial_names:
            if mn in os.listdir(path_models+'/{}/'.format(symbol)):
                branches_terminated[event_type]=True
            else:
                n=datetime.datetime.now()
                print('On {}_{}_{} at {}:{}, calibration of event_type {} has not finished yet'
                      .format(n.year, n.month,n.day, n.hour, n.minute, event_type,))
                time.sleep(time_to_sleep)
            event_type+=1
        all_branches_terminated = np.all(branches_terminated)        


    for model_name in list_of_partial_names:
        with open(path_models+'/{}/'.format(symbol)+model_name, 'rb') as source:
            partial_models.append(pickle.load(source))

    MODEL=sd_hawkes_model.sd_hawkes(
        number_of_event_types=data.number_of_event_types, number_of_states=data.number_of_states,
        number_of_lob_levels=data.n_levels,volume_imbalance_upto_level = data.volume_enc.volume_imbalance_upto_level,
        list_of_n_states=data.state_enc.list_of_n_states, st1_deflationary=data.state_enc.st1_deflationary,
        st1_inflationary=data.state_enc.st1_inflationary, st1_stationary=data.state_enc.st1_stationary
    )

    MODEL.initialise_from_partial(partial_models)  


    print('\nread_and_calibrate.py: END OF FILE')

    sys.stdout=saveout
    fout.close()

    print('\nread_and_calibrate.py: END OF FILE')
        
if __name__=='__main__':
    main()              
