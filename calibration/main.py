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


number_of_event_types=4
#Optional prameters for "read_lobster"
first_read_fromLOBSTER=True
dump_after_reading=False
add_level_to_messagefile=True
#Optional parameters for "calibrate"
n_guesses=2
maxiter=3
store_after_calibration=True







def read_lobster():
    now=datetime.datetime.now()
    print('\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute))
    print('I am reading from lobster')
    print('symbol={}, date={}, time_window={}'.format(symbol,date,time_window))
    path_readout=path_mdata+'_readout'
    print('stdout is being redirected to '+path_readout)
    saveout=sys.stdout
    fout=open(path_readout,'w')
    sys.stdout=fout
    print('\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute))
    print('I am reading from lobster')
    print('symbol={}, date={}, time_window={}'.format(symbol,date,time_window))
    

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
    assert number_of_event_types==data.number_of_event_types
    if not np.isclose(t_0,initial_time,rtol=1.0,atol=1.0):
        message="data.initial_time={}, whereas global initial_time={}".format(t_0,initial_time)
        print(message)
        raise ValueError(message)
    if not np.isclose(t_1,final_time,rtol=1.0,atol=1.0):
        message="data.final_time={}, whereas global final_time={}".format(t_1,final_time)
        print(message)
        raise ValueError(message)

    print('\nData is being stored in {}'.format(path_mdata))
    with open(path_mdata, 'wb') as outfile:
        pickle.dump(data,outfile)
    fout.close()
    sys.stdout=saveout
    print('\nData is being stored in {}'.format(path_mdata))
    

def calibrate(n_guesses=4,maxiter=50):
    now=datetime.datetime.now()
    print('\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute))
    print('I am calibrating on lobster')
    print('symbol={}, date={}, time_window={}'.format(symbol,date,time_window))
    
    array_index=int(os.environ['PBS_ARRAY_INDEX'])
    
    with open(path_mdata,'rb') as source:
        data=pickle.load(source)
    assert symbol==data.symbol    
    assert date==data.date
  
    model=sd_hawkes_model.sd_hawkes(
        number_of_event_types=data.number_of_event_types,  number_of_states = data.number_of_states,
        number_of_lob_levels=data.n_levels,volume_imbalance_upto_level = data.volume_enc.volume_imbalance_upto_level,
        list_of_n_states=data.state_enc.list_of_n_states, st1_deflationary=data.state_enc.st1_deflationary,
        st1_inflationary=data.state_enc.st1_inflationary, st1_stationary=data.state_enc.st1_stationary
    )

    model.get_input_data(data)
    

    for event_type in range(model.number_of_event_types):
        if (event_type == array_index):
            name_of_model_partial=name_of_model+'_partial{}'.format(event_type)
            path_mpartial=path_mmodel+'_partial{}'.format(event_type)
            path_readout=path_mpartial+'_readout'
            print('stdout is being redirected to '+path_readout)
            saveout=sys.stdout
            fout=open(path_readout,'w')
            sys.stdout=fout
            print('\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'
                  .format(now.year,now.month,now.day, now.hour, now.minute))
            print('symbol={}, date={}'.format(symbol,date))
            print('\n\nINSTANTIATE SD_HAWKES MODEL AND CALIBRATE\n')
            model.calibrate_on_input_data_partial(
                event_type, maximum_number_of_iterations=maxiter, number_of_random_guesses=n_guesses,
                store_after_calibration=store_after_calibration, name_of_model=name_of_model_partial,verbose=True)
            n=datetime.datetime.now()
            print('\nCalibration of event_type {} terminates on {}-{:02d}-{:02d} at {}:{:02d}\n'
                  .format(event_type, n.year, n.month, n.day, n.hour, n.minute))
            fout.close()
            sys.stdout=saveout
            print('\nCalibration of event_type {} terminates on {}-{:02d}-{:02d} at {}:{:02d}\n'
                  .format(event_type, n.year, n.month, n.day, n.hour, n.minute))
            
def merge_from_partial():
    now=datetime.datetime.now()
    print('\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute))
    print('I am merging from partial models')
    print('symbol={}, date={}, time_window={}'.format(symbol,date,time_window))
    saveout=sys.stdout
    path_readout=path_mmodel+'_readout'
    print('stdout is being redirected to '+path_readout)
    fout=open(path_readout,'w')
    sys.sdtout=fout
    print('\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute))
    print('I am merging from partial models')
    print('symbol={}, date={}, time_window={}'.format(symbol,date,time_window))
    
    list_of_partial_names=[name_of_model+'_partial{}'.format(e) for e in range(number_of_event_types)]
    partial_models=[]
    for mn in list_of_partial_names:
        with open(path_models+'/{}/'.format(symbol)+mn,'rb') as source:
            partial_models.append(pickle.load(source))
    with open(path_mdata,'rb') as source:
        data=pickle.load(source)
    assert data.number_of_event_types==number_of_event_types
    
    MODEL=sd_hawkes_model.sd_hawkes(
        number_of_event_types=data.number_of_event_types, number_of_states=data.number_of_states,
        number_of_lob_levels=data.n_levels,volume_imbalance_upto_level = data.volume_enc.volume_imbalance_upto_level,
        list_of_n_states=data.state_enc.list_of_n_states, st1_deflationary=data.state_enc.st1_deflationary,
        st1_inflationary=data.state_enc.st1_inflationary, st1_stationary=data.state_enc.st1_stationary
    )

    MODEL.initialise_from_partial(partial_models,store_after_merging=True,name_of_model=name_of_model)  

    n=datetime.datetime.now()
    print('\n Merging has been completed  on {}-{:02d}-{:02d} at {}:{:02d}'.format(n.year,n.month,n.day,n.hour,n.minute))
    sys.stdout=saveout
    fout.close()
    print('\n Merging has been completed  on {}-{:02d}-{:02d} at {}:{:02d}'.format(n.year,n.month,n.day,n.hour,n.minute)) 
        

    


def main():  
    global symbol
    symbol=str(sys.argv[1])
    global date
    date=str(sys.argv[2])
    global initial_time
    initial_time=float(sys.argv[3])
    global final_time
    final_time=float(sys.argv[4])
    global action
    action=str(sys.argv[5])
    global time_window
    time_window=str('{}-{}'.format(int(initial_time),int(final_time)))
    print("$python {} {} {} {} {} {}\n".format(sys.argv[0],symbol,date,initial_time,final_time,action))
    global path_mdata
    path_mdata=path_lobster_data+'/{}/{}_{}_{}'.format(symbol,symbol,date,time_window)
    global path_mmodel
    path_mmodel=path_models+'/{}/{}_{}_{}'.format(symbol,symbol,date,time_window)
    global name_of_model
    name_of_model=symbol+'_sdhawkes_'+date+'_'+time_window
    print("name of model: {}".format(name_of_model))
    if action=='r' or action=='read':
        read_lobster()
    elif action=='c' or action=='calibrate':
        calibrate(n_guesses,maxiter)
    elif action=='m' or action=='merge':
        merge_from_partial()
    print("\nmain.py end of file")    
        
        
        

        
        
if __name__=='__main__':
    main()              
