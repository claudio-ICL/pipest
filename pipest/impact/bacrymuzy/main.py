#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import glob
path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<4):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    print("path_pipest not found. Instead: {}".format(path_pipest))
    raise ValueError("path_pipest not found.")
path_models=path_pipest+'/models'    
path_impact=path_pipest+'/impact'    
path_sdhawkes=path_pipest+'/sdhawkes'
path_modelling = path_sdhawkes+'/modelling'
path_resources = path_sdhawkes+'/resources'
path_lobster=path_pipest+'/lobster'
path_lobster_data=path_lobster+'/data'
path_lobster_pyscripts=path_lobster+'/py_scripts'
path_tests = path_pipest+'/tests'
path_saved_tests = path_tests+'/saved_tests'
path_perfmeas=path_tests+'/performance_measurements'
sys.path.append(path_modelling)
sys.path.append(path_resources)
sys.path.append(path_perfmeas)
import numpy as np
import pandas as pd
import pickle
import datetime
import time
import datetime
import timeit
import model as sd_hawkes_model
import computation

def redirect_stdout(direction= 'from', # or 'to'
                    message= '',
                    path='',
                    fout=None,saveout=None):
    if direction=='from':
        print(message)
        print("stdout is being redirected to {}".format(path))
        saveout=sys.stdout
        fout=open(path,'w')
        sys.stdout = fout
        print(message)
        return fout, saveout
    elif direction=='to':
        print(message)
        fout.close()
        sys.stdout=saveout
        print(message)
    else:
        print("WARNINNG: redirect_stdout failed! direction={} not recognised".format(direction))
        print(message)        
def read(
    symbol="INTC",
    date="2019-01-23",
    time_window="41400-45000",
    simulate=False,
    ):    
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='I am reading from saved models\n'
    message+='symbol={}, date={}, time_window={}'.format(symbol,date,time_window)
    path_readout=path_impact+'/models/{}/{}_{}_{}_readout'.format(symbol, symbol, date, time_window)
    fout, saveout = redirect_stdout(direction='from', message=message, path=path_readout)
    try:
        with open(path_models+"/{}/{}_{}/{}_sdhawkes_{}_{}"
                  .format(symbol, symbol, date, symbol, date, time_window), 'rb') as source:
            calmodel=pickle.load(source)
    except FileNotFoundError:
        try:
            with open(path_models+"/{}/{}_sdhawkes_{}_{}"
                      .format(symbol, symbol, date, time_window), 'rb') as source:
                calmodel=pickle.load(source)
        except FileNotFoundError:
            print("File not found")
            raise FileNotFoundError
    model=sd_hawkes_model.SDHawkes(
        number_of_event_types=calmodel.number_of_event_types,
        list_of_n_states=calmodel.state_enc.list_of_n_states,
        number_of_lob_levels=calmodel.n_levels,
        volume_imbalance_upto_level=\
        calmodel.volume_enc.volume_imbalance_upto_level
    )
    model.get_configuration(calmodel)
    model.create_uq()
    target=computation.avg_rates(model.data.number_of_event_types, 
                                model.data.observed_times,
                                model.data.observed_events, partial=False)
    model.uncertainty_quantification.adjust_baserates(
        target,
        adj_coef=5.0e-2,
        num_iter=15, 
        max_number_of_events=30000
    )
    model.reduce_price_volatility(reduction_coef=0.7)
    model.create_goodness_of_fit(type_of_input='empirical')
    if simulate:    
        time_start=0.0
        time_end=time_start+0.15*60*60
        model.simulate(time_start, time_end,
                       max_number_of_events=50000,
                       add_initial_cond=True,
                       store_results=True, report_full_volumes=False)
        model.store_price_trajectory(type_of_input='simulated', initial_price=model.data.mid_price.iloc[0,1],
                                     ticksize=model.data.ticksize)
    model.store_price_trajectory(type_of_input='empirical', initial_price=model.data.mid_price.iloc[0,1],
                                 ticksize=model.data.ticksize)
    try:
        os.mkdir(path_impact+'/models/{}/{}_{}_{}'.format(symbol, symbol, date, time_window))
    except FileExistsError:
        pass
    model.dump(path=path_impact+'/models/{}/{}_{}_{}'.format(symbol, symbol, date, time_window))
    now=datetime.datetime.now()
    message='\nEnds on {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year, now.month, now.day, now.hour, now.minute)
    redirect_stdout(direction='to', message=message, fout=fout, saveout=saveout)


def measure_impact(
    symbol="INTC",
    date="2019-01-23",
    time_window="41400-45000",
    liquidator_base_rate=0.150,
    type_of_liquid = 'with_the_market', #constant_intensity or with_the_market or against_the_market
    liquidator_control_type='fraction_of_bid_side', # fraction_of_inventory or fraction_of_bid_side
    liquidator_control=0.2,
    count=0
    ):    
    with open(path_impact+'/models/{}/{}_{}_{}/{}_sdhawkes_{}_{}'\
            .format(symbol, symbol, date, time_window, symbol, date, time_window), 'rb') as source:
        model=pickle.load(source)
    i=count
    name=str(model.name_of_model)+'_bm'+str(i)
    path=path_impact+'/models/{}/{}_{}_{}/'.format(symbol, symbol, date, time_window)
    while os.path.exists(path+name):
        i+=6 #this is the number of different cases when panmeasure is executed
        name=name.replace("_bm"+str(i-6), "_bm"+str(i))
    model.set_name_of_model(name)
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='I am measuring bacry-muzy impact\n'
    message+='symbol={}, date={}, time_window={}'.format(symbol,date,time_window)
    path_readout=path_impact+'/models/{}/{}_{}_{}/'.format(symbol, symbol,date,time_window)\
            +name+'_readout'.format(symbol)
    fout, saveout = redirect_stdout(direction='from', message=message, path=path_readout)
    initial_condition_times=np.array(model.data.observed_times[:10000],copy=True)
    initial_condition_events=1+np.array(model.data.observed_events[:10000],copy=True)
    initial_condition_states=np.array(model.data.observed_states[:10000],copy=True)
    initial_condition_volumes=np.array(model.data.observed_volumes[:10000,:],copy=True)
#    initial_condition_times=np.array(model.simulated_times,copy=True)
#    initial_condition_events=1+np.array(model.simulated_events,copy=True)
#    initial_condition_states=np.array(model.simulated_states,copy=True)
#    initial_condition_volumes=np.array(model.simulated_volume,copy=True)
    initial_inventory=10.0
    time_start=float(initial_condition_times[-1])
    time_end=time_start+1.50*60*60
    model.setup_liquidator(initial_inventory=initial_inventory,
                           time_start=time_start,
                           liquidator_base_rate=liquidator_base_rate,
                           type_of_liquid=type_of_liquid,
                           liquidator_control_type=liquidator_control_type,
                           liquidator_control=liquidator_control)
    model.simulate_liquidation(
        time_end,
        initial_condition_events=initial_condition_events,
        initial_condition_states=initial_condition_states,
        initial_condition_times=initial_condition_times,
        initial_condition_volumes=initial_condition_volumes,
        max_number_of_events=2*10**5,
        verbose=False,
        report_history_of_intensities = False,
        store_results=True
    )
    model.make_start_liquid_origin_of_times(delete_negative_times=False)
    model.create_impact_profile(delete_negative_times=False,
                                produce_weakly_defl_pp=False,)
    model.liquidator.impact.store_bm_impact()
    model.store_price_trajectory(type_of_input='simulated', initial_price=model.data.mid_price.iloc[0,1],
                                 ticksize=model.data.ticksize)
    model.store_history_of_intensities()
    model.dump(path=path)
    now=datetime.datetime.now()
    message='\nEnds on {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year, now.month, now.day, now.hour, now.minute)
    redirect_stdout(direction='to', message=message, fout=fout, saveout=saveout)

def panmeasure(
    symbol="INTC",
    date="2019-01-23",
    time_window="41400-45000",
    liquidator_base_rate=0.150,
    liquidator_control=0.2
    ):    
    count=0
    for type_of_liquid in ['constant_intensity', 'with_the_market', 'against_the_market']:
        for liquidator_control_type in ['fraction_of_inventory', 'fraction_of_bid_side']:
            array_index=int(os.environ['PBS_ARRAY_INDEX'])
            if count==array_index:
                measure_impact(symbol, date, time_window,
                        liquidator_base_rate,
                        type_of_liquid,
                        liquidator_control_type,
                        liquidator_control,
			count
			)
            count+=1


def collect_results(
        symbol='INTC', date='2019-01-23', time_window="41400-45000"):
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    with open(path_impact+'/models/{}/{}_{}_{}/{}_sdhawkes_{}_{}'\
            .format(symbol, symbol, date, time_window, symbol, date, time_window), 'rb') as source:
        model=pickle.load(source)
    now=datetime.datetime.now()
    print(message)
    model.create_archive()
    pathlist = glob.glob(path_impact+'/models/{}/{}_{}_{}/*_bm?'.format(symbol, symbol, date, time_window))
    pathlist.append(glob.glob(path_impact+'/models/{}/{}_{}_{}/*_bm??'.format(symbol, symbol, date, time_window)))
    for path in pathlist:
        print(path)
        with open(path, 'rb') as source:
            bm=pickle.load(source)
        model.stack_to_archive(bm.name_of_model, name_of_item=bm.name_of_model)
        model.stack_to_archive(bm.liquidator, name_of_item='liquidator', idx=bm.name_of_model)
        model.stack_to_archive(bm.simulated_times, name_of_item='simulated_times', idx=bm.name_of_model)
        model.stack_to_archive(bm.simulated_events, name_of_item='simulated_events', idx=bm.name_of_model)
        model.stack_to_archive(bm.simulated_states, name_of_item='simulated_states', idx=bm.name_of_model)
        model.stack_to_archive(bm.simulated_intensities, name_of_item='simulated_intensities', idx=bm.name_of_model)
    model.dump(path=path_impact+'/models/{}/{}_{}_{}'.format(symbol, symbol, date, time_window))
    now=datetime.datetime.now()
    message='\nEnds on {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year, now.month, now.day, now.hour, now.minute)
    print(message)


def main():
    print("\n\npython {} {} {} {} {}".format(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
    symbol=str(sys.argv[1])
    date=str(sys.argv[2])
    time_window=str(sys.argv[3])
    action=str(sys.argv[4])
    if action=='-r' or action=='--read':
        read(symbol,date,time_window)
    elif action=='-m' or action=='--measure':
        liquidator_base_rate=float(sys.argv[5])
        type_of_liquid=str(sys.argv[6])
        liquidator_control_type=str(sys.argv[7])
        liquidator_control=float(sys.argv[8])
        measure_impact(symbol, date, time_window,
                liquidator_base_rate,
                type_of_liquid,
                liquidator_control_type,
                liquidator_control)
    elif action=='-pm' or action=='--panmeasure':
        liquidator_base_rate=float(sys.argv[5])
        liquidator_control=float(sys.argv[6])
        panmeasure(symbol, date, time_window,
                liquidator_base_rate,
                liquidator_control)
    elif action=='-c' or action=='--collect':
        collect_results(symbol, date, time_window)

    else:
        print("action: {}".format(action))
        print("Error: action not recognised")
        raise ValueError("Action not recognised")
    
if __name__=="__main__":
    main()
