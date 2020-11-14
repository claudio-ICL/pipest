#!/usr/bin/env python
# coding: utf-8
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

def measure_impact_prep(
    symbol="INTC",
    date="2019-01-23",
    time_window="41400-45000",
    liquidator_base_rate=0.150,
    liquidator_control=0.2,
    count=0
    ):    
    with open(path_impact+'/models/{}/{}_{}_{}/{}_sdhawkes_{}_{}_onesided_thesis'\
            .format(symbol, symbol, date, time_window, symbol, date, time_window), 'rb') as source:
        model=pickle.load(source)
    i=count
    name=str(model.name_of_model)+'_1s'+str(i)
    path=path_impact+'/models/{}/{}_{}_{}/'.format(symbol, symbol, date, time_window)
    while os.path.exists(path+name):
        i+=16 #this is the number of different cases when panmeasure is executed
        name=name.replace("_1s"+str(i-16), "_1s"+str(i))
    model.set_name_of_model(name)
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='I am measuring one-sided impact\n'
    message+="measure_impact_prep\n"
    message+='symbol={}, date={}, time_window={}, count={}'.format(symbol,date,time_window, count)
    print(message)
    path_readout=path_impact+'/models/{}/{}_{}_{}/'.format(symbol, symbol,date,time_window)\
            +name+'_prep_readout'.format(symbol)
    fout, saveout = redirect_stdout(direction='from', message=message, path=path_readout)
    initial_condition_times=np.array(model.data.observed_times[:5],copy=True)
    initial_condition_events=1+np.array(model.data.observed_events[:5],copy=True)
    initial_condition_states=np.array(model.data.observed_states[:5],copy=True)
    initial_condition_volumes=np.array(model.data.observed_volumes[:5,:],copy=True)
    initial_inventory=5.0
    time_start=float(initial_condition_times[-1])
    time_end=time_start+0.750*60*60
    model.setup_liquidator(initial_inventory=initial_inventory,
                           time_start=time_start,
                           liquidator_base_rate=liquidator_base_rate,
                           type_of_liquid='constant_intensity',#Poissonian liquidation ...
                           liquidator_control_type='fraction_of_inventory',# ...with constant order size to be under the assumptions of the explicit formula for the one-sided impact profile
                           liquidator_control=liquidator_control)
    model.simulate_liquidation(
        time_end,
        initial_condition_events=initial_condition_events,
        initial_condition_states=initial_condition_states,
        initial_condition_times=initial_condition_times,
        initial_condition_volumes=initial_condition_volumes,
        max_number_of_events=1*10**4,
        verbose=False,
        report_history_of_intensities = True,
        store_results=True
    )
    model.make_start_liquid_origin_of_times(delete_negative_times=True)
    model.dump(path=path)
    now=datetime.datetime.now()
    message='\nEnds on {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year, now.month, now.day, now.hour, now.minute)
    redirect_stdout(direction='to', message=message, fout=fout, saveout=saveout)
    print(message)

def measure_impact_core(
    symbol="INTC",
    date="2019-01-23",
    time_window="41400-45000",
    liquidator_base_rate=0.150,
    liquidator_control=0.2,
    t0=None, t1=None, quarter=None,
    count=0
    ):    
    name_of_model = '{}_sdhawkes_{}_{}_onesided_thesis'.format(symbol, date, time_window)
    i=count
    name=name_of_model+'_1s'+str(i)
    path=path_impact+'/models/{}/{}_{}_{}/'.format(symbol, symbol, date, time_window)
    with open(path+name, 'rb') as source:
        model=pickle.load(source)
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='measure_impact_core\n'
    message+='symbol={}, date={}, time_window={},  count={}'.format(symbol,date,time_window, count)
    path_readout=path_impact+'/models/{}/{}_{}_{}/'.format(symbol, symbol,date,time_window)\
            +name+'_core-{}_readout'.format(quarter)
    fout, saveout = redirect_stdout(direction='from', message=message, path=path_readout)
    if 'impact' not in model.liquidator.__dict__:
        model.create_impact_profile(delete_negative_times=False,
                                    produce_weakly_defl_pp=True,
                                    mle_estim=True)
    if t0==None:
        t0=0.0
    if t1==None:
        idx = len(model.liquidator.impact.times)-1
        t1=model.liquidator.impact.times[idx]
    if quarter!=None:
        t0 = t0 + (t1-t0)*max(0.0, min(4,quarter)-1.0)/4.0
        t1 = t0 + (t1-t0)*max(0.0, min(4,quarter))/4.0
    model.liquidator.impact.store_impact_profile(t0, t1, num_extra_eval_points = 3)
    model.dump(path=path)
    now=datetime.datetime.now()
    message='\nEnds on {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year, now.month, now.day, now.hour, now.minute)
    redirect_stdout(direction='to', message=message, fout=fout, saveout=saveout)
def measure_impact_conclude(
    symbol="INTC",
    date="2019-01-23",
    time_window="41400-45000",
    liquidator_base_rate=0.150,
    liquidator_control=0.2,
    count=0
    ):    
    name_of_model = '{}_sdhawkes_{}_{}_onesided_thesis'.format(symbol, date, time_window)
    i=count
    name=name_of_model+'_1s'+str(i)
    path=path_impact+'/models/{}/{}_{}_{}/'.format(symbol, symbol, date, time_window)
    with open(path+name, 'rb') as source:
        model=pickle.load(source)
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    message+='measure_impact_conclude\n'
    message+='symbol={}, date={}, time_window={},  count={}'.format(symbol,date,time_window, count)
    path_readout=path_impact+'/models/{}/{}_{}_{}/'.format(symbol, symbol,date,time_window)\
            +name+'_conclude_readout'.format(symbol)
    fout, saveout = redirect_stdout(direction='from', message=message, path=path_readout)
    model.store_price_trajectory(type_of_input='simulated', initial_price=model.data.mid_price.iloc[0,1],
                                 ticksize=model.data.ticksize)
    model.store_history_of_intensities()
    model.dump(path=path)
    now=datetime.datetime.now()
    message='\nEnds on {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year, now.month, now.day, now.hour, now.minute)
    redirect_stdout(direction='to', message=message, fout=fout, saveout=saveout)
    print(message)
