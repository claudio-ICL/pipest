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
from measure_impact import *

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
        try:
            fout.close()
        except:
            pass
        try:
            sys.stdout=saveout
            print(message)
        except:
            pass
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
    path_readout=path_impact+'/models/{}/{}_{}_{}_onesided_thesis_readout'.format(symbol, symbol, date, time_window)
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
    model.set_base_rates(np.array([0.14008215, 0.14451349, 6.4689, 6.0295], dtype=np.float))
#    target=computation.avg_rates(model.data.number_of_event_types, 
#                                model.data.observed_times,
#                                model.data.observed_events, partial=False)
#    model.uncertainty_quantification.adjust_baserates(
#        target,
#        adj_coef=5.0e-2,
#        num_iter=15, 
#        max_number_of_events=30000
#    )
    model.reduce_price_volatility(reduction_coef=0.7)
    #Remarkably, compared to the same function in bacrymuzy/main.py, here we are NOT enforcing price symmetry
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
    model.set_name_of_model('{}_sdhawkes_{}_{}_onesided_thesis'.format(symbol, date, time_window))
    model.dump(path=path_impact+'/models/{}/{}_{}_{}'.format(symbol, symbol, date, time_window))
    now=datetime.datetime.now()
    message='\nEnds on {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year, now.month, now.day, now.hour, now.minute)
    redirect_stdout(direction='to', message=message, fout=fout, saveout=saveout)


def measure_impact(
    symbol="INTC",
    date="2019-01-23",
    time_window="41400-45000",
    liquidator_base_rate=0.150,
    liquidator_control=0.2,
    count=0, 
    phase="prep",
    quarter=None
    ):    
    if phase=="prep":
        measure_impact_prep(symbol, date, time_window, liquidator_base_rate, liquidator_control, count)
    elif phase=="core":
        measure_impact_core(symbol, date, time_window, liquidator_base_rate, liquidator_control, quarter=quarter, count=count)
    elif phase=="conclude":
        measure_impact_conclude(symbol, date, time_window, liquidator_base_rate, liquidator_control, count)
    else:
        print("WARNING: measure_impact: phase not recognised")
        print("Given phase: {}".format(pahse))

def panmeasure(
        symbol='INTC', date='2019-01-23', time_window="41400-45000", phase="prep", quarter=None):
    count=0
    for br in [0.01, 0.05, 0.1, 0.15]:
        for c in [0.05, 0.1, 0.2, 0.5]:
            if count==int(os.environ["PBS_ARRAY_INDEX"]):
                measure_impact(
                     symbol=symbol,
                     date=date,
                     time_window=time_window,
                     liquidator_base_rate=br,
                     liquidator_control=c,
                     count=count, 
                     phase=phase, quarter=quarter)
            count+=1


def collect_results(
        symbol='INTC', date='2019-01-23', time_window="41400-45000"):
    now=datetime.datetime.now()
    message='\ndate of run: {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year,now.month,now.day, now.hour, now.minute)
    with open(path_impact+'/models/{}/{}_{}_{}/{}_sdhawkes_{}_{}_onesided_thesis'\
            .format(symbol, symbol, date, time_window, symbol, date, time_window), 'rb') as source:
        model=pickle.load(source)
    now=datetime.datetime.now()
    print(message)
    model.set_name_of_model = model.name_of_model + "_20201106"
    model.create_archive()
    for path in glob.glob(path_impact+'/models/{}/{}_{}_{}/*_1s*'.format(symbol, symbol, date, time_window)):
        try:
            with open(path, 'rb') as source:
                m1s=pickle.load(source)
            model.stack_to_archive(m1s.name_of_model, name_of_item=m1s.name_of_model)
            model.stack_to_archive(m1s.liquidator, name_of_item='liquidator', idx=m1s.name_of_model)
            model.stack_to_archive(m1s.simulated_times, name_of_item='simulated_times', idx=m1s.name_of_model)
            model.stack_to_archive(m1s.simulated_events, name_of_item='simulated_events', idx=m1s.name_of_model)
            model.stack_to_archive(m1s.simulated_states, name_of_item='simulated_states', idx=m1s.name_of_model)
            try:
                model.stack_to_archive(m1s.simulated_intensities, name_of_item='simulated_intensities', idx=m1s.name_of_model)
            except:
                pass
        except:
           pass
    model.set_name_of_model = model.name_of_model + "_20201106"
    model.dump(path=path_impact+'/models/{}/{}_{}_{}'.format(symbol, symbol, date, time_window))
    now=datetime.datetime.now()
    message='\nEnds on {}-{:02d}-{:02d} at {}:{:02d}\n'.format(now.year, now.month, now.day, now.hour, now.minute)
    print(message)


def main():
    print("\npython {} {} {} {} {}".format(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
    symbol=str(sys.argv[1])
    date=str(sys.argv[2])
    time_window=str(sys.argv[3])
    action=str(sys.argv[4])
    if action=='-r' or action=='--read':
        read(symbol,date,time_window, simulate=True)
    elif action=='-m' or action=='--measure':
        liquidator_base_rate=float(sys.argv[5])
        liquidator_control=float(sys.argv[6])
        try:
            phase=str(sys.argv[7])
            try:
                quarter=int(sys.argv[8])
            except:
                quarter = None
            measure_impact(symbol, date, time_window,
                liquidator_base_rate,
                liquidator_control, 
                phase=phase, quarter=quarter)
        except:
            for phase in ["prep", "core", "conclude"]:
                measure_impact(symbol, date, time_window,
                    liquidator_base_rate,
                    liquidator_control, 
                    phase=phase)
    elif action=='-pm' or action=='--panmeasure':
        phase=sys.argv[5]
        try:
            quarter=int(sys.argv[6])
        except IndexError:
            quarter=None
        panmeasure(symbol, date, time_window, phase, quarter=quarter)
    elif action=='-c' or action=='--collect':
        collect_results(symbol, date, time_window)
    else:
        print("action: {}".format(action))
        print("Error: action not recognised")
        raise ValueError
    
if __name__=="__main__":
    main()
