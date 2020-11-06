#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
path_pipest = os.path.abspath('./')
n=0
while (not os.path.basename(path_pipest)=='pipest') and (n<4):
    path_pipest=os.path.dirname(path_pipest)
    n+=1 
if not os.path.basename(path_pipest)=='pipest':
    print("path_pipest not found. Instead: {}".format(path_pipest))
    raise ValueError("path_pipest not found.")
path_models=path_pipest+'/models'    
path_sdhawkes=path_pipest+'/sdhawkes'
path_modelling = path_sdhawkes+'/modelling'
path_resources = path_sdhawkes+'/resources'
path_impact=path_pipest+'/impact'
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
import glob
import pickle
import datetime
import time
import datetime
import timeit
import model as sd_hawkes_model
import computation

symbol="INTC"
date="2019-01-23"
time_window="37800-41400"

with open(path_impact+"/models/{}/{}_{}_{}/old/{}_sdhawkes_{}_{}_onesided_thesis_1s0"
          .format(symbol, symbol, date, time_window, symbol, date, time_window), 'rb') as source:
    model=pickle.load(source)
print("\n\nPoint-by-point evaluations\n\n")
eval_times = [model.liquidator.impact.times[100*k] for k in range(5)]
print("eval_times: {}".format(eval_times))
#imp_list = [model.liquidator.impact.evaluate_impact_profile(t) for t in eval_times]
#print("results: {}".format(imp_list))

print("\n\nCall to compite_imp_profile_history\n\n")
res = model.liquidator.impact.compute_imp_profile_history(eval_times[0], eval_times[3], num_extra_eval_points = 1)
print("res: {}".format(res))

print("\n\nCall to store_impact_profile\n\n")
model.liquidator.impact.store_impact_profile(eval_times[0], eval_times[3], num_extra_eval_points = 1)
print("model.liquidator.impact.impact_profile: {}".format(model.liquidator.impact.impact_profile))



