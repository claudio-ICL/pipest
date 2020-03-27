#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import sys
sys.path.append('./resources/')
sys.path.append('./modelling/')
sys.path.append('/home/claudio/Documents/analysis_lobster/py_scripts/')


# In[2]:


import pickle
import numpy as np
from scipy.ndimage.interpolation import shift as array_shift
import pandas as pd
import bisect
import copy




# In[3]:


import model as sd_hawkes_model
import lob_model
import computation
import simulation
import goodness_of_fit
import mle_estimation as mle_estim





n_states=[3,5]
n_events = 4  # number of event types, $d_e$
n_levels = 2
upto_level = 2
time_start=np.random.uniform()
time_end=time_start+0.25*60*60
model = sd_hawkes_model.sd_hawkes(
    number_of_event_types=n_events,
    number_of_lob_levels=n_levels,
    list_of_n_states=n_states,
    volume_imbalance_upto_level=upto_level)
tot_n_states = model.state_enc.tot_n_states


# In[6]:


# The transition probabilities $\phi$
phis = model.state_enc.generate_random_transition_prob(n_events=n_events).astype(np.float)

# The base rates $\nu$
nus = 0.001510*np.random.randint(low=1,high=6,size=n_events)

# The impact coefficients $\alpha$
alphas = np.random.uniform(0.0004,0.1875,size=(n_events, tot_n_states, n_events)).astype(np.float)


# The decay coefficients $\beta$
betas = np.random.uniform(1.12725,1.90,size=(n_events, tot_n_states, n_events)).astype(np.float)


# The Dirichlet parameters $\gamma$
gammas = np.random.uniform(low=1.5, high = 5.6,size=(tot_n_states,2*n_levels))


# In[7]:


# symbol='INTC'
# date='2019-01-15'
# with open('./models/{}_{}_h_model'.format(symbol,date),'rb') as source:
#     h_model=pickle.load(source)
# with open('./models/{}_{}_orderBook_model'.format(symbol,date),'rb') as source:
#     ob_model=pickle.load(source)
# with open('./models/{}_{}_messageFile_model'.format(symbol,date),'rb') as source:
#     mf_model=pickle.load(source)    


# In[8]:


# phis = lob_model.correct_null_transition_prob(h_model.transition_probabilities)
# nus=0.1*h_model.base_rates
# alphas=0.5*h_model.impact_coefficients
# betas=h_model.decay_coefficients
# gammas=h_model.dirichlet_param


# In[9]:
print("I am setting the model's parametes")

model.set_hawkes_parameters(nus,alphas,betas)
model.set_dirichlet_parameters(gammas)
model.set_transition_probabilities(phis)
model.enforce_symmetry_in_transition_probabilities()


# In[10]:
print("\nI am starting the simulation without liquidator\n")

pre_times, pre_events, pre_states,pre_volumes = model.simulate(
    time_start, time_end,max_number_of_events=10000,add_initial_cond=True,
    store_results=True,report_full_volumes=False)
time_end_antefact=np.array(pre_times[-1],copy=True)


# In[11]:


print(pre_times.shape)
print(pre_events.shape)
print(pre_volumes.shape)


# In[12]:


pre_intens_history= model.compute_history_of_intensities(
    pre_times,
    pre_events,
    pre_states)
pre_tilda_intens_history= model.compute_history_of_tilda_intensities(
    pre_times,
    pre_events,
    pre_states)


# In[13]:


low_limit=0
up_limit=7200
idx=np.logical_and(pre_times<up_limit,pre_times>low_limit)
idx_history=np.logical_and(pre_intens_history[:,0]<up_limit,pre_intens_history[:,0]>low_limit)
# fig=plot_tools.plot_events_and_intensities(
#     1+pre_events[idx],pre_times[idx],pre_intens_history[idx_history,:],
#     save_fig=False
# )


# In[14]:


states_2D=model.produce_2Dstates(pre_states)
# plot_tools.plot_events_and_states(pre_events+1,pre_times,pre_intens_history,states_2D,
#                                 plot=True,
#                                 save_fig=False,
#                                   name='events_and_states_traject'
#                                )


# In[15]:


model.create_goodness_of_fit()
# model.goodness_of_fit.qq_plot_residuals(index_of_first_event_type=1)


# In[16]:


model.goodness_of_fit.ks_test_on_residuals()


# In[17]:


model.goodness_of_fit.ad_test_on_residuals()


# In[18]:


# print(claudio)


# In[19]:


# model.volume_enc.rejection_sampling.is_target_equal_to_proposal


# In[20]:


# model.volume_enc.prob_volimb_constraint


# In[21]:


# %timeit simulation.sample_volumes(2,model.volume_enc.rejection_sampling.proposal_dir_param,model.volume_enc.rejection_sampling.difference_of_dir_params, model.volume_enc.rejection_sampling.inverse_bound,model.volume_enc.rejection_sampling.is_target_equal_to_proposal,model.state_enc.num_of_st2, model.volume_enc.volimb_limits, upto_lim = 5)


# In[22]:


# %timeit simulation.sample_volumes(5,model.volume_enc.rejection_sampling.proposal_dir_param,model.volume_enc.rejection_sampling.difference_of_dir_params, model.volume_enc.rejection_sampling.inverse_bound,model.volume_enc.rejection_sampling.is_target_equal_to_proposal,model.state_enc.num_of_st2, model.volume_enc.volimb_limits, upto_lim = 5)


# In[23]:


# model.volume_enc.rejection_sampling.inverse_bound


# In[24]:


# print(claudio)


# # Liquidation section
# 

# In[25]:

print('Start of liquidation section')

initial_condition_times=np.array(pre_times,copy=True)
initial_condition_events=np.array(pre_events,copy=True)
initial_condition_states=np.array(pre_states,copy=True)
initial_condition_volumes=np.array(pre_volumes,copy=True)


# In[26]:


initial_inventory=10
liquidator_base_rate=np.amin(nus)
liquidation_strategy='fraction_of_bid_side' # constant_rate or fraction_of_bid_side
liquidator_control=0.2
time_start=np.array(time_end_antefact+3.0,dtype=np.float,copy=True)
time_end=np.array(time_start+2*60*60)


# In[27]:


model.introduce_liquidator(initial_inventory=initial_inventory,
                                 time_start=time_start,
                                 liquidator_base_rate=liquidator_base_rate,
                                 liquidator_control=liquidator_control)
initial_condition_events+=1
time_liquidation_starts=model.liquidator.time_start
print(model.liquidator.initial_inventory)
print(model.liquidator.base_rate)
print(model.liquidator.control_type)


# In[28]:


liquidator_base_rate=0.10
model.configure_liquidator_param(
    initial_inventory=initial_inventory,
    liquidator_base_rate=liquidator_base_rate,
    type_of_liquid='constant_intensity',
    liquidator_control=liquidator_control)
print(model.liquidator.initial_inventory)
print(model.liquidator.base_rate)
print(model.liquidator.control_type)
print(np.linalg.norm(model.liquidator.imp_coef))


# In[ ]:


times, events, states, volumes, inventory, _=model.simulate_liquidation(
    time_end,
    initial_condition_events=initial_condition_events,
    initial_condition_states=initial_condition_states,
    initial_condition_times=initial_condition_times,
    initial_condition_volumes=initial_condition_volumes,
    verbose=True,
    report_history_of_intensities = False,
    store_results=True
)
print('\n')
print('times.shape={}'.format(times.shape))
print('events.shape={}'.format(events.shape))
print('states.shape={}'.format(states.shape))
print('inventory.shape={}'.format(inventory.shape))


# In[ ]:


start_liquidation=time_liquidation_starts
end_liquidation=model.liquidator.termination_time
intens_history= model.compute_history_of_intensities(
    times,
    events,
    states,
    start_time_zero_event=start_liquidation,
    end_time_zero_event=end_liquidation,
    density_of_eval_points=10000,
)
print('intens_history.shape={}'.format(intens_history.shape))
idx=intens_history[:,1]>0
print(np.sum(idx))


# # In[ ]:


# plot_start_index=bisect.bisect_left(times,time_start)-5
# plot_end_index=np.argmin(inventory)+30
# # pd.Series(inventory[plot_start_index:plot_end_index]).plot()


# # In[ ]:


# # plot_tools.plot_liquidation(times,events,inventory,intens_history,
# #                             plot_start_index,plot_end_index,
# #                             save_fig=False)


# # In[ ]:


# df=model.state_enc.translate_labels(states)
# price=np.sum(np.abs(df['st_1'].values))+100+0.85*np.cumsum(df['st_1'].values-1)
# price=0.01*price


# # In[ ]:


# plot_tools.plot_liquidation_with_price(times,events,inventory,intens_history,price,
#                             plot_start_index,plot_end_index,
#                             save_fig=False,path='/home/claudio/Desktop/imperialMF_phdDay/pictures/',
#                             name='plot_liquidation_n'          )


# # In[ ]:


# # plot_tools.plot_liquidator_only(times,events,inventory,intens_history,price,
# #                             plot_start_index,plot_end_index,
# #                             save_fig=False,path='/home/claudio/Desktop/imperialMF_phdDay/pictures/',
# #                             name='plot_liquidator_4' )


# # In[ ]:


# # print(claudio)


# # # Impact profile a\` la Bacry-Muzy

# # In[ ]:


# bm_intensity, bm_profile=model.compute_bm_impact_profile(times,events,states,
#                                   inventory,
#                                   start_liquidation_time = time_liquidation_starts,
#                                   density_of_eval_points=10000)


# # In[ ]:


# # plot_tools.plot_bm_impact_profile(bm_profile,
# #                        bm_intensity,
# #                        time_start=time_liquidation_starts,
# #                        time_end=times[np.argmin(inventory)]+100,
# #                        save_fig=False,
# #                        path='/home/claudio/Desktop/imperialMF_phdDay/pictures/',
# #                        name='bm_impact_profile_3'          )


# # In[ ]:


# plot_tools.plot_bm_impact_profile_full_picture(bm_profile,
#                        times,events,inventory,intens_history,price,
#                        bm_intensity,
#                        time_start=time_liquidation_starts,
#                        time_end=times[np.argmin(inventory)]+100,
#                        save_fig=False,
#                        path='/home/claudio/Desktop/imperialMF_phdDay/pictures/',
#                        name='bm_impact_profile_full_3'                         )


# # In[ ]:


# print(claudio)


# # In[ ]:


# # model.set_transition_probabilities(phis_hat)
# # model.set_hawkes_parameters(nus_hat, alphas_hat, betas_hat)
# # model.set_dirichlet_parameters(gammas_hat)


# # # One-sided impact profile

# # In[ ]:


# model.produce_impact_profile(num_init_guesses=4, maxiter=20)


# # In[ ]:


# model.impact.reduced_weakly_defl_pp.create_goodness_of_fit()


# # In[ ]:


# model.impact.reduced_weakly_defl_pp.goodness_of_fit.qq_plot_residuals(
#     tot_event_types=1, title_per_event_type=0, fig_suptitle='QQ plot residuals of reduced_weakly_defl_pp')


# # In[ ]:


# times=np.array(model.sampled_times,copy=True)
# states=np.array(model.sampled_states,copy=True)
# labelled_times = np.array(model.labelled_times,copy=True)
# count = np.array(model.count, copy=True)
# len_states=len(states)
# eval_time=np.random.uniform(low=model.liquidator.time_start+10,
#                             high=model.liquidator.termination_time,
#                            )
# print('eval_time={}'.format(eval_time))
# nu_zero=model.base_rates[0]

# t=np.random.uniform(low = 0.1, high = times[-100])
# P_T_geq_t=impact_profile.evaluate_anti_cdf_execution_horizon(t,nu_zero,model.liquidator.num_orders)
# print(t,P_T_geq_t)


# # In[ ]:


# run_time=-time.time()
# compensator=model.impact.evaluate_compensator_of_weakly_defl_pp(eval_time)
# expectation=model.impact.reduced_weakly_defl_pp.compute_expectation(eval_time)
# run_time+=time.time()
# print('compesator={}, expectation={}'.format(compensator,expectation))
# print('run_time={}'.format(run_time))


# # In[ ]:


# run_time=-time.time()
# res=model.impact.evaluate_impact_profile(eval_time)
# run_time+=time.time()
# print('run_time={}'.format(run_time))
# print(res)


# # In[ ]:


# T=np.random.uniform(low=model.sampled_times[10], high=model.sampled_times[150])
# print('T={}'.format(T))
# run_time=-time.time()
# res=model.impact.compute_imp_profile_history(T,num_extra_eval_points=5)
# run_time+=time.time()
# print('run_time={}'.format(run_time))
# print(res)


# # In[ ]:





# # In[ ]:


# plot_tools.plot_impact_profile(
#     res,
#     times,events,inventory,intens_history,price,
#     time_start=model.liquidator.time_start,
#     time_end=model.liquidator.termination_time,
#     save_fig=False,
#     path='/home/claudio/Desktop/imperialMF_phdDay/pictures/',
#     name='onesided_impact_profile'
# )


# # In[ ]:





# # In[ ]:





# # In[ ]:


# print(claudio)


# # In[ ]:


# p_prof=pd.Series(res[:,1],index=res[:,0])
# p_prof.plot()


# # In[ ]:





# # In[ ]:





# # # Goodness of fit

# # In[ ]:


# model.create_goodness_of_fit()


# # In[ ]:


# model.goodness_of_fit.qq_plot_residuals()


# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:




