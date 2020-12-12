#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dfFear = pd.read_parquet(r'./data/Singularity_Eval_With_Locs_Slimmed_fearGhost_temp.parquet')

dfNormal = pd.read_parquet(r'./data/Singularity_Eval_With_Locs_Slimmed_normalOnly_temp.parquet')

dfNormalPlus = pd.read_parquet(r'./data/Singularity_Eval_With_Locs_Slimmed_normalPlus_temp.parquet')

dfPower = pd.read_parquet(r'./data/Singularity_Eval_With_Locs_Slimmed_powerPill_temp.parquet')

dfStand = pd.read_parquet(r'./data/Singularity_Eval_With_Locs_Slimmed_standard_temp.parquet')

dfCombo = pd.read_parquet(r'./data/Singularity_Eval_With_Locs_Slimmed_combo_df.parquet')


# In[4]:


Fearkey_states = dfFear[dfFear.key_state.eq(True)]
Normalkey_states = dfNormal[dfNormal.key_state.eq(True)]
NormalPluskey_states = dfNormalPlus[dfNormalPlus.key_state.eq(True)]
Powerkey_states = dfPower[dfPower.key_state.eq(True)]
Standkey_states = dfStand[dfStand.key_state.eq(True)]


# In[5]:


Fearkey_states_with_context = dfFear[dfFear.context_state.eq(True)]
Normalkey_states_with_context = dfNormal[dfNormal.context_state.eq(True)]
NormalPluskey_states_with_context = dfNormalPlus[dfNormalPlus.context_state.eq(True)]
Powerkey_states_with_context = dfPower[dfPower.context_state.eq(True)]
Standkey_states_with_context = dfStand[dfStand.context_state.eq(True)]


# In[11]:


print(dfFear.columns)


# In[23]:


print(dfCombo.columns)


# In[12]:


corr = dfFear.corr()
# create a mask to pass it to seaborn and only show half of the cells 
# corr between x and y is the same as the y and x
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

fig = plt.figure(figsize = (10, 5))

ax = sns.heatmap(corr, 
                 mask = mask, 
                 vmax = 0.3, 
                 square = True,  
                 cmap = "viridis")
ax.set_title("Heatmap of Fear Ghost");


# In[13]:


corr = dfNormal.corr()
# create a mask to pass it to seaborn and only show half of the cells 
# because corr between x and y is the same as the y and x

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

fig = plt.figure(figsize = (10, 5))

ax = sns.heatmap(corr, 
                 mask = mask, 
                 vmax = 0.3, 
                 square = True,  
                 cmap = "viridis")
ax.set_title("Heatmap of Normal Pills");


# In[15]:


corr = dfNormalPlus.corr()
# create a mask to pass it to seaborn and only show half of the cells 
# because corr between x and y is the same as the y and x

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

fig = plt.figure(figsize = (10, 5))

ax = sns.heatmap(corr, 
                 mask = mask, 
                 vmax = 0.3, 
                 square = True,  
                 cmap = "viridis")
ax.set_title("Heatmap of Normal Plus Pills");


# In[16]:


corr = dfPower.corr()
# create a mask to pass it to seaborn and only show half of the cells 
# because corr between x and y is the same as the y and x
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

fig = plt.figure(figsize = (10, 5))

ax = sns.heatmap(corr, 
                 mask = mask, 
                 vmax = 0.3, 
                 square = True,  
                 cmap = "viridis")
ax.set_title("Heatmap of Power Pills");


# In[17]:


corr = dfStand.corr()
# create a mask to pass it to seaborn and only show half of the cells 
# because corr between x and y is the same as the y and x
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

fig = plt.figure(figsize = (10, 5))

ax = sns.heatmap(corr, 
                 mask = mask, 
                 vmax = 0.3, 
                 square = True,  
                 cmap = "viridis")
ax.set_title("Heatmap of Standard");


# In[18]:


corr = dfCombo.corr()
# create a mask to pass it to seaborn and only show half of the cells 
# because corr between x and y is the same as the y and x
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

fig = plt.figure(figsize = (10, 5))

ax = sns.heatmap(corr, 
                 mask = mask, 
                 vmax = 0.3, 
                 square = True,  
                 cmap = "viridis")
ax.set_title("Heatmap of Combo");


# In[24]:


sns.lineplot(x = "action", y = "episode_reward", data = dfFear, color='r',linestyle='-')
sns.lineplot(x = "action", y = "episode_reward", data = dfNormal, color='g',linestyle='--')
sns.lineplot(x = "action", y = "episode_reward", data = dfPower, color='b',linestyle=':')
sns.lineplot(x = "action", y = "episode_reward", data = dfStand, color='k',linestyle='-.')
sns.lineplot(x = "action", y = "episode_reward", data = dfNormalPlus, color='y',linestyle='solid')


# In[25]:


sns.lineplot(x = "action", y = "total_reward", data = dfFear,label='Fear')
sns.lineplot(x = "action", y = "total_reward", data = dfNormal,label='Normal')
sns.lineplot(x = "action", y = "total_reward", data = dfPower,label='Power')
sns.lineplot(x = "action", y = "total_reward", data = dfStand,label='Standard')
sns.lineplot(x = "action", y = "total_reward", data = dfNormalPlus,label='NormalPlus')


# In[34]:


sns.lineplot(x = "action", y = "epoch_reward", data = dfFear,label='Fear')
sns.lineplot(x = "action", y = "epoch_reward", data = dfNormal,label='Normal')
sns.lineplot(x = "action", y = "epoch_reward", data = dfNormalPlus,label='NormalPlus')
sns.lineplot(x = "action", y = "epoch_reward", data = dfPower,label='Power')
sns.lineplot(x = "action", y = "epoch_reward", data = dfStand,label='Standard')


# In[35]:


sns.lineplot(x = "action", y = "reward", data = dfFear,label='Fear')
sns.lineplot(x = "action", y = "reward", data = dfNormal,label='Normal')
sns.lineplot(x = "action", y = "reward", data = dfNormalPlus,label='NormalPlus')
sns.lineplot(x = "action", y = "reward", data = dfPower,label='Power')
sns.lineplot(x = "action", y = "reward", data = dfStand,label='Standard')


# In[30]:


sns.lineplot(x = "state", y = "episode_reward", data = dfFear,label='Fear')
sns.lineplot(x = "state", y = "episode_reward", data = dfNormal,label='Normal')
sns.lineplot(x = "state", y = "episode_reward", data = dfNormalPlus,label='NormalPlus')
sns.lineplot(x = "state", y = "episode_reward", data = dfPower,label='Power')
sns.lineplot(x = "state", y = "episode_reward", data = dfStand,label='Standard')
sns.scatterplot(x='state', y='episode_reward', data = Fearkey_states)
sns.scatterplot(x='state', y='episode_reward', data = Normalkey_states)
sns.scatterplot(x='state', y='episode_reward', data = NormalPluskey_states)
sns.scatterplot(x='state', y='episode_reward', data = Powerkey_states)
sns.scatterplot(x='state', y='episode_reward', data = Standkey_states)


# In[28]:


sns.lineplot(x = "state", y = "total_reward", data = dfFear,label='Fear')
sns.lineplot(x = "state", y = "total_reward", data = dfNormal,label='Normal')
sns.lineplot(x = "state", y = "total_reward", data = dfPower,label='Power')
sns.lineplot(x = "state", y = "total_reward", data = dfStand,label='Standard')
sns.lineplot(x = "state", y = "total_reward", data = dfNormalPlus,label='NormalPlus')
sns.scatterplot(x='state', y='total_reward', data = Fearkey_states)
sns.scatterplot(x='state', y='total_reward', data = Normalkey_states)
sns.scatterplot(x='state', y='total_reward', data = NormalPluskey_states)
sns.scatterplot(x='state', y='total_reward', data = Powerkey_states)
sns.scatterplot(x='state', y='total_reward', data = Standkey_states)
sns.scatterplot(x='state', y='total_reward', data = Standkey_states)


# In[31]:


sns.lineplot(x = "state", y = "epoch_reward", data = dfFear,label='Fear')
sns.lineplot(x = "state", y = "epoch_reward", data = dfNormal,label='Normal')
sns.lineplot(x = "state", y = "epoch_reward", data = dfNormalPlus,label='NormalPlus')
sns.lineplot(x = "state", y = "epoch_reward", data = dfPower,label='Power')
sns.lineplot(x = "state", y = "epoch_reward", data = dfStand,label='Standard')
sns.scatterplot(x='state', y='epoch_reward', data = Fearkey_states)
sns.scatterplot(x='state', y='epoch_reward', data = Normalkey_states)
sns.scatterplot(x='state', y='epoch_reward', data = NormalPluskey_states)
sns.scatterplot(x='state', y='epoch_reward', data = Powerkey_states)
sns.scatterplot(x='state', y='epoch_reward', data = Standkey_states)


# In[36]:


sns.lineplot(x = "state", y = "reward", data = dfFear,label='Fear')
sns.lineplot(x = "state", y = "reward", data = dfNormal,label='Normal')
sns.lineplot(x = "state", y = "reward", data = dfNormalPlus,label='NormalPlus')
sns.lineplot(x = "state", y = "reward", data = dfPower,label='Power')
sns.lineplot(x = "state", y = "reward", data = dfStand,label='Standard')
sns.scatterplot(x='state', y='reward', data = Fearkey_states)
sns.scatterplot(x='state', y='reward', data = Normalkey_states)
sns.scatterplot(x='state', y='reward', data = NormalPluskey_states)
sns.scatterplot(x='state', y='reward', data = Powerkey_states)
sns.scatterplot(x='state', y='reward', data = Standkey_states)


# In[8]:


fig, ax = plt.subplots(1, 2,figsize = (16,6), dpi = 80)
plt.suptitle('Reward at the end of episodes')
sns.lineplot(x = "episode_step", y = "total_reward", data = dfFear,label='Fear',ax=ax[0])
sns.lineplot(x = "episode_step", y = "total_reward", data = dfNormal,label='NormalOnly', ax=ax[0])
sns.lineplot(x = "episode_step", y = "total_reward", data = dfNormalPlus,label='NormalPlus', ax=ax[0])
sns.lineplot(x = "episode_step", y = "total_reward", data = dfPower,label='Power', ax=ax[0])
sns.lineplot(x = "episode_step", y = "total_reward", data = dfStand,label='Standard', ax=ax[0])

sns.lineplot(x = "episode_step", y = "episode_reward", data = dfFear,label='Fear', ax=ax[1])
sns.lineplot(x = "episode_step", y = "episode_reward", data = dfNormal,label='NormalOnly', ax=ax[1])
sns.lineplot(x = "episode_step", y = "episode_reward", data = dfNormalPlus,label='NormalPlus', ax=ax[1])
sns.lineplot(x = "episode_step", y = "episode_reward", data = dfPower,label='Power', ax=ax[1])
sns.lineplot(x = "episode_step", y = "episode_reward", data = dfStand,label='Standard', ax=ax[1])


# In[37]:


sns.lineplot(x = "episode_step", y = "episode_reward", data = dfFear,label='Fear')
sns.lineplot(x = "episode_step", y = "episode_reward", data = dfNormal,label='Normal')
sns.lineplot(x = "episode_step", y = "episode_reward", data = dfNormalPlus,label='NormalPlus')
sns.lineplot(x = "episode_step", y = "episode_reward", data = dfPower,label='Power')
sns.lineplot(x = "episode_step", y = "episode_reward", data = dfStand,label='Standard')


# In[38]:


sns.lineplot(x = "episode_step", y = "epoch_reward", data = dfFear,label='Fear')
sns.lineplot(x = "episode_step", y = "epoch_reward", data = dfNormal,label='Normal')
sns.lineplot(x = "episode_step", y = "epoch_reward", data = dfNormalPlus,label='NormalPlus')
sns.lineplot(x = "episode_step", y = "epoch_reward", data = dfPower,label='Power')
sns.lineplot(x = "episode_step", y = "epoch_reward", data = dfStand,label='Standard')


# In[39]:


sns.lineplot(x = "episode_step", y = "reward", data = dfFear,label='Fear')
sns.lineplot(x = "episode_step", y = "reward", data = dfNormal,label='Normal')
sns.lineplot(x = "episode_step", y = "reward", data = dfNormalPlus,label='NormalPlus')
sns.lineplot(x = "episode_step", y = "reward", data = dfPower,label='Power')
sns.lineplot(x = "episode_step", y = "reward", data = dfStand,label='Standard')


# In[17]:


sns.lineplot(x = "episode_step", y = "total_reward", data = dfCombo,label='Combo')


# In[18]:


import stumpy
from matplotlib.patches import Rectangle
plt.rcParams["figure.figsize"] = [20, 6]  # width, height
plt.rcParams['xtick.direction'] = 'out'


# In[19]:


m = 640
fig, axs = plt.subplots(2)
axs[0].set_ylabel("Total Reward")
axs[0].plot(dfFear['total_reward'], alpha=0.5, linewidth=1)
axs[0].plot(dfFear['total_reward'].iloc[643:643+m])
axs[0].plot(dfFear['total_reward'].iloc[8724:8724+m])
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Total Reward")
axs[1].plot(dfFear['total_reward'].values[643:643+m], color='C1')
axs[1].plot(dfFear['total_reward'].values[8724:8724+m], color='C2')
plt.show()


# In[20]:


m = 640
fig, axs = plt.subplots(2)
axs[0].set_ylabel("Episode Reward")
axs[0].plot(dfFear['episode_reward'], alpha=0.5, linewidth=1)
axs[0].plot(dfFear['episode_reward'].iloc[643:643+m])
axs[0].plot(dfFear['episode_reward'].iloc[8724:8724+m])
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Episode Reward")
axs[1].plot(dfFear['episode_reward'].values[643:643+m], color='C1')
axs[1].plot(dfFear['episode_reward'].values[8724:8724+m], color='C2')
plt.show()


# In[22]:


m = 640
fig, axs = plt.subplots(2)
axs[0].set_ylabel("Episode Reward")
axs[0].plot(dfNormal['episode_reward'], alpha=0.5, linewidth=1)
axs[0].plot(dfNormal['episode_reward'].iloc[643:643+m])
axs[0].plot(dfNormal['episode_reward'].iloc[8724:8724+m])
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Episode Reward")
axs[1].plot(dfNormal['episode_reward'].values[643:643+m], color='C1')
axs[1].plot(dfNormal['episode_reward'].values[8724:8724+m], color='C2')
plt.show()


# In[23]:


m = 640
fig, axs = plt.subplots(2)
axs[0].set_ylabel("Episode Reward")
axs[0].plot(dfCombo['episode_reward'], alpha=0.5, linewidth=1)
axs[0].plot(dfCombo['episode_reward'].iloc[643:643+m])
axs[0].plot(dfCombo['episode_reward'].iloc[8724:8724+m])
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Episode Reward")
axs[1].plot(dfCombo['episode_reward'].values[643:643+m], color='C1')
axs[1].plot(dfCombo['episode_reward'].values[8724:8724+m], color='C2')
plt.show()


# In[24]:


m = 640
fig, axs = plt.subplots(2)
axs[0].set_ylabel("Episode Reward")
axs[0].plot(dfNormalPlus['episode_reward'], alpha=0.5, linewidth=1)
axs[0].plot(dfNormalPlus['episode_reward'].iloc[643:643+m])
axs[0].plot(dfNormalPlus['episode_reward'].iloc[8724:8724+m])
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Episode Reward")
axs[1].plot(dfNormalPlus['episode_reward'].values[643:643+m], color='C1')
axs[1].plot(dfNormalPlus['episode_reward'].values[8724:8724+m], color='C2')
plt.show()


# In[25]:


m = 640
fig, axs = plt.subplots(2)
axs[0].set_ylabel("Episode Reward")
axs[0].plot(dfStand['episode_reward'], alpha=0.5, linewidth=1)
axs[0].plot(dfStand['episode_reward'].iloc[643:643+m])
axs[0].plot(dfStand['episode_reward'].iloc[8724:8724+m])
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Episode Reward")
axs[1].plot(dfStand['episode_reward'].values[643:643+m], color='C1')
axs[1].plot(dfStand['episode_reward'].values[8724:8724+m], color='C2')
plt.show()


# In[26]:


m = 640
fig, axs = plt.subplots(2)
axs[0].set_ylabel("Episode Reward")
axs[0].plot(dfPower['episode_reward'], alpha=0.5, linewidth=1)
axs[0].plot(dfPower['episode_reward'].iloc[643:643+m])
axs[0].plot(dfPower['episode_reward'].iloc[8724:8724+m])
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Episode Reward")
axs[1].plot(dfPower['episode_reward'].values[643:643+m], color='C1')
axs[1].plot(dfPower['episode_reward'].values[8724:8724+m], color='C2')
plt.show()


# In[28]:


m = 640
mp = stumpy.stump(dfFear['episode_reward'], m)

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery')

axs[0].plot(dfFear['episode_reward'].values)
axs[0].set_ylabel('Episode Reward')
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile')
axs[1].axvline(x=643, linestyle="dashed")
axs[1].axvline(x=8724, linestyle="dashed")
axs[1].plot(mp[:, 0])
plt.show()


# In[29]:


m = 640
mp = stumpy.stump(dfNormal['episode_reward'], m)

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery')

axs[0].plot(dfNormal['episode_reward'].values)
axs[0].set_ylabel('Episode Reward')
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile')
axs[1].axvline(x=643, linestyle="dashed")
axs[1].axvline(x=8724, linestyle="dashed")
axs[1].plot(mp[:, 0])
plt.show()


# In[30]:


m = 640
mp = stumpy.stump(dfNormalPlus['episode_reward'], m)

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery')

axs[0].plot(dfNormalPlus['episode_reward'].values)
axs[0].set_ylabel('Episode Reward')
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile')
axs[1].axvline(x=643, linestyle="dashed")
axs[1].axvline(x=8724, linestyle="dashed")
axs[1].plot(mp[:, 0])
plt.show()


# In[31]:


m = 640
mp = stumpy.stump(dfPower['episode_reward'], m)

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery')

axs[0].plot(dfPower['episode_reward'].values)
axs[0].set_ylabel('Episode Reward')
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile')
axs[1].axvline(x=643, linestyle="dashed")
axs[1].axvline(x=8724, linestyle="dashed")
axs[1].plot(mp[:, 0])
plt.show()


# In[32]:


m = 640
mp = stumpy.stump(dfStand['episode_reward'], m)

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery')

axs[0].plot(dfStand['episode_reward'].values)
axs[0].set_ylabel('Episode Reward')
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile')
axs[1].axvline(x=643, linestyle="dashed")
axs[1].axvline(x=8724, linestyle="dashed")
axs[1].plot(mp[:, 0])
plt.show()


# In[33]:


m = 640
mp = stumpy.stump(dfCombo['episode_reward'], m)

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery')

axs[0].plot(dfCombo['episode_reward'].values)
axs[0].set_ylabel('Episode Reward')
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile')
axs[1].axvline(x=643, linestyle="dashed")
axs[1].axvline(x=8724, linestyle="dashed")
axs[1].plot(mp[:, 0])
plt.show()


# In[39]:


m = 640
mp = stumpy.stump(dfFear['episode_reward'], m)
true_P = mp[:, 0]


# In[40]:


def compare_approximation(true_P, approx_P):
    fig, ax = plt.subplots(gridspec_kw={'hspace': 0})
    ax.set_xlabel('Time')
    ax.axvline(x=643, linestyle="dashed")
    ax.axvline(x=8724, linestyle="dashed")
    ax.set_ylim((5, 28))
    ax.plot(approx_P, color='C1', label="Approximate Matrix Profile")
    ax.plot(true_P, label="True Matrix Profile")
    ax.legend()
    plt.show()


# In[41]:


approx = stumpy.scrump(dfFear['episode_reward'], m, percentage=0.01, pre_scrump=False)
approx.update()
approx_P = approx.P_


# In[42]:


seed = np.random.randint(100000)
np.random.seed(seed)
approx = stumpy.scrump(dfFear['episode_reward'], m, percentage=0.01, pre_scrump=False)


# In[43]:


compare_approximation(true_P, approx_P)


# In[44]:


m = 640
mp = stumpy.stump(dfNormal['episode_reward'], m)
true_P = mp[:, 0]

def compare_approximation(true_P, approx_P):
    fig, ax = plt.subplots(gridspec_kw={'hspace': 0})
    ax.set_xlabel('Time')
    ax.axvline(x=643, linestyle="dashed")
    ax.axvline(x=8724, linestyle="dashed")
    ax.set_ylim((5, 28))
    ax.plot(approx_P, color='C1', label="Approximate Matrix Profile")
    ax.plot(true_P, label="True Matrix Profile")
    ax.legend()
    plt.show()
    
approx = stumpy.scrump(dfNormal['episode_reward'], m, percentage=0.01, pre_scrump=False)
approx.update()
approx_P = approx.P_

seed = np.random.randint(100000)
np.random.seed(seed)
approx = stumpy.scrump(dfNormal['episode_reward'], m, percentage=0.01, pre_scrump=False)

compare_approximation(true_P, approx_P)


# In[52]:


sns.lineplot(x = "episode", y = "episode_reward", data = dfFear,label='Fear')
sns.lineplot(x = "episode", y = "episode_reward", data = dfNormal,label='Normal')
sns.lineplot(x = "episode", y = "episode_reward", data = dfNormalPlus,label='NormalPlus')
sns.lineplot(x = "episode", y = "episode_reward", data = dfPower,label='Power')
sns.lineplot(x = "episode", y = "episode_reward", data = dfStand,label='Standard')


# In[53]:


sns.lineplot(x = "episode", y = "total_reward", data = dfFear,label='Fear')
sns.lineplot(x = "episode", y = "total_reward", data = dfNormal,label='Normal')
sns.lineplot(x = "episode", y = "total_reward", data = dfNormalPlus,label='NormalPlus')
sns.lineplot(x = "episode", y = "total_reward", data = dfPower,label='Power')
sns.lineplot(x = "episode", y = "total_reward", data = dfStand,label='Standard')


# In[21]:


print(dfFear.dtypes)


# In[72]:


dfFear.head(6)


# In[78]:


dfFear["action"] = dfFear["action"].astype('float')
dfNormal["action"] = dfNormal["action"].astype('float')
dfNormalPlus["action"] = dfNormalPlus["action"].astype('float')
dfStand["action"] = dfStand["action"].astype('float')
dfPower["action"] = dfPower["action"].astype('float')


# In[ ]:


Index(['action', 'reward', 'episode_reward', 'epoch_reward', 'total_reward',
       'lives', 'end_of_episode', 'end_of_epoch', 'episode', 'episode_step',
       'epoch', 'epoch_step', 'state', 'mean_reward', 'to_pill_one',
       'to_pill_two', 'to_pill_three', 'to_pill_four', 'to_red_ghost',
       'to_pink_ghost', 'to_blue_ghost', 'to_orange_ghost', 'pacman_coord_x',
       'pacman_coord_y', 'red_ghost_coord_x', 'red_ghost_coord_y',
       'pink_ghost_coord_x', 'pink_ghost_coord_y', 'blue_ghost_coord_x',
       'blue_ghost_coord_y', 'orange_ghost_coord_x', 'orange_ghost_coord_y',
       'pacman_direction', 'red_ghost_direction', 'pink_ghost_direction',
       'blue_ghost_direction', 'orange_ghost_direction',
       'dark_blue_ghost1_coord_x', 'dark_blue_ghost1_coord_y',
       'dark_blue_ghost2_coord_x', 'dark_blue_ghost2_coord_y',
       'dark_blue_ghost3_coord_x', 'dark_blue_ghost3_coord_y',
       'dark_blue_ghost4_coord_x', 'dark_blue_ghost4_coord_y',
       'action 1 episode sum', 'action 1 total sum', 'action 2 episode sum',
       'action 2 total sum', 'action 3 episode sum', 'action 3 total sum',
       'action 4 episode sum', 'action 4 total sum', 'importance',
       'epoch_score', 'key_state', 'context_state', 'keyNum', 'beforeLifeLoss',
       'bigRewardNum', 'agent', 'to_pill_mean', 'to_top_pills_mean',
       'to_bottom_pills_mean', 'to_ghosts_mean', 'to_db1', 'to_db2', 'to_db3',
       'to_db4', 'diff_to_red', 'diff_to_orange', 'diff_to_blue',
       'diff_to_pink', 'diff_to_dbg1', 'diff_to_dbg2', 'diff_to_dbg3',
       'diff_to_dbg4', 'diff_to_pill1', 'diff_to_pill2', 'diff_to_pill3',
       'diff_to_pill4'],
      dtype='object')


# In[6]:


newFear=dfFear.drop(dfFear.iloc[:, 6:76].columns, axis = 1) 


# In[7]:


newNormal=dfNormal.drop(dfNormal.iloc[:, 5:81].columns, axis = 1) 


# In[8]:


newNormalPlus=dfNormalPlus.drop(dfNormalPlus.iloc[:, 5:81].columns, axis = 1)


# In[9]:


newStand=dfStand.drop(dfStand.iloc[:, 5:81].columns, axis = 1) 


# In[10]:


newPower=dfPower.drop(dfPower.iloc[:, 5:81].columns, axis = 1) 


# In[ ]:


action = dfFear({"action" : [1:, 2, 3]})
[dfFear.action.{"A" : [1, 2, 3]}]
{"A" : [1, 2, 3]}


# In[ ]:


newdF=dfFear[['action', 'episode reward', 'epoch reward', 'total reward', 'lives','epoch_score']].rename({'action_name':'Fear_action_name', 'episode reward':'Fear_episode reward', 'epoch reward':'Fear_epoch reward', 'total reward':'Fear_total reward', 'lives':'Fear_lives','epoch_score':'Fear_epoch_score'}, axis=1)


# In[17]:


newStand.head()


# In[15]:


cols = newStand.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = newStand[cols] 

from pandas.plotting import andrews_curves
plt.figure(figsize=(10,5))
andrews_curves(df,'action',colormap='rainbow')
plt.show()


# In[18]:


cols = newPower.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = newPower[cols] 

from pandas.plotting import andrews_curves
plt.figure(figsize=(10,5))
andrews_curves(df,'action',colormap='rainbow')
plt.show()


# In[16]:


from pandas.plotting import andrews_curves
plt.figure(figsize=(10,5))
andrews_curves(newFear,'action',colormap='rainbow')
plt.show()


# In[16]:


from pandas.plotting import andrews_curves
plt.figure(figsize=(10,5))
andrews_curves(newNormal,'action',colormap='rainbow')
plt.show()


# In[66]:


from pandas.plotting import andrews_curves
plt.figure(figsize=(10,5))
andrews_curves(newNormalPlus,'action',colormap='rainbow')
plt.show()


# In[67]:


from pandas.plotting import andrews_curves
plt.figure(figsize=(10,5))
andrews_curves(newStand,'action',colormap='rainbow')
plt.show()


# In[68]:


from pandas.plotting import andrews_curves
plt.figure(figsize=(10,5))
andrews_curves(newPower,'action',colormap='rainbow')
plt.show()

