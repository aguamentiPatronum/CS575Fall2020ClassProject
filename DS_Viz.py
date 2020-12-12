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


dfFear = pd.read_parquet('Singularity_EvaluationsSlimmed_FearGhostsEvalDB.parquet')
dfFear = dfFear.drop(columns=dfFear.columns[0])

dfNormal = pd.read_parquet('Singularity_EvaluationsSlimmed_NormalPillsOnlyEvalDB.parquet')
dfNormal = dfNormal.drop(columns=dfNormal.columns[0])

dfPower = pd.read_parquet('Singularity_EvaluationsSlimmed_PowerPillsOnlyEvalDB.parquet')
dfPower = dfPower.drop(columns=dfPower.columns[0])

dfStand = pd.read_parquet('Singularity_EvaluationsSlimmed_StandardEval2DB.parquet')
dfStand = dfStand.drop(columns=dfStand.columns[0])


# In[4]:


Fearkey_states = dfFear[dfFear.key_state.eq(True)]
Normalkey_states = dfNormal[dfNormal.key_state.eq(True)]
Powerkey_states = dfPower[dfPower.key_state.eq(True)]
Standkey_states = dfStand[dfStand.key_state.eq(True)]


# In[5]:


Fearkey_states_with_context = dfFear[dfFear.context_state.eq(True)]
Normalkey_states_with_context = dfNormal[dfNormal.context_state.eq(True)]
Powerkey_states_with_context = dfPower[dfPower.context_state.eq(True)]
Standkey_states_with_context = dfStand[dfStand.context_state.eq(True)]


# In[10]:


print(dfFear.columns,dfNormal.columns,dfPower.columns,dfStand.columns)


# In[30]:


newdF=dfFear[['action_name', 'episode reward', 'epoch reward', 'total reward', 'lives','epoch_score']].rename({'action_name':'Fear_action_name', 'episode reward':'Fear_episode reward', 'epoch reward':'Fear_epoch reward', 'total reward':'Fear_total reward', 'lives':'Fear_lives','epoch_score':'Fear_epoch_score'}, axis=1)
newdN=dfNormal[['action_name', 'episode reward', 'epoch reward', 'total reward', 'lives','epoch_score']].rename({'action_name':'Normal_action_name', 'episode reward':'Normal_episode reward', 'epoch reward':'Normal_epoch reward', 'total reward':'Normal_total reward', 'lives':'Normal_lives','epoch_score':'Normal_epoch_score'}, axis=1)
newdP=dfPower[['action_name', 'episode reward', 'epoch reward', 'total reward', 'lives','epoch_score']].rename({'action_name':'Power_action_name', 'episode reward':'Power_episode reward', 'epoch reward':'Power_epoch reward', 'total reward':'Power_total reward', 'lives':'Power_lives','epoch_score':'Power_epoch_score'}, axis=1)
newdS=dfStand[['action_name', 'episode reward', 'epoch reward', 'total reward', 'lives','epoch_score']].rename({'action_name':'Stand_action_name', 'episode reward':'Stand_episode reward', 'epoch reward':'Stand_epoch reward', 'total reward':'Stand_total reward', 'lives':'Stand_lives','epoch_score':'Stand_epoch_score'}, axis=1)

newdf=pd.concat([newdF,newdN,newdP,newdS], axis=1)
newdf.head(10)


# In[57]:


# plot to see the distribution of the points of each category
x = dfFear["total reward"]
y = dfFear["action_name"]

ax = sns.stripplot(x, y)
ax.set_title("Distrubution of FearGhosts");


# In[59]:


# plot to see the distribution of the points of each category
x = dfFear["total reward"]
y = dfFear["action"]

ax = sns.stripplot(x, y)
ax.set_title("Distrubution of FearGhosts");


# In[42]:


# plot to see the distribution of the points of each category
x = dfNormal["total reward"]
y = dfNormal["action_name"]

ax = sns.stripplot(x, y)
ax.set_title("Distrubution of FearGhosts");


# In[17]:


# plot to see the distribution of the points of each category
x = dfNormal["action_name"]
y = dfNormal["total reward"]

ax = sns.stripplot(x, y)
ax.set_title("Distrubution of Normal");


# In[18]:


# plot to see the distribution of the points of each category
x = dfPower["action_name"]
y = dfPower["total reward"]

ax = sns.stripplot(x, y)
ax.set_title("Distrubution of Power Pill");


# In[19]:


# plot to see the distribution of the points of each category
x = dfStand["action_name"]
y = dfStand["total reward"]

ax = sns.stripplot(x, y)
ax.set_title("Distrubution of Standard");


# In[6]:


sns.stripplot(x='action_name', y='total reward',hue='episode reward',data=dfFear, jitter=True)
plt.xlabel('Actions')
plt.ylabel('Total reward')
plt.title('Distrubution of FearGhosts')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
sns.stripplot(x='action_name', y='total reward',hue='episode reward',data=dfNormal, jitter=True)
plt.xlabel('Actions')
plt.ylabel('Total reward')
plt.title("Distrubution of NormalPills");
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
sns.stripplot(x='action_name', y='total reward',hue='episode reward',data=dfPower, jitter=True)
plt.xlabel('Actions')
plt.ylabel('Total reward')
plt.title("Distrubution of PowerPills");
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
sns.stripplot(x='action_name', y='total reward',hue='episode reward',data=dfStand, jitter=True)
plt.xlabel('Actions')
plt.ylabel('Total reward')
plt.title("Distrubution of Standard");
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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


# In[11]:


#Categorical estimate plots 
#to show episode reward as a function of 3 categorical factors like live, action and total reward
g = sns.catplot(x="total reward", y="episode reward", hue="lives", col="action_name",
                capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=dfFear)
g.despine(left=True)


# In[12]:


#Categorical estimate plots
#to show episode reward as a function of 3 categorical factors like live, action and total reward
gN = sns.catplot(x="total reward", y="episode reward", hue="lives", col="action_name",
                capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=dfNormal)
gN.despine(left=True)


# In[13]:


#Categorical estimate plots
#to show episode reward as a function of 3 categorical factors like live, action and total reward
gP = sns.catplot(x="total reward", y="episode reward", hue="lives", col="action_name",
                capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=dfPower)
g.despine(left=True)


# In[14]:


#Categorical estimate plots
#to show episode reward as a function of 3 categorical factors like live, action and total reward
gS = sns.catplot(x="total reward", y="episode reward", hue="lives", col="action_name",
                capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=dfStand)
gS.despine(left=True)


# In[31]:


# Useful for: timeseries analysis.
# compare the a series again itself but with some lags.
# plots that graphically summarize the strength of a relationship with an observation 
# in a time series with observations at prior time steps.

fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfFear["total reward"], ax = ax1, lags = 50)
plot_pacf(dfFear["total reward"], ax = ax2, lags = 15);

fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfFear["episode reward"], ax = ax1, lags = 50)
plot_pacf(dfFear["episode reward"], ax = ax2, lags = 15);

fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfFear["epoch reward"], ax = ax1, lags = 50)
plot_pacf(dfFear["epoch reward"], ax = ax2, lags = 15);


# In[32]:


fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfNormal["total reward"], ax = ax1, lags = 50)
plot_pacf(dfNormal["total reward"], ax = ax2, lags = 15);

fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfNormal["episode reward"], ax = ax1, lags = 50)
plot_pacf(dfNormal["episode reward"], ax = ax2, lags = 15);

fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfNormal["epoch reward"], ax = ax1, lags = 50)
plot_pacf(dfNormal["epoch reward"], ax = ax2, lags = 15);


# In[33]:


fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfPower["total reward"], ax = ax1, lags = 50)
plot_pacf(dfPower["total reward"], ax = ax2, lags = 15);

fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfPower["episode reward"], ax = ax1, lags = 50)
plot_pacf(dfPower["episode reward"], ax = ax2, lags = 15);

fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfPower["epoch reward"], ax = ax1, lags = 50)
plot_pacf(dfPower["epoch reward"], ax = ax2, lags = 15);


# In[34]:


fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfStand["total reward"], ax = ax1, lags = 50)
plot_pacf(dfStand["total reward"], ax = ax2, lags = 15);

fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfStand["episode reward"], ax = ax1, lags = 50)
plot_pacf(dfStand["episode reward"], ax = ax2, lags = 15);

fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfStand["epoch reward"], ax = ax1, lags = 50)
plot_pacf(dfStand["epoch reward"], ax = ax2, lags = 15);


# In[13]:


sns.violinplot(x = "action_name", 
               y = "total reward", 
               data = dfFear, 
               scale = 'width', 
               inner = 'quartile'
              )

# get the current figure
ax = plt.gca()
# get the xticks to iterate over
xticks = ax.get_xticks()

# iterate over every xtick and add a vertical line
# to separate different classes
for tick in xticks:
    ax.vlines(tick + 0.5, 0, np.max(dfFear["total reward"]), color = "grey", alpha = .1)
    
# rotate the x and y ticks
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add x and y label
ax.set_xlabel("Action", fontsize = 14)
ax.set_ylabel("Total Reward", fontsize = 14)

# set title
ax.set_title("Violinplot of FearGhosts", fontsize = 14);


# In[17]:


sns.violinplot(x = "action_name", 
               y = "total reward", 
               data = dfNormal, 
               scale = 'width', 
               inner = 'quartile'
              )

# get the current figure
ax = plt.gca()
# get the xticks to iterate over
xticks = ax.get_xticks()

# iterate over every xtick and add a vertical line
# to separate different classes
for tick in xticks:
    ax.vlines(tick + 0.5, 0, np.max(dfNormal["total reward"]), color = "grey", alpha = .1)
    
# rotate the x and y ticks
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add x and y label
ax.set_xlabel("Action", fontsize = 14)
ax.set_ylabel("Total Reward", fontsize = 14)

# set title
ax.set_title("Violinplot of Normal", fontsize = 14);


# In[18]:


sns.violinplot(x = "action_name", 
               y = "total reward", 
               data = dfPower, 
               scale = 'width', 
               inner = 'quartile'
              )

# get the current figure
ax = plt.gca()
# get the xticks to iterate over
xticks = ax.get_xticks()

# iterate over every xtick and add a vertical line
# to separate different classes
for tick in xticks:
    ax.vlines(tick + 0.5, 0, np.max(dfPower["total reward"]), color = "grey", alpha = .1)
    
# rotate the x and y ticks
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add x and y label
ax.set_xlabel("Action", fontsize = 14)
ax.set_ylabel("Total Reward", fontsize = 14)

# set title
ax.set_title("Violinplot of Power Pill", fontsize = 14);


# In[19]:


sns.violinplot(x = "action_name", 
               y = "total reward", 
               data = dfStand, 
               scale = 'width', 
               inner = 'quartile'
              )

# get the current figure
ax = plt.gca()
# get the xticks to iterate over
xticks = ax.get_xticks()

# iterate over every xtick and add a vertical line
# to separate different classes
for tick in xticks:
    ax.vlines(tick + 0.5, 0, np.max(dfStand["total reward"]), color = "grey", alpha = .1)
    
# rotate the x and y ticks
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add x and y label
ax.set_xlabel("Action", fontsize = 14)
ax.set_ylabel("Total Reward", fontsize = 14)

# set title
ax.set_title("Violinplot of Standard", fontsize = 14);


# In[6]:


sns.violinplot(x = "action_name", 
               y = "episode reward", 
               data = dfFear, 
               scale = 'width', 
               inner = 'quartile'
              )

# get the current figure
ax = plt.gca()
# get the xticks to iterate over
xticks = ax.get_xticks()

# iterate over every xtick and add a vertical line
# to separate different classes
for tick in xticks:
    ax.vlines(tick + 0.5, 0, np.max(dfFear["episode reward"]), color = "grey", alpha = .1)
    
# rotate the x and y ticks
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add x and y label
ax.set_xlabel("Action", fontsize = 14)
ax.set_ylabel("episode Reward", fontsize = 14)

# set title
ax.set_title("Violinplot of FearGhosts", fontsize = 14);


# In[7]:


sns.violinplot(x = "action_name", 
               y = "episode reward", 
               data = dfNormal, 
               scale = 'width', 
               inner = 'quartile'
              )

# get the current figure
ax = plt.gca()
# get the xticks to iterate over
xticks = ax.get_xticks()

# iterate over every xtick and add a vertical line
# to separate different classes
for tick in xticks:
    ax.vlines(tick + 0.5, 0, np.max(dfNormal["episode reward"]), color = "grey", alpha = .1)
    
# rotate the x and y ticks
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add x and y label
ax.set_xlabel("Action", fontsize = 14)
ax.set_ylabel("episode Reward", fontsize = 14)

# set title
ax.set_title("Violinplot of Normal", fontsize = 14);


# In[8]:


sns.violinplot(x = "action_name", 
               y = "episode reward", 
               data = dfPower, 
               scale = 'width', 
               inner = 'quartile'
              )

# get the current figure
ax = plt.gca()
# get the xticks to iterate over
xticks = ax.get_xticks()

# iterate over every xtick and add a vertical line
# to separate different classes
for tick in xticks:
    ax.vlines(tick + 0.5, 0, np.max(dfPower["episode reward"]), color = "grey", alpha = .1)
    
# rotate the x and y ticks
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add x and y label
ax.set_xlabel("Action", fontsize = 14)
ax.set_ylabel("episode Reward", fontsize = 14)

# set title
ax.set_title("Violinplot of Power Pill", fontsize = 14);


# In[9]:


sns.violinplot(x = "action_name", 
               y = "episode reward", 
               data = dfStand, 
               scale = 'width', 
               inner = 'quartile'
              )

# get the current figure
ax = plt.gca()
# get the xticks to iterate over
xticks = ax.get_xticks()

# iterate over every xtick and add a vertical line
# to separate different classes
for tick in xticks:
    ax.vlines(tick + 0.5, 0, np.max(dfStand["episode reward"]), color = "grey", alpha = .1)
    
# rotate the x and y ticks
ax.tick_params(axis = 'x', labelrotation = 90, labelsize = 12)
ax.tick_params(axis = 'y', labelsize = 12)

# add x and y label
ax.set_xlabel("Action", fontsize = 14)
ax.set_ylabel("episode Reward", fontsize = 14)

# set title
ax.set_title("Violinplot of Standard", fontsize = 14);


# In[45]:


sns.lineplot(x = "action", y = "episode reward", data = dfFear, color='r',linestyle='-')
sns.lineplot(x = "action", y = "episode reward", data = dfNormal, color='g',linestyle='--')
sns.lineplot(x = "action", y = "episode reward", data = dfPower, color='b',linestyle=':')
sns.lineplot(x = "action", y = "episode reward", data = dfStand, color='k',linestyle='-.')


# In[49]:


sns.lineplot(x = "action", y = "total reward", data = dfFear,label='Fear')
sns.lineplot(x = "action", y = "total reward", data = dfNormal,label='Normal')
sns.lineplot(x = "action", y = "total reward", data = dfPower,label='Power')
sns.lineplot(x = "action", y = "total reward", data = dfStand,label='Standard')


# In[68]:


sns.lineplot(x = "state", y = "total reward", data = dfFear,label='Fear')
sns.lineplot(x = "state", y = "total reward", data = dfNormal,label='Normal')
sns.lineplot(x = "state", y = "total reward", data = dfPower,label='Power')
sns.lineplot(x = "state", y = "total reward", data = dfStand,label='Standard')
sns.scatterplot(x='state', y='total reward', data = Fearkey_states)
sns.scatterplot(x='state', y='total reward', data = Normalkey_states)
sns.scatterplot(x='state', y='total reward', data = Powerkey_states)
sns.scatterplot(x='state', y='total reward', data = Standkey_states)


# In[69]:


sns.lineplot(x = "state", y = "episode reward", data = dfFear,label='Fear')
sns.lineplot(x = "state", y = "episode reward", data = dfNormal,label='Normal')
sns.lineplot(x = "state", y = "episode reward", data = dfPower,label='Power')
sns.lineplot(x = "state", y = "episode reward", data = dfStand,label='Standard')
sns.scatterplot(x='state', y='episode reward', data = Fearkey_states)
sns.scatterplot(x='state', y='episode reward', data = Normalkey_states)
sns.scatterplot(x='state', y='episode reward', data = Powerkey_states)
sns.scatterplot(x='state', y='episode reward', data = Standkey_states)


# In[70]:


sns.lineplot(x = "state", y = "epoch reward", data = dfFear,label='Fear')
sns.lineplot(x = "state", y = "epoch reward", data = dfNormal,label='Normal')
sns.lineplot(x = "state", y = "epoch reward", data = dfPower,label='Power')
sns.lineplot(x = "state", y = "epoch reward", data = dfStand,label='Standard')
sns.scatterplot(x='state', y='epoch reward', data = Fearkey_states)
sns.scatterplot(x='state', y='epoch reward', data = Normalkey_states)
sns.scatterplot(x='state', y='epoch reward', data = Powerkey_states)
sns.scatterplot(x='state', y='epoch reward', data = Standkey_states)


# In[72]:


sns.lineplot(x = "action", y = "epoch reward", data = dfFear,label='Fear')
sns.lineplot(x = "action", y = "epoch reward", data = dfNormal,label='Normal')
sns.lineplot(x = "action", y = "epoch reward", data = dfPower,label='Power')
sns.lineplot(x = "action", y = "epoch reward", data = dfStand,label='Standard')


# In[71]:


sns.lineplot(x = "action", y = "epoch reward", data = dfFear,label='Fear')
sns.lineplot(x = "action", y = "epoch reward", data = dfNormal,label='Normal')
sns.lineplot(x = "action", y = "epoch reward", data = dfPower,label='Power')
sns.lineplot(x = "action", y = "epoch reward", data = dfStand,label='Standard')
sns.scatterplot(x='action', y='epoch reward', data = Fearkey_states)
sns.scatterplot(x='action', y='epoch reward', data = Normalkey_states)
sns.scatterplot(x='action', y='epoch reward', data = Powerkey_states)
sns.scatterplot(x='action', y='epoch reward', data = Standkey_states)


# In[74]:


sns.lineplot(x = "episode step", y = "total reward", data = dfFear,label='Fear')
sns.lineplot(x = "episode step", y = "total reward", data = dfNormal,label='Normal')
sns.lineplot(x = "episode step", y = "total reward", data = dfPower,label='Power')
sns.lineplot(x = "episode step", y = "total reward", data = dfStand,label='Standard')


# In[73]:


sns.lineplot(x = "episode step", y = "reward", data = dfFear,label='Fear')
sns.lineplot(x = "episode step", y = "reward", data = dfNormal,label='Normal')
sns.lineplot(x = "episode step", y = "reward", data = dfPower,label='Power')
sns.lineplot(x = "episode step", y = "reward", data = dfStand,label='Standard')

