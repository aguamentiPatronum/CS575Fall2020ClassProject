#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pywaffle import Waffle


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dfFear = pd.read_parquet('Singularity_EvaluationsSlimmed_FearGhostsEvalDB.parquet')
dfFear = dfFear.drop(columns=dfFear.columns[0])

dfNormal = pd.read_parquet('Singularity_EvaluationsSlimmed_NormalPillsOnlyEvalDB.parquet')
dfNormal = dfNormal.drop(columns=dfNormal.columns[0])

dfPower = pd.read_parquet('Singularity_EvaluationsSlimmed_PowerPillsOnlyEvalDB.parquet')
dfPower = dfPower.drop(columns=dfPower.columns[0])

dfStand = pd.read_parquet('Singularity_EvaluationsSlimmed_StandardEval2DB.parquet')
dfStand = dfStand.drop(columns=dfStand.columns[0])


# In[5]:


Fearkey_states = dfFear[dfFear.key_state.eq(True)]
Normalkey_states = dfNormal[dfNormal.key_state.eq(True)]
Powerkey_states = dfPower[dfPower.key_state.eq(True)]
Standkey_states = dfStand[dfStand.key_state.eq(True)]


# In[6]:


Fearkey_states_with_context = dfFear[dfFear.context_state.eq(True)]
Normalkey_states_with_context = dfNormal[dfNormal.context_state.eq(True)]
Powerkey_states_with_context = dfPower[dfPower.context_state.eq(True)]
Standkey_states_with_context = dfStand[dfStand.context_state.eq(True)]


# In[11]:


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


# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[10]:


g = sns.catplot(x="total reward", y="episode reward", hue="lives", col="action_name",
                capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=dfFear)
g.despine(left=True)


# In[11]:


gN = sns.catplot(x="total reward", y="episode reward", hue="lives", col="action_name",
                capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=dfNormal)
gN.despine(left=True)


# In[12]:


gP = sns.catplot(x="total reward", y="episode reward", hue="lives", col="action_name",
                capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=dfPower)
g.despine(left=True)


# In[13]:


gS = sns.catplot(x="total reward", y="episode reward", hue="lives", col="action_name",
                capsize=.2, palette="YlGnBu_d", height=6, aspect=.75,
                kind="point", data=dfStand)
gS.despine(left=True)


# In[1]:


fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (16,6), dpi = 80)

# plot the data using the built in plots from the stats module
plot_acf(dfFear["total reward"], ax = ax1, lags = 50)
plot_pacf(dfFear["total reward"], ax = ax2, lags = 15);


# In[6]:


from fastparquet import ParquetFile
pf = ParquetFile('Singularity_EvaluationsSlimmed_FearGhostsEvalDB.parquet')
myDataframe = pf.to_pandas()
myDataframe = myDataframe.drop(columns=myDataframe.columns[0])
myDataframe.head(10)


# In[7]:


myDataframe.shape


# In[19]:


print(myDataframe. columns)


# In[17]:


key_states = myDataframe[myDataframe.key_state.eq(True)]
key_states


# In[18]:


key_states_with_context = df[df.context_state.eq(True)]
key_states_with_context


# In[32]:


import matplotlib.colors as mcolors
import matplotlib.cm as cm

plot = sns.stripplot(x='action_name', y='total reward', hue='episode reward', data= myDataframe,
                     palette='ocean', jitter=True, edgecolor='none', alpha=.60)
plot.get_legend().set_visible(False)
sns.despine()
# iris.describe()

# Drawing the side color bar
normalize = mcolors.Normalize(vmin=myDataframe['episode reward'].min(), vmax=myDataframe['episode reward'].max())
colormap = cm.ocean

for n in myDataframe['episode reward']:
    plt.plot(color=colormap(normalize(n)))

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(myDataframe['episode reward'])
plt.colorbar(scalarmappaple)


# In[13]:


x_subset = myDataframe.iloc[0:2000,2:9]
print(x_subset.shape)
y_subset = myDataframe.iloc[0:2000, 1]
print(y_subset.shape)
y_subset.head(5)
print (y_subset.unique())


# In[14]:


sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123


# In[10]:


def EV_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


# In[15]:


from sklearn.manifold import TSNE
EV_tsne = TSNE(random_state=RS).fit_transform(x_subset)
print(EV_tsne)
# EV_scatter(EV_tsne, y_subset)


# In[16]:


plt.scatter(EV_tsne[:,0],EV_tsne[:,1],  c = y_subset, 
            cmap = "coolwarm", edgecolor = "None")
plt.colorbar()
plt.title('TSNE Scatter Plot')


# In[53]:


sns.scatterplot(EV_tsne[:,0],EV_tsne[:,1], hue=y_subset)
plt.title('TSNE Scatter Plot')
plt.show()


# In[14]:


ax.spines["top"].set_color("None")
ax.spines["right"].set_color("None")

# set a specific label for each axis
ax.set_xlabel("Reward")
ax.set_ylabel("Action")

# change the lower limit of the plot, this will allow us to see the legend on the left
ax.set_xlim(-0.01) 
ax.set_title("Bubble plot with encircling")
ax.legend(loc = "upper left", fontsize = 10);


# In[ ]:


sns.pairplot(myDataframe, hue = "action_name");
            


# In[22]:


bar_plot = sns.barplot(x='key_state', y='action_name', data=myDataframe)
bar_plot = sns.barplot(x='context_state', y='action_name', data=myDataframe)

bar_plot.set(xlabel="State", ylabel="action_name", title = "Fear Ghosts Pyramid")


# In[ ]:


# prepare the data for plotting
gb_df = df.groupby(["key_state", "context_state"])["total reward"].sum().to_frame().reset_index()
gb_df.set_index("Stage", inplace = True)

# separate the different groups to be plotted
x_male = gb_df[gb_df["Gender"] == "Male"]["total reward"]
x_female = gb_df[gb_df["Gender"] == "Female"]["total reward"]


# In[2]:


from fastparquet import ParquetFile
pf = ParquetFile('Singularity_TrainingSlimmed_FearGhostsTrainingDB.parquet')
myDataframe2 = pf.to_pandas()
myDataframe2.head(10)


# In[7]:


x_subset = myDataframe2.iloc[0:2000,3:7]
print(x_subset.shape)
y_subset = myDataframe2.iloc[0:2000, 2]
print(y_subset.shape)
y_subset.head(5)
print (y_subset.unique())

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

EV_tsne2 = TSNE(random_state=RS).fit_transform(x_subset)
print(EV_tsne2)

sns.scatterplot(EV_tsne2[:,0],EV_tsne2[:,1], hue=y_subset)
plt.title('TSNE Scatter Plot2')
plt.show()


# In[8]:


plt.scatter(EV_tsne2[:,0],EV_tsne2[:,1],  c = y_subset, 
            cmap = "coolwarm", edgecolor = "None")
plt.colorbar()
plt.title('TSNE Scatter Plot')


# In[ ]:


plt.ylim(0, 1)
y = np.linspace(0, 1)
plt.plot(y, newdf['Fear_episode reward'], '+-m', label = 'Fear')
plt.plot(y, newdf['Normal_episode reward'], 'x-r', label = 'Normal')
plt.xlabel("Episode Reward")
plt.ylabel("Inference Accuracy")
plt.legend() 
plt.show()    

