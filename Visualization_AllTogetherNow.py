#!/usr/bin/env python
# coding: utf-8

# In[151]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import random
sns.set_style("whitegrid")


# In[147]:


# Location of DFs
tempTopDir = os.path.abspath('/Volumes/Britt_SSD')


# In[148]:


#############################
# Training Section
#############################
trainFile = os.path.join(tempTopDir, 'TrainingSlimmed/comboTrain.parquet')
# make training into df
allTrainDF = pd.read_parquet(trainFile)
    


# In[149]:


# Generate list of agents in order
trainAgents = allTrainDF.agent.unique()
trainAgents


# In[146]:


fearDF = pd.DataFrame()
normalOnlyDF = pd.DataFrame()
normalPlusDF = pd.DataFrame()
powerPillDF = pd.DataFrame()
standardDF = pd.DataFrame()
# Array of these DFs
trainDFsList = [fearDF, normalOnlyDF, normalPlusDF, powerPillDF, standardDF]

# Df for each Agent Training
for index, agent in enumerate(trainAgents):
    trainDFsList[index] = allTrainDF[allTrainDF['agent'] == agent]


# In[ ]:


# Visuals For Training


# In[144]:


for index, agent in enumerate(trainDFsList):
    plt.scatter("state", "epoch reward", data = agent, label = trainAgents[index])
    plt.title(trainAgents[index])
    plt.legend(loc="best")
    # call savefig right before show
    filename = randint(1344576,95836265278) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')


# In[ ]:





# In[ ]:





# In[3]:


#############################
# Evaluations Section
#############################
evalsFile = os.path.join(tempTopDir, 'Location_Evals/comboTrain.parquet')
# Make evals to df
allEvalsDF = pd.read_parquet(evalsFile)


# In[4]:


# Generate list of agents in order
evalAgents = allEvalsDF.agent.unique()
evalAgents


# In[22]:


allEvalsDF.columns


# In[129]:


allEvalsDF['red_x_relative'] = np.where(allEvalsDF['red_ghost_coord_x'] < allEvalsDF['pacman_coord_x'], -1, np.where(allEvalsDF['red_ghost_coord_x'] == allEvalsDF['pacman_coord_x'],0,1))


# In[130]:


allEvalsDF['red_y_relative'] = np.where(allEvalsDF['red_ghost_coord_y'] < allEvalsDF['pacman_coord_y'], -1, np.where(allEvalsDF['red_ghost_coord_y'] == allEvalsDF['pacman_coord_y'],0,1))


# In[131]:


allEvalsDF['blue_x_relative'] = np.where(allEvalsDF['blue_ghost_coord_x'] < allEvalsDF['pacman_coord_x'], -1, np.where(allEvalsDF['blue_ghost_coord_x'] == allEvalsDF['pacman_coord_x'],0,1))


# In[132]:


allEvalsDF['blue_y_relative'] = np.where(allEvalsDF['blue_ghost_coord_y'] < allEvalsDF['pacman_coord_y'], -1, np.where(allEvalsDF['blue_ghost_coord_y'] == allEvalsDF['pacman_coord_y'],0,1))


# In[133]:


allEvalsDF['orange_x_relative'] = np.where(allEvalsDF['orange_ghost_coord_x'] < allEvalsDF['pacman_coord_x'], -1, np.where(allEvalsDF['orange_ghost_coord_x'] == allEvalsDF['pacman_coord_x'],0,1))


# In[134]:


allEvalsDF['orange_y_relative'] = np.where(allEvalsDF['orange_ghost_coord_y'] < allEvalsDF['pacman_coord_y'], -1, np.where(allEvalsDF['orange_ghost_coord_y'] == allEvalsDF['pacman_coord_y'],0,1))



# In[135]:


allEvalsDF['pink_x_relative'] = np.where(allEvalsDF['pink_ghost_coord_x'] < allEvalsDF['pacman_coord_x'], -1, np.where(allEvalsDF['pink_ghost_coord_x'] == allEvalsDF['pacman_coord_x'],0,1))


# In[136]:


allEvalsDF['pink_y_relative'] = np.where(allEvalsDF['pink_ghost_coord_y'] < allEvalsDF['pacman_coord_y'], -1, np.where(allEvalsDF['pink_ghost_coord_y'] == allEvalsDF['pacman_coord_y'],0,1))



# In[ ]:


# allEvalsDF['red_x_relative'] = np.where(allEvalsDF['red_ghost_coord_x'] < allEvalsDF['pacman_coord_x'], -1, np.where(allEvalsDF['red_ghost_coord_x'] == allEvalsDF['pacman_coord_x'],0,1))


# In[ ]:


# allEvalsDF['blue_y_relative'] = np.where(allEvalsDF['blue_ghost_coord_y'] < allEvalsDF['pacman_coord_y'], -1, np.where(allEvalsDF['blue_ghost_coord_y'] == allEvalsDF['pacman_coord_y'],0,1))



# In[138]:


# Df for each Agent Training
fearEvalDF = pd.DataFrame()
normalOnlyEvalDF = pd.DataFrame()
normalPlusEvalDF = pd.DataFrame()
powerPillEvalDF = pd.DataFrame()
standardEvalDF = pd.DataFrame()

# Array of these DFs
evalDFsList = [fearEvalDF, normalOnlyEvalDF, normalPlusEvalDF, powerPillEvalDF, standardEvalDF]

# Df for each Agent Training Key States
fearEvalKeyDF = pd.DataFrame()
normalOnlyEvalKeyDF = pd.DataFrame()
normalPlusEvalKeyDF = pd.DataFrame()
powerPillEvalKeyDF = pd.DataFrame()
standardEvalKeyDF = pd.DataFrame()

# Array of these DFs
evalKeysDFsList = [fearEvalKeyDF, normalOnlyEvalKeyDF, normalPlusEvalKeyDF, powerPillEvalKeyDF, standardEvalKeyDF]

# Df for each Agent Evaluation
for index, agent in enumerate(evalAgents):
    evalDFsList[index] = allEvalsDF[allEvalsDF['agent'] == agent]
    
# Df for each Agent Evaluation Key States
for index, agent in enumerate(evalAgents):
    temp = evalDFsList[index]
    evalKeysDFsList[index] = temp[temp['keyNum'] > 0]


# In[58]:


# Now get old data without location information

evalsOldFile = os.path.join(tempTopDir, 'Singularity/EvaluationsSlimmed')

# Df for each Agent Training
fearEvalOldDF = pd.read_parquet(os.path.join(evalsOldFile, 'FearGhostsEvalDB.parquet'))
normalOnlyEvalOldDF = pd.read_parquet(os.path.join(evalsOldFile, 'NormalPillsOnlyEvalDB.parquet'))
normalPlusEvalOldDF = pd.read_parquet(os.path.join(evalsOldFile, 'NmlPillsInGameEvalDB.parquet'))
powerPillEvalOldDF = pd.read_parquet(os.path.join(evalsOldFile, 'PowerPillsOnlyEvalDB.parquet'))
standardEvalOldDF = pd.read_parquet(os.path.join(evalsOldFile, 'StandardEvalDB.parquet'))

# Array of these DFs
evalOldDFsList = [fearEvalOldDF, normalOnlyEvalOldDF, normalPlusEvalOldDF, powerPillEvalOldDF, standardEvalOldDF]

# Df for each Agent Training Key States
fearEvalOldKeyDF = pd.DataFrame()
normalOnlyEvalOldKeyDF = pd.DataFrame()
normalPlusEvalOldKeyDF = pd.DataFrame()
powerPillEvalOldKeyDF = pd.DataFrame()
standardEvalOldKeyDF = pd.DataFrame()

# Array of these DFs
evalKeysOldDFsList = [fearEvalOldKeyDF, normalOnlyEvalOldKeyDF, normalPlusEvalOldKeyDF, powerPillEvalOldKeyDF, standardEvalOldKeyDF]

# Df for each Agent Evaluation Key States
for index, agent in enumerate(evalAgents):
    temp = evalOldDFsList[index]
    evalKeysOldDFsList[index] = temp[temp['key_state'] == True]
    
oldCombDF = pd.concat([fearEvalOldKeyDF,normalOnlyEvalOldKeyDF,normalPlusEvalOldKeyDF,powerPillEvalOldKeyDF,standardEvalOldKeyDF],ignore_index=True)
    


# In[6]:


# And get validation data with location information

evalsValidFile = os.path.join(tempTopDir, 'Location_Evals2_Beluga/comboTrain_Validation.parquet')
# Make evals to df
allEvalsValidDF = pd.read_parquet(evalsValidFile)

# Generate list of agents in order
evalValidAgents = allEvalsValidDF.agent.unique()
evalValidAgents

# Df for each Agent Training
fearEvalValidDF = pd.DataFrame()
normalOnlyEvalValidDF = pd.DataFrame()
normalPlusEvalValidDF = pd.DataFrame()
powerPillEvalValidDF = pd.DataFrame()
standardEvalValidDF = pd.DataFrame()

# Array of these DFs
evalValidDFsList = [fearEvalValidDF, normalOnlyEvalValidDF, normalPlusEvalValidDF, powerPillEvalValidDF, standardEvalValidDF]

# Df for each Agent Training Key States
fearEvalValidKeyDF = pd.DataFrame()
normalOnlyEvalValidKeyDF = pd.DataFrame()
normalPlusEvalValidKeyDF = pd.DataFrame()
powerPillEvalValidKeyDF = pd.DataFrame()
standardEvalValidKeyDF = pd.DataFrame()

# Array of these DFs
evalValidKeysDFsList = [fearEvalValidKeyDF, normalOnlyEvalValidKeyDF, normalPlusEvalValidKeyDF, powerPillEvalValidKeyDF, standardEvalValidKeyDF]

# Df for each Agent Evaluation
for index, agent in enumerate(evalValidAgents):
    evalValidDFsList[index] = allEvalsValidDF[allEvalsValidDF['agent'] == agent]
    
# Df for each Agent Evaluation Key States
for index, agent in enumerate(evalValidAgents):
    temp = evalValidDFsList[index]
    evalValidKeysDFsList[index] = temp[temp['keyNum'] > 0]


# In[6]:


# Visuals For Evaluations


# In[ ]:





# In[154]:


for index, agent in enumerate(evalKeysDFsList):
    plt.scatter("state", "importance", data = agent, alpha=0.7)
    plt.xlabel('Frame Number')
    plt.ylabel('Importance')
    plt.title(evalAgents[index])
    plt.legend(loc="best")
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    


# In[24]:


for index, agent in enumerate(evalDFsList):
    plt.scatter("state", "importance", data = agent, alpha = 0.25)
    plt.scatter("state", "importance", data = evalKeysDFsList[index], c = "b", alpha = 0.25)
    plt.xlabel('Frame Number')
    plt.ylabel('Importance')
    plt.title(evalAgents[index])
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[25]:


fearEvalKeyDF.head(1)


# In[26]:


for index, agent in enumerate(evalKeysDFsList):
    plt.scatter("keyNum", "lives", s='epoch_reward', c= 'b', data = agent, label = "lives left")
    plt.scatter("keyNum", "lives", s=15, data = agent[agent['beforeLifeLoss']>0], color = "red", label = "lost a life")
    plt.ylabel('Lives Left')
    plt.xlabel('Key Situation Number')
    plt.title("Lives Left in Each Key Situation " + evalAgents[index])
    plt.yticks([1,2,3],)
    plt.legend(loc="best")
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[27]:


# sns.set_style("whitegrid")
# for index, agent in enumerate(evalKeysDFsList):
#     for i in agent.keyNum.unique():
#         plt.plot("state", "reward", data = agent[agent['keyNum']==i], c='blue')
#         plt.plot("state", "lives", data = agent[agent['keyNum']==i], c='green')
#         plt.xlabel("Frame Number")
#         plt.ylabel("Rewards Earned")
#         plt.title("Rewards Earned in Situation " + evalAgents[index] + str(i))
#         plt.legend(loc="best")
#         # call savefig right before show
#         # set up filename
#         #filename = agentKeyNames[index] + "RewardsOverTimeInSitch" + str(i) + ".png"
#         #savePath = os.path.join(folderList[index], filename)
#         #plt.savefig(savePath, dpi=300, bbox_inches='tight')
#         plt.show()


# In[28]:


sns.set_style("whitegrid")
for index, agent in enumerate(evalKeysDFsList):
    for i in agent.keyNum.unique():
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.plot("state", "episode_reward", data = agent[agent['keyNum']==i], c='red', linewidth=3, linestyle='--', label = 'rewards this life')
        ax1.plot("state", "epoch_reward", data = agent[agent['keyNum']==i], c='gold', linewidth=15, alpha = 0.62, label = 'rewards this game')
        ax1.scatter("state", "episode_reward", data = agent.loc[(agent['keyNum']==i) & (agent['reward']>0)], c='blue', linewidth=3, alpha = 0.9, label="points scored")
        ax1.set_ylim(0,400)
        ax1.set_ylabel('Points')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color = 'tab:green'

        ax2.plot("state", "lives", data = agent.loc[(agent['keyNum']==i)], c=color, linewidth=4, linestyle=":")
        ax2.set_ylim(0,4)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('Lives')

        plt.title("Map of agent " + evalAgents[index])
        

#         fig.legend(loc="best")
        # call savefig right before show
        # set up filename
        #filename = agentKeyNames[index] + "SubwayMapSitch" + str(i) + ".png"
        #savePath = os.path.join(folderList[index], filename)
        #plt.savefig(savePath, dpi=300, bbox_inches='tight')
        fig.tight_layout()
        # call savefig right before show
        num = random.randint(1344576,95836265278)
        filename = str(num) + ".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
#         temp = evalValidKeysDFsList[index]
#         fig, ax1 = plt.subplots()
#         color = 'tab:red'
#         ax1.plot("state", "episode_reward", data = temp[temp['keyNum']==i], c='red', linewidth=3, linestyle='--', label = 'rewards this life')
#         ax1.plot("state", "epoch_reward", data = temp[temp['keyNum']==i], c='gold', linewidth=15, alpha = 0.62, label = 'rewards this game')
#         ax1.scatter("state", "episode_reward", data = temp.loc[(temp['keyNum']==i) & (temp['reward']>0)], c='blue', linewidth=3, alpha = 0.9, label="points scored")
#         ax1.set_ylim(0,400)
#         ax1.set_ylabel('Points')
#         ax1.tick_params(axis='y', labelcolor=color)

#         ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
#         color = 'tab:green'

#         ax2.plot("state", "lives", data = agent.loc[(temp['keyNum']==i)], c=color, linewidth=4, linestyle=":")
#         ax2.set_ylim(0,4)
#         ax2.tick_params(axis='y', labelcolor=color)
#         ax2.set_ylabel('Lives')

#         plt.title("Map of agent " + evalAgents[index] + " validation state " + str(i))
        

# #         fig.legend(loc="best")
#         # call savefig right before show
#         # set up filename
#         #filename = agentKeyNames[index] + "SubwayMapSitch" + str(i) + ".png"
#         #savePath = os.path.join(folderList[index], filename)
#         #plt.savefig(savePath, dpi=300, bbox_inches='tight')
#         fig.tight_layout()
#         plt.show()
        


# In[126]:


colorList = ['b', 'coral','r','orange','cyan']
for index, agent in enumerate(evalKeysDFsList):
    plt.plot("state", "total_reward", data = evalDFsList[index], label = evalAgents[index])
    plt.scatter('state', 'total_reward', data = agent, s=58, c='green', alpha=0.05, label="")
    plt.scatter('state', 'total_reward', data = evalKeysDFsList[index], s=48, c=colorList[index], label = evalAgents[index])
    plt.xlabel('Frame')
    plt.ylabel('Total Rewards Earned')
    plt.title('Key states during Evaluation')
    plt.legend(loc="best")
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')


# In[123]:


sns.set_style("whitegrid")
for index, agent in enumerate(evalKeysDFsList):
    for i in agent.keyNum.unique():
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.plot("state", "episode_reward", data = agent[agent['keyNum']==i], c='red', linewidth=3, linestyle='--', label = 'rewards this life')
        ax1.plot("state", "epoch_reward", data = agent[agent['keyNum']==i], c='gold', linewidth=15, alpha = 0.62, label = 'rewards this game')
        ax1.scatter("state", "episode_reward", data = agent.loc[(agent['keyNum']==i) & (agent['reward']>0)], c='blue', linewidth=3, alpha = 0.9, label="points scored")
        ax1.set_ylim(0,400)
        ax1.set_ylabel('Points')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color = 'tab:green'

        ax2.plot("state", "lives", data = agent.loc[(agent['keyNum']==i)], c=color, linewidth=4, linestyle=":")
        ax2.set_ylim(0,4)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('Lives')

        plt.title("Map of agent " + evalAgents[index])
        

#         fig.legend(loc="best")
        # call savefig right before show
        # set up filename
        #filename = agentKeyNames[index] + "SubwayMapSitch" + str(i) + ".png"
        #savePath = os.path.join(folderList[index], filename)
        #plt.savefig(savePath, dpi=300, bbox_inches='tight')
        fig.tight_layout()
        plt.show()


# In[122]:


val = 0.1
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("red_ghost_coord_x", "red_ghost_coord_y", alpha=1, data=agent)
    plt.show()
    plt.scatter("blue_ghost_coord_x", "blue_ghost_coord_y", alpha=1, data=agent)
    plt.show()
    plt.scatter("pink_ghost_coord_x", "pink_ghost_coord_y", alpha=1, data=agent)
    plt.show()
    plt.scatter("orange_gohst_coord_x", "orange_gohst_coord_y", alpha=1, data=agent)
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[30]:


# Set the palette using the name of a palette:
sns.set_palette("tab10")
for index, agent in enumerate(evalKeysDFsList): 
    for i in agent.keyNum.unique():
        g.fig.suptitle("Distribution of Actions over Key Situations " + evalAgents[index] + " " + str(i))
        g.set(xlabel=specialACT, ylabel="Percent of All Actions")
        p1=sns.kdeplot(agent[agent['keyNum']==i]['action'], shade=True, legend=False)
        
    plt.figure()


# In[31]:


# Section to grab the end of episode scores and plot them 
for index, agent in enumerate(evalDFsList):
    temp = agent[agent['epoch_score'] != 0]
    plt.plot(temp['episode'],temp['epoch_score'], 'o', c='b')
    plt.title('Score for Each Life ' + evalAgents[index])
    plt.xlabel('Life')
    plt.ylabel('Score Achieved as of Last Frame of Life ' + evalAgents[index])
    plt.ylim(0,600)
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[32]:


markers = ['o','','x','*','|','_','.','>','<']
fig, ax1 = plt.subplots()
for index, agent in enumerate(evalDFsList):
    for i in agent.keyNum.unique():
        plt.plot("state", "reward", data = agent[agent['keyNum']==i], c='blue')
        plt.plot("state", "lives", data = agent[agent['keyNum']==i], c='green')
        plt.plot("state", "episode_reward", data = agent[agent['keyNum']==i], c='red')
        plt.plot("state", "epoch_reward", data = agent[agent['keyNum']==i], c='yellow')
        plt.title("Rewards Earned in Situation " + str(i))
        plt.legend(loc="best")
        # call savefig right before show
        num = random.randint(1344576,95836265278)
        filename = str(num) + ".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()


# In[34]:


# put this into a loop
# Can I get the sums of each action for each epoch?

# for index, agent in enumerate(evalDFsList):
#     actions_df = pd.DataFrame(agent.pivot_table(index='epoch',columns='action_name',aggfunc=sum))
#     action_subset = pd.DataFrame(actions_df['action'])
#     action_subset['epoch'] = range(1, len(action_subset) + 1)
#     action_subset.set_index('epoch',drop=False,inplace=True)
#     actions_only = action_subset.drop(['epoch'], axis=1)
#     data_perc = actions_only.divide(actions_only.sum(axis=1), axis=0)
#     plt.stackplot(range(1,24), data_perc['DOWN'], 
#                   data_perc['LEFT'], 
#                   data_perc['RIGHT'], 
#                   data_perc['UP'], 
#                   labels = ['DOWN', 'LEFT', 'RIGHT', 'UP'])
#     plt.title('Actions Per Life')
#     plt.xlabel('Life')
#     plt.ylabel('Percent of Actions Taken in Life')
#     plt.legend(loc="lower left")


# In[35]:


plt.scatter("state", "total_reward", c='agentNum', data=allEvalsDF)


# In[36]:


temp = allEvalsDF[allEvalsDF["keyNum"]>0]
plt.scatter("state", "keyNum", c='agentNum', alpha=0.1, data=temp)
plt.legend(loc="best")
# call savefig right before show
num = random.randint(1344576,95836265278)
filename = str(num) + ".png"
plt.savefig(filename, dpi=300, bbox_inches='tight')


# In[37]:


val = 0.25
for index, agent in enumerate(evalDFsList):
    plt.scatter("to_red_ghost", "importance", alpha=1-(val*index), data=agent, label = evalAgents[index])
    plt.xlabel("to red ghost")
    plt.ylabel("importance")
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[38]:


val = 0.25
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("to_ghosts_mean", "to_top_pills_mean", alpha=1, data=temp, label = evalAgents[index])
# call savefig right before show
num = random.randint(1344576,95836265278)
filename = str(num) + ".png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.legend(loc="best")


# In[39]:


val = 0.25
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("to_ghosts_mean", "to_bottom_pills_mean", alpha=1, data=temp, label = evalAgents[index])
# call savefig right before show
num = random.randint(1344576,95836265278)
filename = str(num) + ".png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.legend(loc="best")


# In[40]:


val = 0.25
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("to_pill_mean", "keyNum", alpha=1, data=temp, label = evalAgents[index])
#     plt.scatter("to_pill_mean", "keyNum", alpha=0.1, data=agent, label = names[index])
plt.legend(loc="best")


# In[41]:


val = 0.1
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("pacman_coord_x", "pacman_coord_y", alpha=0.8-(val*index), data=temp, label = evalAgents[index])
#     plt.scatter("to_pill_mean", "keyNum", alpha=0.1, data=agent, label = names[index])
plt.legend(loc="best")


# In[62]:


val = 0.1
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("dark_blue_ghost1_coord_x", "dark_blue_ghost1_coord_y", alpha=0.8-(val*index), data=agent)
    plt.scatter("dark_blue_ghost2_coord_x", "dark_blue_ghost2_coord_y", alpha=0.8-(val*index), data=agent)
    plt.scatter("dark_blue_ghost3_coord_x", "dark_blue_ghost3_coord_y", alpha=0.8-(val*index), data=agent)
    plt.scatter("dark_blue_ghost4_coord_x", "dark_blue_ghost4_coord_y", alpha=0.8-(val*index), data=agent)
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[111]:


val = 0.1
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("red_ghost_coord_x", "red_ghost_coord_y", alpha=1, data=agent)
    plt.show()
    plt.scatter("blue_ghost_coord_x", "blue_ghost_coord_y", alpha=1, data=agent)
    plt.show()
    plt.scatter("pink_ghost_coord_x", "pink_ghost_coord_y", alpha=1, data=agent)
    plt.show()
    plt.scatter("orange_gohst_coord_x", "orange_gohst_coord_y", alpha=1, data=agent)
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[125]:


for index, agent in enumerate(evalDFsList):
    plt.scatter("state", "diff_to_orange", data=agent, c="orange", alpha=0.9)
    plt.scatter("state", "diff_to_red", data=agent, c="r", alpha=0.7)
    plt.scatter("state", "diff_to_pink", data=agent, c="green", alpha=0.5)
    plt.scatter("state", "diff_to_blue", data=agent, c="b", alpha=0.2)
    temp = agent.query('diff_to_dbg1 != diff_to_dbg2 & diff_to_dbg2 != diff_to_dbg3 & diff_to_dbg3 != diff_to_dbg4')
    for i in range(1, len(temp)):
        plt.axvline(x=list(temp['state'])[i],alpha=0.01)
        plt.scatter("state", "diff_to_dbg1", data=temp, c="gold")
        plt.scatter("state", "diff_to_dbg2", data=temp, c="gold")
        plt.scatter("state", "diff_to_dbg3", data=temp, c="gold")
        plt.scatter("state", "diff_to_dbg4", data=temp, c="gold")
#     plt.legend(loc="best")
    plt.title(evalAgents[index])
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[105]:


val = 0.1
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]==1]
    plt.scatter("red_ghost_coord_x", "red_ghost_coord_y", alpha=1, c="r", data=temp)
    plt.scatter("pacman_coord_x", "pacman_coord_y", alpha=1, c="gold", data=temp)
    plt.scatter("red_ghost_coord_x", "red_ghost_coord_y", alpha=0.01, c="hotpink", data=agent)
    plt.scatter("pacman_coord_x", "pacman_coord_y", alpha=0.01, c="yellow", data=agent)
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.scatter("blue_ghost_coord_x", "blue_ghost_coord_y", alpha=1, c="b", data=temp)
    plt.scatter("pacman_coord_x", "pacman_coord_y", alpha=1, c="gold",  data=temp)
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.scatter("pink_ghost_coord_x", "pink_ghost_coord_y", alpha=1, c="pink", data=temp)
    plt.scatter("pacman_coord_x", "pacman_coord_y", alpha=1, c="gold",  data=temp)
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.scatter("orange_gohst_coord_x", "orange_gohst_coord_y", alpha=1, c="orange", data=temp)
    plt.scatter("pacman_coord_x", "pacman_coord_y", alpha=1, c="gold",  data=temp)
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[106]:


val = 0.25
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("total_reward", "keyNum", alpha=1, data=temp, label = evalAgents[index])
#     plt.scatter("to_pill_mean", "keyNum", alpha=0.1, data=agent, label = names[index])
plt.legend(loc="best")


# In[107]:


val = 0.25
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("to_ghosts_mean", "importance", alpha=0.2, data=agent, c="reward", label = evalAgents[index])
    plt.scatter("to_ghosts_mean", "importance", alpha=0.72, data=temp, label = evalAgents[index])
    plt.title(evalAgents[index])
    plt.ylabel("importance")
    plt.xlabel("mean distance to ghosts")
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
#     plt.scatter("to_pill_mean", "keyNum", alpha=0.1, data=agent, label = names[index])
# plt.legend(loc="best")


# In[108]:


#########################
# Key State Evals For ML
#########################


# In[109]:


val = 0.25
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("key_state", "to_orange_ghost", alpha=1, data=temp, label = evalAgents[index], c= "darkorange")
    plt.scatter("key_state", "to_red_ghost", alpha=1, data=temp, label = evalAgents[index], c= "firebrick")
    plt.title(evalAgents[index])
    plt.xlabel("key state")
    plt.ylabel("distance to ghosts")
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    #     plt.scatter("to_pill_mean", "keyNum", alpha=0.1, data=agent, label = names[index])
# plt.legend(loc="best")


# In[121]:


val = 0.25
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("key_state", "to_blue_ghost", alpha=1, data=temp, label = evalAgents[index], c= "dodgerblue")
    plt.scatter("key_state", "to_pink_ghost", alpha=1, data=temp, label = evalAgents[index], c= "lightpink")
    plt.title(evalAgents[index])
    plt.xlabel("key state")
    plt.ylabel("distance to ghosts")
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    #     plt.scatter("to_pill_mean", "keyNum", alpha=0.1, data=agent, label = names[index])


# In[113]:


val = 0.25
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("key_state", "to_orange_ghost", alpha=1, data=temp, label = evalAgents[index], c= "darkorange")
    plt.scatter("key_state", "to_red_ghost", alpha=1, data=temp, label = evalAgents[index], c= "firebrick")
    plt.scatter("key_state", "to_blue_ghost", alpha=1, data=temp, label = evalAgents[index], c= "dodgerblue")
    plt.scatter("key_state", "to_pink_ghost", alpha=1, data=temp, label = evalAgents[index], c= "lightpink")
    plt.title(evalAgents[index])
    plt.xlabel("key state")
    plt.ylabel("distance to ghosts")
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    #     plt.scatter("to_pill_mean", "keyNum", alpha=0.1, data=agent, label = names[index])


# In[114]:


val = 0.25
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("key_state", "action 1 total sum", alpha=1, data=agent, label = evalAgents[index], c= "darkorange")
    plt.scatter("key_state", "action 2 total sum", alpha=1, data=agent, label = evalAgents[index], c= "firebrick")
    plt.scatter("key_state", "action 3 total sum", alpha=1, data=agent, label = evalAgents[index], c= "dodgerblue")
    plt.scatter("key_state", "action 4 total sum", alpha=1, data=agent, label = evalAgents[index], c= "lightpink")
    plt.title(evalAgents[index])
    plt.xlabel("key state")
    plt.ylabel("action count")
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    #     plt.scatter("to_pill_mean", "keyNum", alpha=0.1, data=agent, label = names[index])


# In[115]:


##############################################################################################################


# In[116]:


val = 0.25
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("episode_reward", "keyNum", c="importance", alpha=1, data=agent, label = evalAgents[index])
    plt.title(evalAgents[index])
    plt.xlabel("episode reward")
    plt.ylabel("key state")
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
#     plt.scatter("to_pill_mean", "keyNum", alpha=0.1, data=agent, label = names[index])
# plt.legend(loc="best")


# In[117]:


for index, agent in enumerate(evalDFsList):
    ax = sns.lmplot(x="to_blue_ghost",y="beforeLifeLoss",logistic=True,data=agent)
    ax = sns.lmplot(x="to_red_ghost",y="beforeLifeLoss",logistic=True,data=agent)
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[64]:


val = 0.1
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]==1]
    plt.scatter("pacman_coord_x", "pacman_coord_y", alpha=0.5, c="gold", data=temp)
    plt.scatter("red_ghost_coord_x", "red_ghost_coord_y", alpha=0.5, c="r", data=temp)
    plt.title(evalAgents[index])
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[65]:


val = 0.1
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    for num in temp.keyNum.unique():
        temp2 = temp[temp["keyNum"]==num]
        plt.scatter("state", "to_red_ghost", alpha=0.5, c="r", data=temp2)
        plt.scatter("state", "to_blue_ghost", alpha=0.5, c="b", data=temp2)
#         xcoords = temp2[temp2['lives']==1]
#         for xc in xcoords:
#             plt.axvline(x=xc)
        plt.title(evalAgents[index] + " key state " + str(num))
        # call savefig right before show
        num = random.randint(1344576,95836265278)
        filename = str(num) + ".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()


# In[66]:


for i in (0,1,2,3,4):
    plt.plot('state', 'to_db1', data=allEvalsDF[allEvalsDF['agentNum']==i], alpha=0.5)
    plt.plot('state', 'to_pill_three', data=allEvalsDF[allEvalsDF['agentNum']==i], alpha=0.1)
    plt.plot('state', 'to_pill_four', data=allEvalsDF[allEvalsDF['agentNum']==i], alpha=0.1)
#     plt.plot('state', 'dist_to_db4', data=combo_df[combo_df['agentNum']==i])
    plt.scatter("state","reward",data = allEvalsDF[allEvalsDF['agentNum']==i], c='r')
    plt.legend(loc="best")
    plt.title(evalAgents[i])
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[68]:


val = 0.1
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("dark_blue_ghost1_coord_x", "dark_blue_ghost1_coord_y", alpha=0.8-(val*index), data=agent)
    plt.scatter("dark_blue_ghost2_coord_x", "dark_blue_ghost2_coord_y", alpha=0.8-(val*index), data=agent)
    plt.scatter("dark_blue_ghost3_coord_x", "dark_blue_ghost3_coord_y", alpha=0.8-(val*index), data=agent)
    plt.scatter("dark_blue_ghost4_coord_x", "dark_blue_ghost4_coord_y", alpha=0.8-(val*index), data=agent)
    plt.show()


# In[69]:


val = 0.1
for index, agent in enumerate(evalDFsList):
    temp = agent[agent["keyNum"]>0]
    plt.scatter("pacman_coord_x", "pacman_coord_y", alpha=0.8-(val*index), data=temp, label = evalAgents[index])
#     plt.scatter("to_pill_mean", "keyNum", alpha=0.1, data=agent, label = names[index])
plt.legend(loc="best")


# In[118]:


# for index, agent in enumerate(evalDFsList):
#     temp = agent.query('diff_to_dbg1 != diff_to_dbg2 & diff_to_dbg2 != diff_to_dbg3 & diff_to_dbg3 != diff_to_dbg4')
#     print(temp['state'])


# In[119]:


for index, agent in enumerate(evalDFsList):
    plt.scatter("state", "diff_to_orange", data=agent, c="orange", alpha=0.9)
    plt.scatter("state", "diff_to_red", data=agent, c="r", alpha=0.7)
    plt.scatter("state", "diff_to_pink", data=agent, c="green", alpha=0.5)
    plt.scatter("state", "diff_to_blue", data=agent, c="b", alpha=0.2)
    temp = agent.query('diff_to_dbg1 != diff_to_dbg2 & diff_to_dbg2 != diff_to_dbg3 & diff_to_dbg3 != diff_to_dbg4')
    for i in range(1, len(temp)):
        plt.axvline(x=list(temp['state'])[i],alpha=0.2)
        plt.scatter("state", "diff_to_dbg1", data=temp, c="gold")
        plt.scatter("state", "diff_to_dbg2", data=temp, c="gold")
        plt.scatter("state", "diff_to_dbg3", data=temp, c="gold")
        plt.scatter("state", "diff_to_dbg4", data=temp, c="gold")
#     plt.legend(loc="best")
    plt.title(evalAgents[index])
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[120]:


# allEvalsDF.columns


# In[44]:


sns.set_style("whitegrid")
for index, agent in enumerate(evalKeysDFsList):
    for i in agent.keyNum.unique():
        temp = agent[agent['keyNum']==i]
        plt.plot("state", "diff_to_orange", data=temp, label="orange", c="orange", marker=1)
        plt.plot("state", "diff_to_red", data=temp, label="red", c="r", marker=2)
        plt.plot("state", "diff_to_pink", data=temp, label="pink", c="pink", marker=3)
        plt.plot("state", "diff_to_blue", data=temp, label="blue", c="cyan", marker=4)
        plt.scatter("state", "beforeLifeLoss", data=temp[temp["beforeLifeLoss"]==True], label="life lost", c="green")
        plt.xlabel("Frame Number")
        plt.ylabel("Rewards Earned")
        plt.title("Rewards Earned in Situation " + evalAgents[index] + str(i))
        plt.legend(loc="best")
        # call savefig right before show
        num = random.randint(1344576,95836265278)
        filename = str(num) + ".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()


# In[45]:


sns.set_style("whitegrid")
for index, agent in enumerate(evalKeysDFsList):
    for i in agent.keyNum.unique():
        temp = agent[agent['keyNum']==i]
        plt.plot("state", "diff_to_dbg1", data=temp, c="orange", marker=1)
        plt.plot("state", "diff_to_dbg2", data=temp, c="r", marker=2)
        plt.plot("state", "diff_to_dbg3", data=temp, c="pink", marker=3)
        plt.plot("state", "diff_to_dbg4", data=temp, c="cyan", marker=4)
        plt.scatter("state", "beforeLifeLoss", data=temp[temp["beforeLifeLoss"]==True], label="life lost", c="green")
        plt.scatter("state", "lives", data=temp[temp["reward"]>10], label="power pill", c="black")
        
        plt.xlabel("Frame Number")
        plt.ylabel("Rewards Earned")
        plt.title("Rewards Earned in Situation " + evalAgents[index] + str(i))
        # call savefig right before show
        num = random.randint(1344576,95836265278)
        filename = str(num) + ".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()


# In[46]:


for index, agent in enumerate(evalKeysDFsList):
    temp = agent.query('diff_to_dbg1 != diff_to_dbg2 & diff_to_dbg2 != diff_to_dbg3 & diff_to_dbg3 != diff_to_dbg4')
    plt.plot("state", "diff_to_dbg1", data=temp, label="orange", c="orange", marker=1)
    plt.plot("state", "diff_to_dbg2", data=temp, label="red", c="r", marker=2)
    plt.plot("state", "diff_to_dbg3", data=temp, label="pink", c="pink", marker=3)
    plt.plot("state", "diff_to_dbg4", data=temp, label="blue", c="cyan", marker=4)
    plt.scatter("state","reward", data=temp[temp['reward'] > 10])
    plt.scatter("state", "keyNum", data=temp[temp["keyNum"]>0], c="grey", label=temp["keyNum"])
    plt.legend(loc="best")
    plt.title(evalAgents[index])
    plt.show()


# In[77]:


# for index, agent in enumerate(evalDFsList):
#     temp = agent.query('diff_to_dbg1 != diff_to_dbg2 & diff_to_dbg2 != diff_to_dbg3 & diff_to_dbg3 != diff_to_dbg4')
#     plt.plot("state", "diff_to_dbg1", data=temp, label="orange", c="orange", marker=1)
#     plt.plot("state", "diff_to_dbg2", data=temp, label="red", c="r", marker=2)
#     plt.plot("state", "diff_to_dbg3", data=temp, label="pink", c="pink", marker=3)
#     plt.plot("state", "diff_to_dbg4", data=temp, label="blue", c="cyan", marker=4)
#     plt.scatter("state","reward", data=temp[temp['reward'] > 10])
#     plt.scatter("state", "keyNum", data=temp[temp["keyNum"]>0], c="grey", label=temp["keyNum"])
# #     plt.legend(loc="best")
#     plt.title(evalAgents[index])
#     plt.show()


# In[47]:


for index, agent in enumerate(evalKeysDFsList):
    plt.scatter("state","reward", data=agent[agent['reward'] > 10], label="big reward")
    plt.scatter("state", "keyNum", data=agent[agent["keyNum"]>0], c="grey", label="keyNum")
    plt.legend(loc="best")
    plt.title(evalAgents[index])
    plt.show()


# In[49]:


for index, agent in enumerate(evalDFsList):
    temp = agent.query('diff_to_dbg1 != diff_to_dbg2 & diff_to_dbg2 != diff_to_dbg3 & diff_to_dbg3 != diff_to_dbg4')
    plt.plot("state", "diff_to_dbg1", data=temp, label="orange", c="orange", marker=1)
    plt.plot("state", "diff_to_dbg2", data=temp, label="red", c="r", marker=2)
    plt.plot("state", "diff_to_dbg3", data=temp, label="pink", c="pink", marker=3)
    plt.plot("state", "diff_to_dbg4", data=temp, label="blue", c="cyan", marker=4)
    plt.plot("state", "diff_to_blue", data=temp, label="blue", c="black", marker=4)
    plt.title(evalAgents[index])
    plt.show()


# In[143]:


for index, agent in enumerate(evalKeysDFsList):
    for i in agent.keyNum.unique():
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.plot("state","red_x_relative", data=agent[agent['keyNum']==i], label="red")
        ax1.set_ylabel('Red Ghost')
        ax1.set_ylim(-2.5,2.5)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color = 'tab:green'

        ax2.scatter("state", "action", data=agent[agent['keyNum']==i], c="grey", label="action")
        ax2.set_ylim(0,5)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('Action')

        plt.title("Map of agent " + evalAgents[index])
        # call savefig right before show
        num = random.randint(1344576,95836265278)
        filename = str(num) + ".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    


# In[127]:


for index, agent in enumerate(evalKeysDFsList):
    for i in agent.keyNum.unique():
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.plot("state","diff_to_blue", data=agent[agent['keyNum']==i], label="red")
        ax1.set_ylabel('Red Ghost')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color = 'tab:green'

        ax2.scatter("state", "action", data=agent[agent['keyNum']==i], c="grey", label="action")
        ax2.set_ylim(0,5)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('Action')

        plt.title("Map of agent " + evalAgents[index])
        # call savefig right before show
        num = random.randint(1344576,95836265278)
        filename = str(num) + ".png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    


# In[53]:


for index, agent in enumerate(evalDFsList):
    temp = agent[agent['lives']==1]
    plt.scatter("state", "importance", data = temp, alpha = 0.25)
    plt.scatter("state", "importance", data = evalKeysDFsList[index], c = "b", alpha = 0.25)
    plt.xlabel('Frame Number')
    plt.ylabel('Importance')
    plt.title(evalAgents[index])
    
    plt.show()


# In[55]:


# for index, agent in enumerate(evalDFsList):
#     temp = agent[agent['lives']==2]
#     plt.scatter("state", "importance", data = temp, alpha = 0.25)
#     plt.scatter("state", "importance", data = evalKeysDFsList[index], c = "b", alpha = 0.25)
#     plt.xlabel('Frame Number')
#     plt.ylabel('Importance')
#     plt.title(evalAgents[index])
    
#     plt.show()


# In[54]:


for index, agent in enumerate(evalDFsList):
    temp = agent[agent['lives']==3]
    plt.scatter("state", "importance", data = temp, alpha = 0.25)
    plt.scatter("state", "importance", data = evalKeysDFsList[index], c = "b", alpha = 0.25)
    plt.xlabel('Frame Number')
    plt.ylabel('Importance')
    plt.title(evalAgents[index])
    
    plt.show()


# In[51]:


for index, agent in enumerate(evalDFsList):
    plt.subplot(1, 3, 1)
    temp = agent.loc[(agent['lives']==3) & (agent['importance'] > 0.15)]
    plt.scatter("state", "importance", data = temp, alpha = 0.25, c="importance")
    plt.scatter("state", "importance", data = evalKeysDFsList[index].loc[(evalKeysDFsList[index]['importance'] > 0.15)], c = "importance", alpha = 0.25)
    plt.xlabel('Frame Number')
    plt.ylabel('Importance')
    plt.title(evalAgents[index] + " 3 lives left")
    
    plt.subplot(1, 3, 2)
    temp = agent.loc[(agent['lives']==2) & (agent['importance'] > 0.15)]
    plt.scatter("state", "importance", data = temp, alpha = 0.25, c="importance")
    plt.scatter("state", "importance", data = evalKeysDFsList[index].loc[(evalKeysDFsList[index]['importance'] > 0.15)], c = "importance", alpha = 0.25)
    plt.xlabel('Frame Number')
    plt.ylabel('Importance')
    plt.title(evalAgents[index] + " 2 lives left")
    
    plt.subplot(1, 3, 3)
    temp = agent.loc[(agent['lives']==1) & (agent['importance'] > 0.15)]
    plt.scatter("state", "importance", data = temp, alpha = 0.25, c="importance")
    plt.scatter("state", "importance", data = evalKeysDFsList[index].loc[(evalKeysDFsList[index]['importance'] > 0.15)], c = "importance", alpha = 0.25)
    plt.xlabel('Frame Number')
    plt.ylabel('Importance')
    plt.title(evalAgents[index] + " 1 lives left")
    # call savefig right before show
    num = random.randint(1344576,95836265278)
    filename = str(num) + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# In[56]:


temp_df = pd.DataFrame()

for index, agent in enumerate(evalDFsList):
    for i in range(1,4):
        temp = agent.loc[(agent['lives']==i) & (agent['importance'] > 0.15)]
        tempTemp = pd.DataFrame(columns=['avg_import', 'agent', "lives"])
        tempTemp.loc[0] = [temp['importance'].mean(), evalAgents[index], i]
        temp_df = pd.concat([temp_df,tempTemp])


# In[59]:


temp_df2 = pd.DataFrame()
for index, agent in enumerate(evalOldDFsList):
    for i in range(1,4):
        temp = agent.loc[(agent['lives']==i) & (agent['importance'] > 0.15)]
        tempTemp = pd.DataFrame(columns=['avg_import', 'agent', "lives"])
        tempTemp.loc[0] = [temp['importance'].mean(), evalAgents[index], i]
        temp_df2 = pd.concat([temp_df2,tempTemp])


# In[61]:


for index, agent in enumerate(evalAgents):
    temp = temp_df[temp_df['agent']==agent]
    plt.plot("lives", "avg_import", data=temp)
    plt.title(agent)
    plt.show()
    
    temp = temp_df2[temp_df2['agent']==agent]
    plt.plot("lives", "avg_import", data=temp)
    plt.title(agent + "pre-location data")
    plt.show()


# In[ ]:


# temp_df2 = pd.DataFrame()

# for index, agent in enumerate(evalValidDFsList):
#     for i in range(1,4):
#         temp = agent.loc[(agent['lives']==i) & (agent['importance'] > 0.15)]
#         tempTemp = pd.DataFrame(columns=['avg_import', 'agent', "lives"])
#         tempTemp.loc[0] = [temp['importance'].mean(), evalAgents[index], i]
#         temp_df2 = pd.concat([temp_df2,tempTemp])


# In[ ]:





# In[ ]:





# In[9]:


# for index, agent in enumerate(evalAgents):
#     plt.subplot(1, 3, 1)
#     temp = evalOldDFsList[index]
#     plt.plot("episode", "importance", data=temp)
#     plt.title(agent + " Old")
#     plt.show()
    
#     plt.subplot(1, 3, 2)
#     temp = evalDFsList[index]
#     plt.plot("episode", "importance", data=temp)
#     plt.title(agent + " Now")
#     plt.show()
    
#     plt.subplot(1, 3, 3)
#     temp = evalValidDFsList[index]
#     plt.plot("episode", "importance", data=temp)
#     plt.title(agent + " Valid")
#     plt.show()


# In[10]:


# for index, agent in enumerate(evalAgents):
#     plt.subplot(1, 3, 1)
#     temp1 = evalDFsList[index]
#     temp2 = evalValidDFsList[index]
#     plt.plot(temp1['action'], temp2['action'])
#     plt.title(agent + " Old")
#     plt.show()


# In[11]:


# temp1 = evalDFsList[0]
# temp2 = evalValidDFsList[0]

# temp1['action'] == temp2['action']


# In[12]:


# evalKeysOldDFsList


# In[13]:



# temp1 = evalValidKeysDFsList[2]
# temp2 = evalKeysDFsList[2]
# temp1[temp1["key_state"] == 1]


# In[14]:


# temp2[temp2["key_state"] == 1]


# In[15]:


# for index, agent in enumerate(evalValidKeysDFsList):
#     for i in range(1,8):
#         temp1 = evalValidKeysDFsList[index]
#         temp2 = evalKeysDFsList[index]
#         plt.plot(temp1[temp1["keyNum"]==i]['state'], temp1[temp1["keyNum"]==i]['action'])
#         plt.plot(temp2[temp2["keyNum"]==i]['state'], temp2[temp2["keyNum"]==i]['action'])
#         plt.show()


# In[ ]:




