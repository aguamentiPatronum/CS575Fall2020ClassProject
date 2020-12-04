"""
Make the imports of python packages needed
"""
import pandas as pd
import numpy as np
import os
import matplotlib as matplotlib
from tabulate import tabulate
import pyarrow as pa
import pyarrow.parquet as pq
from io import StringIO


class dbMaker():

    complete_df = pd.DataFrame()
    tempTopDir = os.path.abspath('/Volumes/Britt_SSD/Singularity2/eval/PacmanNmlPillsInGame')

    def __init__(self):
        print("Initialized")
        
    def pullCSVs(self):
        # For training data, need CSVs folder only (maybe make this the same for all?)
        main_file = os.path.join(self.tempTopDir, 'df1main.csv')
        q_file = os.path.join(self.tempTopDir, 'df3q_values_df.csv')
        obs_file = os.path.join(self.tempTopDir, 'df4obs_df.csv')
        arg_file = os.path.join(self.tempTopDir, 'df5argmax_df.csv')
        
        self.combineCSV(main_file, q_file, obs_file, arg_file)
        return
        
    def addEpochScores(self):
        # get scores for epochs from text document
        scores_path = os.path.join(self.tempTopDir, 'scores.txt')
        
        f = open(scores_path, 'r')
        data_txt = f.read()
        data_txt = data_txt.replace('[', '')
        data_txt = data_txt.replace(']', '')
        print(data_txt)
        epoch_scores = np.fromstring(data_txt, dtype=float, sep=',')
        print("scores")
        for item in epoch_scores:
            print(item)
        
        indices = self.complete_df.index[self.complete_df['end of epoch']].tolist()
        indices.append(len(self.complete_df) - 1)
        print("indices")
        print(indices)
        
        key_states = np.load(os.path.join(self.tempTopDir, 'summary_states.npy')).tolist()
        temp_context_states = np.load(os.path.join(self.tempTopDir, 'summary_states_with_context.npy'))
        print(key_states)
        print(temp_context_states)
        context_states = [i for i in temp_context_states if i not in key_states]
        
        self.complete_df['epoch_score'] = 0
        self.complete_df['key_state'] = False
        self.complete_df['context_state'] = False
        
        #if we are going to iterate, let's just iterate once
        all_indices = np.hstack((indices, key_states, context_states))
        print("all indices; ")
        print(all_indices)
        
        for state in all_indices:
            if ((state < len(self.complete_df)) and (state > 0)) :
                # if it is the end of an epoch, fill in the score
                if state in indices:
                    print("Found item in indices")
                    list_index = indices.index(state)
                    if (self.complete_df.iloc[state]['epoch_score'] == 0):
                        self.complete_df['epoch_score'][state] = epoch_scores[list_index-1]
                        print(self.complete_df.iloc[state]['epoch_score'])
                # if is a context state, fill in that column
                if state in context_states:
                    print("found item in context and state it: " + str(state))
                    print("columns are ")
                    print(self.complete_df.columns)
                    print("DF length: ")
                    print(len(self.complete_df))
                    print("Context States are: ")
                    print(context_states)
                    context_index = context_states.index(state)
                    print(self.complete_df.iloc[state])
                    if (self.complete_df.iloc[state]['context_state'] == False):
                        print("false found")
                        self.complete_df['context_state'][state] = True
                        print(self.complete_df.iloc[state]['context_state'])
                # if is a key state, fill in that column
                if state in key_states:
                    print("found item in keys")
                    key_index = key_states.index(state)
                    if (self.complete_df.iloc[state]['key_state'] == False):
                        print("false found")
                        self.complete_df['key_state'][state] = True
                        print(self.complete_df.iloc[state]['key_state'])

        
#        print(self.complete_df.to_markdown())
        return
        
    def checkImport(self):
        import_temp = pd.read_csv(os.path.join(self.tempTopDir, 'state_features_impoartance.csv'), index_col=0)
        print(import_temp.columns)
        print(self.complete_df.columns)
        import_temp.sort_values('state',inplace=True)
        
        self.complete_df = pd.merge(left=self.complete_df, right=import_temp, on='state')
        print(self.complete_df.columns)
#        print(import_temp['state'].to_markdown())
        return
        
    def combineCSV(self, main_file, q_file, obs_file, arg_file):
        first_df = pd.read_csv(main_file)
        print("1")
#        print(first_df.head(3).to_markdown())
        #first_df.head(10).to_csv('temp.csv')
        
        q_df = pd.read_csv(q_file)
        print("2")
#        print(q_df.head(3))
        self.complete_df = pd.merge(first_df, q_df)
        
        self.complete_df.drop(["info", "action 5 episode sum", "action 5 total sum", "action 6 episode sum", "action 6 total sum", "action 7 episode sum", "action 7 total sum", "action 8 episode sum", "action 8 total sum"], axis=1, inplace=True)

        obs_df = pd.read_csv(obs_file)
        print("3")
#        print(q_df.head(3))
        self.complete_df = pd.merge(self.complete_df, obs_df)
        
        arg_df = pd.read_csv(arg_file)
        print("3")
#        print(q_df.head(3))
        self.complete_df = pd.merge(self.complete_df, arg_df)

#        print(self.complete_df.head(3).to_markdown())
        
        
#        print(self.complete_df.head(5).to_markdown())
        print(self.complete_df.episode.unique)
        print(self.complete_df.epoch.unique)
        
        print("columns: ")
        print(self.complete_df.columns)
        self.checkImport()
        
        self.addEpochScores()
        return
    
        
        
    def saveDF(self):
        #Sanity Check file
        self.complete_df.tail(10).to_csv('temp.csv')
        #Save to parquet temporary file
        # Convert DataFrame to Apache Arrow Table
        table = pa.Table.from_pandas(self.complete_df)
        # Parquet with Brotli compression
        pq.write_table(table, 'NmlPillsInGameEvalDB.parquet', compression='BROTLI')
        # And save to CSV for permanent storage
        self.complete_df.to_csv('NmlPillsInGameEvalDB.csv')
        return
        
    def testPQ(self):
        temp_test = pd.read_parquet('NmlPillsInGameEvalDB.parquet')
        print(temp_test['state'].iloc[-1])


def main():
    print("Hello World!")
    
    # Make an arg parser, and then  pass in the singularity/training/PacmanX folder
    
    # make a dbmaker object
    builder = dbMaker()
    # Send the top-level singularity folder to the pullCSVs fxn
    builder.pullCSVs()
    # and save created db to necessary file formats
    builder.saveDF()
    builder.testPQ()

if __name__ == "__main__":
    main()
