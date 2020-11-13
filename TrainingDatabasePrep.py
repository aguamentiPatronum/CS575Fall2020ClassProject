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


class dbMaker():

    complete_df = pd.DataFrame()

    def __init__(self):
        print("Initialized")
        
    def pullCSVs(self):
        tempTopDir = os.path.abspath('/Volumes/Britt_SSD/Singularity2/Training/PacmanPowerPills')
        
        # For training data, need CSVs folder only (maybe make this the same for all?)
        main_file = os.path.join(tempTopDir, 'CSVs/df1.csv')
        secondary_file = os.path.join(tempTopDir, 'CSVs/df3.csv')
        
        self.makeDFs(main_file, secondary_file)
        return
        
        
    def makeDFs(self,main_file, secondary_file):
        first_df = pd.read_csv(main_file)
#        print("1")
#        print(first_df.head(3).to_markdown())
#        first_df.head(10).to_csv('temp.csv')
        second_df = pd.read_csv(secondary_file)
#        print("2")
#        print(second_df.head(3))
        self.complete_df = pd.merge(first_df, second_df)
        
#        print(self.complete_df.head(3).to_markdown())
        self.complete_df.drop(["mean reward", "action 5 episode sum", "action 5 total sum", "action 6 episode sum", "action 6 total sum", "action 7 episode sum", "action 7 total sum", "action 8 episode sum", "action 8 total sum"], axis=1, inplace=True)
        self.complete_df.head(10).to_csv('temp.csv')
        print(self.complete_df.describe())
        return
        
    def saveDF(self):
        #Sanity Check file
        self.complete_df.head(10).to_csv('temp.csv')
        #Save to parquet temporary file
        # Convert DataFrame to Apache Arrow Table
        table = pa.Table.from_pandas(self.complete_df)
        # Parquet with Brotli compression
        pq.write_table(table, 'PowerPillsOnlyDB.parquet', compression='BROTLI')
        # And save to CSV for permanent storage
        self.complete_df.to_csv('PowerPillsOnlyDB.csv')
        return
        
    def testPQ(self):
        temp_test = pd.read_parquet('PowerPillsOnlyDB.parquet')
        print(temp_test.head(3).to_markdown())


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
