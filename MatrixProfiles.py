import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matrixprofile as mp
from matplotlib.patches import Rectangle
import stumpy as sp

if __name__ == "__main__":
    FearDF = pd.read_parquet('/Volumes/Britt_SSD/Singularity/TrainingSlimmed/FearGhostsTrainingDB.parquet')
    FearDF = FearDF.drop(columns=FearDF.columns[0])
    NormalOnlyDF = pd.read_parquet('/Volumes/Britt_SSD/Singularity/TrainingSlimmed/NormalPillsOnlyDB.parquet')
    NormalOnlyDF = NormalOnlyDF.drop(columns=NormalOnlyDF.columns[0])
    NormalPlusDF = pd.read_parquet('/Volumes/Britt_SSD/Singularity/TrainingSlimmed/NormalPillsInGameDB.parquet')
    NormalPlusDF = NormalPlusDF.drop(columns=NormalPlusDF.columns[0])
    PowerDF = pd.read_parquet('/Volumes/Britt_SSD/Singularity/TrainingSlimmed/PowerPillsOnlyDB.parquet')
    PowerDF = PowerDF.drop(columns=PowerDF.columns[0])
    StdDF = pd.read_parquet('/Volumes/Britt_SSD/Singularity/TrainingSlimmed/StandardScoringOnlyDB.parquet')
    StdDF = StdDF.drop(columns=StdDF.columns[0])

    df_array = [FearDF, NormalOnlyDF, NormalPlusDF, PowerDF, StdDF]
    names_array = ['fear', 'normalOnly', 'normalPlus', 'power', 'standard']
    
    actions_sequence = np.array(FearDF['action'], dtype=int)
    actions_sequence[:20]
    print(actions_sequence.dtype)
