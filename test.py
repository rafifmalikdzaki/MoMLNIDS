import numpy as np
import pandas as pd
import csv
from torch.utils.data import DataLoader
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from pathlib import Path


#%%
from skripsi_code.utils.domain_dataset import MultiChunkDataset
from skripsi_code.utils.dataloader import random_split_dataloader
#%%
DATA_PATH = "./skripsi_code/data/interim/"
DATA_LIST = ["NF-BoT-IoT-v2",
             "NF-CSE-CIC-IDS2018-v2",
             "NF-ToN-IoT-v2",
             "NF-UNSW-NB15-v2"]
#%%

if __name__ == '__main__':
    train_data = MultiChunkDataset(DATA_PATH, DATA_LIST, domain=DATA_LIST)
    train_dataloader = DataLoader(train_data, batch_size=1024, shuffle=False,pin_memory=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    for X, Y in tqdm(train_dataloader, total=len(train_dataloader)):
        continue
