import pandas as pd
import numpy as np
from statics import statics
from data_module import mongodb as mdb

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset

from data_module import curation

STATICS = statics.StaticVars()
torch.device("cuda")
def fetch_merged_data_mongo():
    """Function to fetch Boligsiden data from MongoDB

    Returns:
        Dataframe: Merged data from Boligsiden
    """

    # get client for mongodb
    client = mdb.mongodb_client()

    # fetch raw data
    data_raw = mdb.fetch_all_data_mongodb(client, STATICS.BoligsidenDatabase, STATICS.BoligsidenDataRaw) 

    # fetch address info
    data_address = mdb.fetch_all_data_mongodb(client, STATICS.BoligsidenDatabase, STATICS.BoligsidenDataAddress) 

    # remove nan-filled columns
    data_address = data_address.dropna(thresh=0.5*len(data_address), axis=1)

    # merge data
    data_merged = data_raw.merge(data_address, on="address_id")

    return data_merged

class CustomDataset(Dataset):
    def __init__(self, X, y, numeric_columns=None, scaler_x=None, scaler_y=None, device="cuda"):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.X_original = X
        self.y_original = y

        if numeric_columns is not None:
            X = X[numeric_columns]

        X = X.values
        if scaler_x is not None:
            X = scaler_x.transform(X)
  
        self.dataset = torch.tensor(X, device=device, dtype=torch.float32)

        y = y.values.reshape(len(y),1)
        if scaler_y is not None:
            y = scaler_y.transform(y)

        self.y = torch.tensor(y, device=device, dtype=torch.float32)

    def __getitem__(self, index):
        return (self.dataset[index], self.y[index])

    def __len__(self):
        return self.dataset.shape[0]


def get_datasets(device="cuda", kommune=None, numeric_columns=None, N_top_kommuner=None, outlier_level=None):
    assert numeric_columns is not None

    dataset = pd.read_parquet(r"C:\Users\Thomas\Documents\Projects\pytorch_boligsiden\data\data_curated.parquet")
    dataset = dataset[dataset["year"].isin(list(np.arange(2009, 2022)))]
    if kommune is not None:
        dataset = dataset[dataset["kommune"] == kommune]
        if "kommune_encoded" in numeric_columns:
            numeric_columns.remove("kommune_encoded")
    else:
        if N_top_kommuner is not None:
            top_kommuner = dataset["kommune"].value_counts().index[:N_top_kommuner]
            dataset = dataset[dataset["kommune"].isin(top_kommuner)]

    if outlier_level is not None:
        mask = curation.get_outlier_mask(dataset, columns=["price_sqrm"], level=1)
        dataset = dataset[mask]

    target = "price"
    X = dataset[dataset.columns.drop(target)]
    y = dataset[target]

    random_state = 42
    train_val_test_split = [0.6, 0.2, 0.2]

    train_test_size =  round(train_val_test_split[1] + train_val_test_split[2], 2)
    test_val_size = round(train_val_test_split[2]/ (train_val_test_split[1] + train_val_test_split[2]), 2)

    print ( train_test_size )
    print (test_val_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_val_size, random_state=random_state) 

    print (X_train.shape, X_val.shape, X_test.shape)
    print (y_train.shape, y_val.shape, y_test.shape)

    X_numeric = X[numeric_columns].values
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_numeric)
    X_scaled = scaler_x.transform(X_numeric)

    y_numeric = y.values.reshape(len(y),1)
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_numeric)
    y_scaled = scaler_y.transform(y_numeric)

    dataset_train = CustomDataset(X_train, y_train, numeric_columns=numeric_columns,
                                  scaler_x=scaler_x, scaler_y=scaler_y, device=device)
                                  
    dataset_val   = CustomDataset(X_val,   y_val,   numeric_columns=numeric_columns,
                                  scaler_x=scaler_x, scaler_y=scaler_y, device=device)

    dataset_test  = CustomDataset(X_test,  y_test,  numeric_columns=numeric_columns,
                                  scaler_x=scaler_x, scaler_y=scaler_y, device=device)

    return dataset_train, dataset_val, dataset_test




