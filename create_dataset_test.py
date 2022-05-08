import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset

dataset = pd.read_parquet("data/data_curated.parquet")

target = "price"
# breakpoint()
numeric_columns = ["kommune_encoded", "sqrm", "year_encoded", "city_encoded", "værelser", "floor_encoded", 'Tog', 'Bus', 'Sø', 'Skov', 'Kyst', 'Hospital']
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

X_scaled_r = scaler_x.inverse_transform(X_scaled)
y_scaled_r = scaler_y.inverse_transform(y_scaled)

allclose_X = np.allclose(X_numeric, X_scaled_r)
print (f"allclose X: {allclose_X}")
assert allclose_X

allclose_y = np.allclose(y_numeric, y_scaled_r)
print (f"allclose X: {allclose_y}")
assert allclose_y


# set device to cuda is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (f"Device: {device}")
torch.device(device)

# breakpoint()

class CustomDataset(Dataset):
    def __init__(self, X, y, numeric_columns=None, scaler_x=None, scaler_y=None):
        if numeric_columns is not None:
            X = X[numeric_columns]

        X = X.values
        if scaler_x is not None:
            X = scaler_x.transform(X)
  
        self.dataset = torch.tensor(X, device=device, dtype=torch.float32).cuda()

        y = y.values.reshape(len(y),1)
        if scaler_y is not None:
            y = scaler_y.transform(y)

        self.y = torch.tensor(y, device=device, dtype=torch.float32).cuda()

    def __getitem__(self, index):
        return (self.dataset[index], self.y[index])

    def __len__(self):
        return self.dataset.shape[0]
    

dataset_train = CustomDataset(X_train, y_train, numeric_columns=numeric_columns,scaler_x=scaler_x, scaler_y=scaler_y)
dataset_val   = CustomDataset(X_val,   y_val,   numeric_columns=numeric_columns,scaler_x=scaler_x, scaler_y=scaler_y)
dataset_test  = CustomDataset(X_test,  y_test,  numeric_columns=numeric_columns,scaler_x=scaler_x, scaler_y=scaler_y)

X_train_numeric = X_train[numeric_columns].values
X_train_dataset = scaler_x.inverse_transform(dataset_train.dataset.cpu().numpy())
allclose_train = np.allclose(X_train_numeric, X_train_dataset, atol=10)
assert allclose_train

X_val_numeric = X_val[numeric_columns].values
X_val_dataset = scaler_x.inverse_transform(dataset_val.dataset.cpu().numpy())
allclose_val = np.allclose(X_val_numeric, X_val_dataset, atol=10)
assert allclose_val

X_test_numeric = X_test[numeric_columns].values
X_test_dataset = scaler_x.inverse_transform(dataset_test.dataset.cpu().numpy())
allclose_test = np.allclose(X_test_numeric, X_test_dataset, atol=10)
assert allclose_test
