import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

def MAPE_loss(y_pred, y):
    loss = (y - y_pred).abs() /( (y.abs() + y_pred.abs())/2 )
    loss[loss == float('inf')] = 0
    loss = torch.mean(loss)
    return loss

class RegressionModel(pl.LightningModule):
    def __init__(self, n_in, n_hidden1, n_hidden2, learning_rate, device="cpu"):
        super().__init__()
        self.training_device = device
        self.learning_rate = learning_rate
        self.input_layer = nn.Linear(n_in, n_hidden1)
        self.hidden_layer_1 = nn.Linear(n_hidden1, n_hidden2)
        self.hidden_layer_2 = nn.Linear(n_hidden2, n_hidden2)
        self.output_layer = nn.Linear(n_hidden2, 1)
        self.activation = torch.relu
        self.dropout = nn.Dropout(0.5)
        
    
    def forward(self, x):
        x = x.to(self.training_device)
        y_pred = self.activation(self.input_layer(x).to(self.training_device))
        y_pred = self.activation(self.hidden_layer_1(y_pred))
        y_pred = self.dropout(y_pred)
        y_pred = self.activation(self.hidden_layer_2(y_pred))
        y_pred = self.dropout(y_pred)
        y_pred = self.output_layer(y_pred)
        return y_pred

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_pred = self.forward(x)
        y_pred = y_pred.to(self.training_device)
        loss_mse = F.mse_loss(y_pred, y)
        self.log("train_loss_mse", loss_mse)
        loss_mape = MAPE_loss(y_pred, y)
        self.log("train_loss_mape", loss_mape)
        # return loss_mse
        return loss_mape

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss_mse = F.mse_loss(y_pred, y)
        self.log("val_loss_mse", loss_mse)
        loss_mape = MAPE_loss(y_pred, y)
        self.log("val_loss_mape", loss_mape)
        # return loss_mse
        return loss_mape

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
