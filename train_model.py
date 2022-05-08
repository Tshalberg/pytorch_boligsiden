import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger

from models import regression_model

from data_module import model_data as md

columns = ["kommune_encoded", "sqrm", "year_encoded", 
            "city_encoded", "værelser", "floor_encoded", 
            'Tog', 'Bus', 'Sø', 'Skov', 'Kyst', 'Hospital']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"
print (f"Device: {device}")
torch.device(device)

dataset_train, dataset_val, dataset_test = md.get_datasets(device=device, 
                                                           numeric_columns=columns,
                                                           kommune=None)

batchsize = 256
train_dataloader = DataLoader(dataset_train, batch_size=batchsize)
val_dataloader = DataLoader(dataset_val, batch_size=batchsize)
test_dataloader = DataLoader(dataset_test, batch_size=batchsize)


wandb_logger = WandbLogger(project="Pytorch Lightning Test")

experiment = wandb_logger.experiment
experiment_name = experiment.name

# breakpoint()
seed_everything(42)

# init model
n_in = len(columns)
n_hidden1 = 100
n_hidden2 = 200
learning_rate = 0.00005

# early stopping    
patience = 5
monitor = "val_loss_mape"
mode = "min"

hyperparameters = {
                    "n_hidden1":n_hidden1,
                    "n_hidden_2":n_hidden2,
                    "learning_rate":learning_rate,
                    "batchsize":batchsize
                }

wandb_logger.log_hyperparams(hyperparameters)

regression_model = regression_model.RegressionModel(n_in, n_hidden1, n_hidden2, learning_rate)

early_stop_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode)
checkpoint_callback = ModelCheckpoint(dirpath=f"data/models/{experiment_name}/", save_top_k=2, monitor="val_loss_mape",
                                      filename="model-{epoch:02d}-{val_loss_mape:.2f}")
callbacks = [early_stop_callback, checkpoint_callback]

if device == "cuda":
    trainer = pl.Trainer(callbacks=callbacks, logger=wandb_logger)
else:
    trainer = pl.Trainer(gpus=1, callbacks=callbacks, logger=wandb_logger)

trainer.fit(model=regression_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
