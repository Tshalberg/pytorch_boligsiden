import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

import numpy as np

from models import regression_model

from data_module import model_data as md

def train_model(features, hyperparameters):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hyperparameters["device"] = device
    seed_everything(hyperparameters["seed"])
    # device = "cpu"
    print (f"Device: {device}")
    torch.device(device)
    fp = "data/data_curated.parquet"

    dataset_train, dataset_val, dataset_test = md.get_datasets(fp, device=device, 
                                                            numeric_columns=features,
                                                            kommune=None,
                                                            N_top_kommuner=hyperparameters["N_top_kommuner"],
                                                            outlier_level= hyperparameters["outlier_level"])

    train_dataloader = DataLoader(dataset_train, batch_size=hyperparameters["batchsize"])
    val_dataloader = DataLoader(dataset_val, batch_size=hyperparameters["batchsize"])
    test_dataloader = DataLoader(dataset_test, batch_size=hyperparameters["batchsize"])

    wandb_logger = WandbLogger(project="Bayesian Optimization")

    experiment = wandb_logger.experiment
    experiment_name = experiment.name
    # wandb_logger = False
    # experiment_name = "test"

    # early stopping    
    patience = 5
    monitor = "val_loss_mse"
    mode = "min"

    wandb_logger.log_hyperparams(hyperparameters)
    model = regression_model.RegressionModel(**hyperparameters)
    model = model.cuda()
    early_stop_callback = EarlyStopping(monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode)
    # checkpoint_callback = ModelCheckpoint(dirpath=f"D:/Pytorch/boligsiden/models/bayesian_test/{experiment_name}/", save_top_k=2, monitor="val_loss_mape",
    #                                     filename="model-{epoch:02d}-{val_loss_mape:.3f}")
    callbacks = [early_stop_callback] #checkpoint_callback
    max_epochs = None
    gradient_clip_val = 0.5
    if device != "cuda":
        trainer = pl.Trainer(callbacks=callbacks, logger=wandb_logger, max_epochs=max_epochs, gradient_clip_val=gradient_clip_val)
    else:
        trainer = pl.Trainer(gpus=1, callbacks=callbacks, logger=wandb_logger, max_epochs=max_epochs, gradient_clip_val=gradient_clip_val)

    trainer.fit(model, train_dataloader, val_dataloader)
    model = model.cuda()
    try:
        y_predict = model(val_dataloader.dataset.dataset)
        wandb.finish()
        model.freeze()
        val_loss_MAPE = regression_model.MAPE_loss(y_predict, val_dataloader.dataset.y)
        val_loss_MAPE = float(val_loss_MAPE.cpu().data.numpy())
    except:
        breakpoint()
    return val_loss_MAPE

features = ["kommune_encoded", "sqrm", "year_encoded", 
            "city_encoded", "værelser", "floor_encoded", 
            'Tog', 'Bus', 'Sø', 'Skov', 'Kyst', 'Hospital']

from bayes_opt import BayesianOptimization, UtilityFunction

# pbounds = {"batchsize": [5, 4000]}
pbounds = {"n_hidden1": [1, 50]}

optimizer = BayesianOptimization(f = None, 
                                 pbounds = pbounds, 
                                 verbose = 2, random_state = 1234)

                                #  "n_hidden1": [50, 4000],
                                            # "n_hidden2": [50, 4000],
# Specify the acquisition function (bayes_opt uses the term
# utility function) to be the upper confidence bounds "ucb".
# We set kappa = 1.96 to balance exploration vs exploitation.
# xi = 0.01 is another hyper parameter which is required in the
# arguments, but is not used by "ucb". Other acquisition functions
# such as the expected improvement "ei" will be affected by xi.
utility = UtilityFunction(kind = "ucb", kappa = 1.96, xi = 0.01)

hyperparameters = {
                "n_in":len(features),
                "n_hidden1":None,
                "n_hidden2":None,
                "learning_rate":0.0001,
                "batchsize":2048,
                "seed":666,
                "dropout_rate":0.1,
                "N_top_kommuner":None,
                "outlier_level":None
            }

# np.random.seed(1337)
hyperparameters["seed"] = 30765
hyperparameters["n_hidden1"] = 10
hyperparameters["n_hidden2"] = 10
target = train_model(features, hyperparameters)
for dropout_rate in [0.2, 0.5]:
    for N in [10, 100, 1000, 2000]:
        for i in range(10):
            hyperparameters["seed"] = np.random.randint(1, 100000)
            hyperparameters["n_hidden1"] = N
            hyperparameters["n_hidden2"] = N
            hyperparameters["dropout_rate"] = dropout_rate
            target = train_model(features, hyperparameters)

# for i in range(20):
    # Get optimizer to suggest new parameter values to try using the
    # specified acquisition function.
    # try:
    # next_point = optimizer.suggest(utility)
    # except:
        # breakpoint()
    # Force degree from float to int.
    # hyperparameters["n_hidden1"] = int(next_point["n_hidden1"])
    # hyperparameters["n_hidden2"] = int(next_point["n_hidden2"])
    # hyperparameters["batchsize"] = int(next_point["batchsize"])
    # breakpoint()
    # Evaluate the output of the black_box_function using 
    # the new parameter values.
    # target = train_model(features, hyperparameters)
    # make problem into minimization instead of maximization
    # if target > 1.0:
    #     print (f"Target is {target}. Skipping optimization...")
    #     continue
    # target = -target
    # try:
    #     # Update the optimizer with the evaluation results. 
    #     # This should be in try-except to catch any errors!
    #     optimizer.register(params = next_point, target = target)
    # except:
    #     pass
    # breakpoint()

# breakpoint()
# for n_hidden_1 in [200, 400, 600, 800]:
#     # n_hidden_1 = 200
#     hyperparameters = {
#                         "n_in":len(features),
#                         "n_hidden1":n_hidden_1,
#                         "n_hidden2":800,
#                         "learning_rate":0.0001,
#                         "batchsize":1024,
#                         "seed":666,
#                         "dropout_rate":0.1,
#                         "N_top_kommuner":None,
#                         "outlier_level":None
#                     }

#     val_loss_MAPE = train_model(features, hyperparameters)
#     breakpoint()

