from sklearn.model_selection import KFold
from loguru import logger
import time
from functools import partial
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import optuna
from loguru import logger


def get_params_deep(trial, model):
    """Gets model type and returns parameters for optimization

    Args:
        trial (trial): Optuna trial
        model (str): Model type

    Returns:
        parameters (dict): Parameters for model
    """

    if model == "fcnn":
        return {
            "batch_size": trial.suggest_int("batch_size", 4, 256, step=4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "dim": trial.suggest_int("dim", 50, 150, step=10),
            "n_layers": trial.suggest_categorical("n_layers", [4,5])
        }
    
    elif model == "gcnn":
        return {
            "batch_size": trial.suggest_int("batch_size", 4, 256, step=4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        }
    else: 
        logger.error("Wrong type of model")

    
class FCNN(nn.Module):
    def __init__(self, n_layers: int = 5, hidden_size: int = 150, in_hw: int = 32, dp: int = 0.3):
        super(FCNN, self).__init__()
        self.kwargs = {'n_layers': n_layers, 'hidden_size': hidden_size, 'in_hw': in_hw, 'dp': dp}
        self.n_layers = n_layers
        if self.n_layers == 4:
            self.linear1 = nn.Linear(in_hw, 2*hidden_size)
            self.linear2 = nn.Linear(2*hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, hidden_size//2)
            self.linear4 = nn.Linear(int(hidden_size/2), 1)
            self.batch = torch.nn.BatchNorm1d(2*hidden_size)
        elif self.n_layers == 5:
            self.linear1 = nn.Linear(in_hw, 2*hidden_size)
            self.linear2 = nn.Linear(2*hidden_size, 3*hidden_size)
            self.linear3 = nn.Linear(3*hidden_size, 2*hidden_size)
            self.linear4 = nn.Linear(2*hidden_size, hidden_size//2)
            self.linear5 = nn.Linear(int(hidden_size/2), 1)
            self.batch = torch.nn.BatchNorm1d(3*hidden_size)
        self.dropout = torch.nn.Dropout(dp)
        
    def forward(self, x):
        if self.n_layers == 4:
            y_pred = F.relu(self.batch(self.linear1(x)))
            y_pred = F.relu(self.linear2(y_pred))
            y_pred = self.dropout(y_pred)
            y_pred = F.relu(self.linear3(y_pred))
            y_pred = self.dropout(y_pred)
            y_pred = F.relu(self.linear4(y_pred))
        elif self.n_layers == 5:
            y_pred = F.relu(self.linear1(x))
            y_pred = self.dropout(y_pred)
            y_pred = F.relu(self.batch(self.linear2(y_pred)))
            y_pred = F.relu(self.linear3(y_pred))
            y_pred = self.dropout(y_pred)
            y_pred = F.relu(self.linear4(y_pred))
            y_pred = self.dropout(y_pred)
            y_pred = self.linear5(y_pred)
        return y_pred
