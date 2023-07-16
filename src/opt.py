from shallow_models import get_params, get_model
from deep_models import get_params_deep, FCNN, GCNN, GraphFeaturizer, GraphDataset
from torch_geometric.loader import DataLoader as GraphDataLoader
import train_deep
from seed import seedEverything

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from loguru import logger
import numpy as np
from sklearn.linear_model import Ridge
import pathlib
import shutil
import tempfile
import torch
from torch.utils.data import TensorDataset, DataLoader


def objective(trial, model_type, train_x, train_y, model_seed, data_seed):
    """Main function for optuna optimization (shallow models)

    Args:
        trial (trial): Optuna trial
        model_type (str): Type of trained model
        train_x (dataframe): Data for train (features)
        train_y (dataframe): Target 
        model_seed (int): Seed for model
        data_seed (int): Seed for hyperaprameter optimization

    Returns:
        mse(float): metrice for test data
    """
    seedEverything(model_seed)
    params = get_params(trial,model_type)

    second_splits = KFold(n_splits=5, shuffle=True, random_state=data_seed)
    mses = 0
    logger.info(F'Parameters: {params}')
    for fold, (train_val_idx,val_idx) in enumerate(second_splits.split(np.arange(len(train_x)))):
            logger.info(F'Fold {fold+1} - data split hyperparameter optimization')
            train_val_x = train_x.iloc[train_val_idx]
            val_x = train_x.iloc[val_idx]
            train_val_y = train_y.iloc[train_val_idx]
            val_y = train_y.iloc[val_idx]
            logger.info(F'There are {len(train_val_x)} train and {len(val_x)} validation examples')
            #Train
            model = get_model(model_type, params)
            model.fit(train_val_x, train_val_y)
            mses += mean_squared_error(val_y, model.predict(val_x))
    mse = mses/5
    logger.info(F'Mean MSE: {mse:.4f}')
    return mse

def objective_deep_fcnn(trial, model_type, train_x, train_y, model_seed, data_seed, epochs):
    """Main function for optuna optimization (FCNN)

    Args:
        trial (trial): Optuna trial
        model_type (str): Type of trained model
        train_x (dataframe): Data for train (features)
        train_y (dataframe): Target 
        model_seed (int): Seed for model
        data_seed (int): Seed for hyperaprameter optimization

    Returns:
        mse(float): metrice for test data
    """
    seedEverything(model_seed)
    params = get_params_deep(trial,model_type)

    second_splits = KFold(n_splits=5, shuffle=True, random_state=data_seed)
    mses = 0
    logger.info(F'Parameters: {params}')
    for fold, (train_val_idx,val_idx) in enumerate(second_splits.split(np.arange(len(train_x)))):
            logger.info(F'Fold {fold+1} - data split hyperparameter optimization')
            train_val_x = train_x.iloc[train_val_idx]
            val_x = train_x.iloc[val_idx]
            train_val_y = train_y.iloc[train_val_idx]
            val_y = train_y.iloc[val_idx]
            logger.info(F'There are {len(train_val_x)} train and {len(val_x)} validation examples')
            # Prepare data                               
            train_dataset = TensorDataset(torch.FloatTensor(train_val_x.values), torch.FloatTensor(train_val_y.values.reshape(-1, 1)))
            if(len(train_dataset)%params['batch_size'] == 1): drop_last = True
            else: drop_last = False
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last)

            valid_dataset = TensorDataset(torch.FloatTensor(val_x.values), torch.FloatTensor(val_y.values.reshape(-1, 1)))
            valid_loader = DataLoader(valid_dataset, batch_size=val_y.values.shape[0], shuffle=False)

            #Train
            in_hw = train_val_x.shape[1]
            _, score = train_deep.train_fcnn(train_loader, valid_loader, in_hw, params, epochs, mode='optimization')
            mses += score
    mse = mses/5
    logger.info(F'Mean MSE: {mse:.4f}')
    return mse

def objective_deep_gcnn(trial, model_type, train_x, train_y, model_seed, data_seed, epochs):
    """Main function for optuna optimization (GCNN)

    Args:
        trial (trial): Optuna trial
        model_type (str): Type of trained model
        train_x (dataframe): Data for train (features)
        train_y (dataframe): Target 
        model_seed (int): Seed for model
        data_seed (int): Seed for hyperaprameter optimization

    Returns:
        mse(float): metrice for test data
    """
    seedEverything(model_seed)
    params = get_params_deep(trial,model_type)

    second_splits = KFold(n_splits=5, shuffle=True, random_state=data_seed)
    mses = 0
    logger.info(F'Parameters: {params}')
    for fold, (train_val_idx,val_idx) in enumerate(second_splits.split(np.arange(len(train_x)))):
            logger.info(F'Fold {fold+1} - data split hyperparameter optimization')
            train_val_x = train_x.iloc[train_val_idx]
            val_x = train_x.iloc[val_idx]
            train_val_y = train_y.iloc[train_val_idx]
            val_y = train_y.iloc[val_idx]
            logger.info(F'There are {len(train_val_x)} train and {len(val_x)} validation examples')
            # Prepare data
            featurizer = GraphFeaturizer(y_column='target')
            train_graphs = featurizer(train_val_x)
            val_graphs = featurizer(val_x)
            temp_dir = tempfile.mkdtemp()
            train_dataset = GraphDataset(train_graphs, train_val_y, root=pathlib.Path(temp_dir)/'train')
            valid_dataset = GraphDataset(val_graphs, val_y, root=pathlib.Path(temp_dir)/'val')
            shutil.rmtree(temp_dir, ignore_errors=True)
            train_loader = GraphDataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
            valid_loader = GraphDataLoader(valid_dataset, batch_size=val_y.values.shape[0], shuffle=False)
            #Train
            _, score = train_deep.train_gcnn(train_loader, valid_loader, params, epochs, mode='optimization')
            mses += score
    mse = mses/5
    logger.info(F'Mean MSE: {mse:.4f}')
    return mse

    
