from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from loguru import logger
import time
from functools import partial
import pickle
import numpy as np
import optuna

import opt 
from deep_models import FCNN
from seed import seedEverything
from eval import evaluate
from plots import plot_train

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_fcnn(train_loader, test_loader, in_hw, params, epochs, mode='train'):
    best_value = 1000

    if mode != 'train':
        epochs = epochs//2
    else:
        writer = SummaryWriter()

    device = torch.device('cuda:0')
    
    lr = params["learning_rate"]
            
    model = FCNN(params["n_layers"], params["dim"], in_hw,  params["dropout"])
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    train_losses = []
    test_losses = []
    train_r2s = []
    test_r2s = []

    logger.info("Train FCNN")

    for epoch in range(1, epochs+1):

        logger.info(f'################## EPOCH {epoch} ##################')

        model.train()
        train_correct = 0
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += r2_score(target.cpu().detach().numpy(), output.cpu().detach().numpy())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_r2 = train_correct/len(train_loader)
        train_r2s.append(train_r2)
        if mode == 'train':
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("R2/train", train_r2, epoch)

        model.eval()

        test_loss = 0
        test_correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                test_correct += r2_score(target.cpu().detach().numpy(), output.cpu().detach().numpy())

        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        test_r2 =test_correct/len(test_loader)
        test_r2s.append(test_r2)
        if mode == 'train':
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("R2/test", test_r2, epoch)

        if test_loss<best_value:
            best_value = test_loss

    if mode=='train':
        writer.flush()
        writer.close()
        return model, train_losses, test_losses, train_r2s, test_r2s
    else:
        return model, best_value
    
def eval_fcnn(model, loader):
    device = torch.device('cuda:0')
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions += list(output.cpu().detach().numpy())
    return np.array(predictions)


def train_with_cv_fcnn(X, y, model_type, model_seed, data_seed, data_split_seed, n_trials, dataset, data_type, epochs):
    """Trains model with hyperparameter optimization

    Args:
        X (dataframe): Data for train (features)
        y (dataframe): Target 
        model_type (str): Type of trained model
        model_seed (int): Seed for model
        data_seed (int): Seed for hyperaprameter optimization
        data_split_seed (int): Seed for data split
        n_trials (int): Numer of trails for optuna 
        dataset (str): Dataset type
        data_type (str): Dataset type
        epochs(int): Number of epochs

    Returns:
        mse, mae, rmse, r2 (float): metrices for test data
        mse_2, mae_2, rmse_2, r2_2 (float): metrices for train data
    """
    seedEverything(model_seed)
    
    first_splits = KFold(n_splits=5, shuffle=True, random_state=data_split_seed)
    mse = [] # for test
    mae = [] 
    rmse = []
    r2 = [] 
    mse_2 = [] # for train
    mae_2 = [] 
    rmse_2 = [] 
    r2_2 = [] 
    tr_losses = []
    ts_losses = []
    tr_r2s = []
    ts_r2s = []
    for fold_1, (train_idx,test_idx) in enumerate(first_splits.split(np.arange(len(X)))):
        logger.info(F'Fold {fold_1+1} - data split for training')
        train_x = X.iloc[train_idx]
        test_x = X.iloc[test_idx]
        train_y = y.iloc[train_idx]
        test_y = y.iloc[test_idx]
        logger.info(F'There are {len(train_x)} train and {len(test_x)} test examples')
        # Hyperparameters optimization
        logger.info("Hyperparameter optimization")
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            direction = "minimize",
            pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3),
        )
        hs_start_time = time.time()
        study.optimize(partial(opt.objective_deep, model_type=model_type, train_x=train_x, train_y=train_y, model_seed=model_seed, data_seed=data_seed, epochs=epochs), n_trials=n_trials)
        logger.info(('Hyperparameters search finished, it took {:.2f} minutes').format((time.time() - hs_start_time)/60))
        best_params = study.best_params
        logger.info(F'Best params: {best_params}')
        logger.info(F'Best value: {study.best_value}')

        #Prepare data 
        train_dataset = TensorDataset(torch.FloatTensor(train_x.values), torch.FloatTensor(train_y.values.reshape(-1, 1)))
        if(len(train_dataset)%best_params['batch_size'] == 1): drop_last = True
        else: drop_last = False
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=False, drop_last=drop_last)
        test_dataset = TensorDataset(torch.FloatTensor(test_x.values), torch.FloatTensor(test_y.values.reshape(-1, 1)))
        test_loader = DataLoader(test_dataset, batch_size=test_y.values.shape[0], shuffle=False)

        # Train
        in_hw = train_x.shape[1]
        model, train_losses, test_losses, train_r2s, test_r2s = train_fcnn(train_loader, test_loader, in_hw, best_params, epochs, mode='train')
        # Plots
        tr_losses.append(train_losses)
        ts_losses.append(test_losses)
        tr_r2s.append(train_r2s)
        ts_r2s.append(test_r2s)
        # Save 
        model_path = 'models/{}_{}_{}_{}.pth'.format(dataset, data_type, model_type,fold_1+1)
        torch.save([model.kwargs, model.state_dict()], model_path)
        # Eval
        train_preds = eval_fcnn(model, train_loader)
        test_preds = eval_fcnn(model, test_loader)
        mae_i, mse_i, rmse_i, r2_square_i = evaluate(test_y, test_preds) #test
        mae_j, mse_j, rmse_j, r2_square_j = evaluate(train_y, train_preds) #train
        logger.info(F'For test set: MAE: {mae_i:.4f}, MSE: {mse_i:.4f}, RMSE: {rmse_i:.4f}, R2: {r2_square_i:.4f}')
        logger.info(F'For train set: MAE: {mae_j:.4f}, MSE: {mse_j:.4f}, RMSE: {rmse_j:.4f}, R2: {r2_square_j:.4f}')
        mse.append(mse_i)
        mae.append(mae_i)
        rmse.append(rmse_i)
        r2.append(r2_square_i)
        mse_2.append(mse_j)
        mae_2.append(mae_j)
        rmse_2.append(rmse_j)
        r2_2.append(r2_square_j)
    # Plot
    plot_train(tr_losses, ts_losses, tr_r2s, ts_r2s, dataset, data_type, model_type)
    return mse, mae, rmse, r2, mse_2, mae_2, rmse_2, r2_2 