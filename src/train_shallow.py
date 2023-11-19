from sklearn.model_selection import KFold
from loguru import logger
import time
from functools import partial
import optuna
import pickle
import numpy as np
from shallow_models import get_model
from eval import evaluate
from opt import objective
from seed import seedEverything

def train_with_cv_shallow(X, y, model_type, model_seed, data_seed, data_split_seed, n_trials, dataset, data_type):
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
        study.optimize(partial(objective, model_type=model_type, train_x=train_x, train_y=train_y, model_seed=model_seed, data_seed=data_seed), n_trials=n_trials)
        logger.info(('Hyperparameters search finished, it took {:.2f} minutes').format((time.time() - hs_start_time)/60))
        best_params = study.best_params
        logger.info(F'Best params: {best_params}')
        logger.info(F'Best value: {study.best_value}')
        # Train
        model = get_model(model_type, best_params)
        model.fit(train_x, train_y)
        # Save 
        model_path = 'models/{}_{}_{}_{}.pkl'.format(dataset, data_type, model_type,fold_1+1)
        pickle.dump(model, open(model_path, 'wb'))
        # Eval
        mae_i, mse_i, rmse_i, r2_square_i = evaluate(test_y, model.predict(test_x)) # test
        mae_j, mse_j, rmse_j, r2_square_j = evaluate(train_y, model.predict(train_x)) #train
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
    return mse, mae, rmse, r2, mse_2, mae_2, rmse_2, r2_2 