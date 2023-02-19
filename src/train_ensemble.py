from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from loguru import logger
from shallow_models import get_model
import time
from functools import partial
import optuna
import pickle
import numpy as np
import pandas as pd
from eval import evaluate
from opt import objective
from seed import seedEverything

def calculate_weights(mses):
    """Calculates weights for models

    Args:
        mmses (list): List mses from models

    Returns:
        weights (tuple): Weights for models
    """
    sum_mses = sum(mses)
    all_w1 = [sum_mses/mse for mse in mses]
    sum_mses2 = sum(all_w1)
    all_w2 = [w1/sum_mses2 for w1 in all_w1]
    return tuple(all_w2)


def average_predictions(models, weights, X):
    """Calculates average predictions

    Args:
        models (list): List with models
        weights (list): List with weights for models
        X (dataframe): Data (features)

    Returns:
        predictions (dataframe): Averaged predictions
    """
    return ((weights[0] * models[0].predict(X)) + \
            (weights[1] * models[1].predict(X)) + \
            (weights[2] * models[2].predict(X)) + \
            (weights[3] * models[3].predict(X)) + \
            (weights[4] * models[4].predict(X)))


def train_with_cv_ensemble_avg(X, y, model_type, model_seed, data_seed, data_split_seed, n_trials, dataset, data_type):
    """Trains model with hyperparameter optimization, ensemble model with base models and weights

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
        # Load models
        logger.info(F'Load 5 models...')
        rf = pickle.load(open('models/{}_{}_{}_{}.pkl'.format(dataset, data_type, 'rf', fold_1+1), 'rb'))
        xgboost = pickle.load(open('models/{}_{}_{}_{}.pkl'.format(dataset, data_type, 'xgboost', fold_1+1), 'rb'))
        lgbm = pickle.load(open('models/{}_{}_{}_{}.pkl'.format(dataset, data_type, 'lgbm', fold_1+1), 'rb'))
        hist = pickle.load(open('models/{}_{}_{}_{}.pkl'.format(dataset, data_type, 'hist', fold_1+1), 'rb'))
        svr = pickle.load(open('models/{}_{}_{}_{}.pkl'.format(dataset, data_type, 'svr', fold_1+1), 'rb'))     
        # Predict
        logger.info(F'Get predictions...')
        rf_preds = rf.predict(test_x)
        xgboost_preds = xgboost.predict(test_x)
        lgbm_preds = lgbm.predict(test_x)
        hist_preds = hist.predict(test_x)
        svr_preds = svr.predict(test_x)
        # Calculate weights
        logger.info(F'Calculate weights...')
        rf_mse = mean_squared_error(test_y, rf_preds)
        xgboost_mse = mean_squared_error(test_y, xgboost_preds)
        lgbm_mse = mean_squared_error(test_y, lgbm_preds)
        hist_mse = mean_squared_error(test_y, hist_preds)
        svr_mse = mean_squared_error(test_y, svr_preds)
        rf_w, xgboost_w, lgbm_w, hist_w, svr_w = calculate_weights([rf_mse, xgboost_mse, lgbm_mse, hist_mse, svr_mse])
        logger.info(F'Weight for rf: {rf_w}, Xgboost: {xgboost_w}, Lgbm: {lgbm_w}, Hist: {hist_w}, SVR: {svr_w}')
        # Calculate predictions
        avg_preds = ((rf_w * rf_preds) + (xgboost_w  * xgboost_preds) + (lgbm_w * lgbm_preds) + (hist_w  * hist_preds) + (svr_w * svr_preds))
        # Eval
        mae_i, mse_i, rmse_i, r2_square_i = evaluate(test_y, avg_preds) # test
        mae_j, mse_j, rmse_j, r2_square_j = evaluate(train_y, average_predictions([rf, xgboost, lgbm, hist, svr], [rf_w, xgboost_w, lgbm_w, hist_w, svr_w], train_x)) #train
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

def train_with_cv_ensemble_meta(X, y, model_type, model_seed, data_seed, data_split_seed, n_trials, dataset, data_type):
    """Trains model with hyperparameter optimization, ensemble model with base models and meta model

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
        # Load models
        logger.info(F'Load 5 models...')
        rf = pickle.load(open('models/{}_{}_{}_{}.pkl'.format(dataset, data_type, 'rf', fold_1+1), 'rb'))
        xgboost = pickle.load(open('models/{}_{}_{}_{}.pkl'.format(dataset, data_type, 'xgboost', fold_1+1), 'rb'))
        lgbm = pickle.load(open('models/{}_{}_{}_{}.pkl'.format(dataset, data_type, 'lgbm', fold_1+1), 'rb'))
        hist = pickle.load(open('models/{}_{}_{}_{}.pkl'.format(dataset, data_type, 'hist', fold_1+1), 'rb'))
        svr = pickle.load(open('models/{}_{}_{}_{}.pkl'.format(dataset, data_type, 'svr', fold_1+1), 'rb'))     
        # Predict
        logger.info(F'Get predictions...')
        meta_train = pd.DataFrame(data={
            "rf": rf.predict(train_x).tolist(),
            "xgbr": xgboost.predict(train_x).tolist(),
            "lgbm": lgbm.predict(train_x).tolist(),
            "hist": hist.predict(train_x).tolist(),
            "svr": svr.predict(train_x).tolist()
            })
        meta_test = pd.DataFrame(data={
            "rf": rf.predict(test_x).tolist(),
            "xgbr": xgboost.predict(test_x).tolist(),
            "lgbm": lgbm.predict(test_x).tolist(),
            "hist": hist.predict(test_x).tolist(),
            "svr": svr.predict(test_x).tolist()
            })

        logger.info("Hyperparameter optimization")
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            direction = "minimize",
            pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3),
        )
        hs_start_time = time.time()
        study.optimize(partial(objective, model_type="ridge", train_x=meta_train, train_y=train_y, model_seed=model_seed, data_seed=data_seed), n_trials=n_trials)
        logger.info(('Hyperparameters search finished, it took {:.2f} minutes').format((time.time() - hs_start_time)/60))
        best_params = study.best_params
        logger.info(F'Best params: {best_params}')
        logger.info(F'Best value: {study.best_value}')
        # Train
        model = get_model("ridge", best_params)
        model.fit(meta_train, train_y)
        # Eval
        mae_i, mse_i, rmse_i, r2_square_i = evaluate(test_y, model.predict(meta_test)) # test
        mae_j, mse_j, rmse_j, r2_square_j = evaluate(train_y, model.predict(meta_train)) #train
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

def train_with_cv_ensemble_meta_kfold(X, y, model_type, model_seed, data_seed, data_split_seed, n_trials, dataset, data_type):
    """Trains model with hyperparameter optimization, ensemble model with base models, meta model and cross validation

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
    pass