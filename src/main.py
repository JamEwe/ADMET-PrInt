"""
Main file for pipeline

"""

import pandas as pd
import numpy as np
import yaml
from loguru import logger
from datetime import datetime
import argparse
import preprocess
from train_shallow import train_with_cv_shallow
from train_ensemble import train_with_cv_ensemble_avg, train_with_cv_ensemble_meta, train_with_cv_ensemble_meta_kfold
from train_deep import train_with_cv_fcnn
from eval import mean_value, std_value

def main_fc(args, params):
    """Gets arguments and params, trains model and saves logs file

    Args:
        args (dict): Parameters from params.yaml file
        params (dict): Parsed argumnets

    Returns:
        None
    """

    # 1. Load data
    logger.info("Load data")
    data_path = 'data/processed/{}_{}_all.csv'.format(args.dataset, args.data_type)
    df = pd.read_csv(data_path)
    logger.info(df.head())
    logger.info(F'There are {df.shape[0]} records')
    logger.info(F'There are {df.shape[1]-1} features')

    # 2. Split data to X and y 
    logger.info("Split data to X and y")
    X = df.loc[:, df.columns != df.columns[0]]
    y = df[df.columns[0]]

    # 3. Remove low variance features
    if args.data_type == 'klek' or args.data_type == 'pubchem':
        logger.info("Remove low variance features")
        idxs = preprocess.remove_low_variance(X, threshold=params['preprocess']['threshold'])
        X = X[idxs.tolist()]
        logger.info(F'Now there are {X.shape[1]} features')
        preprocess.save_idxs_for_processed(idxs, args.dataset, args.data_type)
    
    # 4. Hyperparameter optimization and train
    logger.info("Hyperparameter optimization and train")
    if args.model in ["ridge","rf","xgboost","lgbm","hist","svr"]:
        mse, mae, rmse, r2, mse_2, mae_2, rmse_2, r2_2 = train_with_cv_shallow(X=X, y=y, model_type=args.model, model_seed=params['train']['model_seed'], data_seed=params['hyperparameter_search']['seed'], data_split_seed=params['train']['data_seed'], n_trials=params['hyperparameter_search']['trials'], dataset=args.dataset, data_type=args.data_type)
    elif args.model in ["avg_ensemble", "meta_model_ensemble", "cv_meta_model_ensemble"]:
        if args.model=="avg_ensemble":
            mse, mae, rmse, r2, mse_2, mae_2, rmse_2, r2_2 = train_with_cv_ensemble_avg(X=X, y=y, model_type=args.model, model_seed=params['train']['model_seed'], data_seed=params['hyperparameter_search']['seed'], data_split_seed=params['train']['data_seed'], n_trials=params['hyperparameter_search']['trials'], dataset=args.dataset, data_type=args.data_type)
        elif args.model=="meta_model_ensemble":
            mse, mae, rmse, r2, mse_2, mae_2, rmse_2, r2_2 = train_with_cv_ensemble_meta(X=X, y=y, model_type=args.model, model_seed=params['train']['model_seed'], data_seed=params['hyperparameter_search']['seed'], data_split_seed=params['train']['data_seed'], n_trials=params['hyperparameter_search']['trials'], dataset=args.dataset, data_type=args.data_type)
        elif args.model=="cv_meta_model_ensemble":
            mse, mae, rmse, r2, mse_2, mae_2, rmse_2, r2_2 = train_with_cv_ensemble_meta_kfold(X=X, y=y, model_type=args.model, model_seed=params['train']['model_seed'], data_seed=params['hyperparameter_search']['seed'], data_split_seed=params['train']['data_seed'], n_trials=params['hyperparameter_search']['trials'], dataset=args.dataset, data_type=args.data_type)    
    elif args.model=="fcnn":
        mse, mae, rmse, r2, mse_2, mae_2, rmse_2, r2_2 = train_with_cv_fcnn(X=X, y=y, model_type=args.model, model_seed=params['train']['model_seed'], data_seed=params['hyperparameter_search']['seed'], data_split_seed=params['train']['data_seed'], n_trials=params['hyperparameter_search']['trials'], dataset=args.dataset, data_type=args.data_type, epochs=params['train']['epochs'])
    elif args.model=="gcnn":
        pass
    else:
        logger.error("Wrong type of model")

    # 5. Eval - mean, std
    mean_mse_test, mean_mae_test, mean_rmse_test, mean_r2_test = mean_value(mse, mae, rmse, r2)
    mean_mse_train, mean_mae_train, mean_rmse_train, mean_r2_train = mean_value(mse_2, mae_2, rmse_2, r2_2)

    std_mse_test, std_mae_test, std_rmse_test, std_r2_test = std_value(mse, mae, rmse, r2)
    std_mse_train, std_mae_train, std_rmse_train, std_r2_train = std_value(mse_2, mae_2, rmse_2, r2_2)

    logger.info(F'For test set:')
    logger.info(F'Mean MSE: {mean_mse_test:.4f}±{std_mse_test:.4f}')
    logger.info(F'Mean MAE: {mean_mae_test:.4f}±{std_mae_test:.4f}')
    logger.info(F'Mean RMSE: {mean_rmse_test:.4f}±{std_rmse_test:.4f}')
    logger.info(F'Mean R2: {mean_r2_test:.4f}±{std_r2_test:.4f}')

    logger.info(F'For train set:')
    logger.info(F'Mean MSE: {mean_mse_train:.4f}±{std_mse_train:.4f}')
    logger.info(F'Mean MAE: {mean_mae_train:.4f}±{std_mae_train:.4f}')
    logger.info(F'Mean RMSE: {mean_rmse_train:.4f}±{std_rmse_train:.4f}')
    logger.info(F'Mean R2: {mean_r2_train:.4f}±{std_r2_train:.4f}')
   

def main(args):

    # Set logger
    log_file_path = 'logs/{}-{}-{}-{:%Y-%m-%d-%H:%M}.log'.format(args.dataset, args.data_type, args.model, datetime.now())
    open(log_file_path, 'w').close()
    logger.add(log_file_path, format='{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}')
    logger.info('Run {} model with {} dataset in {} format'.format(args.model, args.dataset, args.data_type))

    # Load params
    with open("params.yaml", "r") as file:
        try:
            params = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Run
    main_fc(args, params)
    

if __name__=="__main__":
    # Parse args
    parser = argparse.ArgumentParser(description="Main")
    parser.add_argument("-dataset", type=str)
    parser.add_argument("-data_type", type=str)
    parser.add_argument("-model", type=str)
    args = parser.parse_args()
    main(args)