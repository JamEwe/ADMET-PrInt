import optuna
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from loguru import logger


def get_params(trial, model):
    """Gets model type and returns parameters for optimization

    Args:
        trial (trial): Optuna trial
        model (str): Model type

    Returns:
        parameters (dict): Parameters for model
    """

    if model == "ridge":
        return {
            "alpha": trial.suggest_int("alpha", 0, 100),
            "solver": trial.suggest_categorical("solver", ['cholesky', 'svd','lsqr']),
            "tol": trial.suggest_float("tol", 1e-4, 1e-3, log=True),
        }
    elif model == "rf":
        return {
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "n_estimators": trial.suggest_int("n_estimators", 100, 4000, step=100),
        }
    elif model == "xgboost":
        return {
            "eta": trial.suggest_float("eta", 0.0, 1.0, step=0.1),
            "lambda": trial.suggest_float("lambda", 0.4, 0.8, step=0.1),
            "gamma": trial.suggest_float("gamma", 0.1, 0.8, step=0.1),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "n_estimators": trial.suggest_int("n_estimators", 100, 4000, step=100),
            "min_child_weight": trial.suggest_int("min_child_weight", 80, 160, step=5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0, step=0.05),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9, step=0.1),
        }
    elif model == "lgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 50, 80, step=10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "n_estimators": trial.suggest_int("n_estimators", 100, 4000, step=100),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 100, step=5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 4.0, step=0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e-1, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.6, step=0.05),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9, step=0.1),
        }
    elif model == "hist":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1e-1, step=0.01),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
        }
    elif model == "svr":
        return {
            "kernel": trial.suggest_categorical("kernel", ['rbf', 'sigmoid']),
            "gamma": trial.suggest_categorical("gamma", ['scale', 'auto']),
            "C": trial.suggest_float("C", 1.0, 30.0, step=0.1),
            "epsilon": trial.suggest_float("epsilon", 0.1, 8.0, log=True),
        }
    else: 
        logger.error("Wrong type of model")

def get_model(model_type, params):
    """Gets model type and returns model object wit parameters

    Args:
        model_type (str): Model type
        params (dict): Parameters for model

    Returns:
        model (Sklearn model): Model object with parameters
    """

    models_dict = {
        "ridge": Ridge,
        "rf": RandomForestRegressor,
        "xgboost": xgb.XGBRegressor,
        "lgbm": lgbm.LGBMRegressor,
        "hist": HistGradientBoostingRegressor,
        "svr": SVR
    }
    return models_dict[model_type](**params)












