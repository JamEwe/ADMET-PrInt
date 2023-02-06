from shallow_models import get_params, get_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from loguru import logger
import numpy as np
from sklearn.linear_model import Ridge
from seed import seedEverything


def objective(trial, model_type, train_x, train_y, model_seed, data_seed):
    """Main function for optuna optimization

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



    
