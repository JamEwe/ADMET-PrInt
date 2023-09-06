from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def evaluate(true, predicted):
    """Calculates metrices

    Args:
        true (series): true values (target)
        predicted (numpy arr): predicted values 

    Returns:
        mse, mae, rmse, r2_square (float): metrices for test data
    """
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def mean_value(value1, value2, value3, value4):
    """Calculates mean value for metrices

    Args:
        value1 (list): list with values for mse
        value2 (list): list with values for mae
        value3 (list): list with values for rmse
        value4 (list): list with values for r2_square

    Returns:
        mse, mae, rmse, r2_square (float): mean value for metrices
    """
    return np.mean(value1), np.mean(value2), np.mean(value3), np.mean(value4)

def std_value(value1, value2, value3, value4):
    return np.std(value1), np.std(value2), np.std(value3), np.std(value4)