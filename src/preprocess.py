from sklearn.feature_selection import VarianceThreshold
import json


def remove_low_variance(input_data, threshold=0.1):
    """Removes features with low variance

    Args:
        input_data (dataframe): Data for preprocessing
        threshold (float): Threshold for variance

    Returns:
        output_data (datframe): Processed data
    """
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data.columns[selection.get_support(indices=True)]


def save_idxs_for_processed(idxs, dataset, data_type):
    """Saves indexes for features

    Args:
        idxs (numpy arr): Indexes to save
        dataset (str): Type of dataset
        data_type (str): Type of data (features)

    Returns:
        None
    """
    '''
    idxs_dict = {
        dataset: idxs.tolist()
        }
    path = '{}_idxs.txt'.format(data_type)
    with open(path, 'w') as json_file:
        json.dump(idxs_dict, json_file)
    '''

    path = '{}_idxs.txt'.format(data_type)
    with open(path, 'r+') as json_file:
        json_data = json.load(json_file)
        json_data[dataset] = idxs.tolist()
        json_file.seek(0)
        json.dump(json_data, json_file)
        json_file.truncate()
