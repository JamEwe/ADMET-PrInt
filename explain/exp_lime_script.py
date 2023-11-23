"""
Main file for pipeline

"""
from loguru import logger
from datetime import datetime
import argparse
import random
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import KFold

import lime
from lime import lime_tabular
import preprocess

import warnings
warnings.filterwarnings('ignore')
import json
import scipy.cluster.hierarchy as hc
from scipy.cluster.hierarchy import cut_tree

from rdkit import Chem  
from rdkit.Chem import MolFromSmiles as smi2mol 

from rdkit.Chem import AllChem  
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity, TanimotoSimilarity 
from rdkit.Chem import Mol  

def get_explanations(train_x, test_x, model, idxs, data_type):
    explanations = []
    if data_type == 'klek':
        new_columns_names = [int(i.replace('KRFP', ''))-1 for i in idxs]
    elif data_type == 'pubchem':
        new_columns_names = [int(i.replace('PubchemFP', '')) for i in idxs]
    else:
        new_columns_names = range(len(idxs))
    logger.info("Explainer")
    explainer = lime_tabular.LimeTabularExplainer(train_x, feature_names = new_columns_names, class_names=['permeability'], mode="regression", random_state=42)
    logger.info("Explanations")
    for example in test_x:
        explanation = explainer.explain_instance(example, model.predict, num_features=len(new_columns_names))
        explanations += explanation.as_map()[1]
    return explanations

def exp_lime(args):
    """Gets arguments and get lime explanations 

    Args:
        args (dict): Parsed argumnets

    Returns:
        None
    """

    # 1. Load smiles data
    logger.info("Load smiles data")
    data_path = '../data/processed/{}_smiles_all.csv'.format(args.dataset)
    df_smi = pd.read_csv(data_path)

    # 2. Split data to X and y 
    X_smi = df_smi[df_smi.columns[0]]
    y_smi = df_smi[df_smi.columns[1]]

    # 3. Cluster
    logger.info("Cluster")
    
    ECFP4 = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles), 2) for smiles in X_smi.values]
    M = np.array([BulkTanimotoSimilarity(f, ECFP4) for f in ECFP4])
    M2 = 1 - M
    dist_df = pd.DataFrame(M2, index = X_smi.values, columns= X_smi.values)
    clustered = hc.linkage(M2, method='complete')
    clusters = cut_tree(clustered, n_clusters = 20).T
    df_clustered = pd.DataFrame({'Column1': list(X_smi.values), 'Column2': clusters[0]})
    
    random.seed(42)
    np.random.seed(42)
    
    n_clusters = 20
    x = 1000/len(X_smi.values)
    samples = []
    for i in range(n_clusters):
        C_i = np.where(df_clustered.Column2 == i)[0].tolist() 
        n_i = len(C_i)
        sample_i = random.sample(C_i, round(x * n_i)) 
        samples += list(sample_i)
    
    logger.info(f"Number of samples: {len(samples)}")

    # 4. Load data
    logger.info("Load data")
    data_path = '../data/processed/{}_{}_all.csv'.format(args.dataset, args.data_type)
    df = pd.read_csv(data_path)

    # 5. Split data to X and y 
    logger.info("Split data to X and y")
    X = df.loc[:, df.columns != df.columns[0]]
    y = df[df.columns[0]]

    # 6. Remove low variance features
    if args.data_type == 'klek' or args.data_type == 'pubchem':
        idxs = preprocess.remove_low_variance(X, threshold=0.01)
        X = X[idxs.tolist()]
    else:
        idxs = np.array(range(166))

    X = np.array(X)
    y = np.array(y)

    # 7. Get explanations
    for model_type in ['rf', 'hist', 'lgbm', 'xgboost', 'svr']:
        logger.info(f"For {model_type}")
        first_splits = KFold(n_splits=5, shuffle=True, random_state=42)
        explanations = []
        for fold_1, (train_idx,test_idx) in enumerate(first_splits.split(np.arange(len(X)))):
            logger.info(f"Fold {fold_1+1}")
            test_idx = np.array(list(set(test_idx) & set(samples)))
            #train_idx = random.sample(list(train_idx), 200) 
            train_x = X[train_idx]
            test_x = X[test_idx]
            model = pickle.load(open(f'../models/{args.dataset}_{args.data_type}_{model_type}_{1}.pkl', 'rb'))
            explanations = explanations + get_explanations(train_x, test_x, model, idxs, args.data_type)
            del model
        pd.DataFrame(explanations).to_csv(f'explanations/explanations_{args.dataset}_{args.data_type}_{model_type}.csv')


def main(args):
    # Set logger
    log_file_path = 'logs/lime_{}-{}-{:%Y-%m-%d-%H:%M}.log'.format(args.dataset, args.data_type, datetime.now())
    open(log_file_path, 'w').close()
    logger.add(log_file_path, format='{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}')
    logger.info('Run {} dataset in {} format'.format(args.dataset, args.data_type))

    # Run
    exp_lime(args)
    

if __name__=="__main__":
    # Parse args
    parser = argparse.ArgumentParser(description="Exp_main")
    parser.add_argument("-dataset", type=str)
    parser.add_argument("-data_type", type=str)
    args = parser.parse_args()
    main(args)