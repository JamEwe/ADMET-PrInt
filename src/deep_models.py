from sklearn.model_selection import KFold
from loguru import logger
import time
from functools import partial
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter

from sklearn import metrics
from rdkit import Chem
import matplotlib.pyplot as plt
from tqdm import tqdm

import optuna
from loguru import logger


def get_params_deep(trial, model):
    """Gets model type and returns parameters for optimization

    Args:
        trial (trial): Optuna trial
        model (str): Model type

    Returns:
        parameters (dict): Parameters for model
    """

    if model == "fcnn":
        return {
            "batch_size": trial.suggest_int("batch_size", 4, 256, step=4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "dim": trial.suggest_int("dim", 50, 150, step=10),
            "n_layers": trial.suggest_categorical("n_layers", [4,5])
        }
    
    elif model == "gcnn":
        return {
            "batch_size": trial.suggest_int("batch_size", 4, 256, step=4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "dim": trial.suggest_int("dim", 50, 150, step=10),
            "n_layers": trial.suggest_categorical("n_layers", [4,5])
        }
    else: 
        logger.error("Wrong type of model")

    
class FCNN(nn.Module):
    def __init__(self, n_layers: int = 5, hidden_size: int = 150, in_hw: int = 32, dp: int = 0.3):
        super(FCNN, self).__init__()
        self.kwargs = {'n_layers': n_layers, 'hidden_size': hidden_size, 'in_hw': in_hw, 'dp': dp}
        self.n_layers = n_layers
        if self.n_layers == 4:
            self.linear1 = nn.Linear(in_hw, 2*hidden_size)
            self.linear2 = nn.Linear(2*hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, hidden_size//2)
            self.linear4 = nn.Linear(hidden_size//2, 1)
            self.batch = torch.nn.BatchNorm1d(2*hidden_size)
        elif self.n_layers == 5:
            self.linear1 = nn.Linear(in_hw, 2*hidden_size)
            self.linear2 = nn.Linear(2*hidden_size, 3*hidden_size)
            self.linear3 = nn.Linear(3*hidden_size, 2*hidden_size)
            self.linear4 = nn.Linear(2*hidden_size, hidden_size//2)
            self.linear5 = nn.Linear(hidden_size//2, 1)
            self.batch = torch.nn.BatchNorm1d(3*hidden_size)
        self.dropout = torch.nn.Dropout(dp)
        
    def forward(self, x):
        if self.n_layers == 4:
            y_pred = F.relu(self.batch(self.linear1(x)))
            y_pred = F.relu(self.linear2(y_pred))
            y_pred = self.dropout(y_pred)
            y_pred = F.relu(self.linear3(y_pred))
            y_pred = self.dropout(y_pred)
            y_pred = F.relu(self.linear4(y_pred))
        elif self.n_layers == 5:
            y_pred = F.relu(self.linear1(x))
            y_pred = self.dropout(y_pred)
            y_pred = F.relu(self.batch(self.linear2(y_pred)))
            y_pred = F.relu(self.linear3(y_pred))
            y_pred = self.dropout(y_pred)
            y_pred = F.relu(self.linear4(y_pred))
            y_pred = self.dropout(y_pred)
            y_pred = self.linear5(y_pred)
        return y_pred


class GCNN(torch.nn.Module):
    def __init__(self, n_layers: int = 5, hidden_size: int = 150, dp: int = 0.3):
        super(GCNN, self).__init__()
        self.kwargs = {'n_layers': n_layers, 'hidden_size': hidden_size,'dp': dp}
        self.n_layers = n_layers
        torch.manual_seed(42)
        self.final_conv_acts = None
        self.final_conv_grads = None

        if self.n_layers == 4:
            self.conv1 = GCNConv(42, hidden_size)
            self.conv2 = GCNConv(hidden_size, 2*hidden_size)
            self.conv3 = GCNConv(2*hidden_size, 2*hidden_size)
            self.conv4 = GCNConv(2*hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, hidden_size//2)
            self.out2 = nn.Linear(hidden_size//2, 1)
        elif self.n_layers == 5:
            self.conv1 = GCNConv(42, hidden_size)
            self.conv2 = GCNConv(hidden_size, 2*hidden_size)
            self.conv3 = GCNConv(2*hidden_size, 3*hidden_size)
            self.conv4 = GCNConv(3*hidden_size, 2*hidden_size)
            self.conv5 = GCNConv(2*hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, hidden_size//2)
            self.out2 = nn.Linear(hidden_size//2, 1)
            self.batch = torch.nn.BatchNorm1d(3*hidden_size)
        self.dropout = torch.nn.Dropout(dp)
    
    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, edge_index, batch_index):

        if self.n_layers == 4:
            out = self.conv1(x, edge_index)
            out = F.relu(out)
            out = self.conv2(out, edge_index)
            out = F.relu(out)
            out = self.conv3(out, edge_index)
            out = F.relu(out)
            out = F.dropout(out, training=self.training)
            
            with torch.enable_grad():
                self.final_conv_acts = self.conv4(out, edge_index)
            self.final_conv_acts.register_hook(self.activations_hook)
            
            out = F.relu(self.final_conv_acts)
            out = gap(out, batch_index) # global_mean_pool
            out = self.out(out)
            out = F.relu(out)
            out = self.out2(out)
        elif self.n_layers == 5:
            out = self.conv1(x, edge_index)
            out = F.relu(out)
            out = self.conv2(out, edge_index)
            out = F.relu(out)
            out = F.dropout(out, training=self.training)
            out = self.conv3(out, edge_index)
            out = F.relu(out)
            out = self.conv4(out, edge_index)
            out = F.relu(out)
            out = F.dropout(out, training=self.training)
            
            with torch.enable_grad():
                self.final_conv_acts = self.conv5(out, edge_index)
            self.final_conv_acts.register_hook(self.activations_hook)
            
            out = F.relu(self.final_conv_acts)
            out = gap(out, batch_index) # global_mean_pool
            out = self.out(out)
            out = F.relu(out)
            out = self.out2(out)
        return out

class Featurizer:
    def __init__(self, y_column, **kwargs):
        self.y_column = y_column
        self.__dict__.update(kwargs)
    
    def __call__(self, df):
        raise NotImplementedError()

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
        

class GraphFeaturizer(Featurizer):
    def __call__(self, df):
        graphs = []
        for smiles in df:
            mol = Chem.MolFromSmiles(smiles)
            
            edges = []
            for bond in mol.GetBonds():
                edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            edges = np.array(edges)
            
            nodes = []
            for atom in mol.GetAtoms():
                results = one_of_k_encoding_unk(
                    atom.GetSymbol(),
                    [
                        'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Unknown'
                    ]
                ) + one_of_k_encoding(
                    atom.GetDegree(),
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                ) + one_of_k_encoding_unk(
                    atom.GetImplicitValence(),
                    [0, 1, 2, 3, 4, 5, 6]
                ) + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + one_of_k_encoding_unk(
                    atom.GetHybridization(),
                    [
                        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                        Chem.rdchem.HybridizationType.SP3D2
                    ]
                ) + [atom.GetIsAromatic()] + one_of_k_encoding_unk(
                    atom.GetTotalNumHs(),
                    [0, 1, 2, 3, 4]
                )
                nodes.append(results)
            nodes = np.array(nodes)
            
            graphs.append((nodes, edges.T))
        return graphs
    
class GraphDataset(InMemoryDataset):  
    def __init__(self, X, y, root, transform=None, pre_transform=None):
        self.dataset = (X, y)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        x_g, edge_index_g =list(zip(*self.dataset[0]))

        data = [Data(
            x=torch.FloatTensor(x),
            edge_index=torch.LongTensor(edge_index),
            y = y) for x, edge_index, y in zip(x_g,edge_index_g,self.dataset[1])
        ]

        torch.save(data, self.raw_paths[0])

    def process(self):
        data_list = torch.load(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
