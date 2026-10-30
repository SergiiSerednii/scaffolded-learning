
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from optuna import Trial


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLPDataset(Dataset):
    """PyTorch Dataset for MLP training"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def _compute_auc(current_model, data_loader, device="cpu"):
    """Compute ROC AUC per target for a given loader; return NaN if undefined."""
    current_model.eval()
    all_probs = [[] for _ in range(current_model.n_targets)]
    all_labels = [[] for _ in range(current_model.n_targets)]
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            outputs = current_model(features)
            labels_np = labels.numpy()
            for i in range(current_model.n_targets):
                probs_i = outputs[i].detach().cpu().numpy()
                all_probs[i].extend(probs_i)
                all_labels[i].extend(labels_np[:, i])

    aucs = []
    for i in range(current_model.n_targets):
        try:
            auc_i = roc_auc_score(all_labels[i], all_probs[i])
        except ValueError:
            auc_i = float("nan")
        aucs.append(auc_i)
    return aucs

def _resolve_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    mapping = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation '{name}'")
    return mapping[name]()

def _resolve_suggestion(trial: Trial, name: str):
    """Return a bound suggestion method on `trial` selected by short `name`.

    Usage:
        suggest = _resolve_suggestion(trial, "categorical")
        value = suggest("batch_size", choices=[256, 512])
    """
    name = name.lower()
    mapping = {
        "categorical": trial.suggest_categorical,
        "float": trial.suggest_float,
        "int": trial.suggest_int,
        "discrete_uniform": trial.suggest_discrete_uniform,
        "loguniform": trial.suggest_loguniform,
        "uniform": trial.suggest_uniform,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported suggestion '{name}'")
    return mapping[name]