import copy
import datetime as dt
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from scaffolded_learning.utils import _resolve_activation, _resolve_suggestion, _set_seed

class MLPDataset(Dataset):   
    def __init__(self, features, labels, device="cpu"):
        self.features = torch.FloatTensor(features).to(device)
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long).to(device)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class _MLPBackbone(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate, activation_name):
        super().__init__()
        if len(hidden_sizes) == 0:
            self.net = nn.Identity()
            self.out_features = input_size
            return

        layers = []
        in_features = input_size
        for width in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_features, int(width)),
                    _resolve_activation(activation_name),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_features = int(width)
        self.net = nn.Sequential(*layers)
        self.out_features = in_features

    def forward(self, x):
        return self.net(x)


class _PenultimateScaffoldMLP(nn.Module):
    """MLP immediately before the last one, plus a multiclass scaffold head."""

    def __init__(self, input_size, hidden_size, n_classes, dropout_rate, activation_name):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            _resolve_activation(activation_name),
            nn.Dropout(dropout_rate),
        )
        self.target_head = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        hidden = self.block(x)
        return hidden, self.target_head(hidden)


class _MainMLPHead(nn.Module):
    """Last MLP plus the multiclass main-target head."""

    def __init__(self, input_size, hidden_size, n_classes, dropout_rate, activation_name):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            _resolve_activation(activation_name),
            nn.Dropout(dropout_rate),
        )
        self.target_head = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        return self.target_head(self.block(x))


def _check_class_index(values: np.ndarray, n_classes: int, name: str) -> None:
    if values.size == 0:
        raise ValueError(f"{name} has no labels")
    vmin = int(values.min())
    vmax = int(values.max())
    if vmin < 0 or vmax >= n_classes:
        raise ValueError(
            f"{name} must be integer class indices in [0, {n_classes - 1}], got min={vmin}, max={vmax}"
        )


def _n_classes_from_column(frame: pd.DataFrame, column: str) -> int:
    values = frame[column].to_numpy()
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(values == np.floor(values)):
            raise ValueError(f"{column} must contain integer class indices")
        values = values.astype(np.int64)
    vmin = int(values.min())
    vmax = int(values.max())
    if vmin != 0:
        raise ValueError(f"{column} class indices must start at 0, got min={vmin}")
    present = set(int(v) for v in np.unique(values))
    missing = [i for i in range(vmax + 1) if i not in present]
    if missing:
        raise ValueError(
            f"{column} is missing class indices {missing}. "
            "Pass n_scaffold_classes / n_main_classes explicitly if a split is incomplete."
        )
    return vmax + 1


class ScaffoldedMLPMultiClass(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_scaffold_classes: int,
        n_main_classes: int,
        layer_size: int,
        dropout_rate: float = 0.3,
        hidden_activation: str = "relu",
        use_scaffold: bool = True,
    ):
        super().__init__()
        if n_scaffold_classes < 2:
            raise ValueError("n_scaffold_classes must be >= 2")
        if n_main_classes < 2:
            raise ValueError("n_main_classes must be >= 2")        

        self.n_scaffold_classes = int(n_scaffold_classes)
        self.n_main_classes = int(n_main_classes)        
        self.use_scaffold = bool(use_scaffold)
        self.hidden_activation = hidden_activation
        
        self.backbone = _MLPBackbone(input_size, [layer_size], dropout_rate, hidden_activation)
        self.scaffold_module = _PenultimateScaffoldMLP(
            input_size=self.backbone.out_features,
            hidden_size=layer_size,
            n_classes=self.n_scaffold_classes,
            dropout_rate=dropout_rate,
            activation_name=hidden_activation,
        )
        self.output_head = _MainMLPHead(
            input_size=layer_size,
            hidden_size=layer_size,
            n_classes=self.n_main_classes,
            dropout_rate=dropout_rate,
            activation_name=hidden_activation,
        )

    def forward(self, x):
        hidden = self.backbone(x)
        hidden, scaffold_logits = self.scaffold_module(hidden)
        main_logits = self.output_head(hidden)
        return [scaffold_logits, main_logits]
   
    def _aggregate_losses(self, losses: Sequence[torch.Tensor]) -> torch.Tensor:        
        if self.use_scaffold:
            return sum(losses)
        return losses[-1]

    def fit(
        self,
        train_data,
        val_data,
        test_data,
        target_column: str = None,
        scaffold_target_column: str = None,
        features: Optional[Sequence[str]] = None,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        early_stop_metric: str = "target_accuracy",
        early_stop_patience: int = 10,
        random_state: int = 1223,
        device: str = "cpu",
        epochs: int = None,
        val_share: float = None,
        verbose: int = 1,
        show_epochs: int = 1,        
    ):
        # > Inputs validation
        if isinstance(train_data, str):
            train_path = train_data
            if train_data.endswith(".parquet"):
                train_data = pd.read_parquet(train_data)
            elif train_data.endswith(".csv"):
                train_data = pd.read_csv(train_data)
            else:
                raise ValueError(f"train_data must be a parquet or csv file, got {train_path}")
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Loaded train data from {train_path} with shape {train_data.shape}")
        elif isinstance(train_data, pd.DataFrame):
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Use as train data dataframe with shape {train_data.shape}")
        else:
            raise ValueError(f"train_data must be a string or a pandas DataFrame, got {type(train_data)}")

        if isinstance(val_data, str):
            val_path = val_data
            if val_data.endswith(".parquet"):
                val_data = pd.read_parquet(val_data)
            elif val_data.endswith(".csv"):
                val_data = pd.read_csv(val_data)
            else:
                raise ValueError(f"val_data must be a parquet or csv file, got {val_path}")
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Loaded val data from {val_path} with shape {val_data.shape}")
        elif isinstance(val_data, pd.DataFrame):
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Use as val data dataframe with shape {val_data.shape}")
        else:
            if val_share is None:
                if len(train_data) <= 200:
                    val_share = 0.5
                elif len(train_data) <= 1000:
                    val_share = 0.3
                elif len(train_data) <= 10000:
                    val_share = 0.2
                else:
                    val_share = 0.1
                if verbose >= 1:
                    print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Use validation share {val_share} for val data")

            if val_share <= 0 or val_share >= 1:
                raise ValueError("val_share must be in range (0, 1)")

            if target_column is None:
                target_column = train_data.columns[-1]
            train_data, val_data = train_test_split(
                train_data,
                test_size=val_share,
                random_state=random_state,
                stratify=train_data[target_column],
            )
            if verbose >= 1:
                print(
                    f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Split train data into train and val data "
                    f"with shapes {train_data.shape} and {val_data.shape}"
                )

        if isinstance(test_data, str):
            test_path = test_data
            if test_data.endswith(".parquet"):
                test_data = pd.read_parquet(test_data)
            elif test_data.endswith(".csv"):
                test_data = pd.read_csv(test_data)
            else:
                raise ValueError(f"test_data must be a parquet or csv file, got {test_path}")
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Loaded test data with shape {test_data.shape}")
        elif isinstance(test_data, pd.DataFrame):
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Use as test data dataframe with shape {test_data.shape}")
        elif test_data is not None:
            raise ValueError(
                f"test_data must be a string path, a pandas DataFrame, or None, got {type(test_data)}"
            )

        if batch_size is None:
            if len(train_data) <= 200:
                batch_size = 128
            elif len(train_data) <= 1000:
                batch_size = 256
            elif len(train_data) <= 10000:
                batch_size = 512
            else:
                batch_size = 1024
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Use batch size {batch_size}")

        if epochs is None:
            if len(train_data) <= 200:
                epochs = 100
            elif len(train_data) <= 1000:
                epochs = 200
            elif len(train_data) <= 10000:
                epochs = 500
            else:
                epochs = 1000
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Use {epochs} epochs")

        if len(train_data) < 100:
            raise ValueError("train_data must have at least 100 rows")
        if len(val_data) < 100:
            raise ValueError("val_data must have at least 100 rows")

        if target_column is None:
            target_column = train_data.columns[-1]
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Use {target_column} as target column")
        
        _set_seed(random_state)
        self.to(device)        
        if verbose >= 1:
            print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Start training, using device: {device}")
            
        label_columns = [scaffold_target_column, target_column]
        if features is not None:
            feature_columns = list(features)
            missing = [c for c in feature_columns if c not in train_data.columns]
            if missing:
                raise ValueError(f"features not present in train_data columns: {missing}")
        else:
            feature_columns = [c for c in train_data.columns if c not in label_columns]

        
        X_train = train_data[feature_columns].values
        y_train = train_data[label_columns].astype(int).values
        X_val = val_data[feature_columns].values
        y_val = val_data[label_columns].astype(int).values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        test_loader = None
        if test_data is not None:
            missing_test = [c for c in feature_columns if c not in test_data.columns]
            if missing_test:
                raise ValueError(f"test_data is missing feature columns: {missing_test}")
            missing_targets = [c for c in label_columns if c not in test_data.columns]
            if missing_targets:
                raise ValueError(f"test_data is missing target columns: {missing_targets}")
            X_test = test_data[feature_columns].values
            y_test = test_data[label_columns].astype(int).values
            X_test_scaled = scaler.transform(X_test)
            test_dataset = MLPDataset(X_test_scaled, y_test, device=device)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)        

        train_dataset = MLPDataset(X_train_scaled, y_train, device=device)
        val_dataset = MLPDataset(X_val_scaled, y_val, device=device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        if verbose >= 1:
            print("\nModel architecture:")
            print(self)
            print(f"\nTotal parameters: {sum(p.numel() for p in self.parameters()):,}")

        n_train = int(X_train_scaled.shape[0])
        n_val = int(X_val_scaled.shape[0])
        n_test = int(X_test_scaled.shape[0]) if test_loader is not None else None

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if early_stop_metric not in {"total_val_loss", "target_val_loss", "target_accuracy"}:
            raise ValueError(
                "early_stop_metric must be one of: 'total_val_loss', 'target_val_loss', 'target_accuracy'"
            )
        
        best_metric = float("inf") if early_stop_metric in {"total_val_loss", "target_val_loss"} else -float("inf")
        best_model_state = copy.deepcopy(self.state_dict())
        best_epoch = 0
        patience_counter = 0

        self.train_losses = []
        self.val_losses = []
        self.train_losses_per_target = []
        self.val_losses_per_target = []
        self.train_accuracy = []
        self.val_accuracy = []

        if verbose >= 1:
            print("\nStarting training...")
            print(f"Device: {device}")
            print(f"Epochs: {epochs}")
            print(f"Learning rate: {learning_rate}\n")

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            train_loss_per_target = np.zeros(2, dtype=np.float64)
            train_accuracy = np.zeros(2, dtype=np.float64)

            for features, labels in train_loader:
                # features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(features)
                losses = [criterion(outputs[i], labels[:, i]) for i in range(2)]
                total_loss = self._aggregate_losses(losses)
                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item()
                for i, loss_i in enumerate(losses):
                    train_loss_per_target[i] += loss_i.item()
                    train_accuracy[i] += (outputs[i].argmax(dim=1) == labels[:, i]).sum().item()
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_loss_per_target = np.zeros(2, dtype=np.float64)
            val_accuracy = np.zeros(2, dtype=np.float64)
            with torch.no_grad():
                for features, labels in val_loader:
                    # features, labels = features.to(device), labels.to(device)
                    outputs = self(features)
                    losses = [criterion(outputs[i], labels[:, i]) for i in range(2)]
                    total_loss = self._aggregate_losses(losses)
                    val_loss += total_loss.item()
                    for i, loss_i in enumerate(losses):
                        val_loss_per_target[i] += loss_i.item()
                        val_accuracy[i] += (outputs[i].argmax(dim=1) == labels[:, i]).sum().item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_loss_per_target /= len(train_loader)
            val_loss_per_target /= len(val_loader)
            train_accuracy /= len(train_dataset)
            val_accuracy /= len(val_dataset)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_losses_per_target.append(train_loss_per_target)
            self.val_losses_per_target.append(val_loss_per_target)
            self.train_accuracy.append(train_accuracy)
            self.val_accuracy.append(val_accuracy)

            # > Check if results are better 
            if early_stop_metric == "total_val_loss":
                current_metric = val_loss
                is_better = current_metric < best_metric
            elif early_stop_metric == "target_val_loss":
                current_metric = val_loss_per_target[-1]
                is_better = current_metric < best_metric            
            else:
                current_metric = val_accuracy[-1]
                is_better = current_metric > best_metric

            if verbose >= 1 and ((epoch + 1) % show_epochs == 0 or epoch == 0):                
                per_target_stats = " | ".join(
                    [
                        f"T{i}: TL={train_loss_per_target[i]:.4f}, VL={val_loss_per_target[i]:.4f}, "
                        f"TA={train_accuracy[i]:.4f}, VA={val_accuracy[i]:.4f}"
                        for i in range(2)
                    ]
                )
                print(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | {per_target_stats}"
                )

            if is_better:
                best_metric = current_metric
                best_model_state = copy.deepcopy(self.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    if verbose >= 1:
                        print(
                            f"\nEarly stopping at epoch {epoch + 1} "
                            f"with metric={early_stop_metric}, best={best_metric:.4f}"
                        )
                    break

        self.load_state_dict(best_model_state)
        
        self.best_epoch = best_epoch
        self.best_metric = best_metric
        self.scaler_ = scaler
        self.feature_columns_ = feature_columns
        self.train_loss = self.train_losses[best_epoch - 1]
        self.val_loss = self.val_losses[best_epoch - 1]
        self.train_loss_per_target = self.train_losses_per_target[best_epoch - 1]
        self.val_loss_per_target = self.val_losses_per_target[best_epoch - 1]
        self.train_accuracy = self.train_accuracy[best_epoch - 1]
        self.train_accuracy_scaffold = self.train_accuracy[0]
        self.train_accuracy_main = self.train_accuracy[1]
        self.val_accuracy = self.val_accuracy[best_epoch - 1]
        self.val_accuracy_scaffold = self.val_accuracy[0]
        self.val_accuracy_main = self.val_accuracy[1]

        # Calculate on test data
        if test_loader is not None:
            test_loss = 0.0
            test_loss_per_target = np.zeros(2, dtype=np.float64)
            test_accuracy = np.zeros(2, dtype=np.float64)
            with torch.no_grad():
                for features, labels in test_loader:
                    # features, labels = features.to(device), labels.to(device)
                    outputs = self(features)
                    losses = [criterion(outputs[i], labels[:, i]) for i in range(2)]
                    total_loss = self._aggregate_losses(losses)
                    test_loss += total_loss.item()
                    for i, loss_i in enumerate(losses):
                        test_loss_per_target[i] += loss_i.item()
                        test_accuracy[i] += (outputs[i].argmax(dim=1) == labels[:, i]).sum().item()
            test_loss /= len(test_loader)
            test_loss_per_target /= len(test_loader)
            test_accuracy /= len(test_dataset)
            self.test_loss = test_loss
            self.test_loss_per_target = test_loss_per_target
            self.test_accuracy = test_accuracy
            self.test_accuracy_scaffold = test_accuracy[0]
            self.test_accuracy_main = test_accuracy[1]
  
        if verbose >= 1:
            print(
                f"\nTraining completed. Best early-stop metric ({early_stop_metric}): "
                f"{best_metric:.4f}, best epoch: {best_epoch}"
            )
            test_acc = (
                f"  test_acc={[f'{x:.4f}' for x in self.test_accuracy]}"
            )
            print(f"Best-model accuracy [scaffold, main] test={test_acc}")

    def save(self, path: str):
        raise NotImplementedError("Saving is not implemented yet")

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError("Loading is not implemented yet")

    @classmethod
    def init_and_fit(
        cls,
        train_data: Union[str, pd.DataFrame],        
        target_column: Optional[str] = None,
        scaffold_target_column: Optional[str] = None,
        features: Optional[Sequence[str]] = None,
        n_scaffold_classes: Optional[int] = None,
        n_main_classes: Optional[int] = None,
        hidden_activation: str = "relu",
        use_scaffold: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        val_data: Union[str, pd.DataFrame, None] = None,
        test_data: Union[str, pd.DataFrame, None] = None,
        val_share: Optional[float] = None,
        early_stop_metric: str = "target_accuracy",
        early_stop_patience: int = 10,
        random_state: int = 1223,
        epochs: Optional[int] = None,
        verbose: int = 1,
        show_epochs: int = 1,
        layer_size: Union[int, Tuple[str, Dict[str, Any]], None] = None,
        batch_size: Union[int, Tuple[str, Dict[str, Any]], None] = None,
        learning_rate: Union[float, Tuple[str, Dict[str, Any]], None] = None,
        dropout_rate: Union[float, Tuple[str, Dict[str, Any]], None] = None,        
        n_trials: int = 100,
        study_name: Optional[str] = None,        
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        journal_path: Optional[str] = None,
    ):        
        if scaffold_target_column is None or target_column is None:
            raise ValueError("scaffold_target_column and target_column must be provided")
        if not isinstance(scaffold_target_column, str):
            raise ValueError(f"scaffold_target_column must be a string, got {type(scaffold_target_column)}")
        if not isinstance(target_column, str):
            raise ValueError(f"target_column must be a string, got {type(target_column)}")

        if isinstance(train_data, str):
            train_path = train_data
            if train_data.endswith(".parquet"):
                train_data = pd.read_parquet(train_data)
            elif train_data.endswith(".csv"):
                train_data = pd.read_csv(train_data)
            else:
                raise ValueError(f"train_data must be a parquet or csv file, got {train_path}")
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError(
                f"train_data must be a string path or a pandas DataFrame, got {type(train_data)}"
            )

        if isinstance(val_data, str):
            if val_data.endswith(".parquet"):
                val_data = pd.read_parquet(val_data)
            elif val_data.endswith(".csv"):
                val_data = pd.read_csv(val_data)
            else:
                raise ValueError(f"val_data must be a parquet or csv file, got {val_data}")

        if isinstance(test_data, str):
            if test_data.endswith(".parquet"):
                test_data = pd.read_parquet(test_data)
            elif test_data.endswith(".csv"):
                test_data = pd.read_csv(test_data)
            else:
                raise ValueError(f"test_data must be a parquet or csv file, got {test_data}")        

        if n_scaffold_classes is None:
            n_scaffold_classes = _n_classes_from_column(train_data, scaffold_target_column[0])
        if n_main_classes is None:
            n_main_classes = _n_classes_from_column(train_data, target_column)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Use {device} as device")

        if study_name is None:
            study_name = f"ScaffoldedMLPMultiClassGPU_hyperopt_{dt.datetime.now():%Y-%m-%d_%H-%M-%S}"

        search_params = [
            layer_size,
            batch_size,
            learning_rate,            
            dropout_rate,
        ]        
        hyperopt_flag = any(isinstance(param, tuple) for param in search_params)                 

        if hyperopt_flag:
            if batch_size is None:
                batch_size = ("categorical", {"choices": [256, 512, 1024, 2048]})
            if learning_rate is None:
                learning_rate = ("float", {"low": 1e-3, "high": 1e-2})
            if layer_size is None:
                layer_size = ("categorical", {"choices": [64, 128, 256, 512]})            
            if dropout_rate is None:
                dropout_rate = ("float", {"low": 0.05, "high": 0.5})
        else:
            if learning_rate is None:
                learning_rate = 0.001
            if dropout_rate is None:
                dropout_rate = 0.3

        if features is None:
            features_full = [c for c in train_data.columns if c not in [scaffold_target_column, target_column]]
        else:
            features_full = list(features)
        input_size = len(features_full)

        if val_data is None:
            train_data, val_data = train_test_split(
                train_data,
                test_size=val_share,
                random_state=random_state,
                stratify=train_data[target_column],
            )        

        def _build(layer_size, dropout_rate, scaffold_flag):
            _set_seed(random_state)
            model = cls(
                input_size=input_size,
                n_scaffold_classes=n_scaffold_classes,
                n_main_classes=n_main_classes,
                layer_size=layer_size,                
                dropout_rate=dropout_rate,
                hidden_activation=hidden_activation,
                use_scaffold=scaffold_flag,
            )
            model.to(device)
            return model

        if not hyperopt_flag:
            model = _build(layer_size, dropout_rate, use_scaffold)
            model.fit(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                target_column=target_column,
                scaffold_target_column=scaffold_target_column,
                features=features_full,
                batch_size=batch_size,
                learning_rate=learning_rate,
                early_stop_metric=early_stop_metric,
                early_stop_patience=early_stop_patience,
                random_state=random_state,
                device=device,
                epochs=epochs,
                val_share=val_share,
                verbose=verbose,
                show_epochs=show_epochs,                
            )
            return model

        if early_stop_metric not in {"total_val_loss", "target_val_loss", "target_accuracy"}:
            raise ValueError(
                "Hyperopt early_stop_metric must be 'total_val_loss', 'target_val_loss', 'target_accuracy'"
            )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        journal_rows: List[Dict[str, Any]] = []
        best_models_per_trial: Dict[int, "ScaffoldedMLPMultiClassGPU"] = {}

        def _suggest_or_const(trial, value, param_name):
            if isinstance(value, tuple):
                suggested = _resolve_suggestion(trial, value[0])(param_name, **value[1])
                return suggested
            return value

        def objective(trial: optuna.Trial) -> float:
            batch_size_trial = _suggest_or_const(trial, batch_size, "batch_size")
            learning_rate_trial = round(float(_suggest_or_const(trial, learning_rate, "learning_rate")), 6)
            layer_size_trial = _suggest_or_const(trial, layer_size, "layer_size")
            dropout_rate_trial = round(float(_suggest_or_const(trial, dropout_rate, "dropout_rate")), 6)
            
            try:
                model = _build(layer_size_trial, dropout_rate_trial, use_scaffold)
                model.fit(
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    target_column=target_column,
                    scaffold_target_column=scaffold_target_column,
                    features=features_full,
                    batch_size=int(batch_size_trial),
                    learning_rate=learning_rate_trial,
                    early_stop_metric=early_stop_metric,
                    early_stop_patience=early_stop_patience,
                    random_state=random_state,
                    device=device,
                    epochs=epochs,
                    val_share=val_share,
                    verbose=max(verbose - 1, 0),
                    show_epochs=show_epochs,
                )
            except Exception as exc:
                print(f"  [Trial {trial.number}] FAILED: {exc}")
                return float("nan")

            def _last(values):
                if values is None:
                    return float("nan")
                return float(values[-1])

            row = {
                "trial": trial.number,
                "use_scaffold": bool(use_scaffold),
                "scaffold_target": scaffold_target_column,
                "batch_size": int(batch_size_trial),
                "learning_rate": learning_rate_trial,
                "layer_size": int(layer_size_trial),
                "dropout_rate": dropout_rate_trial,
                "best_epoch": getattr(model, "best_epoch", -1),
                "train_loss": getattr(model, "train_loss", float("nan")),
                "val_loss": getattr(model, "val_loss", float("nan")),
                "train_accuracy_scaffold": getattr(model, "train_accuracy_scaffold", float("nan")),
                "val_accuracy_scaffold": getattr(model, "val_accuracy_scaffold", float("nan")),
                "train_accuracy_main": getattr(model, "train_accuracy_main", float("nan")),
                "val_accuracy_main": getattr(model, "val_accuracy_main", float("nan")),
                "test_accuracy_scaffold": getattr(model, "test_accuracy_scaffold", float("nan")),
                "test_accuracy_main": getattr(model, "test_accuracy_main", float("nan")),
            }
            journal_rows.append(row)
            best_models_per_trial[trial.number] = model

            metric_value = row["val_accuracy_main"]
            
            print(
                f"[Trial {trial.number}] use_scaffold={use_scaffold}, "
                f"bs={int(batch_size_trial)}, lr={learning_rate_trial:.6f}, "
                f"hs={int(layer_size_trial)}, dr={dropout_rate_trial:.6f} | "
                f"best_ep={row['best_epoch']}, "
                f"main train_acc={row['train_accuracy_main']:.4f} "
                f"val_acc={row['val_accuracy_main']:.4f} "
                f"test_acc={row['test_accuracy_main']:.4f}"
            )
            return metric_value

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=random_state)

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,            
            sampler=sampler,            
        )
        study.optimize(objective, n_trials=n_trials)

        journal_df = pd.DataFrame(journal_rows)
        if journal_path is not None:
            journal_df.to_csv(journal_path, index=False)
            if verbose >= 1:
                print(f"\nExperiments journal saved to {journal_path}")

        try:
            best_trial_number = study.best_trial.number
            best_model = best_models_per_trial.get(best_trial_number)
        except ValueError:
            best_model = None

        if verbose >= 1 and best_model is not None:
            print("\n" + "=" * 70)
            print("BEST TRIAL")
            print("=" * 70)
            print(f"  Trial number : {study.best_trial.number}")
            print(f"  Val {early_stop_metric}: {study.best_trial.value:.4f}")
            print(f"  use_scaffold : {use_scaffold}")
            for key, value in study.best_trial.params.items():
                print(f"  {key}: {value}")

        return {"model": best_model, "study": study, "journal": journal_df}