import datetime as dt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
from typing import Optional, Union, Sequence, Tuple, Any, Dict, List
from torch.utils.data import DataLoader

import optuna

from scaffolded_learning.blocks import MLPBackbone, BinaryClassificationHead, MLPScaffoldModule
from scaffolded_learning.blocks import VGGBackbone, VGGScaffoldModule, VGGMultiClassClassificationHead

from scaffolded_learning.utils import MLPDataset
from scaffolded_learning.utils import _compute_auc
from scaffolded_learning.utils import _resolve_activation
from scaffolded_learning.utils import _resolve_suggestion
from scaffolded_learning.utils import _set_seed

class ScaffoldedMLPBinary(nn.Module):
    def __init__(
        self,
        input_size,
        n_targets,
        backbone_sizes = [128, 64, 32],
        scaffold_sizes=[32],
        dropout_rate=0.3,
        hidden_activation="relu",
        use_scaffold: bool = True,
    ):
        super(ScaffoldedMLP, self).__init__()
        if n_targets < 1:
            raise ValueError("n_targets must be >= 1")

        self.n_targets = n_targets
        # Switcher only affects training losses; architecture / init stay identical.
        self.use_scaffold = bool(use_scaffold)

        act = _resolve_activation(hidden_activation)

        if len(backbone_sizes) > 0:
            stage_sizes = [backbone_sizes[-1]]
        else:
            stage_sizes = [input_size]

        while len(stage_sizes) <= n_targets:
            next_idx = min(len(stage_sizes), len(scaffold_sizes) - 1)
            stage_sizes.append(scaffold_sizes[next_idx])

        self.backbone = MLPBackbone(input_size, backbone_sizes, dropout_rate, act)

        self.scaffold_modules = nn.ModuleList(
            [MLPScaffoldModule(stage_sizes[i], stage_sizes[i + 1], dropout_rate, act) for i in range(n_targets - 1)]
        )
        
        self.output_head = BinaryClassificationHead(stage_sizes[-2], stage_sizes[-1], dropout_rate, act)
                
    def forward(self, x):
        probs = []
        h = self.backbone(x)
        for scaffold_module in self.scaffold_modules:
            h, target_head = scaffold_module(h)
            probs.append(target_head.squeeze(1))
        
        probs.append(self.output_head(h).squeeze(1))
        return probs

    def set_scaffold_training(self, enabled: bool):
        """Turn scaffold-target supervision on or off (architecture unchanged)."""
        self.use_scaffold = bool(enabled)

    def weights_fingerprint(self) -> Dict[str, Any]:
        """Compact fingerprint of current parameters (sum + sha256 hash).

        Useful to verify that two freshly constructed models share the same
        initialisation without comparing every tensor element-wise.
        """
        import hashlib

        total_sum = 0.0
        hasher = hashlib.sha256()
        numel = 0
        for name, param in self.named_parameters():
            tensor = param.detach().float().cpu().contiguous()
            total_sum += float(tensor.sum().item())
            numel += tensor.numel()
            hasher.update(name.encode("utf-8"))
            hasher.update(tensor.numpy().tobytes())
        return {"sum": total_sum, "hash": hasher.hexdigest(), "numel": numel}

    @staticmethod
    def compare_init_weights(
        model_a: "ScaffoldedMLP",
        model_b: "ScaffoldedMLP",
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> Dict[str, Any]:
        """Compare parameter fingerprints of two model instances.

        Returns a dict with ``equal`` (hash match), both fingerprints, and
        whether parameter sums agree within ``rtol``/``atol``.
        """
        fa = model_a.weights_fingerprint()
        fb = model_b.weights_fingerprint()
        sum_close = abs(fa["sum"] - fb["sum"]) <= (atol + rtol * abs(fb["sum"]))
        return {
            "equal": fa["hash"] == fb["hash"],
            "sum_close": sum_close,
            "fingerprint_a": fa,
            "fingerprint_b": fb,
        }

    def _aggregate_losses(self, losses: Sequence[torch.Tensor]) -> torch.Tensor:
        """Sum all target losses when scaffold training is on; else final only."""
        if self.use_scaffold:
            return sum(losses)
        return losses[-1]

    def fit(self,
        train_data: Union[str, pd.DataFrame], 
        val_data: Union[str, pd.DataFrame] = None, 
        test_data: Union[str, pd.DataFrame, None] = None,
        target_column: str = None,
        scaffold_targets: Sequence[str] = [],
        features: Optional[Sequence[str]] = None,
        batch_size: int = None,
        learning_rate: float = 0.001,
        early_stop_metric: str = "total_val_loss",
        early_stop_patience: int = 10,
        random_state: int = 1223,
        device: str = "cpu",
        epochs: int = None,        
        val_share: float = None,        
        verbose: int = 1,
        show_epochs: int = 1,
        use_scaffold: Optional[bool] = None,
    ):
        # > Inputs validation
        # --> train_data validation
        if isinstance(train_data, str):
            if train_data.endswith(".parquet"):
                train_data = pd.read_parquet(train_data)
            elif train_data.endswith(".csv"):
                train_data = pd.read_csv(train_data)
            else:
                raise ValueError(f"train_data must be a parquet or csv file, got {train_data}")
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Loaded train data from {train_data} with shape {train_data.shape}")
        elif isinstance(train_data, pd.DataFrame):
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Use as train data dataframe with shape {train_data.shape}")
        else:
            raise ValueError(f"train_data must be a string or a pandas DataFrame, got {type(train_data)}")

        # --> val_data validation
        if isinstance(val_data, str):
            if val_data.endswith(".parquet"):
                val_data = pd.read_parquet(val_data)
            elif val_data.endswith(".csv"):
                val_data = pd.read_csv(val_data)
            else:
                raise ValueError(f"val_data must be a parquet or csv file, got {val_data}")
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Loaded val data from {val_data} with shape {val_data.shape}")
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
            
            train_data, val_data = train_test_split(train_data, test_size=val_share, random_state=random_state, stratify=train_data[[target_column]])
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Split train data into train and val data with shapes {train_data.shape} and {val_data.shape}")

        # --> test_data validation (optional)
        if isinstance(test_data, str):
            if test_data.endswith(".parquet"):
                test_data = pd.read_parquet(test_data)
            elif test_data.endswith(".csv"):
                test_data = pd.read_csv(test_data)
            else:
                raise ValueError(f"test_data must be a parquet or csv file, got {test_data}")
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
                
        # > Start training
        if use_scaffold is not None:
            self.set_scaffold_training(use_scaffold)
        _set_seed(random_state)
        if verbose >= 1:
            print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Start training, using device: {device}")
            print(
                f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: "
                f"Scaffold training {'ON' if self.use_scaffold else 'OFF'}"
            )
        
        # > Data preparation
        if features is not None:
            feature_columns = list(features)
            missing = [c for c in feature_columns if c not in train_data.columns]
            if missing:
                raise ValueError(f"features not present in train_data columns: {missing}")
        else:
            feature_columns = [c for c in train_data.columns if c not in list(scaffold_targets) + [target_column]]
        X_train = train_data[feature_columns].values
        y_train = train_data[scaffold_targets + [target_column]].values
        X_val = val_data[feature_columns].values
        y_val = val_data[scaffold_targets + [target_column]].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        test_loader = None
        if test_data is not None:
            missing_test = [c for c in feature_columns if c not in test_data.columns]
            if missing_test:
                raise ValueError(f"test_data is missing feature columns: {missing_test}")
            missing_targets = [c for c in list(scaffold_targets) + [target_column] if c not in test_data.columns]
            if missing_targets:
                raise ValueError(f"test_data is missing target columns: {missing_targets}")
            X_test = test_data[feature_columns].values
            y_test = test_data[scaffold_targets + [target_column]].values
            X_test_scaled = scaler.transform(X_test)
            test_dataset = MLPDataset(X_test_scaled, y_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if verbose >= 1:
            print("\nData splits:")
            print(f"Train: {X_train_scaled.shape[0]} samples")
            print(f"Validation: {X_val_scaled.shape[0]} samples")
            if test_loader is not None:
                print(f"Test: {X_test_scaled.shape[0]} samples")

        train_dataset = MLPDataset(X_train_scaled, y_train)
        val_dataset = MLPDataset(X_val_scaled, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        if verbose >= 1:
            print("\nModel architecture:")
            print(self)
            print(f"\nTotal parameters: {sum(p.numel() for p in self.parameters()):,}")

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if early_stop_metric not in {"total_val_loss", "target_val_loss", "target_auc"}:
            raise ValueError(
                "early_stop_metric must be one of: 'total_val_loss', 'target_val_loss', 'target_auc'"
            )

        best_metric = float("inf") if early_stop_metric in {"total_val_loss", "target_val_loss"} else -float("inf")
        best_model_state = copy.deepcopy(self.state_dict())
        best_epoch = 0
        patience_counter = 0

        self.train_losses = []
        self.val_losses = []

        if verbose >= 1:
            print("\nStarting training...")
            print(f"Device: {device}")
            print(f"Epochs: {epochs}")
            print(f"Learning rate: {learning_rate}\n")

        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_loss_per_target = np.zeros(self.n_targets, dtype=np.float64)
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self(features)
                losses = [criterion(outputs[i], labels[:, i]) for i in range(self.n_targets)]
                total_loss = self._aggregate_losses(losses)
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                for i, loss_i in enumerate(losses):
                    train_loss_per_target[i] += loss_i.item()

            # Validation phase
            self.eval()
            val_loss = 0.0
            val_loss_per_target = np.zeros(self.n_targets, dtype=np.float64)
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = self(features)
                    losses = [criterion(outputs[i], labels[:, i]) for i in range(self.n_targets)]
                    total_loss = self._aggregate_losses(losses)
                    val_loss += total_loss.item()
                    for i, loss_i in enumerate(losses):
                        val_loss_per_target[i] += loss_i.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_loss_per_target /= len(train_loader)
            val_loss_per_target /= len(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # > Check if results are better 
            if early_stop_metric == "total_val_loss":
                current_metric = val_loss
                is_better = current_metric < best_metric
            elif early_stop_metric == "target_val_loss":
                current_metric = val_loss_per_target[-1]
                is_better = current_metric < best_metric
            else:  # "target_auc"
                # Use last target validation AUC; larger is better.            
                val_auc = _compute_auc(self, val_loader, device)
                current_metric = val_auc[-1]
                is_better = current_metric > best_metric

            # > Print progress (including AUC on train and validation)
            if verbose >= 1 and ((epoch + 1) % show_epochs == 0 or epoch == 0):
                train_auc = _compute_auc(self, train_loader, device)
                if early_stop_metric != "target_auc":
                    val_auc = _compute_auc(self, val_loader, device)                
                per_target_stats = " | ".join(
                    [
                        f"T{i}: TL={train_loss_per_target[i]:.4f}, VL={val_loss_per_target[i]:.4f}, "
                        f"TAUC={train_auc[i]:.4f}, VAUC={val_auc[i]:.4f}"
                        for i in range(self.n_targets)
                    ]
                )
                print(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | {per_target_stats}"
                )

            # > Early stopping 
            if is_better:
                best_metric = current_metric
                best_model_state = copy.deepcopy(self.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    if verbose >= 1:
                        print(f"\nEarly stopping at epoch {epoch+1} with metric={early_stop_metric}, best={best_metric:.4f}")
                    break

        # > Restore best model weights before returning
        self.load_state_dict(best_model_state)

        # > Compute final per-target AUC on best model (for reporting / hyperopt)
        self.train_auc_final = _compute_auc(self, train_loader, device)
        self.val_auc_final = _compute_auc(self, val_loader, device)
        self.test_auc_final = (
            _compute_auc(self, test_loader, device) if test_loader is not None else None
        )
        self.best_epoch = best_epoch
        self.best_metric = best_metric

        if verbose >= 1:
            print(
                f"\nTraining completed. Best early-stop metric ({early_stop_metric}): "
                f"{best_metric:.4f}, best epoch: {best_epoch}"
            )
            test_auc_str = (
                f"  test={[f'{x:.4f}' for x in self.test_auc_final]}"
                if self.test_auc_final is not None else ""
            )
            print(
                f"Best-model AUC per target  train={[f'{x:.4f}' for x in self.train_auc_final]}  "
                f"val={[f'{x:.4f}' for x in self.val_auc_final]}{test_auc_str}"
            )


    def save(self, path: str):
        raise NotImplementedError("Saving is not implemented yet")

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError("Loading is not implemented yet")

    @classmethod
    def init_and_fit(
        cls,
        train_data: Union[str, pd.DataFrame],
        targets: Optional[Sequence[str]] = None,
        target_column: Optional[str] = None,        
        scaffold_targets: Optional[Sequence[str]] = None,        
        features: Optional[Sequence[str]] = None,
        n_targets: Optional[int] = None,
        backbone_sizes: Optional[Sequence[int]] = None,
        hidden_sizes: Optional[Sequence[int]] = None,        
        hidden_activation: str = "relu",
        use_scaffold: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        val_data: Union[str, pd.DataFrame, None] = None,
        test_data: Union[str, pd.DataFrame, None] = None,
        val_share: Optional[float] = None,        
        early_stop_metric: str = "target_auc",
        early_stop_patience: int = 10,
        random_state: int = 1223,
        epochs: Optional[int] = None,
        verbose: int = 1,
        show_epochs: int = 1,
        # Hyperopt parameters        
        batch_size: Union[int, Tuple[str, Dict[str, Any]], None] = None,
        learning_rate: Union[float, Tuple[str, Dict[str, Any]]] = None,
        dropout_rate: Union[float, Tuple[str, Dict[str, Any]]] = None,
        first_layer_size: Optional[Union[int, Tuple[str, Dict[str, Any]]]] = None,
        intermediate_layers_size: Optional[Union[int, Tuple[str, Dict[str, Any]]]] = None,
        scaffold_levels: Union[int, Tuple[str, Dict[str, Any]], None] = None,        
        n_trials: int = 100,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        journal_path: Optional[str] = None,
    ):
        """Build and fit a `ScaffoldedMLP`, optionally running Optuna hyperopt.

        Hyperopt is triggered when any of `batch_size`, `learning_rate`,
        `first_layer_size`, `intermediate_layers_size`, `dropout_rate`, or
        `scaffold_levels` is a tuple `(suggest_kind, kwargs)` – e.g.
        `("categorical", {"choices": [256, 512]})` or
        `("float", {"low": 1e-4, "high": 1e-2})` – or when
        `scaffold_levels` is an int strictly smaller than `len(scaffold_targets)`.

        Returns
        -------
        - When hyperopt is OFF: the trained `ScaffoldedMLP` instance.
        - When hyperopt is ON: dict with keys `model` (best trained model),
          `study` (Optuna study) and `journal` (DataFrame of trial results).
        """        
        # > Inputs validation
        # --> targets validation
        if targets is None and (scaffold_targets is None or target_column is None):
            raise ValueError("targets or (scaffold_targets and target_column) must be provided")
        if targets is not None:
            scaffold_targets = targets[:-1]
            target_column = targets[-1]   
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Use {target_column} as target column")     

        # > Load train_data once if it is a path so that we can reuse it across trials
        if isinstance(train_data, str):
            if train_data.endswith(".parquet"):
                train_data = pd.read_parquet(train_data)
            elif train_data.endswith(".csv"):
                train_data = pd.read_csv(train_data)
            else:
                raise ValueError(f"train_data must be a parquet or csv file, got {train_data}")
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

        if backbone_sizes is None:
            backbone_sizes = []

        # --> device validation
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if verbose >= 1:
                print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Use {device} as device")

        # --> study_name validation
        if study_name is None:
            study_name = f"ScaffoldedMLP_hyperopt_{dt.datetime.now():%Y-%m-%d_%H-%M-%S}"            

        # > Detect hyperopt mode        
        # if any of the hyperopt_params is None, then suggest it
        if batch_size is None:
            batch_size = ("categorical", {"choices": [128, 256, 512, 1024, 2048]})
        if learning_rate is None:
            learning_rate = ("float", {"low": 1e-3, "high": 1e-2})
        if first_layer_size is None:
            first_layer_size = ("categorical", {"choices": [64, 128, 256, 512, 1024]})
        if intermediate_layers_size is None:
            intermediate_layers_size = ("categorical", {"choices": [32, 64, 128, 256, 512]})
        if dropout_rate is None:    
            dropout_rate = ("float", {"low": 0.05, "high": 0.95})
        if scaffold_levels is None:
            scaffold_levels = ("int", {"low": 0, "high": len(scaffold_targets)})
        
        hyperopt_flag = any(isinstance(p, tuple) for p in [batch_size, learning_rate, first_layer_size, intermediate_layers_size, dropout_rate, scaffold_levels])
        if isinstance(scaffold_levels, int): 
            if scaffold_levels < len(scaffold_targets):
                hyperopt_flag = True

        # > Feature set 
        if features is None:
            features_full = [c for c in train_data.columns if c not in scaffold_targets + [target_column]]
        else:
            features_full = features

        if not hyperopt_flag:
            # ── Plain training path ────────────────────────────────────────
            n_targets_resolved = n_targets if n_targets is not None else len(scaffold_targets) + 1
            if n_targets_resolved != len(scaffold_targets) + 1:
                raise ValueError(
                    f"n_targets ({n_targets_resolved}) must equal len(scaffold_targets)+1 "
                    f"({len(scaffold_targets) + 1})"
                )

            hidden_sizes_resolved = hidden_sizes
            if hidden_sizes_resolved is None:
                if first_layer_size is not None and intermediate_layers_size is not None:
                    hidden_sizes_resolved = [int(first_layer_size), int(intermediate_layers_size)]
                else:
                    hidden_sizes_resolved = [128, 64, 32]

            input_size = len(features_full)

            _set_seed(random_state)
            model = cls(
                input_size=input_size,
                n_targets=n_targets_resolved,
                backbone_sizes=backbone_sizes,
                scaffold_sizes=hidden_sizes_resolved,
                dropout_rate=dropout_rate,
                hidden_activation=hidden_activation,
                use_scaffold=use_scaffold,
            )
            model.to(device)
            model.fit(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                target_column=target_column,
                scaffold_targets=scaffold_targets,
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
                use_scaffold=use_scaffold,
            )
            return model

        # ── Hyperopt path ─────────────────────────────────────────────────
        if early_stop_metric != "target_auc":
            raise ValueError("early_stop_metric must be 'target_auc' for hyperopt fine tuning")

        # Silence Optuna's per-trial "Trial N finished with value..." log;
        # the objective already prints a more informative line per trial.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        journal_rows: List[Dict[str, Any]] = []
        best_models_per_trial: Dict[int, "ScaffoldedMLP"] = {}

        def _suggest_or_const(trial, value, param_name):
            """If `value` is a (kind, kwargs) tuple, use the trial to suggest;
            otherwise return `value` unchanged."""
            if isinstance(value, tuple):
                return _resolve_suggestion(trial, value[0])(param_name, **value[1])
            return value

        def objective(trial: optuna.Trial) -> float:
            batch_size_trial = _suggest_or_const(trial, batch_size, "batch_size")
            learning_rate_trial = round(float(_suggest_or_const(trial, learning_rate, "learning_rate")), 6)            
            first_layer_size_trial = _suggest_or_const(trial, first_layer_size, "first_layer_size")
            intermediate_layers_size_trial = _suggest_or_const(
                trial, intermediate_layers_size, "intermediate_layers_size"
            )
            dropout_rate_trial = round(float(_suggest_or_const(trial, dropout_rate, "dropout_rate")), 6)

            if isinstance(scaffold_levels, tuple):
                scaffold_levels_trial = _resolve_suggestion(trial, scaffold_levels[0])(
                    "scaffold_levels", **scaffold_levels[1]
                )
            elif isinstance(scaffold_levels, int):
                scaffold_levels_trial = scaffold_levels
            else:
                scaffold_levels_trial = len(scaffold_targets)

            # > Pick a deterministic-but-trial-specific subset of scaffold_targets
            if scaffold_levels_trial >= len(scaffold_targets):
                scaffold_targets_trial = list(scaffold_targets)
            else:
                rng = random.Random(random_state + trial.number)
                pool = list(scaffold_targets)
                scaffold_targets_trial = []
                for _ in range(scaffold_levels_trial):
                    j = rng.randint(0, len(pool) - 1)
                    scaffold_targets_trial.append(pool.pop(j))

            # > Resolve hidden sizes
            if first_layer_size_trial is not None and intermediate_layers_size_trial is not None:
                hidden_sizes_trial = [int(first_layer_size_trial), int(intermediate_layers_size_trial)]
            elif hidden_sizes is not None:
                hidden_sizes_trial = list(hidden_sizes)
            else:
                hidden_sizes_trial = [128, 64, 32]

            n_targets_trial = len(scaffold_targets_trial) + 1
            input_size_trial = len(features_full)

            try:
                _set_seed(random_state)
                model = cls(
                    input_size=input_size_trial,
                    n_targets=n_targets_trial,
                    backbone_sizes=backbone_sizes,
                    scaffold_sizes=hidden_sizes_trial,
                    dropout_rate=dropout_rate_trial,
                    hidden_activation=hidden_activation,
                    use_scaffold=use_scaffold,
                )
                model.to(device)
                model.fit(
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    target_column=target_column,
                    scaffold_targets=scaffold_targets_trial,
                    features=features_full,
                    batch_size=batch_size_trial,
                    learning_rate=learning_rate_trial,
                    early_stop_metric=early_stop_metric,
                    early_stop_patience=early_stop_patience,
                    random_state=random_state,
                    device=device,
                    epochs=epochs,
                    val_share=val_share,
                    verbose=max(verbose - 1, 0),
                    show_epochs=show_epochs,
                    use_scaffold=use_scaffold,
                )
            except Exception as exc:
                print(f"  [Trial {trial.number}] FAILED: {exc}")
                return float("nan")

            train_auc_last = (
                model.train_auc_final[-1] if getattr(model, "train_auc_final", None) else float("nan")
            )
            val_auc_last = (
                model.val_auc_final[-1] if getattr(model, "val_auc_final", None) else float("nan")
            )
            test_auc_last = (
                model.test_auc_final[-1]
                if getattr(model, "test_auc_final", None) is not None and model.test_auc_final
                else float("nan")
            )
            best_epoch_trial = getattr(model, "best_epoch", -1)

            row = {
                "trial": trial.number,
                "scaffold_levels": scaffold_levels_trial,
                "scaffold_targets": ",".join(scaffold_targets_trial),
                "batch_size": batch_size_trial,
                "learning_rate": learning_rate_trial,
                "first_layer_size": first_layer_size_trial,
                "intermediate_layers_size": intermediate_layers_size_trial,
                "hidden_sizes": str(hidden_sizes_trial),
                "dropout_rate": dropout_rate_trial,
                "n_targets": n_targets_trial,
                "best_epoch": best_epoch_trial,
                "train_auc_last_target": train_auc_last,
                "val_auc_last_target": val_auc_last,
                "test_auc_last_target": test_auc_last,
            }
            journal_rows.append(row)
            best_models_per_trial[trial.number] = model

            test_auc_str = f"  test={test_auc_last:.4f}" if test_data is not None else ""
            print(
                f"[Trial {trial.number}] "
                f"scaffold_levels={scaffold_levels_trial}, "
                f"scaffold_targets={scaffold_targets_trial}, "
                f"bs={batch_size_trial}, lr={float(learning_rate_trial):.6f}, "
                f"hs={hidden_sizes_trial}, dr={float(dropout_rate_trial):.6f} | "
                f"best_ep={best_epoch_trial}, "
                f"AUC last target  train={train_auc_last:.4f}  val={val_auc_last:.4f}"
                f"{test_auc_str}"
            )

            return val_auc_last

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=random_state)

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            load_if_exists=storage is not None,
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
            print(f"  Val AUC (last target): {study.best_trial.value:.4f}")
            for k, v in study.best_trial.params.items():
                print(f"  {k}: {v}")

        return {"model": best_model, "study": study, "journal": journal_df}