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

class ScaffoldedVGG(nn.Module):
    def __init__(
        self,
        n_classes_scaffold,
        n_classes_output,
        vgg_type: str = "vgg16",
        dropout_rate=0.5,
    ):
        # > Input validation
        if vgg_type not in ["vgg16", "vgg19"]:
            raise ValueError("Invalid VGG type")
        if vgg_type == "vgg19":
            raise ValueError("VGG19 is not supported yet")
        
        super(ScaffoldedVGG, self).__init__()
        self.backbone = VGGBackbone(vgg_type, dropout_rate)
        self.scaffold_module = VGGScaffoldModule(n_classes_scaffold, dropout_rate)        
        self.output_head = VGGMultiClassClassificationHead(n_classes_output, dropout_rate)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = self.backbone(x)        
        h, logits_scaffold = self.scaffold_module(h)
        logits_output = self.output_head(h)

        return logits_scaffold, logits_output

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 250,
        learning_rate: float = 0.1,
        lr_decay: float = 1e-6,
        lr_drop: int = 20,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        label_smoothing: float = 0.0,
        scaffold_loss_weight: float = 1.0,
        output_loss_weight: float = 1.0,
        early_stop_metric: str = "output_val_acc",
        early_stop_patience: int = 20,
        random_state: int = 1223,
        device: str = "cpu",
        use_amp: bool = False,
        verbose: int = 1,
        show_epochs: int = 1,
    ):
        """Train the scaffolded VGG

        Each batch yielded by the loaders must be a tuple
        ``(inputs, scaffold_target, output_target)`` where ``scaffold_target``
        holds the coarse (scaffold) class indices and ``output_target`` holds
        the fine (output) class indices. The dual-head loss is the weighted
        sum of the cross-entropy on the scaffold head and the output head.
        """
        # > Inputs validation
        if early_stop_metric not in {
            "total_val_loss",
            "output_val_loss",
            "scaffold_val_loss",
            "output_val_acc",
            "scaffold_val_acc",
        }:
            raise ValueError(
                "early_stop_metric must be one of: 'total_val_loss', "
                "'output_val_loss', 'scaffold_val_loss', 'output_val_acc', "
                "'scaffold_val_acc'"
            )

        # When no validation loader is provided, early stopping and best-model
        # selection fall back to the training metrics.
        eval_loader = val_loader if val_loader is not None else train_loader

        _set_seed(random_state)
        self.to(device)

        if verbose >= 1:
            print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Start training, using device: {device}")
            print("\nModel architecture:")
            print(self)
            print(f"\nTotal parameters: {sum(p.numel() for p in self.parameters()):,}")

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = optim.SGD(
            self.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        scaler = None
        if use_amp and str(device).startswith("cuda"):
            scaler = torch.cuda.amp.GradScaler()

        higher_is_better = early_stop_metric in {"output_val_acc", "scaffold_val_acc"}
        best_metric = -float("inf") if higher_is_better else float("inf")
        best_model_state = copy.deepcopy(self.state_dict())
        best_epoch = 0
        patience_counter = 0
        global_iters = 0

        self.train_losses = []
        self.val_losses = []
        self.history = {
            "train_loss": [],
            "train_scaffold_acc": [],
            "train_output_acc": [],
            "val_loss": [],
            "val_scaffold_acc": [],
            "val_output_acc": [],
        }

        if verbose >= 1:
            print("\nStarting training...")
            print(f"Epochs: {epochs}")
            print(f"Learning rate: {learning_rate}\n")

        for epoch in range(epochs):            
            lr_epoch = learning_rate * (0.5 ** (epoch // lr_drop))

            # > Training phase
            self.train()
            train_loss = 0.0
            train_scaffold_loss = 0.0
            train_output_loss = 0.0
            train_correct_scaffold = 0
            train_correct_output = 0
            train_total = 0
            for inputs, scaffold_target, output_target in train_loader:                
                global_iters += 1
                lr_batch = lr_epoch / (1.0 + lr_decay * global_iters)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_batch

                inputs = inputs.to(device, non_blocking=True)
                scaffold_target = scaffold_target.to(device, non_blocking=True)
                output_target = output_target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits_scaffold, logits_output = self(inputs)
                        loss_scaffold = criterion(logits_scaffold, scaffold_target)
                        loss_output = criterion(logits_output, output_target)
                        total_loss = (
                            scaffold_loss_weight * loss_scaffold
                            + output_loss_weight * loss_output
                        )
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits_scaffold, logits_output = self(inputs)
                    loss_scaffold = criterion(logits_scaffold, scaffold_target)
                    loss_output = criterion(logits_output, output_target)
                    total_loss = (
                        scaffold_loss_weight * loss_scaffold
                        + output_loss_weight * loss_output
                    )
                    total_loss.backward()
                    optimizer.step()

                bs = inputs.size(0)
                train_loss += total_loss.item() * bs
                train_scaffold_loss += loss_scaffold.item() * bs
                train_output_loss += loss_output.item() * bs
                train_correct_scaffold += (logits_scaffold.argmax(dim=1) == scaffold_target).sum().item()
                train_correct_output += (logits_output.argmax(dim=1) == output_target).sum().item()
                train_total += bs

            train_total = max(train_total, 1)
            train_loss /= train_total
            train_scaffold_loss /= train_total
            train_output_loss /= train_total
            train_scaffold_acc = train_correct_scaffold / train_total
            train_output_acc = train_correct_output / train_total
            self.train_losses.append(train_loss)

            # > Validation phase
            val_metrics = self._evaluate(
                eval_loader,
                criterion,
                device,
                scaffold_loss_weight,
                output_loss_weight,
            )
            self.val_losses.append(val_metrics["total_loss"])

            self.history["train_loss"].append(train_loss)
            self.history["train_scaffold_acc"].append(train_scaffold_acc)
            self.history["train_output_acc"].append(train_output_acc)
            self.history["val_loss"].append(val_metrics["total_loss"])
            self.history["val_scaffold_acc"].append(val_metrics["scaffold_acc"])
            self.history["val_output_acc"].append(val_metrics["output_acc"])

            # > Check if results are better
            if early_stop_metric == "total_val_loss":
                current_metric = val_metrics["total_loss"]
            elif early_stop_metric == "output_val_loss":
                current_metric = val_metrics["output_loss"]
            elif early_stop_metric == "scaffold_val_loss":
                current_metric = val_metrics["scaffold_loss"]
            elif early_stop_metric == "output_val_acc":
                current_metric = val_metrics["output_acc"]
            else:  # "scaffold_val_acc"
                current_metric = val_metrics["scaffold_acc"]

            is_better = (
                current_metric > best_metric if higher_is_better else current_metric < best_metric
            )

            # > Print progress
            if verbose >= 1 and ((epoch + 1) % show_epochs == 0 or epoch == 0):
                print(
                    f"Epoch [{epoch+1}/{epochs}] - lr: {lr_batch:.5f} | "
                    f"Train: loss={train_loss:.4f}, "
                    f"scaffold_acc={train_scaffold_acc:.4f}, output_acc={train_output_acc:.4f} | "
                    f"Val: loss={val_metrics['total_loss']:.4f}, "
                    f"scaffold_acc={val_metrics['scaffold_acc']:.4f}, "
                    f"output_acc={val_metrics['output_acc']:.4f}"
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
                        print(
                            f"\nEarly stopping at epoch {epoch+1} with metric="
                            f"{early_stop_metric}, best={best_metric:.4f}"
                        )
                    break

        # > Restore best model weights before returning
        self.load_state_dict(best_model_state)

        # > Compute final metrics on the best model (for reporting / hyperopt)
        self.train_metrics_final = self._evaluate(
            train_loader, criterion, device, scaffold_loss_weight, output_loss_weight
        )
        self.val_metrics_final = (
            self._evaluate(val_loader, criterion, device, scaffold_loss_weight, output_loss_weight)
            if val_loader is not None
            else None
        )
        self.test_metrics_final = (
            self._evaluate(test_loader, criterion, device, scaffold_loss_weight, output_loss_weight)
            if test_loader is not None
            else None
        )
        self.best_epoch = best_epoch
        self.best_metric = best_metric

        if verbose >= 1:
            print(
                f"\nTraining completed. Best early-stop metric ({early_stop_metric}): "
                f"{best_metric:.4f}, best epoch: {best_epoch}"
            )
            print(
                f"Best-model output accuracy  train={self.train_metrics_final['output_acc']:.4f}"
                + (
                    f"  val={self.val_metrics_final['output_acc']:.4f}"
                    if self.val_metrics_final is not None
                    else ""
                )
                + (
                    f"  test={self.test_metrics_final['output_acc']:.4f}"
                    if self.test_metrics_final is not None
                    else ""
                )
            )

        return self

    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        device: str,
        scaffold_loss_weight: float = 1.0,
        output_loss_weight: float = 1.0,
    ) -> Dict[str, float]:
        """Compute loss and accuracy per head over a loader."""
        self.eval()
        total_loss = 0.0
        scaffold_loss_sum = 0.0
        output_loss_sum = 0.0
        correct_scaffold = 0
        correct_output = 0
        total = 0
        for inputs, scaffold_target, output_target in loader:
            inputs = inputs.to(device, non_blocking=True)
            scaffold_target = scaffold_target.to(device, non_blocking=True)
            output_target = output_target.to(device, non_blocking=True)
            logits_scaffold, logits_output = self(inputs)
            loss_scaffold = criterion(logits_scaffold, scaffold_target)
            loss_output = criterion(logits_output, output_target)
            batch_total = (
                scaffold_loss_weight * loss_scaffold + output_loss_weight * loss_output
            )
            bs = inputs.size(0)
            total_loss += batch_total.item() * bs
            scaffold_loss_sum += loss_scaffold.item() * bs
            output_loss_sum += loss_output.item() * bs
            correct_scaffold += (logits_scaffold.argmax(dim=1) == scaffold_target).sum().item()
            correct_output += (logits_output.argmax(dim=1) == output_target).sum().item()
            total += bs
        n = max(total, 1)
        return {
            "total_loss": total_loss / n,
            "scaffold_loss": scaffold_loss_sum / n,
            "output_loss": output_loss_sum / n,
            "scaffold_acc": correct_scaffold / n,
            "output_acc": correct_output / n,
        }

    def save(self, path: str):
        raise NotImplementedError("Saving is not implemented yet")

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError("Loading is not implemented yet")

    @classmethod
    def init_and_fit(
        cls,
        n_classes_scaffold: int,
        n_classes_output: int,
        vgg_type: str = "vgg16",
        dropout_rate: float = 0.5,
        **fit_kwargs,
    ):
        """Construct a `ScaffoldedVGG` and immediately train it.

        Constructor arguments (`n_classes_scaffold`, `n_classes_output`,
        `vgg_type`, `dropout_rate`) configure the network; all remaining
        keyword arguments are forwarded to `fit` (e.g. `train_loader`,
        `val_loader`, `epochs`, `learning_rate`, `device`).
        """
        # Seed *before* constructing the model so weight initialisation is
        # reproducible (`fit` re-seeds again before the training loop).
        _set_seed(fit_kwargs.get("random_state", 1223))
        model = cls(
            n_classes_scaffold=n_classes_scaffold,
            n_classes_output=n_classes_output,
            vgg_type=vgg_type,
            dropout_rate=dropout_rate,
        )
        model.fit(**fit_kwargs)
        return model
