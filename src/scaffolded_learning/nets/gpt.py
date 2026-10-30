"""Scaffolded GPT model for next-token prediction with an auxiliary class head."""

import copy
import datetime as dt
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scaffolded_learning.blocks import GPTBackbone, GPTScaffoldModule, GPTMultiClassClassificationHead
from scaffolded_learning.utils import _set_seed


class ScaffoldedGPT(nn.Module):
    def __init__(
        self,
        n_classes_scaffold,
        n_classes_output,
        vocab_size,
        block_size,
        n_embd,
        n_head,
        n_blocks,
        dropout_rate=0.5,
    ):
        # > Input validation
        if n_blocks < 2:
            raise ValueError("n_blocks must be >= 2")

        super(ScaffoldedGPT, self).__init__()

        self.n_classes_scaffold = n_classes_scaffold
        self.n_classes_output = n_classes_output
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_blocks = n_blocks
        self.dropout_rate = dropout_rate

        self.backbone = GPTBackbone(vocab_size, block_size, n_embd, n_head, n_blocks - 2, dropout_rate)
        self.scaffold_module = GPTScaffoldModule(n_embd, n_head, n_classes_scaffold, block_size, dropout_rate)
        self.output_head = GPTMultiClassClassificationHead(n_embd, n_head, n_classes_output, block_size, dropout_rate)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x):        
        h = self.backbone(x)        
        h, logits_scaffold = self.scaffold_module(h)
        logits_output = self.output_head(h)

        return logits_scaffold, logits_output

    def save(self, path: str):
        raise NotImplementedError("Saving is not implemented yet")

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError("Loading is not implemented yet")

    def fit(
        self,
        train_data: Union[torch.Tensor, Sequence[int]],
        scaff_train: Union[torch.Tensor, Sequence[int]],
        val_data: Union[torch.Tensor, Sequence[int], None] = None,
        scaff_val: Union[torch.Tensor, Sequence[int], None] = None,
        block_size: Optional[int] = None,
        batch_size: int = 64,
        max_iters: int = 5000,
        eval_interval: int = 50,
        eval_iters: int = 200,
        learning_rate: float = 3e-4,
        scaffold_loss_weight: float = 1.0,
        output_loss_weight: float = 1.0,
        random_state: int = 1337,
        device: str = "cpu",
        verbose: int = 1,
    ):
        """Train the scaffolded GPT on a contiguous stream of token ids.

        The model is trained on a character/token-level next-token prediction
        task (the *output* head) jointly with an auxiliary class prediction
        task (the *scaffold* head). Both heads are trained with cross-entropy
        and predict the *next* position:

        - ``output`` target: the next token id (``vocab_size`` classes).
        - ``scaffold`` target: ``scaff_train`` / ``scaff_val`` at that same next
          position (``n_classes_scaffold`` classes).

        ``scaff_train`` is aligned with ``train_data`` and ``scaff_val`` with
        ``val_data``.

        Parameters
        ----------
        train_data, val_data : 1-D tensors (or sequences) of token ids.
        scaff_train, scaff_val : 1-D tensors of scaffold class ids, same
            length as the matching token stream. ``scaff_val`` is required when
            ``val_data`` is given.
        """
        _set_seed(random_state)

        if block_size is None:
            block_size = self.block_size

        self.to(device)

        # > Normalise inputs to long tensors on the target device
        train_data = torch.as_tensor(train_data, dtype=torch.long).to(device)
        scaff_train = torch.as_tensor(scaff_train, dtype=torch.long).to(device)
        if val_data is not None:
            val_data = torch.as_tensor(val_data, dtype=torch.long).to(device)
            if scaff_val is None:
                raise ValueError("scaff_val is required when val_data is given")
            scaff_val = torch.as_tensor(scaff_val, dtype=torch.long).to(device)
        elif scaff_val is not None:
            raise ValueError("scaff_val was given without val_data")

        if len(train_data) <= block_size:
            raise ValueError("train_data must contain more than block_size tokens")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate)

        def get_batch(split):
            source = train_data if (split == "train" or val_data is None) else val_data
            scaff = scaff_train if (split == "train" or scaff_val is None) else scaff_val
            ix = torch.randint(len(source) - block_size, (batch_size,))
            x = torch.stack([source[i:i + block_size] for i in ix])
            y_out = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
            y_scaffold = torch.stack([scaff[i + 1:i + block_size + 1] for i in ix])
            return x, y_scaffold, y_out

        @torch.no_grad()
        def estimate_loss():
            self.eval()
            splits = ["train"] + (["val"] if val_data is not None else [])
            out = {}
            for split in splits:
                agg = {
                    "total_loss": 0.0, "output_loss": 0.0, "scaffold_loss": 0.0,
                    "output_acc": 0.0, "scaffold_acc": 0.0,
                }
                for _ in range(eval_iters):
                    xb, scaffold_b, out_b = get_batch(split)
                    logits_scaffold, logits_output = self(xb)
                    bt = out_b.numel()
                    logits_output_flat = logits_output.reshape(bt, -1)
                    logits_scaffold_flat = logits_scaffold.reshape(bt, -1)
                    out_flat = out_b.reshape(bt)
                    scaffold_flat = scaffold_b.reshape(bt)
                    loss_output = criterion(logits_output_flat, out_flat)
                    loss_scaffold = criterion(logits_scaffold_flat, scaffold_flat)
                    total = output_loss_weight * loss_output + scaffold_loss_weight * loss_scaffold
                    agg["total_loss"] += total.item()
                    agg["output_loss"] += loss_output.item()
                    agg["scaffold_loss"] += loss_scaffold.item()
                    agg["output_acc"] += (logits_output_flat.argmax(dim=-1) == out_flat).float().mean().item()
                    agg["scaffold_acc"] += (logits_scaffold_flat.argmax(dim=-1) == scaffold_flat).float().mean().item()
                out[split] = {k: v / eval_iters for k, v in agg.items()}
            self.train()
            return out

        self.eval_steps = []
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_output_loss": [], "val_output_loss": [],
            "train_scaffold_loss": [], "val_scaffold_loss": [],
            "train_output_acc": [], "val_output_acc": [],
            "train_scaffold_acc": [], "val_scaffold_acc": [],
        }

        best_metric = float("inf")
        best_model_state = copy.deepcopy(self.state_dict())
        best_iter = 0

        if verbose >= 1:
            print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}: Start training, using device: {device}")
            print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
            print(f"Iterations: {max_iters} | batch size: {batch_size} | block size: {block_size}")
            print(f"Learning rate: {learning_rate}\n")

        self.train()
        for it in range(max_iters):
            # > Periodic evaluation on both splits
            if it % eval_interval == 0 or it == max_iters - 1:
                metrics = estimate_loss()
                self.eval_steps.append(it)
                tr = metrics["train"]
                va = metrics.get("val")
                self.history["train_loss"].append(tr["total_loss"])
                self.history["train_output_loss"].append(tr["output_loss"])
                self.history["train_scaffold_loss"].append(tr["scaffold_loss"])
                self.history["train_output_acc"].append(tr["output_acc"])
                self.history["train_scaffold_acc"].append(tr["scaffold_acc"])
                if va is not None:
                    self.history["val_loss"].append(va["total_loss"])
                    self.history["val_output_loss"].append(va["output_loss"])
                    self.history["val_scaffold_loss"].append(va["scaffold_loss"])
                    self.history["val_output_acc"].append(va["output_acc"])
                    self.history["val_scaffold_acc"].append(va["scaffold_acc"])

                # > Track best model by output (next-token) loss
                ref = va if va is not None else tr
                if ref["output_loss"] < best_metric:
                    best_metric = ref["output_loss"]
                    best_model_state = copy.deepcopy(self.state_dict())
                    best_iter = it

                if verbose >= 1:
                    val_str = (
                        f", val loss {va['total_loss']:.4f} "
                        f"(output {va['output_loss']:.4f}, scaffold {va['scaffold_loss']:.4f}) | "
                        f"val acc output {va['output_acc']:.4f}, scaffold {va['scaffold_acc']:.4f}"
                        if va is not None else ""
                    )
                    print(
                        f"step {it}: train loss {tr['total_loss']:.4f} "
                        f"(output {tr['output_loss']:.4f}, scaffold {tr['scaffold_loss']:.4f}) | "
                        f"train acc output {tr['output_acc']:.4f}, scaffold {tr['scaffold_acc']:.4f}"
                        f"{val_str}"
                    )

            # > Training step
            xb, scaffold_b, out_b = get_batch("train")
            logits_scaffold, logits_output = self(xb)
            bt = out_b.numel()
            loss_output = criterion(logits_output.reshape(bt, -1), out_b.reshape(bt))
            loss_scaffold = criterion(logits_scaffold.reshape(bt, -1), scaffold_b.reshape(bt))
            total_loss = output_loss_weight * loss_output + scaffold_loss_weight * loss_scaffold

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

        # > Restore the best model weights before returning
        self.load_state_dict(best_model_state)
        self.best_iter = best_iter
        self.best_metric = best_metric

        if verbose >= 1:
            print(
                f"\nTraining completed. Best output loss: {best_metric:.4f} at step {best_iter}"
            )

        return self

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """Autoregressively sample ``max_new_tokens`` from the output (vocab) head.

        ``idx`` is a ``(B, T)`` tensor of token ids; returns the extended
        ``(B, T + max_new_tokens)`` tensor.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            _, logits_output = self(idx_cond)
            logits = logits_output[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @classmethod
    def init_and_fit(
        cls,
        n_classes_scaffold: int,
        n_classes_output: int,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_blocks: int,        
        dropout_rate: float = 0.5,
        **fit_kwargs,
    ):
        """Construct a `ScaffoldedGPT` and immediately train it.

        Constructor arguments configure the network; all remaining keyword
        arguments are forwarded to `fit` (e.g. `train_data`, `scaff_train`,
        `val_data`, `scaff_val`, `max_iters`, `learning_rate`, `device`).
        """
        # Seed *before* constructing the model so weight initialisation is
        # reproducible (`fit` re-seeds again before the training loop).
        _set_seed(fit_kwargs.get("random_state", 1223))
        model = cls(
            n_classes_scaffold=n_classes_scaffold,
            n_classes_output=n_classes_output,
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            n_blocks=n_blocks,
            dropout_rate=dropout_rate,
        )
        model.fit(**fit_kwargs)
        return model
