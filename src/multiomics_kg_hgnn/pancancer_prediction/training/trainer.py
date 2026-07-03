"""Training / validation / test loop for the heterogeneous model.

Graph classification with macro-F1 as the primary metric (proposta sez. 5).
Kept intentionally small and dependency-light; mirrors the role of MOGNN-TF's
Trainer without pulling in its BRCA-specific machinery.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score


class HeteroTrainer:
    def __init__(self, model, optimizer, device="cpu", class_weights=None,
                 patience=20, scheduler=None, logger=print):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.patience = patience
        self.log = logger
        w = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        self.criterion = nn.CrossEntropyLoss(weight=w)
        self._best_state = None
        self._best_val = -1.0

    def _run_epoch(self, loader, train):
        self.model.train(train)
        total_loss, ys, preds = 0.0, [], []
        torch.set_grad_enabled(train)
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += float(loss) * batch.num_graphs
            ys.append(batch.y.cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
        torch.set_grad_enabled(True)
        y = np.concatenate(ys); p = np.concatenate(preds)
        return {
            "loss": total_loss / len(y),
            "macro_f1": f1_score(y, p, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, p),
        }

    def fit(self, train_loader, val_loader, num_epochs):
        for epoch in range(1, num_epochs + 1):
            tr = self._run_epoch(train_loader, train=True)
            va = self._run_epoch(val_loader, train=False)
            if self.scheduler is not None:
                self.scheduler.step()
            self.log(f"epoch {epoch:03d} | train loss {tr['loss']:.4f} f1 {tr['macro_f1']:.4f} "
                     f"| val loss {va['loss']:.4f} f1 {va['macro_f1']:.4f} acc {va['accuracy']:.4f}")
            if va["macro_f1"] > self._best_val:
                self._best_val = va["macro_f1"]
                self._best_state = copy.deepcopy(self.model.state_dict())
                self._since_improve = 0
            else:
                self._since_improve = getattr(self, "_since_improve", 0) + 1
                if self._since_improve >= self.patience:
                    self.log(f"early stopping at epoch {epoch} (best val macro-F1 {self._best_val:.4f})")
                    break
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return self._best_val

    @torch.no_grad()
    def evaluate(self, loader):
        return self._run_epoch(loader, train=False)
