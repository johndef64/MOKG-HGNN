"""Training / validation / test loop for the heterogeneous model.

Graph classification with macro-F1 as the primary metric (proposta sez. 5).
Kept intentionally small and dependency-light; mirrors the role of MOGNN-TF's
Trainer without pulling in its BRCA-specific machinery.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score


def _fmt_hms(seconds):
    """Seconds -> compact h/m/s string for logs (e.g. '1h03m', '4m12s', '38s')."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


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
        self.history = []  # per-epoch metrics, for logging/plots

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
        fit_start = time.perf_counter()
        for epoch in range(1, num_epochs + 1):
            t0 = time.perf_counter()
            tr = self._run_epoch(train_loader, train=True)
            va = self._run_epoch(val_loader, train=False)
            if self.scheduler is not None:
                # reduce_on_plateau needs the monitored metric; drive it on val loss
                if getattr(self.scheduler, "is_plateau", False):
                    self.scheduler.metric = va["loss"]
                self.scheduler.step()
            epoch_time = time.perf_counter() - t0
            # ETA from the running average epoch time over the full schedule
            avg = (time.perf_counter() - fit_start) / epoch
            eta = avg * (num_epochs - epoch)
            self.history.append({
                "epoch": epoch, "train_loss": tr["loss"], "train_macro_f1": tr["macro_f1"],
                "val_loss": va["loss"], "val_macro_f1": va["macro_f1"], "val_accuracy": va["accuracy"],
                "epoch_seconds": round(epoch_time, 2)})
            self.log(f"epoch {epoch:03d} | train loss {tr['loss']:.4f} f1 {tr['macro_f1']:.4f} "
                     f"| val loss {va['loss']:.4f} f1 {va['macro_f1']:.4f} acc {va['accuracy']:.4f} "
                     f"| {epoch_time:.1f}s/epoch ETA {_fmt_hms(eta)}")
            if va["macro_f1"] > self._best_val:
                self._best_val = va["macro_f1"]
                self._best_state = copy.deepcopy(self.model.state_dict())
                self._since_improve = 0
            else:
                self._since_improve = getattr(self, "_since_improve", 0) + 1
                if self._since_improve >= self.patience:
                    self.log(f"early stopping at epoch {epoch} (best val macro-F1 {self._best_val:.4f})")
                    break
        total = time.perf_counter() - fit_start
        self.log(f"training done in {_fmt_hms(total)} "
                 f"({len(self.history)} epochs, {total / max(len(self.history), 1):.1f}s/epoch avg)")
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return self._best_val

    @torch.no_grad()
    def evaluate(self, loader):
        return self._run_epoch(loader, train=False)

    @torch.no_grad()
    def predict(self, loader):
        """Return (y_true, y_pred) over the loader, for per-class metrics."""
        self.model.eval()
        ys, preds = [], []
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            ys.append(batch.y.cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(ys), np.concatenate(preds)
