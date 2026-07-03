from pathlib import Path
from torch import nn
import wandb
from .early_stopping import EarlyStopper
from sklearn.metrics import accuracy_score, f1_score
from multiomics_gnn.utils.logger import get_logger
import numpy as np
import torch
import time
import optuna

class Trainer():
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            test_loader,
            optimizer,
            scheduler,
            device,
            args,
            result_path,
            l2_reg_factor=5e-4,
            class_weights=None,
            debug=False,
            logger=None,
            wandb_run=None
            ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        self.result_path = result_path
        self.run_dir = Path(result_path).parent
        self.chk_dir = self.run_dir / "checkpoints"
        self.chk_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run

        self.l2_reg_factor = l2_reg_factor
        self.debug = debug
        self.logger = logger if logger is not None else get_logger(__name__)
        self.logger.info(f"Wandb run: {self.wandb_run}")
        self.start_epoch = 0
        self.early_stopping = None
        if args['early_stopping']['enabled']:
            self.early_stopping = EarlyStopper(args['early_stopping']['strategy'],
                                              args['early_stopping']['patience'])

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": [],
            "lr": [],
            "epoch": []
        }
        self.early_stopping_metric = args['early_stopping'].get('metric', 'accuracy')
        self.logger.info(f"Early stopping enabled: {self.early_stopping is not None}, metric: {self.early_stopping_metric}")
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.logger.info(f"Initialized Trainer with early stopping: {self.early_stopping is not None}, early stopping metric: {self.early_stopping_metric}, class weights: {'set' if class_weights is not None else 'None'}")
    def _process_batch(self, batch):
        batch = batch.to(self.device)
        B = batch.num_graphs
        N = batch.num_nodes // B
        batch_x = batch.x.view(B, N, -1)
        batch_y = batch.y.view(-1).long()
        batch_edge_index = batch.edge_index
        batch_edge_weight = getattr(batch, "edge_weight", None)
        #self.logger.info(f"[DEBUG] batch_x shape: {batch_x.shape}")
        #self.logger.info(f"[DEBUG] batch_y shape: {batch_y.shape}")
        #self.logger.info(f"[DEBUG] edge_index shape: {batch_edge_index.shape}")
        #if batch_edge_weight is None:
        #    self.logger.info("[DEBUG] edge_weight: None")
        #else:
        #    self.logger.info(f"[DEBUG] edge_weight shape: {batch_edge_weight.shape}")

        return batch_x, batch_edge_index, batch_y, batch_edge_weight
    
    def predict(self, loader, load_best: bool = True):
        """
        Ritorna (y_true, y_pred) su qualunque loader (train/val/test).
        Se load_best=True carica i pesi best salvati in self.result_path (come test()).
        """
        if load_best:
            self.model.load_state_dict(torch.load(self.result_path, map_location=self.device))

        self.model.eval()
        all_true, all_pred, all_probs = [], [], []

        with torch.no_grad():
            for batch in loader:
                batch_x, batch_edge_index, batch_y, batch_edge_weight = self._process_batch(batch)

                if self.args["train"]["decoder"] == True:
                    _, out = self.model(batch_x, batch_edge_index, batch_edge_weight)
                else:
                    out = self.model(batch_x, batch_edge_index, batch_edge_weight)

                
                prob = torch.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                
                all_true.append(batch_y.detach().cpu().numpy())
                all_pred.append(preds.detach().cpu().numpy())
                all_probs.append(prob.detach().cpu().numpy())

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        y_prob = np.concatenate(all_probs) if all_probs else None
        return y_true, y_pred, y_prob

    def train_epoch(self):
        self.model.train()
        loss_sum = 0.0
        total_samples = 0
        all_true = []
        all_pred = []
        # --- START: epoch label distribution (after sampler) ---
        num_classes = 27
        epoch_counts = torch.zeros(num_classes, dtype=torch.long)
        # --- END ---
        opt_param_ids = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        missing = [n for n,p in self.model.named_parameters() if id(p) not in opt_param_ids]
        print("NOT IN OPTIMIZER:", missing)

        for i, batch in enumerate(self.train_loader):
            batch_x, batch_edge_index, batch_y, batch_edge_weight = self._process_batch(batch)

            self.optimizer.zero_grad()
            if self.args['train']['decoder'] == True:
                x_reconstruct, out = self.model(batch_x, batch_edge_index, batch_edge_weight)
                # decoder output shapes (omitted per-batch)
                loss_batch = self.model.loss(x_reconstruct, batch_x, out, batch_y, self.l2_reg_factor, class_weights=self.class_weights)
            else:
                out = self.model(batch_x, batch_edge_index, batch_edge_weight)
                loss_batch = self.model.loss(batch_x.view(batch_x.size(0), -1), batch_x, out, batch_y, self.l2_reg_factor, class_weights=self.class_weights)

            loss_batch.backward()
            # --- INIZIO DEBUG GRADIENTE ---
            if i == 0 and self.debug: 
                total_norm = 0.0
                grad_w_ratios = []
                grad_vars = []
                import pandas as pd
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # Calcola la norma del gradiente per questo layer
                        param_norm = param.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                        
                        # Statistiche del gradiente
                        grad_mean = param.grad.data.mean().item()
                        grad_max = param.grad.data.max().item()
                        grad_var = param.grad.data.var().item()
                        
                        self.logger.info(f"Layer: {name:<20} | Norm: {param_norm:.6f} | Mean: {grad_mean:.8f} | Max: {grad_max:.8f}")
                        
                        # Accumula metriche per calcolare le medie
                        weight_norm = param.data.norm(2).item()
                        self.logger.info(f"Layer: {name:<20} | Weight Norm: {weight_norm:.6f}")
                        self.logger.info(f"Layer: {name:<20} | Grad/Weight Ratio: {(param_norm / (weight_norm + 1e-8)):.6f}")
                        self.logger.info(f"Layer: {name:<20} | Grad Variance: {grad_var:.6f}")
                        grad_w_ratios.append(param_norm / (weight_norm + 1e-8))
                        grad_vars.append(param.grad.data.var().item())
                        
                
                # Calcola metriche aggregate
                total_norm = total_norm ** 0.5
                mean_grad_w_ratio = np.mean(grad_w_ratios) if grad_w_ratios else 0.0
                mean_grad_var = np.mean(grad_vars) if grad_vars else 0.0
                
                self.logger.info(f"--> Total Gradient Norm: {total_norm:.6f}")
                self.logger.info(f"--> Gradient-to-Weight Ratio (mean): {mean_grad_w_ratio:.6f}")
                self.logger.info(f"--> Gradient Variance (mean): {mean_grad_var:.6f}")
                
                if total_norm < 1e-5:
                    self.logger.warning("!!! ATTENZIONE: VANISHING GRADIENT (Il modello non impara) !!!")
            # --- FINE DEBUG GRADIENTE ---
        
            
            self.optimizer.step()
            # ------ Debug Sample per epoch all batches ------
             # --- conteggio label viste ---
            if self.debug:
                y = batch.y.detach().view(-1)
                epoch_counts += torch.bincount(y.cpu(), minlength=num_classes)

            #t_model += time.time() - t1
            #self.logger.info(f"[DEBUG] Batch {i+1}/{len(self.train_loader)} - Data time: {t_data:.4f}s, Model time: {t_model:.4f}s")

            bs = int(batch_y.size(0))
            loss_sum += float(loss_batch.item()) * bs
            total_samples += bs

            preds = out.argmax(dim=1)
            all_true.append(batch_y.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())

        # --- fine epoch: log distribuzione label viste ---
        # --- START: log distribution at end of epoch ---
        if self.debug:
            total = int(epoch_counts.sum())
            probs = (epoch_counts.float() / max(total, 1)).tolist()
            self.logger.info(f"[Sampler Debug] Epoch label counts: {epoch_counts.tolist()}")
            self.logger.info(f"[Sampler Debug] Epoch label min/max: {int(epoch_counts.min())}/{int(epoch_counts.max())}")
            # opzionale: salva su file
            with open("epoch_class_counts.txt", "a") as f:
                f.write(f"counts={epoch_counts.tolist()}\n")
                f.write(f"probs={probs}\n\n")
        # --- END ---


        y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
        y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)
        
        avg_loss = loss_sum / total_samples if total_samples > 0 else 0.0
        acc = accuracy_score(y_true, y_pred) if y_true.size else 0.0
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return avg_loss, acc, f1_macro
    
    def eval_epoch(self, loader):
        self.model.eval()
        loss_sum = 0.0
        total_samples = 0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for batch in loader:
                batch_x, batch_edge_index, batch_y, batch_edge_weight = self._process_batch(batch)

                if self.args['train']['decoder'] == True:
                    x_reconstruct, out = self.model(batch_x, batch_edge_index, batch_edge_weight)
                    loss_batch = self.model.loss(x_reconstruct, batch_x, out, batch_y, self.l2_reg_factor, class_weights=self.class_weights)
                else:
                    out = self.model(batch_x, batch_edge_index, batch_edge_weight)
                    loss_batch = self.model.loss(batch_x.view(batch_x.size(0), -1), batch_x, out, batch_y, self.l2_reg_factor, class_weights=self.class_weights)
                
                bs = int(batch_y.size(0))
                loss_sum += float(loss_batch.item()) * bs
                total_samples += bs

                preds = out.argmax(dim=1)
                all_true.append(batch_y.detach().cpu().numpy())
                all_pred.append(preds.detach().cpu().numpy())

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        
        avg_loss = loss_sum / total_samples if total_samples > 0 else 0.0
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return avg_loss, acc, f1_macro

    def test(self):
        self.model.load_state_dict(torch.load(self.result_path, map_location=self.device))
        self.model.eval()
        all_true = []
        all_pred = []
        self.logger.info("Evaluating on test set...")
        self.logger.info(f"Length of test loader: {len(self.test_loader)}")

        with torch.no_grad():
            for batch in self.test_loader:
                batch_x, batch_edge_index, batch_y, batch_edge_weight = self._process_batch(batch)

                if self.args['train']['decoder'] == True:
                    _, out = self.model(batch_x, batch_edge_index, batch_edge_weight)
                else:
                    out = self.model(batch_x, batch_edge_index, batch_edge_weight)
                
                preds = out.argmax(dim=1)
                all_true.append(batch_y.detach().cpu().numpy())
                all_pred.append(preds.detach().cpu().numpy())

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return acc, f1_macro, y_true, y_pred

    def fit(self, trial: "optuna.Trial | None" = None):
        #debug prints

        self.logger.info(f"Saving directory: {self.run_dir}")
        self.logger.info(f"Checkpoint directory: {self.chk_dir}")
        self.logger.info(f"Result path: {self.result_path}") 
        num_epochs = self.args['train']['num_epochs']
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        if self.start_epoch >= num_epochs:
            self.logger.info("Training already completed in previous run.")
            return
        resume_path = self.args["train"].get("resume_checkpoint", None)
        self.logger.info(f"Resume path: {resume_path}")
        if resume_path is not None:
            self.resume_checkpoint(resume_path)

        for epoch in range(self.start_epoch, num_epochs):
            t0 = time.perf_counter()
            train_loss_epoch, train_acc_epoch, train_f1_epoch = self.train_epoch()
            # cambiamento qui: passare val_loader a eval_epoch
            # dato che c'è il dropout dobbiamo chiamare eval_epoch sul train set
            train_loss, train_acc, train_f1 = self.eval_epoch(self.train_loader)
             # valutazione sul validation set
            val_loss, val_acc, val_f1 = self.eval_epoch(self.val_loader)
            # log train with and without dropout per debug
            self.logger.info(f"Epoch {epoch+1} - Train (with dropout) Loss: {train_loss_epoch:.4f}, Acc: {train_acc_epoch:.4f}, F1: {train_f1_epoch:.4f} | "
                  f"Train (eval mode) Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            # OPTUNA: report intermediate value per early stopping di Optuna
            # report a Optuna la metrica che stai ottimizzando
            if trial is not None:
                # se stai massimizzando F1:
                trial.report(float(val_f1), step=epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            t_train = time.perf_counter() - t0
            self.logger.info(f"Epoch {epoch+1} training time: {t_train:.2f} seconds")

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch+1}/{self.args['train']['num_epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.6f}")
            

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)
            self.history['epoch'].append(epoch + 1)
            # wandb logging
            if self.wandb_run:
                self.logger.info("Logging metrics to Wandb")
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": float(train_loss),
                    "train/acc": float(train_acc),
                    "train/f1": float(train_f1),
                    "val/loss": float(val_loss),
                    "val/acc": float(val_acc),
                    "val/f1": float(val_f1),
                    "lr": float(current_lr),
                }, step=epoch + 1)
            # 1) save latest checkpoint
            t1 = time.perf_counter()
            #self.save_checkpoint(f"latest_checkpoint.pt", epoch)
            t_chk = time.perf_counter() - t1
            self.logger.info(f"Epoch {epoch+1} checkpoint saving time: {t_chk:.2f} seconds")

            # 2) early stopping + best model saving
            if self.args["early_stopping"]["enabled"]:
                if self.early_stopping_metric == "accuracy":
                    is_best = self.early_stopping(val_acc, epoch)
                elif self.early_stopping_metric == "f1":
                    is_best = self.early_stopping(val_f1, epoch)
                else:
                    is_best = self.early_stopping(val_loss, epoch)  # default to accuracy

                if is_best:
                # delete previous best model file
                    # if self.best_checkpoint_path is not None and Path(self.best_checkpoint_path).exists():
                    #     Path(self.best_checkpoint_path).unlink()
                    #     self.logger.info(f"Removed previous best checkpoint")


                # best weights "semplice" per il tuo test()
                    self.logger.info(f"New best model found at epoch {epoch+1}!")
                    t2 = time.perf_counter()
                    torch.save(self.model.state_dict(), self.result_path)

                # best checkpoint completo
                # saving best checkpoint name + epoch number
                    checkpoint_name = f"best_checkpoint_epoch_{epoch+1}.pth"
                    self.save_checkpoint(checkpoint_name, epoch)
                    t_best = time.perf_counter() - t2
                    self.logger.info(f"Epoch {epoch+1} best model saving time: {t_best:.2f} seconds")
                # stop
                if self.early_stopping.early_stop:
                    self.logger.info(
                        f"Early stopping at epoch {epoch+1}. Best epoch={self.early_stopping.best_epoch}, best_metric={self.early_stopping.best_metric}"
                    )
                    break

            

    def save_checkpoint(self, checkpoint_name: str, epoch: int):
        if self.early_stopping is None:
            best_metric = None
            best_epoch = None
        else:
            best_metric = self.early_stopping.best_metric
            best_epoch = self.early_stopping.best_epoch
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            #"config": flatten_dict(wandb.config),
            #"wandb_id": wandb.run.id,
        }

        filename = str(self.chk_dir / checkpoint_name)
        torch.save(state, filename)
        self.logger.info(f"Saved checkpoint: {filename}")

    def resume_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.logger.info("Loading checkpoint")
        self.logger.info("Loading Model")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info("Loading Optimizer state dict")
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.early_stopping is not None:
            self.early_stopping.best_metric = checkpoint["best_metric"]
            self.early_stopping.best_epoch = checkpoint["best_epoch"]

        self.logger.info(f"Checkpoint epoch: {int(checkpoint['epoch'])}")
        self.start_epoch = int(checkpoint["epoch"]) + 1 
        self.logger.info(f"Next epoch is: {self.start_epoch + 1}")  