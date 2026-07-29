from multiomics_gnn.utils.logger import get_logger
logger = get_logger(__name__)

class EarlyStopper():
    def __init__(self, strategy="minimize", patience=1):
        assert strategy.lower() in [
            "minimize",
            "maximize",
        ], "Please select 'minimize' or 'maximize' for strategy"
        self.strategy = strategy.lower()
        self.patience = int(patience)
        self.best_metric = None
        self.best_epoch = None
        self.early_stop = False
        self.counter = 0
        logger.info(f"EarlyStopper initialized with strategy={self.strategy}, patience={self.patience}")
        

    def __call__(self, metric: float, epoch: int) -> bool:
        """Early stopping update call

        Args:
            metric (float): Metric for early stopping
            epoch (int): Current epoch

        Returns:
            bool: Returns true if the model is performing better than the current best model,
                otherwise false
        """
        if self.best_metric is None:
            self.best_metric = metric
            self.best_epoch = epoch
            return True
        else:
            if self.strategy == "minimize":
                if self.best_metric >= metric:
                    self.best_metric = metric
                    self.best_epoch = epoch
                    self.counter = 0
                    #wandb.run.summary["Best-Epoch"] = epoch
                    #wandb.run.summary["Best-Metric"] = metric
                    return True
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                    return False
            elif self.strategy == "maximize":
                if self.best_metric <= metric:
                    self.best_metric = metric
                    self.best_epoch = epoch
                    self.counter = 0
                    #wandb.run.summary["Best-Epoch"] = epoch
                    #wandb.run.summary["Best-Metric"] = metric
                    return True
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True
                    return False