import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4, save_path=None, warmup=0):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.warmup = warmup
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.epoch = 0

    def step(self, metric, model=None):
        self.epoch += 1

        if self.epoch <= self.warmup:
            return

        score = -metric

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)

    def _save_checkpoint(self, model):
        if self.save_path and model:
            torch.save(model.state_dict(), self.save_path)
