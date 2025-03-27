import subprocess
from extra.utils import get_best_iteration


class YOLOTrainer:
    """Handles the training process of the YOLO model using Optuna-generated parameters"""

    def __init__(self, config_path, runs_path, train_index_manager, trial_epochs=10):
        self.train_index_manager = train_index_manager
        self.params = ["lr0", "lrf", "momentum", "weight_decay",
                       "warmup_epochs", "warmup_momentum", "box", "cls", "batch"]
        self.trial_epochs = trial_epochs
        self.runs_path = runs_path
        self.config_path = config_path

    def train(self, trial):
        """Trains model for a trial and returns best metrics"""
        train_index = self.train_index_manager.get_index()
        params = self._get_trial_params(trial)
        subprocess.run([
            "yolo", "train",
            f"data={self.config_path}",
            "optimizer=AdamW",
            "model=yolov8n.pt",
            f"epochs={self.trial_epochs}",
            "imgsz=640",
            "device=cuda:0",
            "workers=1"] + [
            f"{p}={params[p]}" for p in self.params
        ])

        best_metrics = get_best_iteration(
            f"{self.runs_path}/train{train_index}/results.csv",
            metrics_names=[
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ],
            return_metrics=True,
        )

        return best_metrics

    def _get_trial_params(self, trial):
        """Gets Optuna suggested params for input trial"""
        return {
            "lr0": trial.suggest_float("lr0", 1e-5, 1e-1, log=True),
            "lrf": trial.suggest_float("lrf", 0.01, 1.0),
            "momentum": trial.suggest_float("momentum", 0.6, 0.98),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.001),
            "warmup_epochs": trial.suggest_float("warmup_epochs", 0.0, 5.0),
            "warmup_momentum": trial.suggest_float("warmup_momentum", 0.0, 0.95),
            "box": trial.suggest_float("box", 0.02, 0.2),
            "cls": trial.suggest_float("cls", 0.2, 4.0),
            "batch": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
        }
