import subprocess
from extra.utils import get_best_iteration


class YOLOTrainer:
    """Handles the training process of the YOLO model using Optuna-generated parameters"""

    def __init__(self, *, config_path, runs_path, train_index_manager, trial_epochs=10):
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
            "model=yolov8l.pt",
            f"epochs={self.trial_epochs}",
            "imgsz=640",
            "device=cuda:0",
            # Manage color augmentations for mammograms
            "hsv_h=0.0", 
            "hsv_s=0.0", 
            "hsv_v=0.2",
            # Manage geometric augmentations for mammograms (flips already applied)
            "degrees=15.0",
            "fliplr=0.0",
            "flipud=0.0",
            "translate=0.1",
            "scale=0.2",
            # Disable inappropriate augmentations for mammograms
            "shear=0.0",
            "mosaic=0.0",
            "mixup=0.0",
            "copy_paste=0.0",
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
        
        return list(best_metrics.values())

    def _get_trial_params(self, trial):
        """Gets Optuna suggested params for input trial"""
        return {
            "lr0": trial.suggest_float("lr0", 1e-5, 0.05, log=True), # real range: 1e-5 to 1e-1
            "lrf": trial.suggest_float("lrf", 0.01, 0.05),   # real range: 0.01 to 1.0
            "momentum": trial.suggest_float("momentum", 0.6, 0.98),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.001),
            "warmup_epochs": trial.suggest_float("warmup_epochs", 0.0, 5.0),
            "warmup_momentum": trial.suggest_float("warmup_momentum", 0.0, 0.95),
            "box": trial.suggest_float("box", 0.02, 0.2),
            "cls": trial.suggest_float("cls", 0.2, 4.0),
            "batch": trial.suggest_categorical("batch_size", [4, 8, 16])    # real range: 4 to 32
        }
