from config import config
from ultralytics import YOLO
import torch
from extra.utils import get_best_iteration
print("CUDA is available:", torch.cuda.is_available())  # Debe imprimir True si CUDA está disponible
print("Number of GPUs:",torch.cuda.device_count())  # Número de GPUs detectadas
print("CUDA PyTorch-supported version:",torch.version.cuda)  # Versión de CUDA soportada por PyTorch


def run_model(best_trials_path=config.best_trials_path, epochs=128):
    # Get available device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    device = torch.device("cpu")

    # Create model from YOLOv8l
    model = YOLO("yolov8l.pt")
    model.to(device)    # Link to device

    # Get best hyperparameters
    best_hyperparams = get_best_iteration(
        best_trials_path,
        metrics_names=["precision", "recall", "mAP50", "mAP50_95"]
    )

    # Train model
    model.train(
        data=config.yolo_config_path,
        optimizer="AdamW",
        # Load best hyperparameters from Optuna best trials
        lr0=best_hyperparams["lr0"],
        lrf=best_hyperparams["lrf"],
        momentum=best_hyperparams["momentum"],
        weight_decay=best_hyperparams["weight_decay"],
        warmup_epochs=best_hyperparams["warmup_epochs"],
        warmup_momentum=best_hyperparams["warmup_momentum"],
        box=best_hyperparams["box"],
        cls=best_hyperparams["cls"],
        batch=int(best_hyperparams["batch"]),
        # Manage color augmentations for mammograms
        hsv_h=0.0, 
        hsv_s=0.0, 
        hsv_v=0.2,
        # Manage geometric augmentations for mammograms (flips already applied)
        degrees=15.0,
        fliplr=0.0,
        flipud=0.0,
        translate=0.1,
        scale=0.2,
        # Disable inappropriate augmentations for mammograms
        shear=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        # Remaining training parameters
        epochs=epochs,
        imgsz=640,
        device=0,   # Use GPU 0
        workers=1,
        save_period=10, # Save every 10 epochs
        patience=40, # Early stopping if there is no improvement
    )
