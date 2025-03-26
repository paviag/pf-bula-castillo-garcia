from config import config
from ultralytics import YOLO
import torch
from src.extra.utils import get_best_iteration
print(torch.cuda.is_available())  # Debe imprimir True si CUDA está disponible
print(torch.cuda.device_count())  # Número de GPUs detectadas
print(torch.version.cuda)  # Versión de CUDA soportada por PyTorch


def run_model(best_trials_path=config.best_trials_path):
    # Get available device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    device = torch.device("cpu")

    # Create model from YOLOv8n
    model = YOLO("yolov8n.pt")
    model.to(device)    # Link to device

    # Get best hyperparameters
    best_hyperparams = get_best_iteration(
        best_trials_path,
        metrics_names=["precision", "recall", "mAP50", "mAP50_95"]
    )

    # Train model
    model.train(
        data="pf-bula-castillo-garcia\data.yaml",
        optimizer="AdamW",
        lr0=best_hyperparams["lr0"],
        lrf=best_hyperparams["lrf"],
        momentum=best_hyperparams["momentum"],
        weight_decay=best_hyperparams["weight_decay"],
        warmup_epochs=best_hyperparams["warmup_epochs"],
        warmup_momentum=best_hyperparams["warmup_momentum"],
        box=best_hyperparams["box"],
        cls=best_hyperparams["cls"],
        batch=int(best_hyperparams["batch"]),
        epochs=128,
        imgsz=640,
        device=0,
        workers=1,
        patience=10,
    )
