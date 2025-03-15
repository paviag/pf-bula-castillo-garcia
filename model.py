from ultralytics import YOLO
import torch
from utils import get_best_iteration
print(torch.cuda.is_available())  # Debe imprimir True si CUDA está disponible
print(torch.cuda.device_count())  # Número de GPUs detectadas
print(torch.version.cuda)  # Versión de CUDA soportada por PyTorch

def main():
    model = YOLO("yolov8n.pt") 
    print("model device before", model.device)
    gpu0 = torch.device("cuda:0")
    model.to(gpu0)
    print("model device after", model.device)
    best_hyperparams = get_best_iteration(
        "pf-bula-castillo-garcia/optuna_best_trials.csv",
        metrics_names=["precision", "recall", "mAP50", "mAP50_95"]
        )

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
    
if __name__ == "__main__":
    main()