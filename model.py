from ultralytics import YOLO
import torch
from pandas import read_csv
print(torch.cuda.is_available())  # Debe imprimir True si CUDA está disponible
print(torch.cuda.device_count())  # Número de GPUs detectadas
print(torch.version.cuda)  # Versión de CUDA soportada por PyTorch

def get_best_hyperparams(best_trials_path="pf-bula-castillo-garcia/optuna_best_trials.csv"):
    # Read best trials obtained from Optuna as DataFrame
    best_trials = read_csv(best_trials_path)
    
    # Find the maximum value for each metric across all trials
    max_values = [
        max(best_trials.iloc[j][i] for j in range(len(best_trials))) 
        for i in ['precision', 'recall', 'mAP50', 'mAP50_95']
        ]
    
    # For each trial, calculate the "balance score" for metrics
    best_hyperparams = None
    best_balance_score = float('-inf')
    for i in range(len(best_trials)):
        metrics = best_trials.iloc[i][['precision', 'recall', 'mAP50', 'mAP50_95']]
        # Calculate the sum of differences from the max values to determine how close the list of metrics is
        differences = [abs(i - v) for i, v in zip(metrics, max_values)]
        balance_score = -sum(differences)  # Negate the sum of differences (higher means better balance)

        if balance_score > best_balance_score:
            best_balance_score = balance_score
            best_hyperparams = best_trials.iloc[i][['learning_rate', 'batch_size', 'weight_decay']]
    
    return best_hyperparams.to_dict()

def main():
    model = YOLO("yolov8n.pt") 
    print("model device before", model.device)
    gpu0 = torch.device("cuda:0")
    model.to(gpu0)
    print("model device after", model.device)
    best_hyperparams = get_best_hyperparams()
    print(best_hyperparams)

    model.train(
        data="pf-bula-castillo-garcia\data.yaml", 
        optimizer="AdamW",
        lr0=best_hyperparams["learning_rate"],
        batch=int(best_hyperparams["batch_size"]),
        weight_decay=best_hyperparams["weight_decay"],
        epochs=128, 
        imgsz=640, 
        device=0, 
        workers=1,
    )     
    
if __name__ == "__main__":
    main()